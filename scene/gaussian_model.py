#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation

class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, sh_degree : int):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._normal = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)




        self._adaptive_gate = torch.empty(0)
        self._det_support = torch.empty(0)
        self._edge_score = torch.empty(0)
        self._mv_reliability = torch.empty(0)
        self._det_class = torch.empty(0, dtype=torch.long)
        self._residual_ema = torch.empty(0)
        self._hf_ema = torch.empty(0)
        self._residual_count = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.small_gaussian = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()

    def capture(self):
        return {
            "active_sh_degree": self.active_sh_degree,
            "xyz": self._xyz,
            "normal": self._normal,
            "features_dc": self._features_dc,
            "features_rest": self._features_rest,
            "scaling": self._scaling,
            "rotation": self._rotation,
            "opacity": self._opacity,
            "adaptive_gate": self.get_adaptive_gate,
            "det_support": self.get_det_support,
            "edge_score": self.get_edge_score,
            "mv_reliability": self.get_mv_reliability,
            "det_class": self.get_det_class,
            "residual_ema": self.get_residual_ema,
            "hf_ema": self.get_hf_ema,
            "residual_count": self.get_residual_count,
            "max_radii2D": self.max_radii2D,
            "xyz_gradient_accum": self.xyz_gradient_accum,
            "denom": self.denom,
            "optimizer": self.optimizer.state_dict(),
            "spatial_lr_scale": self.spatial_lr_scale,
        }

    def restore(self, model_args, training_args, finetune):
        if not isinstance(model_args, dict):
            raise TypeError("DAGS checkpoints must store the model state as a dictionary.")

        required = {
            "active_sh_degree", "xyz", "normal", "features_dc", "features_rest",
            "scaling", "rotation", "opacity", "adaptive_gate", "det_support",
            "edge_score", "mv_reliability", "det_class", "residual_ema",
            "hf_ema", "residual_count", "max_radii2D", "xyz_gradient_accum",
            "denom", "optimizer", "spatial_lr_scale",
        }
        missing = sorted(required.difference(model_args))
        if missing:
            raise KeyError(f"Incomplete DAGS checkpoint; missing keys: {missing}")

        self.active_sh_degree = model_args["active_sh_degree"]
        self._xyz = model_args["xyz"]
        self._normal = model_args["normal"]
        self._features_dc = model_args["features_dc"]
        self._features_rest = model_args["features_rest"]
        self._scaling = model_args["scaling"]
        self._rotation = model_args["rotation"]
        self._opacity = model_args["opacity"]
        self._adaptive_gate = model_args["adaptive_gate"]
        self._det_support = model_args["det_support"]
        self._edge_score = model_args["edge_score"]
        self._mv_reliability = model_args["mv_reliability"]
        self._det_class = model_args["det_class"]
        self._residual_ema = model_args["residual_ema"]
        self._hf_ema = model_args["hf_ema"]
        self._residual_count = model_args["residual_count"]
        self.max_radii2D = model_args["max_radii2D"]
        xyz_gradient_accum = model_args["xyz_gradient_accum"]
        denom = model_args["denom"]
        optimizer_state = model_args["optimizer"]
        self.spatial_lr_scale = model_args["spatial_lr_scale"]

        self.training_setup(training_args, finetune=finetune)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(optimizer_state)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_normal(self):
        return self._normal
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    def _metadata_or_default(self, tensor, width=1, value=0.0, dtype=torch.float32):
        n = self._xyz.shape[0]
        expected = (n,) if width == 0 else (n, width)
        if tensor.numel() == 0 or tuple(tensor.shape) != expected:
            return torch.full(expected, value, dtype=dtype, device=self._xyz.device)
        return tensor.to(device=self._xyz.device)

    def _initialize_detection_metadata(self, count, gate_value=1.0):
        device = self._xyz.device
        self._adaptive_gate = torch.full((count,), float(gate_value), device=device)
        self._det_support = torch.zeros((count, 1), device=device)
        self._edge_score = torch.zeros((count, 1), device=device)
        self._mv_reliability = torch.zeros((count, 1), device=device)
        self._det_class = torch.full((count, 1), -1, dtype=torch.long, device=device)
        self._residual_ema = torch.zeros((count, 1), device=device)
        self._hf_ema = torch.zeros((count, 1), device=device)
        self._residual_count = torch.zeros((count, 1), device=device)

    @property
    def get_adaptive_gate(self):
        return self._metadata_or_default(self._adaptive_gate, width=0, value=1.0)

    @property
    def get_det_support(self):
        return self._metadata_or_default(self._det_support)

    @property
    def get_edge_score(self):
        return self._metadata_or_default(self._edge_score)

    @property
    def get_mv_reliability(self):
        return self._metadata_or_default(self._mv_reliability)

    @property
    def get_det_class(self):
        return self._metadata_or_default(self._det_class, value=-1, dtype=torch.long)

    @property
    def get_residual_ema(self):
        return self._metadata_or_default(self._residual_ema)

    @property
    def get_hf_ema(self):
        return self._metadata_or_default(self._hf_ema)

    @property
    def get_residual_count(self):
        return self._metadata_or_default(self._residual_count)

    def set_adaptive_gate(self, gate):
        gate = gate.to(device=self._xyz.device, dtype=torch.float32).reshape(-1)
        if gate.shape[0] != self._xyz.shape[0]:
            raise ValueError(f"adaptive_gate length {gate.shape[0]} does not match Gaussian count {self._xyz.shape[0]}")
        self._adaptive_gate = gate.clamp(0.0, 1.0).contiguous()

    def set_detection_metadata(self, det_support, edge_score, mv_reliability, det_class, adaptive_gate):
        n = self._xyz.shape[0]
        tensors = [det_support, edge_score, mv_reliability, det_class]
        if any(t.shape[0] != n for t in tensors):
            raise ValueError("Detection metadata must match the current Gaussian count")
        self._det_support = det_support.to(self._xyz.device, torch.float32).reshape(n, 1).contiguous()
        self._edge_score = edge_score.to(self._xyz.device, torch.float32).reshape(n, 1).contiguous()
        self._mv_reliability = mv_reliability.to(self._xyz.device, torch.float32).reshape(n, 1).contiguous()
        self._det_class = det_class.to(self._xyz.device, torch.long).reshape(n, 1).contiguous()
        self.set_adaptive_gate(adaptive_gate)

    @torch.no_grad()
    def update_residual_ema(self, indices, residual_values, hf_values, decay=0.9):
        if indices.numel() == 0:
            return
        decay = float(max(0.0, min(0.9999, decay)))
        residual = self.get_residual_ema
        hf = self.get_hf_ema
        count = self.get_residual_count
        residual[indices] = decay * residual[indices] + (1.0 - decay) * residual_values.to(residual)
        hf[indices] = decay * hf[indices] + (1.0 - decay) * hf_values.to(hf)
        count[indices] += 1.0
        self._residual_ema, self._hf_ema, self._residual_count = residual, hf, count
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 2), dtype=torch.float, device="cuda"))
        normal =  0.6*torch.ones((fused_point_cloud.shape[0], 3), dtype=torch.float, device="cuda")

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._normal = nn.Parameter(normal.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self._initialize_detection_metadata(fused_point_cloud.shape[0], gate_value=1.0)
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def training_setup(self, training_args, normal_lr=0.003, finetune=False):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")


        if finetune:
            l = [
                {'params': [self._normal], 'lr': 0.003, "name": "normal"},
                {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            ]
        else:
            l = [
                {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
                {'params': [self._normal], 'lr': 0.003, "name": "normal"},
                {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
                {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
                {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
                {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
                {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
            ]
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

    def update_learning_rate(self, iteration):
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz', 'adaptive_gate', 'det_support', 'edge_score', 'mv_reliability', 'det_class', 'residual_ema', 'hf_ema']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity1')
        l.append('opacity2')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normal = self._normal.detach().cpu().numpy()
        metadata = np.concatenate((
            self.get_adaptive_gate.detach().cpu().numpy()[:, None],
            self.get_det_support.detach().cpu().numpy(),
            self.get_edge_score.detach().cpu().numpy(),
            self.get_mv_reliability.detach().cpu().numpy(),
            self.get_det_class.detach().to(torch.float32).cpu().numpy(),
            self.get_residual_ema.detach().cpu().numpy(),
            self.get_hf_ema.detach().cpu().numpy(),
        ), axis=1)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normal, metadata, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.02))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def generate_small_cov(self, cov_input, n):

        cov_input = cov_input.to(torch.float64)
        n = n.to(torch.float64)

        num = cov_input.shape[0]

        n_norm = torch.norm(n, dim=1, keepdim=True)
        n = n / n_norm

        device = n.device


        cov = torch.zeros((num, 3, 3), dtype=torch.float64).to(device)
        cov[:, 0, 0] = cov_input[:, 0]
        cov[:, 0, 1] = cov[:, 1, 0] = cov_input[:, 1]
        cov[:, 0, 2] = cov[:, 2, 0] = cov_input[:, 2]
        cov[:, 1, 1] = cov_input[:, 3]
        cov[:, 1, 2] = cov[:, 2, 1] = cov_input[:, 4]
        cov[:, 2, 2] = cov_input[:, 5]



        v1 = torch.zeros((num, 3), dtype=torch.float64).to(device)
        v2 = torch.zeros((num, 3), dtype=torch.float64).to(device)

        zero_indices = (n[:, 0] == 0) & (n[:, 1] == 0)

        v1[zero_indices] = torch.tensor([1, 0, 0], dtype=torch.float64).to(device)
        v2[zero_indices] = torch.tensor([0, 1, 0], dtype=torch.float64).to(device)


        v1[~zero_indices] = torch.stack(
            [n[~zero_indices][:, 1], -n[~zero_indices][:, 0], torch.zeros_like(n[~zero_indices][:, 0]).to(device)],
            dim=1)
        v1[~zero_indices] = v1[~zero_indices] / torch.norm(v1[~zero_indices], dim=1, keepdim=True)
        v2[~zero_indices] = torch.cross(n[~zero_indices], v1[~zero_indices])
        v2[~zero_indices] = v2[~zero_indices] / torch.norm(v2[~zero_indices], dim=1, keepdim=True)


        R_transform = torch.stack([v1, v2, n], dim=2)
        basis = R_transform.transpose(1, 2)


        cov_transformed = torch.matmul(basis, torch.matmul(cov, R_transform))
        cov_inv2 = torch.linalg.inv(cov_transformed)


        twoD_mat = cov_inv2[:, :2, :2]

        eig_value, eig_vector = torch.linalg.eig(twoD_mat)

        eig_value = eig_value.real
        eig_vector = eig_vector.real
        eig_value_sorted, indices = torch.sort(eig_value, descending=False)
        eig_vector_sorted = torch.gather(eig_vector, 2, indices.unsqueeze(1).expand(-1, eig_vector.size(1), -1))

        lambda1 = eig_value_sorted[:, 0]
        lambda2 = eig_value_sorted[:, 1]

        v1_2d = eig_vector_sorted[:, :, 0]
        v2_2d = eig_vector_sorted[:, :, 1]


        v1_2d = torch.stack([v1_2d[:, 0], v1_2d[:, 1], torch.zeros_like(v1_2d[:, 0])], dim=1)
        v2_2d = torch.stack([v2_2d[:, 0], v2_2d[:, 1], torch.zeros_like(v2_2d[:, 0])], dim=1)

        v3_2d = torch.tensor([0, 0, 1], dtype=torch.float64).expand(num, -1).to(device)

        v_3d = torch.stack([v1_2d, v2_2d, v3_2d], dim=2)

        lam_mat = torch.zeros((num, 3, 3), dtype=torch.float64).to(device)

        lam_mat[:, 0, 0] = 1 / lambda1
        lam_mat[:, 1, 1] = 1 / lambda2
        lam_mat[:, 2, 2] = 0.001 / torch.max(lambda1, lambda2)


        cov_new_3d = torch.matmul(v_3d, torch.matmul(lam_mat, v_3d.transpose(1, 2)))

        cov_new_3d = torch.matmul(R_transform, torch.matmul(cov_new_3d, basis))


        cov_six_elements = torch.zeros((cov_new_3d.shape[0], 6), dtype=torch.float64, device=cov_new_3d.device)
        cov_six_elements[:, 0] = cov_new_3d[:, 0, 0]
        cov_six_elements[:, 1] = cov_new_3d[:, 0, 1]
        cov_six_elements[:, 2] = cov_new_3d[:, 0, 2]
        cov_six_elements[:, 3] = cov_new_3d[:, 1, 1]
        cov_six_elements[:, 4] = cov_new_3d[:, 1, 2]
        cov_six_elements[:, 5] = cov_new_3d[:, 2, 2]

        return cov_six_elements.float()

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        property_names = {p.name for p in plydata.elements[0].properties}
        if "adaptive_gate" in property_names:
            adaptive_gate = np.asarray(plydata.elements[0]["adaptive_gate"], dtype=np.float32)
        else:
            adaptive_gate = np.ones((xyz.shape[0],), dtype=np.float32)
        def read_meta(name, default, dtype=np.float32):
            if name in property_names:
                return np.asarray(plydata.elements[0][name], dtype=dtype)
            return np.full((xyz.shape[0],), default, dtype=dtype)
        det_support = read_meta("det_support", 0.0)
        edge_score = read_meta("edge_score", 0.0)
        mv_reliability = read_meta("mv_reliability", 0.0)
        det_class = read_meta("det_class", -1.0, np.float32).astype(np.int64)
        residual_ema = read_meta("residual_ema", 0.0)
        hf_ema = read_meta("hf_ema", 0.0)

        if "opacity1" in property_names:
            normal = np.stack((np.asarray(plydata.elements[0]["nx"]),
                            np.asarray(plydata.elements[0]["ny"]),
                            np.asarray(plydata.elements[0]["nz"])),  axis=1)
            opacities1 = np.asarray(plydata.elements[0]["opacity1"])
            opacities2 = np.asarray(plydata.elements[0]["opacity2"])
        else:
            normal = np.ones(np.shape(xyz))*0.6

            opacities1 = np.asarray(plydata.elements[0]["opacity"])
            opacities2 = np.asarray(plydata.elements[0]["opacity"])

        opacities = np.stack((opacities1,opacities2), axis=1)

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._normal = nn.Parameter(torch.tensor(normal, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._adaptive_gate = torch.tensor(adaptive_gate, dtype=torch.float32, device="cuda")
        self._det_support = torch.tensor(det_support[:, None], dtype=torch.float32, device="cuda")
        self._edge_score = torch.tensor(edge_score[:, None], dtype=torch.float32, device="cuda")
        self._mv_reliability = torch.tensor(mv_reliability[:, None], dtype=torch.float32, device="cuda")
        self._det_class = torch.tensor(det_class[:, None], dtype=torch.long, device="cuda")
        self._residual_ema = torch.tensor(residual_ema[:, None], dtype=torch.float32, device="cuda")
        self._hf_ema = torch.tensor(hf_ema[:, None], dtype=torch.float32, device="cuda")
        self._residual_count = torch.zeros((xyz.shape[0], 1), dtype=torch.float32, device="cuda")
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

        self.small_gaussian = self.generate_small_cov(self.get_covariance(), self._normal)


    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        metadata = {
            "adaptive_gate": self.get_adaptive_gate[valid_points_mask].contiguous(),
            "det_support": self.get_det_support[valid_points_mask].contiguous(),
            "edge_score": self.get_edge_score[valid_points_mask].contiguous(),
            "mv_reliability": self.get_mv_reliability[valid_points_mask].contiguous(),
            "det_class": self.get_det_class[valid_points_mask].contiguous(),
            "residual_ema": self.get_residual_ema[valid_points_mask].contiguous(),
            "hf_ema": self.get_hf_ema[valid_points_mask].contiguous(),
            "residual_count": self.get_residual_count[valid_points_mask].contiguous(),
        }
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._normal = optimizable_tensors["normal"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._adaptive_gate = metadata["adaptive_gate"]
        self._det_support = metadata["det_support"]
        self._edge_score = metadata["edge_score"]
        self._mv_reliability = metadata["mv_reliability"]
        self._det_class = metadata["det_class"]
        self._residual_ema = metadata["residual_ema"]
        self._hf_ema = metadata["hf_ema"]
        self._residual_count = metadata["residual_count"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_normal, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_metadata):
        d = {"xyz": new_xyz,
        "normal": new_normal,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        old_metadata = {
            "adaptive_gate": self.get_adaptive_gate,
            "det_support": self.get_det_support,
            "edge_score": self.get_edge_score,
            "mv_reliability": self.get_mv_reliability,
            "det_class": self.get_det_class,
            "residual_ema": self.get_residual_ema,
            "hf_ema": self.get_hf_ema,
            "residual_count": self.get_residual_count,
        }
        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._normal = optimizable_tensors["normal"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        for name, old_tensor in old_metadata.items():
            extension = new_metadata[name].to(device=old_tensor.device, dtype=old_tensor.dtype)
            setattr(self, "_" + name, torch.cat((old_tensor, extension), dim=0).contiguous())

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)


        new_normal = self._normal[selected_pts_mask].repeat(N, 1)


        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)


        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        new_metadata = {
            "adaptive_gate": self.get_adaptive_gate[selected_pts_mask].repeat(N),
            "det_support": self.get_det_support[selected_pts_mask].repeat(N, 1),
            "edge_score": self.get_edge_score[selected_pts_mask].repeat(N, 1),
            "mv_reliability": self.get_mv_reliability[selected_pts_mask].repeat(N, 1),
            "det_class": self.get_det_class[selected_pts_mask].repeat(N, 1),
            "residual_ema": torch.zeros_like(self.get_residual_ema[selected_pts_mask]).repeat(N, 1),
            "hf_ema": torch.zeros_like(self.get_hf_ema[selected_pts_mask]).repeat(N, 1),
            "residual_count": torch.zeros_like(self.get_residual_count[selected_pts_mask]).repeat(N, 1),
        }

        self.densification_postfix(new_xyz, new_normal, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_metadata)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_normal = self._normal[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]

        new_opacities = inverse_sigmoid(
            1 - torch.sqrt(1 - torch.sigmoid(new_opacities)))
        self._opacity[selected_pts_mask].copy_(new_opacities.detach())

        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_metadata = {
            "adaptive_gate": self.get_adaptive_gate[selected_pts_mask],
            "det_support": self.get_det_support[selected_pts_mask],
            "edge_score": self.get_edge_score[selected_pts_mask],
            "mv_reliability": self.get_mv_reliability[selected_pts_mask],
            "det_class": self.get_det_class[selected_pts_mask],
            "residual_ema": torch.zeros_like(self.get_residual_ema[selected_pts_mask]),
            "hf_ema": torch.zeros_like(self.get_hf_ema[selected_pts_mask]),
            "residual_count": torch.zeros_like(self.get_residual_count[selected_pts_mask]),
        }

        self.densification_postfix(new_xyz, new_normal, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_metadata)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, det_densify_score=None):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0




        if det_densify_score is not None:
            det_densify_score = det_densify_score.to(device=grads.device, dtype=grads.dtype)
            if det_densify_score.dim() == 1:
                det_densify_score = det_densify_score[:, None]
            if det_densify_score.shape[0] < grads.shape[0]:
                pad = torch.ones((grads.shape[0] - det_densify_score.shape[0], 1), device=grads.device, dtype=grads.dtype)
                det_densify_score = torch.cat([det_densify_score, pad], dim=0)
            elif det_densify_score.shape[0] > grads.shape[0]:
                det_densify_score = det_densify_score[:grads.shape[0]]
            grads = grads * det_densify_score

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        max_elements, max_idxs = torch.max(self.get_opacity,dim=1)

        prune_mask = (max_elements < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)


        self.prune_points(prune_mask)
        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1
