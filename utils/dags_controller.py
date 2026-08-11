
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class DetectionBox:
    class_id: int
    xc: float
    yc: float
    width: float
    height: float
    confidence: float = 1.0


class DetectionLabelStore:

    def __init__(self, label_dirs: Sequence[str], expand_ratio: float = 0.0):
        self.label_dirs = [Path(p) for p in label_dirs]
        self.expand_ratio = float(expand_ratio)
        self.index: Dict[str, Path] = {}
        self.cache: Dict[str, List[DetectionBox]] = {}
        for directory in self.label_dirs:
            if not directory.exists():
                print(f"[DAGS][Warning] label directory does not exist: {directory}")
                continue
            for txt in directory.rglob("*.txt"):
                self.index[txt.stem] = txt

    def read(self, image_name: str) -> List[DetectionBox]:
        stem = Path(str(image_name)).stem
        if stem in self.cache:
            return self.cache[stem]
        path = self.index.get(stem)
        boxes: List[DetectionBox] = []
        if path is not None and path.exists():
            for raw in path.read_text(encoding="utf-8").splitlines():
                parts = raw.strip().split()
                if len(parts) < 5:
                    continue
                try:
                    class_id = int(float(parts[0]))
                    xc, yc, width, height = map(float, parts[1:5])
                    confidence = float(parts[5]) if len(parts) >= 6 else 1.0
                except ValueError:
                    continue
                if width <= 0.0 or height <= 0.0:
                    continue
                boxes.append(
                    DetectionBox(
                        class_id=class_id,
                        xc=max(0.0, min(1.0, xc)),
                        yc=max(0.0, min(1.0, yc)),
                        width=max(0.0, min(1.0, width * (1.0 + self.expand_ratio))),
                        height=max(0.0, min(1.0, height * (1.0 + self.expand_ratio))),
                        confidence=max(0.0, min(1.0, confidence)),
                    )
                )
        self.cache[stem] = boxes
        return boxes

    @staticmethod
    def normalized_bounds(box: DetectionBox) -> Tuple[float, float, float, float]:
        x1 = max(0.0, box.xc - box.width / 2.0)
        x2 = min(1.0, box.xc + box.width / 2.0)
        y1 = max(0.0, box.yc - box.height / 2.0)
        y2 = min(1.0, box.yc + box.height / 2.0)
        return x1, y1, x2, y2

    def pixel_bounds(self, box: DetectionBox, height: int, width: int) -> Tuple[int, int, int, int]:
        x1n, y1n, x2n, y2n = self.normalized_bounds(box)
        x1 = max(0, min(width, int(math.floor(x1n * width))))
        x2 = max(0, min(width, int(math.ceil(x2n * width))))
        y1 = max(0, min(height, int(math.floor(y1n * height))))
        y2 = max(0, min(height, int(math.ceil(y2n * height))))
        return x1, y1, x2, y2


def _sobel_magnitude(image: torch.Tensor) -> torch.Tensor:
    if image.dim() != 3:
        raise ValueError(f"Expected CHW image, got shape {tuple(image.shape)}")
    gray = image.mean(dim=0, keepdim=True).unsqueeze(0)
    kernel_x = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        device=image.device,
        dtype=image.dtype,
    ).view(1, 1, 3, 3)
    kernel_y = kernel_x.transpose(2, 3)
    gx = F.conv2d(gray, kernel_x, padding=1)
    gy = F.conv2d(gray, kernel_y, padding=1)
    magnitude = torch.sqrt(gx.square() + gy.square() + 1e-12)[0]
    flat = magnitude.flatten()
    if flat.numel() > 0:
        q95 = torch.quantile(flat, 0.95).clamp_min(1e-6)
        magnitude = (magnitude / q95).clamp(0.0, 1.0)
    return magnitude


def _robust_normalize(values: torch.Tensor) -> torch.Tensor:
    values = torch.nan_to_num(values.float(), nan=0.0, posinf=0.0, neginf=0.0)
    if values.numel() == 0:
        return values
    flat = values.reshape(-1)
    positive = flat[flat > 0]
    if positive.numel() < 8:
        maximum = flat.max().clamp_min(1e-6)
        return (values / maximum).clamp(0.0, 1.0)
    q05 = torch.quantile(positive, 0.05)
    q95 = torch.quantile(positive, 0.95).clamp_min(q05 + 1e-6)
    return ((values - q05) / (q95 - q05)).clamp(0.0, 1.0)


def _project_points(xyz: torch.Tensor, camera) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    n = xyz.shape[0]
    ones = torch.ones((n, 1), dtype=xyz.dtype, device=xyz.device)
    homo = torch.cat((xyz, ones), dim=1)
    clip = homo @ camera.full_proj_transform.to(device=xyz.device, dtype=xyz.dtype)
    positive = clip[:, 3] > 1e-7
    safe_w = torch.where(positive, clip[:, 3], torch.ones_like(clip[:, 3]))
    ndc = clip[:, :3] / safe_w[:, None]
    width = float(camera.image_width)
    height = float(camera.image_height)
    px = (ndc[:, 0] * 0.5 + 0.5) * width
    py = (0.5 - ndc[:, 1] * 0.5) * height
    visible = positive & (px >= 0.0) & (px < width) & (py >= 0.0) & (py < height)
    return px, py, visible


def _sample_map_nearest(value_map: torch.Tensor, px: torch.Tensor, py: torch.Tensor) -> torch.Tensor:
    height, width = value_map.shape[-2:]
    xi = px.round().long().clamp(0, width - 1)
    yi = py.round().long().clamp(0, height - 1)
    return value_map[0, yi, xi]


class DAGSController:

    def __init__(self, cameras: Sequence, label_dirs: Sequence[str], args):
        self.cameras = list(cameras)
        self.args = args
        self.labels = DetectionLabelStore(label_dirs, expand_ratio=args.dags_box_expand_ratio)
        self.num_classes = max(1, int(args.dags_num_classes))
        self.class_densify_weights = self._parse_class_weights(args.dags_class_densify_weights)
        self.neighbors = self._build_neighbor_index(self.cameras, max_neighbors=max(1, args.dags_neighbor_count))
        self.last_evidence_iteration = -1
        self.evidence_update_count = 0

    def _evidence_cameras_for_update(self) -> List[object]:
        total = len(self.cameras)
        requested = int(getattr(self.args, 'dags_evidence_views_per_update', 48))
        if requested <= 0 or requested >= total:
            return self.cameras
        requested = max(1, requested)
        offset = self.evidence_update_count % total

        indices = []
        seen = set()
        for k in range(requested):
            idx = int(round(k * total / float(requested))) % total
            idx = (idx + offset) % total
            if idx not in seen:
                seen.add(idx)
                indices.append(idx)
        return [self.cameras[i] for i in indices]

    def _parse_class_weights(self, text: str) -> torch.Tensor:
        raw = [item.strip() for item in str(text).split(",") if item.strip()]
        values = []
        for item in raw:
            try:
                values.append(max(1.0, float(item)))
            except ValueError:
                pass
        if not values:
            values = [1.5]
        if len(values) < self.num_classes:
            values.extend([values[-1]] * (self.num_classes - len(values)))
        return torch.tensor(values[: self.num_classes], dtype=torch.float32)

    @staticmethod
    def _build_neighbor_index(cameras: Sequence, max_neighbors: int) -> Dict[int, List[int]]:
        if not cameras:
            return {}
        centers = torch.stack([cam.camera_center.detach().float().cpu() for cam in cameras], dim=0)
        distances = torch.cdist(centers, centers)
        result: Dict[int, List[int]] = {}
        for i in range(len(cameras)):
            order = torch.argsort(distances[i]).tolist()
            result[i] = [j for j in order if j != i][:max_neighbors]
        return result

    def choose_neighbor(self, camera) -> Optional[object]:
        if len(self.cameras) < 2:
            return None
        index = None
        for i, candidate in enumerate(self.cameras):
            if candidate.uid == camera.uid:
                index = i
                break
        if index is None:
            return None
        candidates = self.neighbors.get(index, [])
        if not candidates:
            return None

        target = candidates[(self.last_evidence_iteration if self.last_evidence_iteration >= 0 else 0) % len(candidates)]
        return self.cameras[target]

    @torch.no_grad()
    def update_global_evidence(self, gaussians, iteration: int) -> None:
        xyz = gaussians.get_xyz.detach()
        device = xyz.device
        count = xyz.shape[0]
        if count == 0:
            return

        visible_count = torch.zeros((count,), device=device)
        hit_count = torch.zeros((count,), device=device)
        confidence_sum = torch.zeros((count,), device=device)
        edge_sum = torch.zeros((count,), device=device)
        class_votes = torch.zeros((count, self.num_classes), device=device)
        chunk_size = max(10_000, int(self.args.dags_projection_chunk))

        evidence_cameras = self._evidence_cameras_for_update()
        for camera in evidence_cameras:
            boxes = self.labels.read(camera.image_name)

            if camera.original_image is None:
                continue
            edge_map = _sobel_magnitude(camera.original_image.to(device))
            width = float(camera.image_width)
            height = float(camera.image_height)

            for start in range(0, count, chunk_size):
                end = min(count, start + chunk_size)
                px, py, visible = _project_points(xyz[start:end], camera)
                if not visible.any():
                    continue
                local_indices = torch.arange(start, end, device=device)
                global_visible = local_indices[visible]
                visible_count[global_visible] += 1.0
                edge_values = _sample_map_nearest(edge_map, px[visible], py[visible])
                edge_sum[global_visible] += edge_values

                if not boxes:
                    continue
                local_best_conf = torch.zeros((end - start,), device=device)
                local_best_class = torch.full((end - start,), -1, dtype=torch.long, device=device)
                px_norm = px / max(width, 1.0)
                py_norm = py / max(height, 1.0)
                for box in boxes:
                    x1, y1, x2, y2 = self.labels.normalized_bounds(box)
                    inside = visible & (px_norm >= x1) & (px_norm < x2) & (py_norm >= y1) & (py_norm < y2)
                    better = inside & (box.confidence > local_best_conf)
                    if better.any():
                        local_best_conf[better] = float(box.confidence)
                        local_best_class[better] = int(box.class_id)
                hit = local_best_class >= 0
                if hit.any():
                    global_hit = local_indices[hit]
                    conf = local_best_conf[hit]
                    hit_count[global_hit] += 1.0
                    confidence_sum[global_hit] += conf
                    classes = local_best_class[hit].clamp(0, self.num_classes - 1)
                    class_votes.index_put_((global_hit, classes), conf, accumulate=True)

        safe_visible = visible_count.clamp_min(1.0)
        det_support = (confidence_sum / safe_visible).clamp(0.0, 1.0)
        edge_score = (edge_sum / safe_visible).clamp(0.0, 1.0)
        max_vote, class_id = class_votes.max(dim=1)
        total_vote = class_votes.sum(dim=1).clamp_min(1e-6)
        class_agreement = torch.where(hit_count > 0, max_vote / total_vote, torch.ones_like(max_vote))
        visibility_reliability = (visible_count / max(1.0, float(self.args.dags_min_visible_views))).clamp(0.0, 1.0)
        hit_reliability = (hit_count / max(1.0, float(self.args.dags_min_detection_views))).clamp(0.0, 1.0)


        mv_reliability = visibility_reliability * hit_reliability * class_agreement
        class_id = torch.where(hit_count > 0, class_id, torch.full_like(class_id, -1))





        evidence_decay = float(getattr(self.args, 'dags_evidence_ema_decay', 0.70))
        evidence_decay = min(0.99, max(0.0, evidence_decay))
        if self.evidence_update_count > 0 and gaussians.get_det_support.shape[0] == count:
            old_det = gaussians.get_det_support.squeeze(-1).clamp(0.0, 1.0)
            old_edge = gaussians.get_edge_score.squeeze(-1).clamp(0.0, 1.0)
            old_rel = gaussians.get_mv_reliability.squeeze(-1).clamp(0.0, 1.0)
            old_class = gaussians.get_det_class.squeeze(-1).long()
            det_support = evidence_decay * old_det + (1.0 - evidence_decay) * det_support
            edge_score = evidence_decay * old_edge + (1.0 - evidence_decay) * edge_score
            mv_reliability = evidence_decay * old_rel + (1.0 - evidence_decay) * mv_reliability
            class_id = torch.where(hit_count > 0, class_id, old_class)

        residual = _robust_normalize(gaussians.get_residual_ema.squeeze(-1))
        high_frequency = _robust_normalize(gaussians.get_hf_ema.squeeze(-1))



        outside = max(0.0, min(1.0, float(self.args.dags_outside_gate_allowance)))
        det_envelope = outside + (1.0 - outside) * torch.sqrt(det_support.clamp(0.0, 1.0))
        reliable_det = det_support * (0.25 + 0.75 * mv_reliability)
        logits = (
            float(self.args.dags_gate_beta_det) * reliable_det
            + float(self.args.dags_gate_beta_edge) * edge_score * det_envelope
            + float(self.args.dags_gate_beta_reliability) * mv_reliability
            + float(self.args.dags_gate_beta_residual) * residual * det_envelope
            + float(self.args.dags_gate_beta_hf) * high_frequency * det_envelope
            - float(self.args.dags_gate_threshold)
        ) / max(1e-4, float(self.args.dags_gate_temperature))
        adaptive_gate = torch.sigmoid(logits).clamp(0.0, 1.0)
        gate_sharpen = max(0.5, float(getattr(self.args, 'dags_gate_sharpen', 1.5)))
        if abs(gate_sharpen - 1.0) > 1e-6:
            adaptive_gate = adaptive_gate.pow(gate_sharpen)

        gaussians.set_detection_metadata(
            det_support=det_support[:, None],
            edge_score=edge_score[:, None],
            mv_reliability=mv_reliability[:, None],
            det_class=class_id[:, None],
            adaptive_gate=adaptive_gate,
        )
        self.last_evidence_iteration = int(iteration)
        self.evidence_update_count += 1

    @torch.no_grad()
    def update_residual_statistics(self, camera, gaussians, rendered: torch.Tensor, gt: torch.Tensor, visibility_filter: torch.Tensor) -> None:
        if visibility_filter is None or not visibility_filter.any():
            return
        residual_map = torch.mean(torch.abs(rendered.detach() - gt.detach()), dim=0, keepdim=True)
        rendered_edge = _sobel_magnitude(rendered.detach())
        gt_edge = _sobel_magnitude(gt.detach())
        hf_map = torch.abs(rendered_edge - gt_edge)

        indices = torch.nonzero(visibility_filter, as_tuple=False).squeeze(1)
        xyz = gaussians.get_xyz.detach()[indices]
        px, py, visible = _project_points(xyz, camera)
        if not visible.any():
            return
        indices = indices[visible]
        residual_values = _sample_map_nearest(residual_map, px[visible], py[visible])[:, None]
        hf_values = _sample_map_nearest(hf_map, px[visible], py[visible])[:, None]
        gaussians.update_residual_ema(
            indices,
            residual_values,
            hf_values,
            decay=float(self.args.dags_residual_ema_decay),
        )

    @torch.no_grad()
    def densification_score(self, gaussians, iteration: int = 0) -> torch.Tensor:
        det = gaussians.get_det_support.squeeze(-1).float().clamp(0.0, 1.0)
        reliability = gaussians.get_mv_reliability.squeeze(-1).float().clamp(0.0, 1.0)
        residual = _robust_normalize(gaussians.get_residual_ema.squeeze(-1))
        hf = _robust_normalize(gaussians.get_hf_ema.squeeze(-1))
        coverage = _robust_normalize(torch.log1p(gaussians.max_radii2D.float().clamp_min(0.0)))

        observation_count = gaussians.get_residual_count.squeeze(-1).float().clamp_min(0.0)
        maturity_scale = max(1e-3, float(getattr(self.args, 'dags_densify_maturity_observations', 6.0)))
        maturity_floor = min(1.0, max(0.0, float(getattr(self.args, 'dags_densify_maturity_floor', 0.35))))
        maturity = maturity_floor + (1.0 - maturity_floor) * (
            1.0 - torch.exp(-observation_count / maturity_scale)
        )

        class_id = gaussians.get_det_class.squeeze(-1).long()
        class_weights = self.class_densify_weights.to(det.device)
        class_factor = torch.ones_like(det)
        valid_class = class_id >= 0
        if valid_class.any():
            clipped = class_id[valid_class].clamp(0, class_weights.numel() - 1)
            class_factor[valid_class] = class_weights[clipped]

        detail = (
            float(getattr(self.args, 'dags_densify_lambda_residual', 0.5)) * residual
            + float(getattr(self.args, 'dags_densify_lambda_hf', 0.75)) * hf
            + float(getattr(self.args, 'dags_densify_lambda_coverage', 0.25)) * coverage
        )
        detail = _robust_normalize(detail)

        det_power = max(0.1, float(getattr(self.args, 'dags_densify_det_power', 0.75)))
        base_priority = (
            det.pow(det_power)
            * (0.45 + 0.55 * reliability)
            * (0.50 + 0.50 * detail)
            * maturity
        ).clamp_min(0.0)



        min_support = min(1.0, max(0.0, float(getattr(self.args, 'dags_densify_min_support', 0.08))))
        focus_quantile = min(0.95, max(0.05, float(getattr(self.args, 'dags_densify_focus_quantile', 0.65))))
        focus_temperature = max(1e-3, float(getattr(self.args, 'dags_densify_focus_temperature', 0.15)))
        positive = base_priority[det > min_support]
        if positive.numel() >= 16:
            center = torch.quantile(positive, focus_quantile)
            q10 = torch.quantile(positive, 0.10)
            q90 = torch.quantile(positive, 0.90)
            spread = (q90 - q10).clamp_min(1e-4)
            focus = torch.sigmoid((base_priority - center) / (focus_temperature * spread))
        else:
            focus = _robust_normalize(base_priority)

        support_temperature = max(
            1e-3, float(getattr(self.args, 'dags_densify_support_temperature', 0.03))
        )
        support_gate = torch.sigmoid((det - min_support) / support_temperature)
        priority = (
            focus
            * support_gate
            * (0.55 + 0.45 * det)
            * (0.70 + 0.30 * reliability)
            * class_factor
        ).clamp_min(0.0)

        start_iter = int(getattr(self.args, 'dags_densify_start_iter', 2500))
        ramp_iters = max(1, int(getattr(self.args, 'dags_densify_ramp_iters', 1000)))
        hold_end = max(start_iter + ramp_iters, int(getattr(self.args, 'dags_densify_hold_end_iter', 17000)))
        anneal_end = max(hold_end + 1, int(getattr(self.args, 'dags_densify_anneal_end_iter', 19500)))
        if int(iteration) < start_iter:
            schedule = 0.0
        elif int(iteration) < start_iter + ramp_iters:
            schedule = (int(iteration) - start_iter) / float(ramp_iters)
        elif int(iteration) <= hold_end:
            schedule = 1.0
        elif int(iteration) < anneal_end:
            progress = (int(iteration) - hold_end) / float(anneal_end - hold_end)
            schedule = 0.5 * (1.0 + math.cos(math.pi * progress))
        else:
            schedule = 0.0



        max_boost = max(0.0, float(getattr(self.args, 'dags_densify_max_soft_boost', 0.45)))
        score = 1.0 + max_boost * schedule * priority
        weight_max = max(1.0, float(getattr(self.args, 'dags_densify_weight_max', 1.55)))
        return score.clamp(1.0, weight_max)[:, None]

    def roi_loss_weight_map(self, camera, height: int, width: int, device, dtype) -> torch.Tensor:
        boost = float(self.args.dags_roi_loss_boost)
        weight = torch.ones((1, height, width), device=device, dtype=dtype)
        if boost <= 0.0:
            return weight
        for box in self.labels.read(camera.image_name):
            x1, y1, x2, y2 = self.labels.pixel_bounds(box, height, width)
            if x2 <= x1 or y2 <= y1:
                continue
            value = 1.0 + boost * float(box.confidence)
            weight[:, y1:y2, x1:x2] = torch.maximum(
                weight[:, y1:y2, x1:x2],
                torch.tensor(value, device=device, dtype=dtype),
            )

        return weight / weight.mean().clamp_min(1e-6)

    def roi_mask(self, camera, height: int, width: int, device, dtype, class_id: Optional[int] = None) -> torch.Tensor:
        mask = torch.zeros((1, height, width), device=device, dtype=dtype)
        for box in self.labels.read(camera.image_name):
            if class_id is not None and int(box.class_id) != int(class_id):
                continue
            x1, y1, x2, y2 = self.labels.pixel_bounds(box, height, width)
            if x2 <= x1 or y2 <= y1:
                continue
            value = float(box.confidence)
            mask[:, y1:y2, x1:x2] = torch.maximum(
                mask[:, y1:y2, x1:x2], torch.tensor(value, device=device, dtype=dtype)
            )
        return mask.clamp(0.0, 1.0)

    def roi_high_frequency_loss(self, camera, rendered: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        mask = self.roi_mask(camera, rendered.shape[-2], rendered.shape[-1], rendered.device, rendered.dtype)
        if mask.sum().item() < 1.0:
            return rendered.sum() * 0.0
        kernel = torch.tensor(
            [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
            device=rendered.device, dtype=rendered.dtype,
        ).view(1, 1, 3, 3).repeat(rendered.shape[0], 1, 1, 1)
        lap_render = F.conv2d(rendered.unsqueeze(0), kernel, padding=1, groups=rendered.shape[0])[0]
        lap_gt = F.conv2d(gt.unsqueeze(0), kernel, padding=1, groups=gt.shape[0])[0]
        per_pixel = F.smooth_l1_loss(lap_render, lap_gt, reduction='none', beta=0.02).mean(dim=0, keepdim=True)
        return (per_pixel * mask).sum() / mask.sum().clamp_min(1.0)

    def _scaled_camera(self, camera, scale: float):
        scale = max(0.1, min(1.0, float(scale)))
        return SimpleNamespace(
            uid=camera.uid,
            image_name=camera.image_name,
            image_width=max(8, int(round(camera.image_width * scale))),
            image_height=max(8, int(round(camera.image_height * scale))),
            FoVx=camera.FoVx,
            FoVy=camera.FoVy,
            world_view_transform=camera.world_view_transform,
            full_proj_transform=camera.full_proj_transform,
            camera_center=camera.camera_center,
        )

    def _render_expected_depth(self, camera, gaussians, pipe):
        from gaussian_renderer import render

        device = gaussians.get_xyz.device
        xyz = gaussians.get_xyz
        ones = torch.ones((xyz.shape[0], 1), device=device, dtype=xyz.dtype)
        homo = torch.cat((xyz, ones), dim=1)
        camera_points = homo @ camera.world_view_transform.to(device=device, dtype=xyz.dtype)
        depth = camera_points[:, 2:3].clamp_min(1e-4)
        depth_color = depth.repeat(1, 3)
        one_color = torch.ones_like(depth_color)
        black = torch.zeros((3,), device=device, dtype=xyz.dtype)
        depth_accum = render(camera, gaussians, pipe, black, override_color=depth_color)["render"][0:1]
        alpha = render(camera, gaussians, pipe, black, override_color=one_color)["render"][0:1].clamp(0.0, 1.0)
        expected = depth_accum / alpha.clamp_min(1e-5)
        return expected, alpha

    @staticmethod
    def _local_ncc(a: torch.Tensor, b: torch.Tensor, window: int) -> torch.Tensor:
        padding = window // 2
        mu_a = F.avg_pool2d(a, window, stride=1, padding=padding)
        mu_b = F.avg_pool2d(b, window, stride=1, padding=padding)
        aa = F.avg_pool2d(a * a, window, stride=1, padding=padding) - mu_a * mu_a
        bb = F.avg_pool2d(b * b, window, stride=1, padding=padding) - mu_b * mu_b
        ab = F.avg_pool2d(a * b, window, stride=1, padding=padding) - mu_a * mu_b
        return (ab / torch.sqrt(aa.clamp_min(1e-6) * bb.clamp_min(1e-6))).clamp(-1.0, 1.0)

    def consistency_loss(self, reference_camera, gaussians, pipe) -> Tuple[torch.Tensor, float]:
        source_camera = self.choose_neighbor(reference_camera)
        if source_camera is None:
            zero = gaussians.get_xyz.sum() * 0.0
            return zero, 0.0

        scale = float(self.args.dags_mv_scale)
        ref = self._scaled_camera(reference_camera, scale)
        src = self._scaled_camera(source_camera, scale)
        ref_depth, ref_alpha = self._render_expected_depth(ref, gaussians, pipe)
        src_depth, src_alpha = self._render_expected_depth(src, gaussians, pipe)

        device = ref_depth.device
        dtype = ref_depth.dtype
        height, width = ref.image_height, ref.image_width
        source_height, source_width = src.image_height, src.image_width
        ys, xs = torch.meshgrid(
            torch.arange(height, device=device, dtype=dtype),
            torch.arange(width, device=device, dtype=dtype),
            indexing="ij",
        )
        z = ref_depth[0]
        fx = width / (2.0 * math.tan(ref.FoVx * 0.5))
        fy = height / (2.0 * math.tan(ref.FoVy * 0.5))
        cx = width * 0.5
        cy = height * 0.5
        x_cam = (xs - cx) / fx * z
        y_cam = -(ys - cy) / fy * z
        cam_points = torch.stack((x_cam, y_cam, z, torch.ones_like(z)), dim=-1).reshape(-1, 4)
        world = cam_points @ torch.inverse(ref.world_view_transform.to(device=device, dtype=dtype))

        src_clip = world @ src.full_proj_transform.to(device=device, dtype=dtype)
        src_w = src_clip[:, 3]
        src_ndc = src_clip[:, :3] / src_w.clamp_min(1e-7)[:, None]
        src_px = (src_ndc[:, 0] * 0.5 + 0.5) * source_width
        src_py = (0.5 - src_ndc[:, 1] * 0.5) * source_height
        grid_x = src_px / max(source_width - 1, 1) * 2.0 - 1.0
        grid_y = src_py / max(source_height - 1, 1) * 2.0 - 1.0
        sample_grid = torch.stack((grid_x, grid_y), dim=-1).reshape(1, height, width, 2)

        source_gt = F.interpolate(
            source_camera.original_image.unsqueeze(0).to(device),
            size=(source_height, source_width),
            mode="bilinear",
            align_corners=False,
        )
        reference_gt = F.interpolate(
            reference_camera.original_image.unsqueeze(0).to(device),
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )
        warped_source = F.grid_sample(source_gt, sample_grid, mode="bilinear", padding_mode="zeros", align_corners=True)
        sampled_src_depth = F.grid_sample(src_depth.unsqueeze(0), sample_grid, mode="bilinear", padding_mode="zeros", align_corners=True)[0, 0]
        sampled_src_alpha = F.grid_sample(src_alpha.unsqueeze(0), sample_grid, mode="bilinear", padding_mode="zeros", align_corners=True)[0, 0]

        src_view = world @ src.world_view_transform.to(device=device, dtype=dtype)
        expected_src_z = src_view[:, 2].reshape(height, width).clamp_min(1e-5)
        relative_depth_error = torch.abs(sampled_src_depth - expected_src_z) / expected_src_z
        relative_depth_error = torch.nan_to_num(relative_depth_error, nan=1e6, posinf=1e6, neginf=1e6)


        source_z = sampled_src_depth
        source_x = (src_px.reshape(height, width) - source_width * 0.5) / (
            source_width / (2.0 * math.tan(src.FoVx * 0.5))
        ) * source_z
        source_y = -(src_py.reshape(height, width) - source_height * 0.5) / (
            source_height / (2.0 * math.tan(src.FoVy * 0.5))
        ) * source_z
        source_cam_points = torch.stack(
            (source_x, source_y, source_z, torch.ones_like(source_z)), dim=-1
        ).reshape(-1, 4)
        source_world = source_cam_points @ torch.inverse(src.world_view_transform.to(device=device, dtype=dtype))
        ref_clip = source_world @ ref.full_proj_transform.to(device=device, dtype=dtype)
        ref_ndc = ref_clip[:, :3] / ref_clip[:, 3].clamp_min(1e-7)[:, None]
        ref_px_back = (ref_ndc[:, 0] * 0.5 + 0.5) * width
        ref_py_back = (0.5 - ref_ndc[:, 1] * 0.5) * height
        cycle = torch.sqrt(
            (ref_px_back.reshape(height, width) - xs).square()
            + (ref_py_back.reshape(height, width) - ys).square()
            + 1e-8
        ) / max(float(max(height, width)), 1.0)
        cycle = torch.nan_to_num(cycle, nan=1.0, posinf=1.0, neginf=1.0)

        in_bounds = (
            (src_w.reshape(height, width) > 1e-7)
            & (grid_x.reshape(height, width) >= -1.0)
            & (grid_x.reshape(height, width) <= 1.0)
            & (grid_y.reshape(height, width) >= -1.0)
            & (grid_y.reshape(height, width) <= 1.0)
        )
        valid = (
            in_bounds
            & (ref_alpha[0] >= float(self.args.dags_mv_alpha_threshold))
            & (sampled_src_alpha >= float(self.args.dags_mv_alpha_threshold))
            & (relative_depth_error <= float(self.args.dags_mv_depth_threshold))
            & torch.isfinite(cycle)
        )

        ref_gray = reference_gt.mean(dim=1, keepdim=True)
        warped_gray = warped_source.mean(dim=1, keepdim=True)
        ncc = self._local_ncc(ref_gray, warped_gray, int(self.args.dags_mv_ncc_window))[0, 0]
        valid_patch = F.avg_pool2d(valid.float()[None, None], int(self.args.dags_mv_ncc_window), stride=1,
                                   padding=int(self.args.dags_mv_ncc_window) // 2)[0, 0] > 0.75
        valid = valid & valid_patch

        detection_weight = torch.ones((height, width), device=device, dtype=dtype)
        boost = float(self.args.dags_mv_detection_boost)
        band_ratio = max(0.0, min(0.45, float(self.args.dags_mv_box_band_ratio)))
        band_weight = float(self.args.dags_mv_box_band_weight)
        for box in self.labels.read(reference_camera.image_name):
            x1, y1, x2, y2 = self.labels.pixel_bounds(box, height, width)
            if x2 <= x1 or y2 <= y1:
                continue
            detection_weight[y1:y2, x1:x2] = torch.minimum(
                detection_weight[y1:y2, x1:x2],
                torch.tensor(band_weight, device=device, dtype=dtype),
            )
            bx = int(round((x2 - x1) * band_ratio))
            by = int(round((y2 - y1) * band_ratio))
            ix1, ix2 = min(x2, x1 + bx), max(x1, x2 - bx)
            iy1, iy2 = min(y2, y1 + by), max(y1, y2 - by)
            if ix2 > ix1 and iy2 > iy1:
                detection_weight[iy1:iy2, ix1:ix2] = torch.maximum(
                    detection_weight[iy1:iy2, ix1:ix2],
                    torch.tensor(1.0 + boost * box.confidence, device=device, dtype=dtype),
                )

        weight = valid.float() * detection_weight
        denominator = weight.sum()
        if denominator.item() < 1.0:
            zero = ref_depth.sum() * 0.0
            return zero, 0.0
        cycle_loss = (cycle * weight).sum() / denominator
        ncc_loss = ((1.0 - ncc) * 0.5 * weight).sum() / denominator
        total = (
            float(self.args.dags_mv_cycle_weight) * cycle_loss
            + float(self.args.dags_mv_ncc_weight) * ncc_loss
        )
        return total, float(valid.float().mean().item())
