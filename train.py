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

import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from utils.dags_controller import DAGSController



try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False




def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from,normal_lr, finetune, load_iteration, det_args=None):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)

    scene = Scene(dataset, gaussians)

    gaussians.training_setup(opt, normal_lr=normal_lr, finetune=finetune)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        first_iter = 0
        gaussians.restore(model_params, opt, finetune)


    dags_controller = None
    if det_args is not None and getattr(det_args, 'dags_enable', False):
        if not getattr(det_args, 'det_label_dirs', []):
            raise RuntimeError("--dags_enable requires --det_label_dirs")
        dags_controller = DAGSController(
            scene.getTrainCameras(), getattr(det_args, 'det_label_dirs', []), det_args
        )


        if first_iter < getattr(det_args, 'dags_adaptive_start_iter', 3000):
            gaussians.set_adaptive_gate(torch.zeros((gaussians.get_xyz.shape[0],), device=gaussians.get_xyz.device))

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")

    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):        

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        if dags_controller is not None:
            adaptive_start = int(getattr(det_args, 'dags_adaptive_start_iter', 3000))
            update_interval = max(1, int(getattr(det_args, 'dags_evidence_update_interval', 1000)))
            if iteration >= adaptive_start and (
                iteration == adaptive_start or iteration % update_interval == 0
            ):
                dags_controller.update_global_evidence(gaussians, iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        gt_image = viewpoint_cam.original_image.cuda()

        if dags_controller is not None and iteration % max(1, int(getattr(det_args, 'dags_residual_update_interval', 5))) == 0:
            dags_controller.update_residual_statistics(
                viewpoint_cam, gaussians, image, gt_image, visibility_filter
            )

        if dags_controller is not None:
            dags_weight_map = dags_controller.roi_loss_weight_map(
                viewpoint_cam,
                height=gt_image.shape[1],
                width=gt_image.shape[2],
                device=gt_image.device,
                dtype=gt_image.dtype,
            )
            Ll1 = torch.mean(dags_weight_map * torch.abs(image - gt_image))
        else:
            Ll1 = l1_loss(image, gt_image)

        ssim_value = ssim(image, gt_image)
        ssim_loss_value = 1.0 - ssim_value
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * ssim_loss_value

        hf_loss = image.sum() * 0.0
        hf_effective_weight = 0.0
        if dags_controller is not None:
            hf_start = int(getattr(det_args, 'dags_hf_start_iter', 3000))
            hf_ramp_iters = max(1, int(getattr(det_args, 'dags_hf_ramp_iters', 3000)))
            hf_weight = max(0.0, float(getattr(det_args, 'dags_hf_loss_weight', 0.01)))
            if iteration >= hf_start and hf_weight > 0.0:
                hf_ramp = min(1.0, max(0.0, (iteration - hf_start) / float(hf_ramp_iters)))
                hf_loss = dags_controller.roi_high_frequency_loss(viewpoint_cam, image, gt_image)
                hf_effective_weight = hf_weight * hf_ramp
                loss = loss + hf_effective_weight * hf_loss

        mv_valid_ratio = 0.0
        mv_loss = image.sum() * 0.0
        mv_effective_weight = 0.0
        if dags_controller is not None:
            mv_start = int(getattr(det_args, 'dags_mv_start_iter', 6000))
            mv_interval = max(1, int(getattr(det_args, 'dags_mv_interval', 100)))
            if iteration >= mv_start and iteration % mv_interval == 0:
                mv_loss, mv_valid_ratio = dags_controller.consistency_loss(viewpoint_cam, gaussians, pipe)
                ramp_iters = max(1, int(getattr(det_args, 'dags_mv_ramp_iters', 4000)))
                ramp = min(1.0, max(0.0, (iteration - mv_start) / float(ramp_iters)))
                decay_start = int(getattr(det_args, 'dags_mv_decay_start_iter', 18000))
                decay_end = max(decay_start + 1, int(getattr(det_args, 'dags_mv_decay_end_iter', opt.iterations)))
                final_scale = min(1.0, max(0.0, float(getattr(det_args, 'dags_mv_final_scale', 0.25))))
                if iteration <= decay_start:
                    decay_scale = 1.0
                elif iteration >= decay_end:
                    decay_scale = final_scale
                else:
                    progress = (iteration - decay_start) / float(decay_end - decay_start)
                    decay_scale = 1.0 - progress * (1.0 - final_scale)
                valid_ratio = mv_valid_ratio
                min_valid = float(getattr(det_args, 'dags_mv_min_valid_ratio', 0.03))
                target_valid = max(min_valid + 1e-6, float(getattr(det_args, 'dags_mv_target_valid_ratio', 0.15)))
                if valid_ratio >= min_valid:
                    valid_scale = min(1.0, valid_ratio / target_valid)
                    mv_effective_weight = float(getattr(det_args, 'dags_mv_global_weight', 0.002)) * ramp * decay_scale * valid_scale
                    loss = loss + mv_effective_weight * mv_loss

        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(
                tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end),
                testing_iterations, scene, render, (pipe, background)
            )
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration % 5000==0:
                if finetune:
                    gaussians.optimizer.param_groups[1]['lr'] /= 1.5
                else:
                    gaussians.optimizer.param_groups[4]['lr'] /= 1.4

            if iteration < opt.densify_until_iter and not finetune:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    det_densify_score = None
                    if dags_controller is not None and iteration >= int(getattr(det_args, 'dags_densify_start_iter', 500)):
                        det_densify_score = dags_controller.densification_score(gaussians, iteration=iteration)
                    gaussians.densify_and_prune(
                        opt.densify_grad_threshold, 0.01, scene.cameras_extent, size_threshold,
                        det_densify_score=det_densify_score
                    )

                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()


            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)



            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = (
            {'name': 'test', 'cameras': scene.getTestCameras()},
            {'name': 'train', 'cameras': [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]},
        )

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and idx < 5:
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)

    parser.add_argument("--test_iterations", nargs="+", type=int,
                        default=[7_000, 12_000, 18_000, 24_000, 30_000])

    parser.add_argument("--save_iterations", nargs="+", type=int, default=[30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--normal_lr", type=float, default = 0.003)
    parser.add_argument('--finetune', action='store_true', default=False)
    parser.add_argument('--load_iteration',type=int, default=None)


    parser.add_argument('--dags_enable', action='store_true', default=False,
                        help='Enable multi-view detection evidence, adaptive Gaussian activation, consistency and densification.')
    parser.add_argument('--dags_num_classes', type=int, default=1)
    parser.add_argument('--dags_class_densify_weights', type=str, default='1.5')
    parser.add_argument('--dags_box_expand_ratio', type=float, default=0.0)
    parser.add_argument('--dags_projection_chunk', type=int, default=200000)
    parser.add_argument('--dags_min_visible_views', type=int, default=3)
    parser.add_argument('--dags_min_detection_views', type=int, default=2)
    parser.add_argument('--dags_adaptive_start_iter', type=int, default=3000)
    parser.add_argument('--dags_evidence_update_interval', type=int, default=2000)
    parser.add_argument('--dags_evidence_views_per_update', type=int, default=48,
                        help='Rotating training views per evidence update; 0 uses all views.')
    parser.add_argument('--dags_evidence_ema_decay', type=float, default=0.70,
                        help='EMA used to accumulate rotating multi-view detection evidence.')
    parser.add_argument('--dags_outside_gate_allowance', type=float, default=0.05)
    parser.add_argument('--dags_gate_beta_det', type=float, default=2.5)
    parser.add_argument('--dags_gate_beta_edge', type=float, default=0.75)
    parser.add_argument('--dags_gate_beta_reliability', type=float, default=0.75)
    parser.add_argument('--dags_gate_beta_residual', type=float, default=0.4)
    parser.add_argument('--dags_gate_beta_hf', type=float, default=0.6)
    parser.add_argument('--dags_gate_threshold', type=float, default=0.95)
    parser.add_argument('--dags_gate_temperature', type=float, default=0.25)
    parser.add_argument('--dags_gate_sharpen', type=float, default=1.5,
                        help='Smoothly suppress weak fractional Half-Gaussian activation; >=1.')
    parser.add_argument('--dags_roi_loss_boost', type=float, default=0.10,
                        help='Relative ROI priority; the weight map is normalized to mean 1.')
    parser.add_argument('--dags_residual_update_interval', type=int, default=5)
    parser.add_argument('--dags_residual_ema_decay', type=float, default=0.9)
    parser.add_argument('--dags_densify_start_iter', type=int, default=2500)
    parser.add_argument('--dags_densify_lambda_residual', type=float, default=0.5)
    parser.add_argument('--dags_densify_lambda_hf', type=float, default=0.75)
    parser.add_argument('--dags_densify_lambda_coverage', type=float, default=0.25)
    parser.add_argument('--dags_densify_max_soft_boost', type=float, default=0.45,
                        help='Maximum positive bonus for reliable detected Gaussians; baseline remains 1.0.')
    parser.add_argument('--dags_densify_maturity_observations', type=float, default=6.0,
                        help='Observation count controlling smooth maturity of newly created Gaussians.')
    parser.add_argument('--dags_densify_maturity_floor', type=float, default=0.35,
                        help='Minimum bonus eligibility for newly generated Gaussians; soft, not a hard switch.')
    parser.add_argument('--dags_densify_ramp_iters', type=int, default=1000)
    parser.add_argument('--dags_densify_hold_end_iter', type=int, default=17000)
    parser.add_argument('--dags_densify_anneal_end_iter', type=int, default=19500,
                        help='Iteration where the local detection densification bonus smoothly decays to zero.')
    parser.add_argument('--dags_neighbor_count', type=int, default=3)
    parser.add_argument('--dags_mv_start_iter', type=int, default=6000)
    parser.add_argument('--dags_mv_interval', type=int, default=100)
    parser.add_argument('--dags_mv_ramp_iters', type=int, default=4000)
    parser.add_argument('--dags_mv_global_weight', type=float, default=0.002)
    parser.add_argument('--dags_mv_min_valid_ratio', type=float, default=0.03)
    parser.add_argument('--dags_mv_target_valid_ratio', type=float, default=0.15)
    parser.add_argument('--dags_mv_scale', type=float, default=0.25)
    parser.add_argument('--dags_mv_alpha_threshold', type=float, default=0.6)
    parser.add_argument('--dags_mv_depth_threshold', type=float, default=0.05)
    parser.add_argument('--dags_mv_ncc_window', type=int, default=5)
    parser.add_argument('--dags_mv_cycle_weight', type=float, default=0.5)
    parser.add_argument('--dags_mv_ncc_weight', type=float, default=0.5)
    parser.add_argument('--dags_mv_detection_boost', type=float, default=0.25)
    parser.add_argument('--dags_mv_box_band_ratio', type=float, default=0.08)
    parser.add_argument('--dags_mv_box_band_weight', type=float, default=0.2)

    parser.add_argument('--dags_densify_weight_max', type=float, default=1.55)
    parser.add_argument('--dags_densify_det_power', type=float, default=0.75)
    parser.add_argument('--dags_densify_min_support', type=float, default=0.08)
    parser.add_argument('--dags_densify_focus_quantile', type=float, default=0.65)
    parser.add_argument('--dags_densify_focus_temperature', type=float, default=0.15)
    parser.add_argument('--dags_densify_support_temperature', type=float, default=0.03)
    parser.add_argument('--dags_hf_loss_weight', type=float, default=0.01)
    parser.add_argument('--dags_hf_start_iter', type=int, default=3000)
    parser.add_argument('--dags_hf_ramp_iters', type=int, default=3000)
    parser.add_argument('--dags_mv_decay_start_iter', type=int, default=18000)
    parser.add_argument('--dags_mv_decay_end_iter', type=int, default=30000)
    parser.add_argument('--dags_mv_final_scale', type=float, default=0.25)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training

    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    load_iteration = args.load_iteration
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args.normal_lr, args.finetune, load_iteration, args)

    # All done
    print("\nTraining complete.")
