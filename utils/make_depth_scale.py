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

import numpy as np
import os
import sys
import argparse


sys.path.append(os.getcwd())

from utils.read_write_model import read_model, qvec2rotmat, rotmat2qvec
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal


def get_intrinsics(cam):
    return np.array([
        [cam.params[0], 0, cam.params[2]],
        [0, cam.params[1], cam.params[3]],
        [0, 0, 1]
    ])


def get_extrinsics(qvec, tvec):
    R = qvec2rotmat(qvec)
    t = tvec.reshape(3, 1)
    Rt = np.concatenate([np.concatenate([R, t], 1), np.array([[0, 0, 0, 1]])], 0)
    return np.linalg.inv(Rt)


def get_intrinsics_matrix(K, H, W):
    return np.array([
        [K[0, 0], 0, K[0, 2]],
        [0, K[1, 1], K[1, 2]],
        [0, 0, 1]
    ])


def get_depths(image_dir, intrinsics, extrinsics, points3D):
    depths = []
    for intr, extr in zip(intrinsics, extrinsics):

        proj = intr @ extr[:3, :] @ points3D.T

        depth = proj[2, :]
        depths.append(depth)
    return depths


def get_depth_scales(args, intrinsics, extrinsics, points3D):

    depths = get_depths(args.image_dir, intrinsics, extrinsics, points3D)
    

    depth_ranges = []
    for depth in depths:
        depth_ranges.append((depth.min(), depth.max()))
    

    depth_scales = [1.0] * len(depths)
    return depth_scales


def main(args):

    cameras, images, points3D = read_model(args.model_path)
    

    intrinsics = []
    for cam_id, cam in cameras.items():
        intrinsics.append(get_intrinsics(cam))
    

    extrinsics = []
    for img_id, img in images.items():
        extrinsics.append(get_extrinsics(img.qvec, img.tvec))
    

    points3D_array = np.array([p.xyz for p in points3D.values()])
    

    depth_scales = get_depth_scales(args, intrinsics, extrinsics, points3D_array)
    

    output_path = os.path.join(args.model_path, "depth_scales.txt")
    with open(output_path, "w") as f:
        for i, scale in enumerate(depth_scales):
            f.write(f"{i} {scale}\n")
    
    print(f"深度尺度已保存到: {output_path}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="计算 COLMAP 模型的深度尺度")
    parser.add_argument("--model_path", type=str, required=True, help="COLMAP 模型目录路径")
    parser.add_argument("--image_dir", type=str, required=True, help="图像目录路径")
    args = parser.parse_args()
    
    main(args)
