#!/usr/bin/env python3
"""Debug script to check camera structure."""

import torch
from scene import Scene
from arguments import ModelParams, ModelHiddenParams
from gaussian_renderer import GaussianModel
from argparse import ArgumentParser
from arguments import get_combined_args
import mmcv
from utils.params_utils import merge_hparams

parser = ArgumentParser(description="Debug cameras")
model = ModelParams(parser, sentinel=True)
hyperparam = ModelHiddenParams(parser)
parser.add_argument("--iteration", default=-1, type=int)
parser.add_argument("--configs", type=str)

args = get_combined_args(parser)

if args.configs:
    config = mmcv.Config.fromfile(args.configs)
    args = merge_hparams(args, config)

with torch.no_grad():
    hyper = hyperparam.extract(args)
    gaussians = GaussianModel(model.extract(args).sh_degree, hyper)
    scene = Scene(model.extract(args), gaussians, load_iteration=args.iteration, shuffle=False)
    
    train_cams = scene.getTrainCameras()
    
    print(f"Total train cameras: {len(train_cams)}")
    print("\nFirst 10 cameras:")
    for i, cam in enumerate(train_cams[:10]):
        print(f"  {i}: image_name={cam.image_name}, time={cam.time:.3f}, uid={cam.uid}")
    
    print("\nLast 10 cameras:")
    for i, cam in enumerate(train_cams[-10:], start=len(train_cams)-10):
        print(f"  {i}: image_name={cam.image_name}, time={cam.time:.3f}, uid={cam.uid}")

