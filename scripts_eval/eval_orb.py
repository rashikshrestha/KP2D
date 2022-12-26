# Copyright 2020 Toyota Research Institute.  All rights reserved.
# Example usage: python scripts/eval_keypoint_net.sh --pretrained_model /data/models/kp2d/v4.pth --input_dir /data/datasets/kp2d/HPatches/

import argparse
import os
import pickle
import random
import subprocess

import cv2
import numpy as np
import torch
from PIL import Image
from termcolor import colored
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from kp2d.datasets.patches_dataset import PatchesDataset
from kp2d.evaluation.evaluate import evaluate_orb


def main():

    #! Parse Arguments
    parser = argparse.ArgumentParser(
        description='Script for ORB testing',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input_dir", type=str, default="/mnt/SSD0/rashik/datasets/hpatches", help="Path to hpatches dataset")
    parser.add_argument("--nfeatures", type=int, default=500, help="Number of features")
    parser.add_argument("--fast_threshold", type=int, default=20, help="Fast Threshold")
    args = parser.parse_args()

    print("Using following args:")

    for key, value in vars(args).items():
        print(key, ":", value)

    input("Press any key to continue ...")

    #! Define ORB detector
    orb =  cv2.ORB_create(nfeatures=args.nfeatures, fastThreshold=args.fast_threshold)

    #! Versions of Dataset to use
    eval_params = [{'res': (320, 240), 'top_k': 300, }]
    eval_params += [{'res': (640, 480), 'top_k': 1000, }]

    #! For each version of dataset:
    for params in eval_params:
        hp_dataset = PatchesDataset(root_dir=args.input_dir, use_color=True,
                                    output_shape=params['res'], type='a')
        data_loader = DataLoader(hp_dataset,
                                 batch_size=1,
                                 pin_memory=False,
                                 shuffle=False,
                                 num_workers=8,
                                 worker_init_fn=None,
                                 sampler=None)


        print(colored('Evaluating for {} -- top_k {}'.format(params['res'], params['top_k']),'green'))
        print(f"Available number of datapoints: {hp_dataset.__len__()}")

        rep, loc, c1, c3, c5, mscore, dur = evaluate_orb(
            data_loader,
            orb,
            output_shape=params['res'],
            top_k=params['top_k'])

        print('Repeatability {0:.3f}'.format(rep))
        print('Localization Error {0:.3f}'.format(loc))
        print('Correctness d1 {:.3f}'.format(c1))
        print('Correctness d3 {:.3f}'.format(c3))
        print('Correctness d5 {:.3f}'.format(c5))
        print('MScore {:.3f}'.format(mscore))
        print('Duration {:.3f}'.format(dur))


if __name__ == '__main__':
    main()
