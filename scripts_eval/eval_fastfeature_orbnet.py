import argparse

import cv2
import numpy as np
import torch
from termcolor import colored
from torch.utils.data import DataLoader

from kp2d.datasets.patches_dataset import PatchesDataset
from kp2d.evaluation.evaluate import evaluate_fastfeature_orbnet

#! Stuffs from vfm
from vfm.fastfeature.networks.fastfeature_net_tiny import FastFeatureNetTiny
from vfm.fastfeature.networks.orb_net import OrbNet

def main():

    #! Parse Arguments
    parser = argparse.ArgumentParser(
        description='Script for FastFeature testing',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input_dir", type=str, default="/mnt/Enterprise/PUBLIC_DATASETS/HPatches", help="Path to hpatches dataset")
    parser.add_argument("--pretrained_fastfeaturetiny", default="/home/rashik/workspace/vfm_simple/outputs/homography-tiny-Jan-13-11:35:02/weights/9.pth", type=str, help="pretrained model path")
    parser.add_argument("--pretrained_orbnet", default="/home/rashik/workspace/vfm_simple/outputs/homography-tiny-Jan-13-11:35:02/weights/9.pth_orbnet.pth", type=str, help="pretrained model path")
    parser.add_argument("--nfeatures", type=int, default=500, help="Number of features")
    parser.add_argument("--fast_threshold", type=int, default=20, help="Fast Threshold")
    args = parser.parse_args()

    print("Using following args:")
    print(args)
    input("Press any key to continue ...")

    #! Define ORB detector
    orb =  cv2.ORB_create(nfeatures=args.nfeatures, fastThreshold=args.fast_threshold)


    #! Define FastFeature
    network = FastFeatureNetTiny()
    network.load_state_dict(torch.load(args.pretrained_fastfeaturetiny, map_location='cuda'))
    network = network.cuda()

    orbnet = OrbNet()
    orbnet.load_state_dict(torch.load(args.pretrained_orbnet, map_location='cuda'))
    network = network.cuda()

    print(f"Loaded FastFeature from {args.pretrained_fastfeaturetiny} and {args.pretrained_orbnet}")

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

        print(f"Number of datapoints: {hp_dataset.__len__()}")

        print(colored('Evaluating for {} -- top_k {}'.format(params['res'], params['top_k']),'green'))

        rep, loc, c1, c3, c5, mscore, dur = evaluate_fastfeature_orbnet(
            data_loader,
            orb,
            network,
            orbnet,
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
