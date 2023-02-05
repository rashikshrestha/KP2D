"""
This scripts answers the following question:
Q. Do incorrect matches corresponds to keypoints with less scores? 
   If yes by now much?
A. pending...

"""

import argparse

import numpy as np
import cv2
from termcolor import colored
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

np.set_printoptions(suppress=True, precision=2)

#! KP2D stuffs
from kp2d.datasets.patches_dataset import PatchesDataset
from kp2d.utils.keypoints import warp_keypoints

def select_k_best(points, descriptors, k):
    """ Select the k most probable points (and DON'T strip their probability).
    points has shape (num_points, 3) where the last coordinate is the probability.

    Parameters
    ----------
    points: numpy.ndarray (N,3)
        Keypoint vector, consisting of (x,y,probability).
    descriptors: numpy.ndarray (N,256)
        Keypoint descriptors.
    k: int
        Number of keypoints to select, based on probability.
    Returns
    -------
    
    selected_points: numpy.ndarray (k,2)
        k most probable keypoints.
    selected_descriptors: numpy.ndarray (k,256)
        Descriptors corresponding to the k most probable keypoints.
    """
    sorted_prob = points[points[:, 2].argsort(), :]
    sorted_desc = descriptors[points[:, 2].argsort(), :]
    start = min(k, points.shape[0])
    selected_points = sorted_prob[-start:, :]
    selected_descriptors = sorted_desc[-start:, :]
    return selected_points, selected_descriptors

def compute_matching_score(data, keep_k_points=1000):
    """
    Compute the matching score between two sets of keypoints with associated descriptors.
    
    Parameters
    ----------
    data: dict
        Input dictionary containing:
        image_shape: tuple (H,W)
            Original image shape.
        homography: numpy.ndarray (3,3)
            Ground truth homography.
        prob: numpy.ndarray (N,3)
            Keypoint vector, consisting of (x,y,probability).
        warped_prob: numpy.ndarray (N,3)
            Warped keypoint vector, consisting of (x,y,probability).
        desc: numpy.ndarray (N,256)
            Keypoint descriptors.
        warped_desc: numpy.ndarray (N,256)
            Warped keypoint descriptors.
    keep_k_points: int
        Number of keypoints to select, based on probability.

    Returns
    -------    
    ms: float
        Matching score.
    """
    #! Get data
    shape = data['image_shape']
    real_H = data['homography']

    kpts_scores_0 = data['prob']
    kpts_scores_1 = data['warped_prob']

    desc0 = data['desc']
    desc1 = data['warped_desc']
    
    #! Select top points
    kpts_scores_0, desc0 = select_k_best(
        kpts_scores_0,
        desc0, 
        keep_k_points)

    kpts_scores_1, desc1 = select_k_best(
        kpts_scores_1, 
        desc1, 
        keep_k_points)

    #! Separate kpts and scores
    kpts0, scores0 = kpts_scores_0[:,:2], kpts_scores_0[:,2]
    kpts1, scores1 = kpts_scores_1[:,:2], kpts_scores_1[:,2]

    # print(kpts0.shape, scores0.shape)
    # print(kpts1.shape, scores1.shape)
    
    #! Match the keypoints with the warped_keypoints (no cross check)
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(desc0, desc1)

    #! Rearrange kpts and scores on the basis of matches
    matched_idx0 = np.array([m.queryIdx for m in matches])
    kpts0 = kpts0[matched_idx0, :]
    scores0 = scores0[matched_idx0]

    matched_idx1 = np.array([m.trainIdx for m in matches])
    kpts1 = kpts1[matched_idx1, :]
    scores1 = scores1[matched_idx1]

    # print(kpts0.shape, scores0.shape)
    # print(kpts1.shape, scores1.shape)

    #! Inverse warp (from img1->img0)
    kpts1_warped = warp_keypoints(kpts1, np.linalg.inv(real_H))

    #! Can the inversed warped point be Visualized (lies within) in img0?
    # eg: vis_warped = [T F T T T F F T T ]
    # F = not visible, T = visible
    vis = np.all((kpts1_warped >= 0) & (kpts1_warped <= (np.array(shape)-1)), axis=-1)
    kpts0, scores0, kpts1, scores1 = kpts0[vis], scores0[vis], kpts1[vis], scores1[vis]
    kpts1_warped = kpts1_warped[vis]

    # print(kpts0.shape, scores0.shape)
    # print(kpts1.shape, scores1.shape)
    # print(kpts1_warped.shape)

    #! Find distance between kpts0 and inverse warpped kpts1
    dist = np.linalg.norm(kpts1_warped - kpts0, axis=-1)

    out = np.concatenate(
        (dist.reshape(-1,1), scores0.reshape(-1,1), scores1.reshape(-1,1)),
        axis=1
    )

    # print(out.shape)
    # print(out)


    #! Distance < 3 is considered to be same point
    # correct1 = (norm1 < 3)
    # count1 = np.sum(correct1)



    # matches = bf.match(warped_desc, desc)
    
    # matches_idx = np.array([m.queryIdx for m in matches])
    # m_warped_keypoints = warped_keypoints[matches_idx, :]
    # matches_idx = np.array([m.trainIdx for m in matches])
    # m_keypoints = keypoints[matches_idx, :]

    # true_keypoints = warp_keypoints(m_keypoints, real_H)
    # vis = np.all((true_keypoints >= 0) & (true_keypoints <= (np.array(shape)-1)), axis=-1)
    # norm2 = np.linalg.norm(true_keypoints - m_warped_keypoints, axis=-1)

    # correct2 = (norm2 < 3)
    # count2 = np.sum(correct2 * vis)
    # score2 = count2 / np.maximum(np.sum(vis), 1.0)

    # ms = (score1 + score2) / 2

    return out

def evaluate_orb(data_loader, orb, output_shape=(320, 240), top_k=300):
    """ORB evaluation script. 

    Parameters
    ----------
    data_loader: torch.utils.data.DataLoader
        Dataset loader. 
    orb:
        ORB extractor
    output_shape: tuple
        Original image shape.
    top_k: int
        Number of keypoints to use to compute metrics, 
        selected based on probability.    
    """  
    def get_KSD(image):
        #! Get image in numpy
        image = image[0] # Remove SINGLE batch dimension
        # Swap the dimenstion to be compatable with opencv and change the range from 0-1 to 0-255
        image = image.permute((1,2,0))*255. 
        img_np = image.cpu().numpy()
        img_np = img_np.astype(np.uint8)

        #! Detect kpts
        keypoints, desc = orb.detectAndCompute(img_np, None)
        kpts = []
        scores = []
        for k in keypoints:
            kpts.append([k.pt[0], k.pt[1]])
            scores.append(k.response)
        kpts = np.array(kpts)
        scores = np.array(scores)

        #! Normalize the scores
        scores_norm = np.linalg.norm(scores)
        scores /= scores_norm
        kpts_scores = np.hstack((kpts, scores.reshape(-1,1)))

        #! Return 
        return kpts_scores, desc

    #! Accumulators
    outputs = []

    #! For each data point:
    for i, sample in tqdm(enumerate(data_loader), desc="evaluate_ORB"):
        score_1, desc1 = get_KSD(sample['image'])
        score_2, desc2 = get_KSD(sample['warped_image'])

        #! Prepare data for eval
        data = {'image': sample['image'].numpy().squeeze(),
                'image_shape' : output_shape,
                'warped_image': sample['warped_image'].numpy().squeeze(),
                'homography': sample['homography'].squeeze().numpy(),
                'prob': score_1, 
                'warped_prob': score_2,
                'desc': desc1,
                'warped_desc': desc2}



        #! Compute matching score
        out = compute_matching_score(data, keep_k_points=top_k)
        outputs.append(out)

    all_outputs = np.concatenate(outputs, axis=0)

    print("ALLLLLLLL")
    print(all_outputs.shape)
    input()
    return all_outputs

def main():

    #! Parse Arguments
    parser = argparse.ArgumentParser(
        description='Script for ORB testing',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        "--input_dir", 
        type=str, 
        default="/mnt/SSD0/rashik/datasets/hpatches", 
        help="Path to hpatches dataset")

    parser.add_argument(
        "--nfeatures", 
        type=int, 
        default=500, 
        help="Number of features")

    parser.add_argument(
        "--fast_threshold", 
        type=int, 
        default=20, 
        help="Fast Threshold")

    args = parser.parse_args()

    print("Using following args:")
    for key, value in vars(args).items():
        print(key, ":", value)
    input("Press any key to continue ...")

    #! Define ORB detector
    orb =  cv2.ORB_create(
        nfeatures=args.nfeatures, 
        fastThreshold=args.fast_threshold
    )

    #! Versions of Dataset to use
    eval_params = [{'res': (320, 240), 'top_k': 300, }]
    eval_params += [{'res': (640, 480), 'top_k': 1000, }]

    #! For each version of dataset:
    for params in eval_params:

        hp_dataset = PatchesDataset(
            root_dir=args.input_dir, use_color=True,
            output_shape=params['res'], type='a'
        )
        data_loader = DataLoader(
            hp_dataset,
            batch_size=1,
            pin_memory=False,
            shuffle=False,
            num_workers=8,
            worker_init_fn=None,
            sampler=None
        )


        print(colored(f"Evaluating for {params['res']} -- {params['top_k']}", 
                'green'))
        print(f"Available number of datapoints: {hp_dataset.__len__()}")

        data = evaluate_orb(
            data_loader,
            orb,
            output_shape=params['res'],
            top_k=params['top_k'])

        #! Draw outputs
        scores = data[:,1]
        dist = data[:,0]


        plt.xlim([0,0.3])
        plt.ylim([0,10])
        plt.scatter(scores, dist, s=0.5)
        plt.savefig('scores_vs_distance_with_crosscheck.png')

if __name__ == '__main__':
    main()
