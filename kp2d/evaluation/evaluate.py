# Copyright 2020 Toyota Research Institute.  All rights reserved.

import numpy as np
import torch
import torchvision.transforms as transforms
from tqdm import tqdm
import sys
np.set_printoptions(suppress=True, threshold=sys.maxsize)


from kp2d.evaluation.descriptor_evaluation import (compute_homography,
                                                   compute_matching_score)
from kp2d.evaluation.detector_evaluation import compute_repeatability
from kp2d.utils.image import to_color_normalized, to_gray_normalized


def evaluate_keypoint_net(data_loader, keypoint_net, output_shape=(320, 240), top_k=300, use_color=True):
    """Keypoint net evaluation script. 

    Parameters
    ----------
    data_loader: torch.utils.data.DataLoader
        Dataset loader. 
    keypoint_net: torch.nn.module
        Keypoint network.
    output_shape: tuple
        Original image shape.
    top_k: int
        Number of keypoints to use to compute metrics, selected based on probability.    
    use_color: bool
        Use color or grayscale images.
    """
    keypoint_net.eval()
    keypoint_net.training = False

    conf_threshold = 0.0
    localization_err, repeatability = [], []
    correctness1, correctness3, correctness5, MScore = [], [], [], []

    with torch.no_grad():
        for i, sample in tqdm(enumerate(data_loader), desc="evaluate_keypoint_net"):
            if use_color:
                image = to_color_normalized(sample['image'].cuda())
                warped_image = to_color_normalized(sample['warped_image'].cuda())
            else:
                image = to_gray_normalized(sample['image'].cuda())
                warped_image = to_gray_normalized(sample['warped_image'].cuda())

            score_1, coord_1, desc1 = keypoint_net(image)
            score_2, coord_2, desc2 = keypoint_net(warped_image)
            B, C, Hc, Wc = desc1.shape

            # Scores & Descriptors
            score_1 = torch.cat([coord_1, score_1], dim=1).view(3, -1).t().cpu().numpy()
            score_2 = torch.cat([coord_2, score_2], dim=1).view(3, -1).t().cpu().numpy()
            desc1 = desc1.view(C, Hc, Wc).view(C, -1).t().cpu().numpy()
            desc2 = desc2.view(C, Hc, Wc).view(C, -1).t().cpu().numpy()
            
            # Filter based on confidence threshold
            desc1 = desc1[score_1[:, 2] > conf_threshold, :]
            desc2 = desc2[score_2[:, 2] > conf_threshold, :]
            score_1 = score_1[score_1[:, 2] > conf_threshold, :]
            score_2 = score_2[score_2[:, 2] > conf_threshold, :]


            # Prepare data for eval
            data = {'image': sample['image'].numpy().squeeze(),
                    'image_shape' : output_shape,
                    'warped_image': sample['warped_image'].numpy().squeeze(),
                    'homography': sample['homography'].squeeze().numpy(),
                    'prob': score_1, 
                    'warped_prob': score_2,
                    'desc': desc1,
                    'warped_desc': desc2}

            print(data['prob'].shape)
            print(data['prob'].dtype)
            print(data['desc'].shape)
            print(data['desc'].dtype)

            print(data['prob'][:10])
            print(data['desc'][:10][:10])
            input()
            
            # Compute repeatabilty and localization error
            _, _, rep, loc_err = compute_repeatability(data, keep_k_points=top_k, distance_thresh=3)
            repeatability.append(rep)
            localization_err.append(loc_err)

            # Compute correctness
            c1, c2, c3 = compute_homography(data, keep_k_points=top_k)
            correctness1.append(c1)
            correctness3.append(c2)
            correctness5.append(c3)

            # Compute matching score
            mscore = compute_matching_score(data, keep_k_points=top_k)
            MScore.append(mscore)

    return np.mean(repeatability), np.mean(localization_err), \
           np.mean(correctness1), np.mean(correctness3), np.mean(correctness5), np.mean(MScore)

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
        Number of keypoints to use to compute metrics, selected based on probability.    
    """  
    def get_KSD(image):
        #! Get image in numpy
        image = image[0] # Remove SINGLE batch dimension
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
    conf_threshold = 0.0
    localization_err, repeatability = [], []
    correctness1, correctness3, correctness5, MScore = [], [], [], []

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

        # print(score_1.shape)
        # print(desc1.shape)
        # print(score_1[:10])
        # print(desc1[:10][:4])
        # input()

        #! Compute repeatabilty and localization error
        _, _, rep, loc_err = compute_repeatability(data, keep_k_points=top_k, distance_thresh=3)
        repeatability.append(rep)
        localization_err.append(loc_err)

        #! Compute correctness
        c1, c2, c3 = compute_homography(data, keep_k_points=top_k)
        correctness1.append(c1)
        correctness3.append(c2)
        correctness5.append(c3)

        #! Compute matching score
        mscore = compute_matching_score(data, keep_k_points=top_k)
        MScore.append(mscore)

    return np.mean(repeatability), np.mean(localization_err), \
           np.mean(correctness1), np.mean(correctness3), np.mean(correctness5), np.mean(MScore)
