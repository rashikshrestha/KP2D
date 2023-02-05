# Copyright 2020 Toyota Research Institute.  All rights reserved.

import numpy as np
import torch
import torchvision.transforms as transforms
from tqdm import tqdm
import sys
from timeit import default_timer as timer

np.set_printoptions(suppress=True, threshold=sys.maxsize)


from kp2d.evaluation.descriptor_evaluation import (compute_homography,
                                                   compute_matching_score)
from kp2d.evaluation.detector_evaluation import compute_repeatability
from kp2d.utils.image import to_color_normalized, to_gray_normalized

#! Stuffs from vfm
from vfm.utils.descriptor_loss import sample_descriptors


def evaluate_keypoint_net(data_loader, keypoint_net, output_shape=(320, 240), top_k=300, use_color=True, device='cuda'):
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
    duration = []

    with torch.no_grad():
        for i, sample in tqdm(enumerate(data_loader), desc="evaluate_keypoint_net"):
            #! -----------------------------------------------------------------------------
            start = timer()

            if use_color:
                image = to_color_normalized(sample['image'].to(device))
                warped_image = to_color_normalized(sample['warped_image'].to(device))
            else:
                image = to_gray_normalized(sample['image'].to(device))
                warped_image = to_gray_normalized(sample['warped_image'].to(device))

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

            end = timer()
            duration.append(end-start)
            #! -----------------------------------------------------------------------------

            # print(data['prob'].shape)
            # print(data['prob'].dtype)
            # print(data['desc'].shape)
            # print(data['desc'].dtype)

            # print(data['prob'][:10])
            # print(data['desc'][:10][:10])
            # input()
            
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
           np.mean(correctness1), np.mean(correctness3), np.mean(correctness5), np.mean(MScore), np.mean(duration)


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
    conf_threshold = 0.0
    localization_err, repeatability = [], []
    correctness1, correctness3, correctness5, MScore = [], [], [], []
    dur = []

    #! For each data point:
    for i, sample in tqdm(enumerate(data_loader), desc="evaluate_ORB"):
        #! -----------------------------------------------------------------------------
        start = timer()

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

        end = timer()
        dur.append(end-start)
        #! -----------------------------------------------------------------------------

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
           np.mean(correctness1), np.mean(correctness3), np.mean(correctness5), np.mean(MScore), \
           np.mean(dur)


def evaluate_fastfeature(data_loader, orb, fastfeature, output_shape=(320, 240), top_k=300):

    transform_to_gray = transforms.Grayscale()

    def get_KSD(image):
        #! Get image in numpy
        image_ = image[0] # Remove SINGLE batch dimension
        # Swap the dimenstion to be compatable with opencv and change the range from 0-1 to 0-255
        image_ = image_.permute((1,2,0))*255. 
        img_np = image_.cpu().numpy()
        img_np = img_np.astype(np.uint8)

        #! Detect kpts
        keypoints, orbdesc = orb.detectAndCompute(img_np, None)
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

        #! Use model to get dense_desc 
        img = image.cuda()
        img = transform_to_gray(img)
        dense_desc = fastfeature(img)

        #! Get Batchsize, img height, img width
        B, _, H, W = image.shape

        #! Sample from dense descriptors
        kpts_tensor = torch.from_numpy(kpts)
        sampled_desc = sample_descriptors(kpts_tensor, dense_desc[0], H, W)
        sampled_desc_np = sampled_desc.detach().cpu().numpy()

        # print(sampled_desc_np.shape)

        #! concat with orb desc 
        desc = np.concatenate((orbdesc, sampled_desc_np), 1)

        return kpts_scores, desc


    #! Change the fastfeature model to evaluation mode
    fastfeature.eval()

    #! Accumulatiors
    conf_threshold = 0.0
    localization_err, repeatability = [], []
    correctness1, correctness3, correctness5, MScore = [], [], [], []
    duration = []

    with torch.no_grad():
        #! For each data point:
        for i, sample in tqdm(enumerate(data_loader), desc="evaluate_fastfeature"):
            # if i==50:
            #     break
            #! -----------------------------------------------------------------------------
            start = timer()

            score_1, desc1 = get_KSD(sample['image'])
            score_2, desc2 = get_KSD(sample['warped_image'])
            # print(desc1)

            #! Prepare data for eval
            data = {'image': sample['image'].numpy().squeeze(),
                    'image_shape' : output_shape,
                    'warped_image': sample['warped_image'].numpy().squeeze(),
                    'homography': sample['homography'].squeeze().numpy(),
                    'prob': score_1, 
                    'warped_prob': score_2,
                    'desc': desc1,
                    'warped_desc': desc2}

            end = timer()
            duration.append(end-start)
            #! -----------------------------------------------------------------------------

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
           np.mean(correctness1), np.mean(correctness3), np.mean(correctness5), np.mean(MScore), \
           np.mean(duration)


def evaluate_fastfeature_orbnet(data_loader, orb, fastfeature, orbnet, output_shape=(320, 240), top_k=300):

    transform_to_gray = transforms.Grayscale()

    def get_KSD(image):
        #! Get image in numpy
        image_ = image[0] # Remove SINGLE batch dimension
        # Swap the dimenstion to be compatable with opencv and change the range from 0-1 to 0-255
        image_ = image_.permute((1,2,0))*255. 
        img_np = image_.cpu().numpy()
        img_np = img_np.astype(np.uint8)

        #! Detect kpts
        keypoints, orbdesc = orb.detectAndCompute(img_np, None)
        kpts = []
        scores = []
        for k in keypoints:
            kpts.append([k.pt[0], k.pt[1]])
            scores.append(k.response)
        kpts = np.array(kpts)
        scores = np.array(scores)

        #! Map descriptors from 0-255 to 0-1
        orbdesc = orbdesc.astype(np.float32)
        orbdesc /= 255

        #! Use orbnet
        orbdesc_tensor = torch.as_tensor(orbdesc, dtype=torch.float32)
        orbdesc_refined_tensor = orbnet(orbdesc_tensor)
        orbdesc_refined_np = orbdesc_refined_tensor.detach().cpu().numpy()

        # print("ORB desc shape:")
        # print(orbdesc_refined_np.shape)



        #! Normalize the scores
        scores_norm = np.linalg.norm(scores)
        scores /= scores_norm
        kpts_scores = np.hstack((kpts, scores.reshape(-1,1)))

        #! Use model to get dense_desc 
        img = image.cuda()
        img = transform_to_gray(img)
        dense_desc = fastfeature(img)

        #! Get Batchsize, img height, img width
        B, _, H, W = image.shape

        #! Sample from dense descriptors
        kpts_tensor = torch.from_numpy(kpts)
        sampled_desc = sample_descriptors(kpts_tensor, dense_desc[0], H, W)
        sampled_desc_np = sampled_desc.detach().cpu().numpy()

        # print(sampled_desc_np.shape)

        #! concat with orb desc 
        desc = np.concatenate((orbdesc_refined_np, sampled_desc_np), 1)

        return kpts_scores, desc


    #! Change the fastfeature model to evaluation mode
    fastfeature.eval()

    #! Accumulatiors
    conf_threshold = 0.0
    localization_err, repeatability = [], []
    correctness1, correctness3, correctness5, MScore = [], [], [], []
    duration = []

    with torch.no_grad():
        #! For each data point:
        for i, sample in tqdm(enumerate(data_loader), desc="evaluate_fastfeature"):
            # if i==50:
            #     break
            #! -----------------------------------------------------------------------------
            start = timer()

            score_1, desc1 = get_KSD(sample['image'])
            score_2, desc2 = get_KSD(sample['warped_image'])
            # print(desc1)

            #! Prepare data for eval
            data = {'image': sample['image'].numpy().squeeze(),
                    'image_shape' : output_shape,
                    'warped_image': sample['warped_image'].numpy().squeeze(),
                    'homography': sample['homography'].squeeze().numpy(),
                    'prob': score_1, 
                    'warped_prob': score_2,
                    'desc': desc1,
                    'warped_desc': desc2}

            end = timer()
            duration.append(end-start)
            #! -----------------------------------------------------------------------------

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
           np.mean(correctness1), np.mean(correctness3), np.mean(correctness5), np.mean(MScore), \
           np.mean(duration)
