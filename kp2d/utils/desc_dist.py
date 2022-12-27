from scipy.spatial.distance import cdist
import numpy as np
import cv2


def get_custom_descriptor_distance(desc1, desc2, alpha=0.5):
    #! Descriptor type float
    desc1 = desc1.astype(np.float64)
    desc2 = desc2.astype(np.float64)

    #! split the descriptors into two parts
    part1 = desc1[:32]
    part2 = desc1[32:]
    part3 = desc2[:32]
    part4 = desc2[32:]

    #! calculate the distance(L2 norm) between the corresponding parts
    dist1 = np.sqrt(np.sum((part1 - part3)**2))
    dist2 = np.sqrt(np.sum((part2 - part4)**2))

    #! combine the distances using the hyperparameter alpha
    dist = alpha * dist1 + (1 - alpha) * dist2

    #! return the distance between the descriptors
    return dist


def get_descriptor_distance(desc1, desc2):
    desc1 = desc1.astype(np.float64)
    desc2 = desc2.astype(np.float64)

    return np.sqrt(np.sum((desc1 - desc2)**2))


def bf_matcher(des1, des2, alpha=0.5, cross_check=False):

    # initialize the list of matches
    matches = []

    # compute the distances between the descriptors
    # dists = cdist(des1, des2, lambda x, y: get_descriptor_distance(x, y))
    dists = cdist(des1, des2, lambda x, y: get_custom_descriptor_distance(x, y, alpha=alpha))

    # loop over the descriptors in des1
    for i in range(des1.shape[0]):
        best_match = np.argmin(dists[i])

        if cross_check:
            r_best_match = np.argmin(dists.T[best_match])
            if r_best_match == i:
                m = cv2.DMatch(_queryIdx=i, _trainIdx=best_match, _distance=dists[i][best_match])
                matches.append(m)
        else:
            m = cv2.DMatch(_queryIdx=i, _trainIdx=best_match, _distance=dists[i][best_match])
            matches.append(m)

    return matches