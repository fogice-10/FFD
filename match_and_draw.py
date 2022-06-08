#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from save_load_kps import read_keypoints
#from save_load_matches import save_matches, load_matches
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pydegensac
from time import time
from copy import deepcopy

#from descriptors import read_descriptors

from skimage.measure import ransac as skransac
from skimage.transform import FundamentalMatrixTransform

def decolorize(img):
    return  cv2.cvtColor(cv2.cvtColor(img,cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB)

#Now helper function for running homography RANSAC
def verify_pydegensac_fundam(kps1, kps2, tentatives, th = 1.0,  n_iter = 10000):
    src_pts = np.float32([ kps1[m.queryIdx].pt for m in tentatives ]).reshape(-1,2)
    dst_pts = np.float32([ kps2[m.trainIdx].pt for m in tentatives ]).reshape(-1,2)
    F, mask = pydegensac.findFundamentalMatrix(src_pts, dst_pts, th, 0.999, n_iter, enable_degeneracy_check= True)
    print ('pydegensac found {} inliers'.format(int(deepcopy(mask).astype(np.float32).sum())))
    return F, mask

def draw_matches_F(kps1, kps2, tentatives, img1, img2, mask):
    matchesMask = mask.ravel().tolist()
    h,w,ch = img1.shape
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    draw_params = dict(matchColor = (255,255,0), # draw matches in yellow color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)
    img_out = cv2.drawMatches(decolorize(img1),kps1,
                              decolorize(img2),kps2,tentatives,None,**draw_params)
    plt.figure()
    fig, ax = plt.subplots(figsize=(15, 15))
    ax.imshow(img_out, interpolation='nearest')
    return

def match_draw(img1_path, img2_path, kps1_path, kps2_path, descs1_path, descs2_path):
    img1 = cv2.cvtColor(cv2.imread(img1_path), cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(cv2.imread(img2_path), cv2.COLOR_BGR2RGB)
    kps1 = read_keypoints(kps1_path)
    kps2 = read_keypoints(kps2_path)
    #descs1 = np.float32(read_descriptors(descs1_path))
    #descs2 = np.float32(read_descriptors(descs2_path))
    descs1 = np.float32(np.loadtxt(descs1_path))
    descs2 = np.float32(np.loadtxt(descs2_path))
    
    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descs1,descs2, k=2)

    # Need to draw only good matches, so create a mask
    matchesMask = [False for i in range(len(matches))]

    # SNN ratio test
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.99*n.distance:
            matchesMask[i]=True
    tentatives = [m[0] for i, m in enumerate(matches) if matchesMask[i] ]

    th = 0.5
    n_iter = 50000

    t=time()
    cmp_H, cmp_mask = verify_pydegensac_fundam(kps1,kps2,tentatives, th, n_iter)
    print ("{0:.5f}".format(time()-t), ' sec pydegensac')


    draw_matches_F(kps1, kps2, tentatives, img1, img2, cmp_mask)
    print (f'{cmp_mask.sum()} inliers found')

