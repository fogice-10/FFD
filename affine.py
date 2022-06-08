import sys
import os
import pathlib
import shutil
from pathlib import Path
import cv2
import numpy as np
from multiprocessing.pool import ThreadPool
from utilities import Timer, image_resize
from config import MAX_SIZE
import torch
from ffd_foerstner import detect_foerstner_kp, compute_FFD_keypoints
from save_load_kps import save_kps
from image_matching import init_feature, filter_matches

def affine_skew(tilt, phi, img, mask=None):
    h, w = img.shape[:2]

    if mask is None:
        mask = np.zeros((h, w), np.uint8)
        mask[:] = 255

    A = np.float32([[1, 0, 0], [0, 1, 0]])

    # Rotate image
    if phi != 0.0:
        phi = np.deg2rad(phi)
        s, c = np.sin(phi), np.cos(phi)
        A = np.float32([[c, -s], [s, c]])
        corners = [[0, 0], [w, 0], [w, h], [0, h]]
        tcorners = np.int32(np.dot(corners, A.T))
        x, y, w, h = cv2.boundingRect(tcorners.reshape(1, -1, 2))
        A = np.hstack([A, [[-x], [-y]]])
        img = cv2.warpAffine(img, A, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    # Tilt image (resizing after rotation)
    if tilt != 1.0:
        s = 0.8 * np.sqrt(tilt * tilt - 1)
        img = cv2.GaussianBlur(img, (0, 0), sigmaX=s, sigmaY=0.01)
        img = cv2.resize(img, (0, 0), fx=1.0 / tilt, fy=1.0, interpolation=cv2.INTER_NEAREST)
        A[0] /= tilt

    if phi != 0.0 or tilt != 1.0:
        h, w = img.shape[:2]
        mask = cv2.warpAffine(mask, A, (w, h), flags=cv2.INTER_NEAREST)
    #Ai is an affine transform matrix from skew_img to img
    Ai = cv2.invertAffineTransform(A)

    return img, mask, Ai


def affine_detect(detector, img, pool=None):
    params = [(1.0, 0.0)]

    # Simulate all possible affine transformations
    for t in 2 ** (0.5 * np.arange(1, 6)):
        for phi in np.arange(0, 180, 72.0 / t):
            params.append((t, phi))

    def f(p):
        t, phi = p
        timg, tmask, Ai = affine_skew(t, phi, img)
        ks = compute_FFD_keypoints(timg)
        #keypoints, descrs = detectAndCompute(timg, ks)
        keypointss = []
        for el in ks:
            keypointss.append(cv2.KeyPoint(el[1], el[2], 6, el[3], el[0]))
            
        ks_foer = detect_foerstner_kp(timg)
        
        keypoints = keypointss + ks_foer
        
        keypoints = keypointss
        
        des = detector.compute(timg, keypoints)
        descrs = des[1]
       
        for kp in keypoints:
            x, y = kp.pt
            kp.pt = tuple(np.dot(Ai, (x, y, 1)))

        if descrs is None:
            descrs = []

        return keypoints, descrs, keypointss

    keypoints, descrs, keypointss = [], [], []

    ires = pool.imap(f, params)

    for i, (k, d, kss) in enumerate(ires):
        print(f"Affine sampling: {i + 1} / {len(params)}\r", end='')
        keypoints.extend(k)
        descrs.extend(d)
        keypointss.extend(kss)

    print()

    return keypoints, np.array(descrs), keypointss


def affine_main(image1: str, image2: str, kps_path_save, desc_path_save, detector_name: str = "sift-flann"):
    # Read images
    ori_img1 = cv2.imread(image1, cv2.IMREAD_GRAYSCALE)
    ori_img2 = cv2.imread(image2, cv2.IMREAD_GRAYSCALE)

    # Initialize feature detector and keypoint matcher
    detector, matcher = init_feature(detector_name)

    # Exit when reading empty image
    if ori_img1 is None or ori_img2 is None:
        print("Failed to load images")
        sys.exit(1)

    ratio_1 = 1
    ratio_2 = 1

    if ori_img1.shape[0] > MAX_SIZE or ori_img1.shape[1] > MAX_SIZE:
        ratio_1 = MAX_SIZE / ori_img1.shape[1]
        print("Large input detected, image 1 will be resized")
        img1 = image_resize(ori_img1, ratio_1)
    else:
        img1 = ori_img1

    if ori_img2.shape[0] > MAX_SIZE or ori_img2.shape[1] > MAX_SIZE:
        ratio_2 = MAX_SIZE / ori_img2.shape[1]
        print("Large input detected, image 2 will be resized")
        img2 = image_resize(ori_img2, ratio_2)
    else:
        img2 = ori_img2

    # Profile time consumption of keypoints extraction
    with Timer(f"Extracting keypoints..."):
        pool = ThreadPool(processes=cv2.getNumberOfCPUs())
        kp1, desc1, kps1 = affine_detect(detector, img1, pool=pool)
        kp2, desc2, kps2 = affine_detect(detector, img2, pool=pool)
    
    path_to_file1 = pathlib.Path(kps_path_save+str(Path(image1).stem))
    if os.path.exists(path_to_file1):
        shutil.rmtree(path_to_file1)
    os.makedirs(path_to_file1)
    
    path_to_file2 = pathlib.Path(kps_path_save+str(Path(image2).stem))
    if os.path.exists(path_to_file2):
        shutil.rmtree(path_to_file2)
    os.makedirs(path_to_file2)
    
    cnt=1
    for i in range(0, len(kps1), 5000):
        save_kps(kps1[i:i+5000], kps_path_save+str(Path(image1).stem)+'/'+str(cnt)+'.txt')
        cnt=cnt+1
    
    cnt2=1
    for i in range(0, len(kps2), 5000):
        save_kps(kps2[i:i+5000], kps_path_save+str(Path(image2).stem)+'/'+str(cnt2)+'.txt')
        cnt2=cnt2+1
    
    print('Done')

