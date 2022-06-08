import cv2
import kornia as K
import kornia.feature as KF
import numpy as np
import torch
import os
from save_load_kps import load_kps
from kornia_moons.feature import *

def find_descriptors(folder_path, imgname, hardnet, write_path):
    path, dirs, files = next(os.walk(folder_path))
    file_count = len(files)
    img = cv2.cvtColor(cv2.imread(imgname), cv2.COLOR_BGR2RGB)
    descs=[]
    
    for i in range(1,file_count+1):
        read_from_txt=load_kps(folder_path+str(i)+'.txt')
        descs.append(get_local_descriptors(img, read_from_txt, hardnet))
    
    flat_descs = np.asarray([item for sublist in descs for item in sublist])
    np.savetxt(write_path, flat_descs, fmt='%f')
    
    return flat_descs

def get_local_descriptors(img, cv2_sift_kpts, kornia_descriptor):
    if len(cv2_sift_kpts)==0:
        return np.array([])
  
   #We will not train anything
    with torch.no_grad():
        kornia_descriptor.eval()
        timg = K.color.rgb_to_grayscale(K.image_to_tensor(img, False).float()/255.)
        lafs = laf_from_opencv_SIFT_kpts(cv2_sift_kpts)
        patches = KF.extract_patches_from_pyramid(timg,lafs, 32)
        B, N, CH, H, W = patches.size()
        # Descriptor accepts standard tensor [B, CH, H, W], while patches are [B, N, CH, H, W] shape
        # So we need to reshape a bit
        descs = kornia_descriptor(patches.view(B * N, CH, H, W)).view(B * N, -1)
    return descs.detach().cpu().numpy()