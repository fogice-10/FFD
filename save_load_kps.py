import cv2
import _pickle as cPickle
import os
import numpy as np
import shutil
from pathlib import Path
import pathlib
import os

def save_kps(kps, filename):
    f = open(filename, "w")
    
    for point in kps:
        p = str(point.pt[0]) + "," + str(point.pt[1]) + "," + str(point.size) + "," + str(point.angle) + "," + str(point.response) + "," + str(point.octave) + "," + str(point.class_id) + "\n"
        f.write(p)
    f.close()


def load_kps(filename):
    kps = []
    lines = [line.strip() for line in open(filename)]
    
    for line in lines:
        list = line.split(',')
        kp = cv2.KeyPoint(x=float(list[0]), y=float(list[1]), size=float(list[2]), angle=float(list[3]), response=float(list[4]), octave=int(list[5]), class_id=int(list[6]))
        kps.append(kp)
    
    return kps

def save_descrs_to_files(image_path, path_to_save, desc):
    cnt=0
    os.mkdir(path_to_save+str(os.path.basename(image_path))) 
    for i in range(0, len(desc), 5000):
        np.savetxt(path_to_save+str(os.path.basename(image_path))+'/'+str(cnt)+'.txt', desc[i:i+5000])
        cnt=cnt+1

def read_keypoints(folder_path):
    path, dirs, files = next(os.walk(folder_path))
    file_count = len(files)
    keypoints=[]
    for i in range(1,file_count+1):
        keypoints.append(load_kps(folder_path+str(i)+'.txt'))
    flat_keypoints = [item for sublist in keypoints for item in sublist]
    
    return flat_keypoints

def load_kps_ver2(filename):
    kps = []
    lines = [line.strip() for line in open(filename)]
    
    for line in lines:
        list = line.split(',')
        kp = cv2.KeyPoint(x=float(list[0]), y=float(list[1]), _size=float(list[2]), _angle=float(list[3]), _response=float(list[4]), _octave=int(list[5]), _class_id=int(list[6]))
        kps.append(kp)
    
    return kps

