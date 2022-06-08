import cv2
import numpy as np
from scipy.ndimage.filters import uniform_filter
from scipy.signal import convolve2d
import os
import shutil

def clahe(bgr):
    lab = cv2.cvtColor(bgr, cv2.COLOR_RGB2GRAY) #BGR2LAB
    lab_planes = cv2.split(lab)
    gridsize=8
    clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(gridsize,gridsize))
    lab_planes[0] = clahe.apply(lab_planes[0])
    return lab_planes[0]

def bm3d(image_name):
    import bm3d
    denoised_image = bm3d.bm3d(image_name, sigma_psd=30/255, stage_arg=bm3d.BM3DStages.HARD_THRESHOLDING)
    return denoised_image

def wallis_filter(image):
    win=20
    tars=150
    tarm=150
    b=1
    c=0.9995

    img = cv2.normalize(image.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
    bit = 255

    # Padding
    impad = np.pad(img,(int(win/2), int(win/2)),'symmetric')

    # Loop impad dimensions
    lenx, leny = impad.shape

    # imfilter of impad
    hsize = [win+1, win+1]
    h = np.ones((win+1, win+1), np.float32)/(win**2)
    imf = cv2.filter2D(impad.astype('float32'), -1, h, borderType=cv2.BORDER_CONSTANT)

    # stdfilt of impad
    windowSize = 21
    imstd = window_stdev(impad.astype('float'), windowSize)*np.sqrt(windowSize/(windowSize-1))

    img_wallis = np.ones(impad.shape)
    for i in range(0, lenx):
        for j in range(0, leny):
            img_wallis[i,j] = (impad[i,j]-imf[i,j])*c*tars/(c*imstd[i,j]+(1-c)*tars)+b*tarm+(1-b)*imf[i,j]
            if img_wallis[i,j]<0:
                img_wallis[i,j]=0
            elif img_wallis[i,j]>255:
                img_wallis[i,j]=255

    # De-padding
    end1, end2 = img_wallis.shape
    img_wallis = img_wallis[1+int(win/2):end1-int(win/2), 1+int(win/2):end2-int(win/2)]

    return img_wallis

def window_stdev(X, window_size):
    r,c = X.shape
    X+=np.random.rand(r,c)*1e-6
    c1 = uniform_filter(X, window_size, mode='reflect')
    c2 = uniform_filter(X*X, window_size, mode='reflect')
    return np.sqrt(c2 - c1*c1)

# save images to chosen folder
def save_image(new_path, i, processed_image):
    #write the whole path for a new image with a name under which the image will be saved
    img_name = new_path + str(i) + '.jpg'
    cv2.imwrite(img_name, processed_image)

# main process function: clahe+bm3d+wallis and saving to folder 
def process_images(unprocessed_img, new_path):
    if os.path.exists(new_path):
        shutil.rmtree(new_path)
    os.makedirs(new_path)

    i = 0
    #for every image from folder:
    while i < len(unprocessed_img):
        img = cv2.imread(unprocessed_img[i])
        #process image
        clahe_image = clahe(img)
        denoised_image = bm3d(clahe_image)
        wallis_image = wallis_filter(denoised_image)
        #save image
        save_image(new_path, i, wallis_image)
        i += 1

