# This code extracts patches ROI bounding boxes from annotations and its mask, ensuring enough overlap with the mask and removing white regions in slide patches.

from sklearn.feature_extraction import image
import numpy as np
import cv2
import os
import glob
import sys

# Source and destination paths (placeholders)
src = '../data/img'
dest = '../data/TestData/LumA' #change this for TrainData/ValData and LumA/NLumA
mpath = '../data/mask'

fold = glob.glob(src + '/*')  # Get all folders in the source path
print(fold)

# Parameters
window = 512
win = 512
threshold = 210
slide = 1
tot_count = window * window

# Iterate through each folder in the source path
for f in fold:
    files = glob.glob(f + '/' + '*.png')  # Get all .png files in the folder
    
    # Iterate through each file in the folder
    for f1 in files:
        fname = f1.split('/')[-1]
        dest_fname = fname.split('.png')[0]
        foldname = f1.split('/')[-2]
        
        # Create destination folder if it doesn't exist
        if not os.path.exists(dest + foldname):
            os.makedirs(dest + foldname)
        
        os.chdir(dest + foldname)  # Change directory to destination folder
        img = cv2.imread(f1)  # Read the image
        sr = mpath + foldname + '/' + fname
        mask = cv2.imread(mpath + foldname + '/' + fname, 0)  # Read the mask
        
        row, col = img.shape[:2]  # Get image dimensions
        
        # Iterate through the image in steps of 'win/slide'
        for i in range(0, row, int(win / slide)):
            if (i + win > row):
                rlast_index = row + 1
                rfirst_index = row - win
            else:
                rlast_index = i + win 
                rfirst_index = i       
            for j in range(0, col, int(win / slide)):
                if (j + win > col):
                    clast_index = col + 1
                    cfirst_index = col - win
                else:
                    clast_index = j + win 
                    cfirst_index = j
                
                mask_patch = mask[rfirst_index:rlast_index, cfirst_index:clast_index]
                patch = img[rfirst_index:rlast_index, cfirst_index:clast_index]
                
                black_count = np.sum(mask_patch == 0)  # Count black pixels in the mask patch
                percent_black = black_count * 100 / tot_count
                
                # Check if the percentage of black pixels is less than or equal to 40%
                if percent_black <= 40.0:
                    channel_1 = patch[:, :, 0] > threshold
                    channel_2 = patch[:, :, 1] > threshold
                    channel_3 = patch[:, :, 2] > threshold
                    
                    # Check if all channels have pixel values above the threshold
                    vfunc = np.vectorize(np.logical_and)
                    pixel_and = vfunc(vfunc(channel_1, channel_2), channel_3)
                    pixel_and_count = np.count_nonzero(pixel_and)
                    ratio_white_pixel = float(pixel_and_count * 100 / (window * window))
                    
                    # Save the patch if the conditions are met
                    if (ratio_white_pixel < 40) and patch.shape == (512, 512, 3):
                        cv2.imwrite(dest_fname + '_' + str(i) + '_' + str(j) + '.png', patch)
