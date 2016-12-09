import numpy as np
import cv2
import os
import string

# Image directory
images_dir = '/scratch3/terriyu/maps'
# Output directory
out_dir = '/scratch3/terriyu/maps_augment'

# List of all files in image directory
image_files = os.listdir(images_dir)
#image_files = ['D0090-5242001.tiff', 'D0041-5370006.tiff']

# Kernels for erosion and dilation
kernel_erode = np.ones((3,3), np.uint8)
kernel_dilate = np.ones((3,3), np.uint8)

for image_file in image_files:
    print "Processing file %s" % image_file
    # Read image file
    img = cv2.imread(os.path.join(images_dir, image_file))
    # Extract root of image file name
    im_root = string.split(string.split(image_file, os.sep)[-1], '.')[0]
    # Erode image (make text fatter)
    eroded_img = cv2.erode(img, kernel_erode, iterations = 1)
    # Dilate image (make text skinnier)
    dilated_img = cv2.dilate(img, kernel_dilate, iterations = 1)
    # Write eroded and dilated images to file
    cv2.imwrite(os.path.join(out_dir, im_root + '_eroded.tiff'), eroded_img) 
    cv2.imwrite(os.path.join(out_dir, im_root + '_dilated.tiff'), dilated_img) 
