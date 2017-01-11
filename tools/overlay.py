import numpy as np
import cv2
import os
import string

def adjust_gamma(image, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
 
    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)

def lighten_black(image, val):
    H, W, C = image.shape
    new_image = np.zeros((H,W,C), dtype=np.uint8)
    # Loop through all pixels in image
    for x in xrange(H):
        for y in xrange(W):
            if (np.all(image[x,y,:] == (0,0,0))):
                new_image[x,y,:] = (val,val,val)
            else:
                new_image[x,y,:] = image[x,y,:]

    return new_image

# Image directory
images_dir = '/scratch3/terriyu/crop_images/base'
# Overlays directory
overlays_dir = '/scratch3/terriyu/overlays'
# Output directory
out_dir = '/scratch3/terriyu/maps_overlay'


# List of all files in image directory
image_files = os.listdir(images_dir)
#image_files = ['D0089-5235001_58.jpg']

num_iter = 3
crop_size = 500

for image_file in image_files:
    print "Processing file %s" % image_file
    # Read image file
    im = cv2.imread(os.path.join(images_dir, image_file))
    # Extract root of image file name
    im_root = string.split(string.split(image_file, os.sep)[-1], '.')[0]
    # Extract image dimensions
    H, W, C = im.shape
 
    for i in xrange(1, num_iter + 1):
        # Read and construct overlay
        idx = np.random.randint(1, 44)
        overlay = cv2.imread(os.path.join(overlays_dir, 'overlay' + str(idx) + '.PNG'))
        # Randomly select crop
        x_max = overlay.shape[0] - (H + 2)
        y_max = overlay.shape[1] - (W + 2)
        x_idx = np.random.randint(0, x_max)
        y_idx = np.random.randint(0, y_max)
        overlay = overlay[x_idx:x_idx+H, y_idx:y_idx+W]
        # Make overlay lighter
        overlay = lighten_black(overlay, np.random.randint(0,51))
        random_gamma = 1.0 + np.random.rand()
        overlay = adjust_gamma(overlay, gamma=random_gamma) 
        
        composite = np.zeros((H,W,C), dtype=np.uint8)
        for x in xrange(H):
            for y in xrange(W):
                if (np.all(overlay[x,y,:] == (255,255,255))):
                    # If overlay is white, use original image
                    composite[x,y,:] = im[x,y,:]
                else:
                    # Otherwise substitute overlay
                    composite[x,y,:] = np.uint8(overlay[x,y,:])

        # Write image with overlay to file
        cv2.imwrite(os.path.join(out_dir, im_root + '_overlay' + str(i) + '.tiff'), composite) 
