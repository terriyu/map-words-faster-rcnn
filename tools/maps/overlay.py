# Script for creating augmented images by overlaying lines over original map images

import numpy as np
import cv2
import os
import string

def adjust_gamma(image, gamma=1.0):
    """Adjust gamma of an image (change brightness)

    Args:
        image (2D array) - image to process
        gamma (float) - gamma value that determines brightness

    Returns:
        corrected_image (2D array) - gamma corrected image 
    """

    # Build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")

    # Apply gamma correction using the lookup table
    corrected_image = cv2.LUT(image, table)
    return corrected_image

def lighten_black(image, val):
    """Make black pixels image lighter color (grayish)

    Args:
        image (2D array) - image to process
        val (int) - specifies grayscale color (val, val, val) 

    Returns:
        new_image (2D array) - image with black pixels replaced by grayscale pixels 
    """

    # Extract dimensions of image
    H, W, C = image.shape
    # Initialize lightened image
    new_image = np.zeros((H,W,C), dtype=np.uint8)

    # Loop through all pixels in image
    for x in xrange(H):
        for y in xrange(W):
            if (np.all(image[x,y,:] == (0,0,0))):
                # If pixel is pure black, make it lighter
                new_image[x,y,:] = (val,val,val)
            else:
                # Otherwise, leave pixel unchanged
                new_image[x,y,:] = image[x,y,:]

    return new_image

if __name__ == '__main__':
    # Image directory
    images_dir = '/scratch3/terriyu/crop_images/base'
    # Overlays directory
    overlays_dir = '/scratch3/terriyu/overlays'
    # Output directory
    out_dir = '/scratch3/terriyu/maps_overlay'

    # List of all files in image directory
    image_files = os.listdir(images_dir)
    #image_files = ['D0089-5235001_58.jpg']

    # Number of overlays to create per image
    num_iter = 3
    # Assume crop has the same width and height
    crop_size = 500

    # Loop through all images
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
            # Randomly select a crop of the overlay image
            x_max = overlay.shape[0] - (H + 2)
            y_max = overlay.shape[1] - (W + 2)
            x_idx = np.random.randint(0, x_max)
            y_idx = np.random.randint(0, y_max)
            overlay = overlay[x_idx:x_idx+H, y_idx:y_idx+W]
            # Make overlay lighter
            overlay = lighten_black(overlay, np.random.randint(0,51))
            random_gamma = 1.0 + np.random.rand()
            overlay = adjust_gamma(overlay, gamma=random_gamma)

            # Create composite
            composite = np.zeros((H,W,C), dtype=np.uint8)
            # Loop through all pixels in image
            for x in xrange(H):
                for y in xrange(W):
                    if (np.all(overlay[x,y,:] == (255,255,255))):
                        # If overlay pixel is white, use original image's pixel
                        composite[x,y,:] = im[x,y,:]
                    else:
                        # Otherwise substitute overlay
                        composite[x,y,:] = np.uint8(overlay[x,y,:])

            # Write image with overlay to file
            cv2.imwrite(os.path.join(out_dir, im_root + '_overlay' + str(i) + '.tiff'), composite)
