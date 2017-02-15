# Script to write labels for cropped images

from __future__ import division
import os
import json
import string
import argparse
import cv2
import numpy as np

import matplotlib.pyplot as plt

BLACK = (0, 0, 0)
BOX_COLOR = (0, 0, 255)
LINE_WIDTH = 2

def boxes_contained_in_crop(h_start, h_end, w_start, w_end, boxes, crop_size, image=None, check=False):
    """
    """

    box_color = (0, 0, 255)
    line_width = 2

    boxes_in_crop = None
    idx_in_crop = []

    if boxes.size != 0:
        # Boolean logical array corresponding to whether upper left corner of box
        # is inside the crop
        upper_left_in = (boxes[:, 0] > w_start) & (boxes[:, 0] < w_end) & (boxes[:, 1] > h_start) & (boxes[:, 1] < h_end)
        # Boolean logical array corresponding to whether bottom right corner of box
        # is inside the crop
        bottom_right_in = (boxes[:, 0] + boxes[:, 2] < w_end) & (boxes[:, 1] + boxes[:, 3] < h_end)

        if np.sum(upper_left_in & bottom_right_in) > 0:
            idx_in_crop = np.where(upper_left_in & bottom_right_in)[0].tolist()
            boxes_in_crop = boxes[idx_in_crop, :]
            if check:
                print "Crop bounding coordinates upper left = (%i, %i), lower right = (%i, %i)" % (w_start, w_end, h_start, h_end)
                print "Bounding boxes in crop (upper_left_x, upper_left_y, width, height)"
                print boxes_in_crop

        if check and (boxes_in_crop is not None) and (image is not None):
            num_boxes = boxes_in_crop.shape[0]
            for idx in xrange(num_boxes):
                # Upper left x-coordinate, upper left y-coordinate, box width, box height
                ul_x, ul_y, w, h = np.float32(boxes_in_crop[idx, :])
                # Shift coordinates for cropped image
                ul_x -= np.float32(w_start)
                ul_y -= np.float32(h_start)
                cv2.rectangle(image, (ul_x, ul_y), (ul_x + w, ul_y + h), box_color, line_width)

    return boxes_in_crop, idx_in_crop

def write_crop_and_labels(im_path, im, boxes, w_start, h_start, file_obj, always_write_flag):
    """
    """

    if (boxes is not None) or always_write_flag:
        if boxes is not None:
            num_boxes = boxes.shape[0]
        else:
            num_boxes = 0
        cv2.imwrite(im_path, im)
        print >> file_obj, "./%s" % im_path
        print >> file_obj, "%i" % num_boxes
        for i in xrange(num_boxes):
            print >> file_obj, "%f %f %f %f" % (boxes[i, 0] - w_start, boxes[i, 1] - h_start, boxes[i, 2], boxes[i, 3])

def parse_args():
    """Parse input arguments"""

    parser = argparse.ArgumentParser(description='Write training labels for cropped images')
    parser.add_argument('--image_dir', required=True, dest='image_dir', help='Specify path to directory for image files')
    parser.add_argument('--crop_dir', required=True, dest='crop_dir', help='Specify path to destination directory for cropped image files')
    parser.add_argument('--train_list', required=True, dest='train_file', help='Specify path to text file containing list of training image file names')
    parser.add_argument('--boxes', required=True, dest='boxes_file', help='Specify path to JSON file containing bounding boxes')
    parser.add_argument('--dest', required=True, dest='dest_file', help='Specify path to write file containing training labels')
    parser.add_argument('--crop_size', dest='crop_size', default=500, type=int, help='Size in pixels to crop image (assume square crops)')
    parser.add_argument('--crop_overlap', dest='crop_overlap', default=100, type=int, help='Crop overlap in pixels')
    parser.add_argument('--neg_examples', dest='neg_examples', action='store_true', help='Flag to specify that negative examples will be included in cropped images and output labels')
    parser.add_argument('--check', dest='check', action='store_true', help='Flag to specify that additional info will be printed and bounding boxes will be drawn on images to check everything is working correctly')

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_args()

    # Load bounding boxes
    with open(args.boxes_file) as f:
        boxes_dict = json.load(f)

    # Angles to use
    angles = boxes_dict['angles']
    #angles = [48]

    # Load list of training image file names
    with open(args.train_file) as f:
        lines = f.readlines()

    train_files = [l.rstrip() for l in lines]
    #train_files = [train_files[0]]

    # Open file for writing labels
    f_labels = open(args.dest_file, 'w')

    # Keep stats on labels covered in crops of image
    # These stats are for all images in the directory
    num_boxes_all = 0
    num_boxes_covered_all = 0
    num_missed_all = 0
    num_duplicates_all = 0
    width_missed = []
    height_missed = []

    # Loop through all training images
    for im_file in train_files:
        # Extract image name
        im_root = string.split(im_file, '.')[0]
        print "Processing %s.tiff ...\n" % im_root
        # Load original image
        im_orig = cv2.imread(os.path.join(args.image_dir, im_file))
        # Pad image (for rotation)
        height_padding = boxes_dict['padding'][im_root]['height_padding']
        width_padding = boxes_dict['padding'][im_root]['width_padding']
        im_padded = cv2.copyMakeBorder(im_orig, height_padding, height_padding, width_padding, width_padding, cv2.BORDER_CONSTANT, value = BLACK)
        height, width, _ = im_padded.shape
        # Keep stats on labels covered in crops of image
        # These stats are specific to a single image
        num_boxes_image = 0
        num_boxes_covered_image = 0
        num_missed_image = 0
        num_duplicates_image = 0
        # Loop through all angles
        for angle in angles:
            print "Processing angle = %i degrees" % angle
            # Extract boxes for this angle
            boxes_angle = np.array(boxes_dict['data'][im_root][str(angle)])
            # Keep track of which boxes (annotations) we've captured in the crops, for this angle
            idx_covered = []
            # Rotate image
            rot_mat = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), angle, 1)
            im_rot = cv2.warpAffine(im_padded, rot_mat, (width, height))
            remain_h = height
            row = 0
            while (remain_h > args.crop_size):
                col = 0
                row += 1
                remain_w = width
                h_start = height - remain_h
                h_end = h_start + args.crop_size
                # All of these crops are full-size,
                # have size args.crop_size x args.crop_size
                while (remain_w > args.crop_size):
                    w_start = width - remain_w
                    w_end = w_start + args.crop_size
                    im_crop = im_rot[h_start:h_end, w_start:w_end].copy()
                    # Find any bounding boxes in this crop
                    boxes_crop, idx_crop = boxes_contained_in_crop(h_start, h_end, w_start, w_end, boxes_angle, args.crop_size, im_crop, args.check)
                    idx_covered.extend(idx_crop)
                    # Write crop
                    col += 1
                    im_name = im_root + '_' + str(angle) + 'deg_crop_r' + format(row, '02d') + 'c' + format(col, '02d') + '.tiff'
                    if args.check:
                        print "%s" % im_name

                    im_path = os.path.join(args.crop_dir, im_name)
                    write_crop_and_labels(im_path, im_crop, boxes_crop, w_start, h_start, f_labels, args.neg_examples)
                    # Update width values
                    remain_w -= args.crop_size - args.crop_overlap
                # Last crop in row
                # height = args.crop_size, but width < args.crop_size
                if (remain_w > 0):
                    w_start = width - args.crop_size
                    w_end = width
                    im_crop = im_rot[h_start:h_end, w_start:w_end].copy()
                    # Find any bounding boxes in this crop
                    boxes_crop, idx_crop = boxes_contained_in_crop(h_start, h_end, w_start, w_end, boxes_angle, args.crop_size, im_crop, args.check)
                    idx_covered.extend(idx_crop)
                    # Write crop
                    col += 1
                    im_name = im_root + '_' + str(angle) + 'deg_crop_r' + format(row, '02d') + 'c' + format(col, '02d') + '.tiff'
                    if args.check:
                        print "Wrote %s" % im_name

                    im_path = os.path.join(args.crop_dir, im_name)

                    write_crop_and_labels(im_path, im_crop, boxes_crop, w_start, h_start, f_labels, args.neg_examples)
                # Update height values
                remain_h -= args.crop_size - args.crop_overlap
            # Last row of crops
            if (remain_h > 0):
                col = 0
                row += 1
                remain_w = width
                h_start = height - args.crop_size
                h_end = height
                while (remain_w > args.crop_size):
                    w_start = width - remain_w
                    w_end = w_start + args.crop_size
                    im_crop = im_rot[h_start:h_end, w_start:w_end].copy()
                    # Find any bounding boxes in this crop
                    boxes_crop, idx_crop = boxes_contained_in_crop(h_start, h_end, w_start, w_end, boxes_angle, args.crop_size, im_crop, args.check)
                    idx_covered.extend(idx_crop)
                    # Write crop
                    col += 1
                    im_name = im_root + '_' + str(angle) + 'deg_crop_r' + format(row, '02d') + 'c' + format(col, '02d') + '.tiff'
                    if args.check:
                        print "Wrote %s" % im_name

                    im_path = os.path.join(args.crop_dir, im_name)

                    write_crop_and_labels(im_path, im_crop, boxes_crop, w_start, h_start, f_labels, args.neg_examples)
                    # Update width values
                    remain_w -= args.crop_size - args.crop_overlap
                # Last crop in row
                if (remain_w > 0):
                    w_start = width - args.crop_size
                    w_end = width
                    im_crop = im_rot[h_start:h_end, w_start:w_end].copy()
                    # Find any bounding boxes in this crop
                    boxes_crop, idx_crop = boxes_contained_in_crop(h_start, h_end, w_start, w_end, boxes_angle, args.crop_size, im_crop, args.check)
                    idx_covered.extend(idx_crop)
                    # Write crop
                    col += 1
                    im_name = im_root + '_' + str(angle) + 'deg_crop_r' + format(row, '02d') + 'c' + format(col, '02d') + '.tiff'
                    if args.check:
                        print "Wrote %s" % im_name

                    im_path = os.path.join(args.crop_dir, im_name)

                    write_crop_and_labels(im_path, im_crop, boxes_crop, w_start, h_start, f_labels, args.neg_examples)
            # Check if we covered all the boxes in the crops
            # Compute stats for specific angle, specific image
            num_boxes_angle = boxes_angle.shape[0]
            num_boxes_covered = len(set(idx_covered))
            num_missed = num_boxes_angle - num_boxes_covered
            num_duplicates = len(idx_covered) - num_boxes_covered
            # Update stats for image overall
            num_boxes_image += num_boxes_angle
            num_boxes_covered_image += num_boxes_covered
            num_missed_image += num_missed
            num_duplicates_image += num_duplicates
            # Update stats for all
            idx_all_set = set(range(num_boxes_angle))
            idx_missed = list(idx_all_set.difference(set(idx_covered)))
            width_missed.extend([b[2] for b in boxes_angle[idx_missed]])
            height_missed.extend([b[3] for b in boxes_angle[idx_missed]])
            # Print stats for specific angle, specific image
            print "%s, %i deg: # boxes = %i, # boxes covered = %i, # boxes missed = %i, # duplicates = %i" % (im_root, angle, num_boxes_angle, num_boxes_covered, num_missed, num_duplicates)

        # Update stats for all images in directory
        num_boxes_all += num_boxes_image
        num_boxes_covered_all += num_boxes_covered_image
        num_missed_all += num_missed_image
        num_duplicates_all += num_duplicates_image
        # Print total stats for one image
        print "%s tot: # boxes = %i, # boxes covered = %i, # boxes missed = %i, # duplicates = %i" % (im_root, num_boxes_image, num_boxes_covered_image, num_missed_image, num_duplicates_image)

    # Print total stats for all images in directory
    print "All images: # boxes = %i, # boxes covered = %i, # boxes missed = %i, # duplicates = %i" % (num_boxes_all, num_boxes_covered_all, num_missed_all, num_duplicates_all)

    # Close training labels file
    f_labels.close()

bins_pixel = np.arange(0, 1100, 100)
plt.figure()
n_h, bins_h, patches_h = plt.hist(np.array(height_missed), bins_pixel)
plt.xlabel('Height (pixels)')
plt.ylabel('Count')
plt.title('Heights of missed labels')
plt.figure()
n_w, bins_w, patches_w = plt.hist(np.array(width_missed), bins_pixel)
plt.xlabel('Width (pixels)')
plt.ylabel('Count')
plt.title('Widths of missed labels')

plt.show()
