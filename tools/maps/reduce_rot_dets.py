#!/usr/bin/env python

# Script to reduce the number of detections and eliminate redundant detections

from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import argparse
import string
import json
from sklearn.cluster import KMeans
from scipy.io import savemat

# Choices for method argument sent to parser
VALID_CHOICES = set(['kmeans', 'random_neighbors'])

def parse_args():
    """Parse input arguments"""

    parser = argparse.ArgumentParser(description='Reduce number of detections and eliminate redundant detections')
    parser.add_argument('--image', required=True, dest='image_path', help='Specify image file path for previously computed detections')
    parser.add_argument('--dets', required=True, dest='dets_path', help='Specify detections file path for previously computed detections')
    parser.add_argument('--method', required=True, dest='method', choices=VALID_CHOICES, help='Method for reducing detections (kmeans or random_neighbors)')

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_args()

    # Load detection data
    with open(args.dets_path) as f:
        det_dict = json.load(f)

    det_data = det_dict['det_data']
    det_counts = det_dict['det_counts']
    row_padding = det_dict['row_padding']
    col_padding = det_dict['col_padding']

    # Read original image
    im_orig = cv2.imread(args.image_path)
    # Add padding (padding values according to data file)
    im_composite_reduced = cv2.copyMakeBorder(im_orig, row_padding, row_padding, col_padding, col_padding, cv2.BORDER_CONSTANT, value=(128, 128, 128))

    print "Finished loading files"

    print "Number of detections loaded = %i" % (len(det_data))

    # Convert det_data to NumPy array so we can do logical indexing
    det_data_array = np.array(det_data)

    print "Using %s method to reduce detections" % args.method

    # Reduce detections using kmeans clustering
    if args.method == 'kmeans':
        # Get centers of all detection boxes
        center_coords = [x['center_rot'] for x in det_data_array]
        center_coords = np.vstack(center_coords)

        # Run kmeans clustering algorithm
        num_clusters = max(det_counts.values())
        kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(center_coords)

        # Initialize list of indicies corresponding to detection boxes to keep
        keep = []
        # Loop through clusters
        for cid in xrange(num_clusters):
            cluster_items = det_data_array[kmeans.labels_ == cid]
            # Keep only one detection per cluster, one with largest aspect ratio
            idx_keep = np.where(kmeans.labels_ == cid)[0][np.argmax([x['aspect_ratio'] for x in cluster_items])]
            keep.append(idx_keep)

    # Reduce detections using random neighbors
    elif args.method == 'random_neighbors':
        np.random.seed(42)
        #np.random.seed(137)
        #np.random.seed(0)

        epsilon = 30

        # Get centers of detection boxes and their aspect ratios
        center_coords = np.array([np.float32(x['center_rot']) for x in det_data])
        aspect_ratios = np.array([x['aspect_ratio'] for x in det_data])

        # Make an array of detection indices
        det_indices = np.arange(len(det_data))
        # Initialize list of indices corresponding to detection boxes to keep
        keep = []
        while len(det_indices) > 0:
            # Pick a random detection box
            idx = np.random.choice(det_indices)
            # Obtain center of this box
            idx_center = center_coords[idx]
            # Compute distances of all other detections from current detection
            dists = np.linalg.norm(center_coords[det_indices] - idx_center, axis = 1)
            # Get all indices corresponding to detections in epsilon neighborhood of current detection
            cluster_indices = det_indices[dists < epsilon]
            # Only keep the detection in the neighborhood with the largest aspect ratio
            idx_keep = cluster_indices[np.argmax(aspect_ratios[cluster_indices])]
            keep.append(idx_keep)
            # Remove all indicies corresponding to detections in neighborhood examined
            det_indices = np.delete(det_indices, np.where(dists < epsilon))

    print "Number of detections kept = %i" % (len(keep))

    # Bounding boxes to be saved to .mat file
    # 3D array for bounding boxes to be saved to .mat and .npy files
    # Each entry boxes_3d[:,:,i] = [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
    boxes_3d = np.zeros((4,2,len(keep)))
    # 2D array for bounding boxes to be saved to .csv file
    # Each entry boxes_2d[:,i] = [[x1, y1, x2, y2, x3, y3, x4, y4]]
    boxes_2d = np.zeros((8,len(keep))

    # Loop through all the detections we kept after reduction
    for i, det in enumerate(det_data_array[keep]):
        # Convert values to tuples
        upper_left_rot = np.float32(det['ul_rot'])
        upper_right_rot = np.float32(det['ur_rot'])
        lower_right_rot = np.float32(det['lr_rot'])
        lower_left_rot = np.float32(det['ll_rot'])

        # Save values to arrays
        # Shift values to account for padding
        upper_left_rot_shift = np.float32([upper_left_rot[0] - col_padding, upper_left_rot[1] - row_padding])
        upper_right_rot_shift = np.float32([upper_right_rot[0] - col_padding, upper_right_rot[1] - row_padding])
        lower_right_rot_shift = np.float32([lower_right_rot[0] - col_padding, lower_right_rot[1] - row_padding])
        lower_left_rot_shift = np.float32([lower_left_rot[0] - col_padding, lower_left_rot[1] - row_padding])
        boxes_3d[:,:,i] = np.vstack((upper_left_rot_shift, upper_right_rot_shift, lower_right_rot_shift, lower_left_rot_shift))
        boxes_2d[:,i] = np.hstack((upper_left_rot_shift, upper_right_rot_shift, lower_right_rot_shift, lower_left_rot_shift))

        # Draw bounding box
        red = (0, 0, 255)
        green = (0, 255, 0)
        blue = (255, 0, 0)
        line_color = blue
        cv2.line(im_composite_reduced, tuple(upper_left_rot), tuple(upper_right_rot), line_color, 4)
        cv2.line(im_composite_reduced, tuple(upper_right_rot), tuple(lower_right_rot), line_color, 4)
        cv2.line(im_composite_reduced, tuple(lower_right_rot), tuple(lower_left_rot), line_color, 4)
        cv2.line(im_composite_reduced, tuple(lower_left_rot), tuple(upper_left_rot), line_color, 4)

        # Test shift in box coordinates
        #cv2.line(im_orig, tuple(upper_left_rot_shift), tuple(upper_right_rot_shift), line_color, 4)
        #cv2.line(im_orig, tuple(upper_right_rot_shift), tuple(lower_right_rot_shift), line_color, 4)
        #cv2.line(im_orig, tuple(lower_right_rot_shift), tuple(lower_left_rot_shift), line_color, 4)
        #cv2.line(im_orig, tuple(lower_left_rot_shift), tuple(upper_left_rot_shift), line_color, 4)

    # Crop composite image to remove padding
    im_composite_reduced = im_composite_reduced[row_padding:-row_padding, col_padding:-col_padding]

    # Write composite image with reduced detections
    im_root = string.split(string.split(args.image_path, os.sep)[-1], '.')[0]
    im_composite_reduced_name = im_root + '_' + args.method + '_composite_reduced.tiff'
    out_dir = '/scratch3/terriyu/working/reduce/composite_reduced_det'
    cv2.imwrite(os.path.join(out_dir, im_composite_reduced_name), im_composite_reduced)
    print "Wrote composite image with reduced detections to %s" % os.path.join(out_dir, im_composite_reduced_name)

    #im_composite_reduced_name = im_root + '_' + args.method + '_check_composite.tiff'
    #cv2.imwrite(os.path.join(out_dir, im_composite_reduced_name), im_orig)

    # Write bounding boxes for reduced detections to .npy file
    npy_dir = '/scratch3/terriyu/working/reduce/reduced_det_data/npy'
    npy_fname = im_root + '_boxes.npy'
    np.save(os.path.join(npy_dir, npy_fname), boxes_3d)
    print "Wrote bounding box data to .npy file %s" % os.path.join(npy_dir, npy_fname)

    # Write bounding boxes for reduced detections to .mat file
    mat_dir = '/scratch3/terriyu/working/reduce/reduced_det_data/mat'
    mat_fname = im_root + '_boxes.mat'
    savemat(os.path.join(mat_dir, mat_fname), dict(boxes = boxes_3d))
    print "Wrote bounding box data to .mat file %s" % os.path.join(mat_dir, mat_fname)

    # Write bounding boxes for reduced detections to .csv file
    csv_dir = '/scratch3/terriyu/working/reduce/reduced_det_data/csv'
    csv_fname = im_root + '_boxes.csv'
    np.savetxt(os.path.join(csv_dir, csv_fname), boxes_2d, delimiter=',')
    print "Wrote bounding box data to .csv file %s" % os.path.join(csv_dir, csv_fname)
