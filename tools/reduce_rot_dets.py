#!/usr/bin/env python

from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import argparse
import string
import json
from sklearn.cluster import KMeans

VALID_CHOICES = set(['kmeans', 'random_neighbors'])

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Face Detection using Faster R-CNN')
    parser.add_argument('--image', required=True, dest='image_path', help='Specify image file path for previously computed detections')
    parser.add_argument('--dets', required=True, dest='dets_path', help='Specify detections file path for previously computed detections')
    parser.add_argument('--method', required=True, dest='method', choices=VALID_CHOICES, help='Method for reducing detections (kmeans or random_neighbors)')

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_args()

    im_orig = cv2.imread(args.image_path) 

    diagonal = np.ceil(np.sqrt(np.sum(np.square(im_orig.shape))))
    rows_orig, cols_orig, _ = im_orig.shape
    h_padding = np.int_(np.ceil((diagonal - cols_orig)/2.0))
    v_padding = np.int_(np.ceil((diagonal - rows_orig)/2.0))

    im_composite_reduced = cv2.copyMakeBorder(im_orig, v_padding, v_padding, h_padding, h_padding, cv2.BORDER_CONSTANT, value=(128, 128, 128))

    with open(args.dets_path) as f:
        det_dict = json.load(f)

    print "Finished loading files"

    det_data = det_dict['det_data']
    det_counts = det_dict['det_counts']

    print "Number of detections loaded = %i" % (len(det_data))

    # Convert det_data to NumPy array so we can do logical indexing
    det_data_array = np.array(det_data)
 
    print "Using %s method to reduce detections" % args.method

    if args.method == 'kmeans':
        center_coords = [x['center_rot'] for x in det_data_array]
        center_coords = np.vstack(center_coords)

        num_clusters = max(det_counts.values())
        kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(center_coords)

        keep = []
        for cid in xrange(num_clusters):
            cluster_items = det_data_array[kmeans.labels_ == cid] 
            idx_keep = np.where(kmeans.labels_ == cid)[0][np.argmax([x['aspect_ratio'] for x in cluster_items])]
            keep.append(idx_keep)

    elif args.method == 'random_neighbors':
        np.random.seed(42)
        epsilon = 90 

        center_coords = np.array([np.float32(x['center_rot']) for x in det_data])
        aspect_ratios = np.array([x['aspect_ratio'] for x in det_data])

        det_indices = np.arange(len(det_data))
        keep = [] 
        while len(det_indices) > 0:
            #print len(det_indices)
            #print det_indices
            idx = np.random.choice(det_indices)
            #print idx
            idx_center = center_coords[idx]
            dists = np.linalg.norm(center_coords[det_indices] - idx_center, axis = 1)
            #print dists
            cluster_indices = det_indices[dists < epsilon]
            #print cluster_indices
            idx_keep = cluster_indices[np.argmax(aspect_ratios[cluster_indices])]
            #print idx_keep
            keep.append(idx_keep)
            #print dists < epsilon
            det_indices = np.delete(det_indices, np.where(dists < epsilon))

    print "Number of detections kept = %i" % (len(keep))
 
    for det in det_data_array[keep]:
        # Convert values to tuples
        upper_left_rot = tuple(np.float32(det['ul_rot']))
        upper_right_rot = tuple(np.float32(det['ur_rot']))
        lower_left_rot = tuple(np.float32(det['ll_rot']))
        lower_right_rot = tuple(np.float32(det['lr_rot']))

        # Draw bounding box
        red = (0, 0, 255)
        cv2.line(im_composite_reduced, upper_left_rot, upper_right_rot, red, 4)
        cv2.line(im_composite_reduced, upper_right_rot, lower_right_rot, red, 4)
        cv2.line(im_composite_reduced, lower_right_rot, lower_left_rot, red, 4)
        cv2.line(im_composite_reduced, lower_left_rot, upper_left_rot, red, 4)

    # Crop composite image to remove padding
    im_composite_reduced = im_composite_reduced[v_padding:-v_padding, h_padding:-h_padding]

    im_root = string.split(string.split(args.image_path, os.sep)[-1], '.')[0]
    im_composite_reduced_name = im_root + '_' + args.method + '_composite_reduced.tiff'
    out_dir = 'output'
    cv2.imwrite(os.path.join(out_dir, im_composite_reduced_name), im_composite_reduced)
    print "Wrote composite image with reduced detections to %s" % os.path.join(out_dir, im_composite_reduced_name)
