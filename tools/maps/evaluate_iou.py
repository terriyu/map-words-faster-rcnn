#!/usr/bin/env python

# Script to evaluate precision and recall given a list of ground truth
# bounding polygons and a list of detection boxes

# The detection boxes don't need to be axis-aligned, they can be
# general quadrilaterals

# Note: Requires shapely and SciPy version >= 0.17

from __future__ import division
import numpy as np
import os
import argparse
import string
import json
import cv2
from collections import defaultdict

# Requires scipy version >= 0.17
# This function implements the Hungarian algorithm
from scipy.optimize import linear_sum_assignment

# Disable shapely warnings
import logging
logger = logging.getLogger()
logger.setLevel(logging.ERROR)

from shapely.geometry import Polygon

# Color constants (BGR convention, not RGB)
RED = (0, 0, 255)
YELLOW = (0, 255, 255)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)
LINE_THICKNESS = 4

def parse_args():
    """Parse input arguments."""

    parser = argparse.ArgumentParser(description='Evaluation of map detection results using intersection-over-union')
    parser.add_argument('--det_dir', required=True, dest='det_dir', help='Specify path to directory for previously computed detections (.npy files)')
    parser.add_argument('--gtruth', required=True, dest='gt_file', help='Specify path to JSON file containing ground truth')
    parser.add_argument('--threshold', dest='threshold', default=0.5, help='Threshold for detection, i.e. intersection-over-union >= threshold ==> detection (default value = 0.5)')
    parser.add_argument('--image_dir', dest='image_dir', help='Specify path to directory containing full map images')
    parser.add_argument('--composite_dir', dest='composite_dir', help='Specify path to write composite images showing detection boxes and ground truth annotations')
    parser.add_argument('--correct_invalid', dest='correct', action ='store_true', help='Flag to specify that invalid ground truth polygon should be corrected')
    parser.add_argument('--verbose', dest='verbose', action='store_true', help='Flag to specify that additional information should be printed')

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_args()

    if args.correct:
        print "Correcting invalid ground truth polygons automatically"
    else:
        print "Invalid ground truth polygons will be skipped"

    print "Intersection-over-union threshold is %f\n" % args.threshold

    # Load ground truth bounding polygons
    with open(args.gt_file) as f:
        gtruth_dict = json.load(f)

    det_files = os.listdir(args.det_dir)
    #det_files = ['D0090-5242001_boxes.npy']

    # Initialize a bunch of global counters for statistics

    # Total true positives
    tp_tot = 0
    # Total false negatives
    fn_tot = 0
    # Total false positives
    fp_tot = 0

    # Continous score (see FDDB paper)
    cont_score = 0

    # Total ground truth labels
    labels_tot = 0
    # Total number of invalid ground truth labels
    invalid_tot = 0

    # Histogram for how many matches we find for each label (overall)
    matches_tot_hist = defaultdict(int)

    # Loop through all images
    for det_file in det_files:
        # Load detections for one image
        path = os.path.join(args.det_dir, det_file)
        dets = np.load(path)
        # Root name for image file
        im_root = string.split(det_file, '_')[0]
        # Read image file (if appropriate parser arguments specified)
        if (args.image_dir is not None) and (args.composite_dir is not None):
            im_composite = cv2.imread(os.path.join(args.image_dir, im_root + '.tiff'))

        print "Processing detections for %s" % (im_root + ".tiff")

        # Each box should have 4 x-coordinates and 4 y-coordinates
        if (dets.shape[:2] != (4,2)):
            print "Error: unexpected array shape"

        num_dets = dets.shape[2]

        # Initialize local counters for statistics
        # These are statistics for just one map image as opposed to the entire data set
        true_positives = 0
        false_negatives = 0
        false_positives = 0
        num_invalid = 0  # Number of invalid ground truth labels for this image

        # Dictionaries with statistics
        # Histogram for how many matches we find for each label (for one image)
        matches_image_hist = defaultdict(int)
        # Number of matches for each detection - initialized to 0
        det_matches = dict.fromkeys(range(num_dets), 0)

        # List containing score matrix for matches
        # Each score is intersection-over-union between a detection and a ground truth label
        # We need this matrix for running the Hungarian algorithm later
        scores = []

        # Loop through all ground truth labels for this image
        for label in gtruth_dict[im_root]:
            # Construct polygon for ground truth label
            x_coords, y_coords = label
            exterior_gt = [(x_coords[i], y_coords[i]) for i in xrange(len(x_coords))]
            exterior_gt.append(exterior_gt[0])
            poly_gt = Polygon(exterior_gt)

            # Check if polygon ground truth label is valid polygon
            valid_flag = True
            if (not poly_gt.is_valid):
                if args.verbose:
                    print "Invalid polygon with points"
                    print exterior_gt
                valid_flag = False
                num_invalid += 1

            # If polygon was invalid and we are fixing the polygons, go ahead and fix it
            if ((not valid_flag) and args.correct):
                if args.verbose:
                    print "Correcting invalid polygon -- replacing with convex hull"
                poly_gt = poly_gt.convex_hull
                valid_flag = True

            # If the polygon is valid (or valid because we fixed it),
            # try to find a matching bounding box for the ground truth polygon
            if valid_flag:
                # Draw polygon annotation on composite (if appropriate parser arguments specified)
                if (args.image_dir is not None) and (args.composite_dir is not None):
                    poly_pts = np.array(exterior_gt[:-1], np.int32)
                    poly_pts = poly_pts.reshape((-1, 1, 2))
                    cv2.polylines(im_composite, [poly_pts], True, RED, LINE_THICKNESS)
                # Set flag for whether we found a match for ground truth label
                num_matches = 0
                # Initialize row for score matrix
                scores_row = []
                # Loop through all detections to find a match
                for idx in xrange(num_dets):
                    # Construct polygon for detection box
                    x_coords, y_coords = (dets[:,0,idx], dets[:,1,idx])
                    exterior_det = [(x_coords[i], y_coords[i]) for i in xrange(len(x_coords))]
                    exterior_det.append(exterior_det[0])
                    box_det = Polygon(exterior_det)
                    # Calculate polygon intersection and polygon union
                    poly_intersection = poly_gt.intersection(box_det)
                    poly_union = poly_gt.union(box_det)
                    # Compute intersection over union
                    iou = poly_intersection.area / poly_union.area
                    # Append score
                    # Use negative of iou since Hungarian algorithm does minimization
                    scores_row.append(iou)
                    # Consider a detection a match if IOU >= threshold
                    if (iou >= args.threshold):
                        num_matches += 1
                        det_matches[idx] += 1

                # Update local statistics for this image
                matches_image_hist[num_matches] += 1
                matches_tot_hist[num_matches] += 1
                # Append entire row of scores
                scores.append(scores_row)

        # Draw detection boxes on composite (if appropriate parser arguments specified)
        if (args.image_dir is not None) and (args.composite_dir is not None):
            for idx in xrange(num_dets):
                x_coords, y_coords = (dets[:,0,idx], dets[:,1,idx])
                exterior_det = [(x_coords[i], y_coords[i]) for i in xrange(len(x_coords))]
                box_pts = np.array(exterior_det, np.int32)
                box_pts = box_pts.reshape((-1, 1, 2))
                if det_matches[idx] > 0:
                    cv2.polylines(im_composite, [box_pts], True, GREEN, LINE_THICKNESS)
                else:
                    cv2.polylines(im_composite, [box_pts], True, BLUE, LINE_THICKNESS)
            # Write composite image
            im_composite_name = im_root + '_dets_gt_composite_reduced.tiff'
            im_path = os.path.join(args.composite_dir, im_composite_name)
            cv2.imwrite(im_path, im_composite)
            print "Wrote composite image %s" % im_path

        # Run Hungarian algorithm
        # Convert list to matrix
        # Each row corresponds to a ground truth label,
        # each column corresponds to a detection
        scores_matrix = np.array(scores)
        # Remove any rows that below threshold
        # Each row removed is a false negative
        false_negatives = sum(np.all(scores_matrix < args.threshold, axis=1))
        scores_matrix = scores_matrix[~np.all(scores_matrix < args.threshold, axis=1)]
        # Remove any columns that are all zeros
        # Each column removed is a false positive
        false_positives = sum(np.all(scores_matrix < args.threshold, axis=0))
        scores_matrix = scores_matrix[:, ~np.all(scores_matrix < args.threshold, axis=0)]
        # Final dimensions of score matrix
        row_dim = scores_matrix.shape[0]
        col_dim = scores_matrix.shape[1]
        # Find best assignment (multiply by -1 for minimization)
        row_ind, col_ind = linear_sum_assignment(-1 * scores_matrix)

        # Compute stats for discrete score
        true_positives = len(row_ind)
        if (col_dim > row_dim):
            # More detections than ground truth labels
            false_positives += col_dim - row_dim
        else:
            # More ground truth labels than detections
            false_negatives += row_dim - col_dim
        # Compute continous score for this image
        cont_score_image = scores_matrix[row_ind, col_ind].sum()

        # Update global statistics for entire data set
        tp_tot += true_positives
        fn_tot += false_negatives
        fp_tot += false_positives
        cont_score += cont_score_image

        num_labels = len(gtruth_dict[im_root])
        labels_tot += num_labels
        invalid_tot += num_invalid

        # Print statistics for single map image
        print "Matches histogram for this image"
        print matches_image_hist
        print "Precision = %f" % (true_positives / (true_positives + false_positives))
        print "Recall = %f" % (true_positives / (true_positives + false_negatives))
        print "True pos = %i, false neg = %i, false pos = %i" % (true_positives, false_negatives, false_positives)
        print "Continuous score = %f" % cont_score_image
        print "Number of invalid ground truth polygons = %i, number of gt labels = %i, num detections = %i\n" % (num_invalid, num_labels, num_dets)
        if (true_positives + false_negatives != num_labels):
            print "ERROR: TP + FP != num ground truth labels\n"
        if (true_positives + false_positives != num_dets):
            print "ERROR: TP + FP != num detections\n"

    # Print statistics for entire data set
    print "\nTOTAL Precision = %f" % (tp_tot / (tp_tot + fp_tot))
    print "TOTAL Recall = %f" % (tp_tot / (tp_tot + fn_tot))
    print "TOTAL true pos = %i, false neg = %i, false pos = %i" % (tp_tot, fn_tot, fp_tot)
    print "TOTAL continuous score = %f" % cont_score
    print "TOTAL Ground truth labels = %i" % labels_tot
    print "TOTAL Invalid ground truth polygons = %i" % invalid_tot
    print "Percentage of invalid labels = %f" % (invalid_tot / labels_tot)
    print "Matches histogram overall"
    print matches_tot_hist
