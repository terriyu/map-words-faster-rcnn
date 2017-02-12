#!/usr/bin/env python

# Script to convert bounding box data from Matlab to NumPy format

from __future__ import division
import numpy as np
import os
import argparse
import string
from scipy.io import loadmat

def parse_args():
    """Parse input arguments."""

    parser = argparse.ArgumentParser(description='Convert bounding boxes from Matlab to NumPy format')
    parser.add_argument('--mat_dir', required=True, dest='mat_dir', help='Specify path to directory for previously computed detections (.mat files)')
    parser.add_argument('--dest_dir', required=True, dest='dest_dir', help='Specify path to directory for write converted .npy files')

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_args()

    mat_files = os.listdir(args.mat_dir)

    # Loop through all Matlab files in directory
    for mat_file in mat_files: 

        path = os.path.join(args.mat_dir, mat_file)

        # Load data from Matlab file
        mat_dict = loadmat(path)        
        boxes = mat_dict['boxes']

        # Convert bounding boxes to .npy file
        im_root = string.split(mat_file, '_')[0]
        fname = im_root + '_boxes.npy'
        np.save(os.path.join(args.dest_dir, fname), boxes)
        print "Wrote bounding box data to .npy file %s" % os.path.join(args.dest_dir, fname)
