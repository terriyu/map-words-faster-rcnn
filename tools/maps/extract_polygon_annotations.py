# Script to extract ground truth annotations from MATLAB format
# and save them in a Python dictionary which is written to JSON

# This is a pre-processing step to put all the annotations in a single
# JSON file, so we don't have to deal with extracting the annotations
# when we do evaluation.

from __future__ import division
import os
import string
import json
import numpy as np
import scipy.io as sio

# Need this to test polygon validity
from shapely.geometry import Polygon

#mat_dir = 'annotated_map_word_polygons-20160923122008/data'
mat_dir = 'annotated_map_word_polygons-20170120041514/data'

#mat_files = ['D0090-5242001.mat']
#mat_files = ['D0117-5755018.mat']
mat_files = os.listdir(mat_dir)

# Ground truth Python dictionary in which we will
# save the extracted polygon annotations
gtruth_dict = {}

# Loop through each file
# Each file contains annotations for one map image
for mat_file in mat_files:
    # Extract name of image
    im_root = string.split(mat_file, '.')[0]
    # Initialize ground truth dictionary for image name key
    gtruth_dict[im_root] = []

    # Load polygon annotations from Matlab file
    mat_dict = sio.loadmat(os.path.join(mat_dir, mat_file))

    # Prepare to extract annotations
    polygons = mat_dict['V'][0][0]
    num_cells = polygons.shape[0]

    # Loop through each annotation in the file
    for k in xrange(num_cells):
        poly_cell = polygons[k][0][0]
        cell_length = len(poly_cell)
        for m in xrange(cell_length):
            p = poly_cell[m]
            # Extract x and y coordinates for polygon
            x_coords = p[0, :]
            y_coords = p[1, :]
            # Check if polygon is valid
            # Construct polygon exterior
            exterior_gt = [(x_coords[i], y_coords[i]) for i in xrange(len(x_coords))]
            exterior_gt.append(exterior_gt[0])
            # Create polygon object from exterior
            poly_gt = Polygon(exterior_gt)
            if (not poly_gt.is_valid):
                print "V{1}{%i}{%i}" % (k, m)
                print "Invalid polygon with points"
                print exterior_gt
                print p
            # Add polygon coordinates to ground truth dictionary
            bounding_polygon = np.vstack((x_coords, y_coords)).tolist()
            gtruth_dict[im_root].append(bounding_polygon)

    print "Finished processing %s" % mat_file

# Save extracted annotations to json file
save_file = 'gtruth_polygons.json'
with open(save_file, 'w') as f:
    json.dump(gtruth_dict, f)

print "Wrote polygon annotation data to %s" % save_file
