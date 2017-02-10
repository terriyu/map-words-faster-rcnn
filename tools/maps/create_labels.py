# Script to create labels for supervised training

from __future__ import division
import os
import string
import json
import numpy as np
import scipy.io as sio
import cv2

# Need this to test polygon validity
from shapely.geometry import Polygon

# Need this for plotting histogram
import matplotlib.pyplot as plt

# Raise exception if numpy raises any warnings
np.seterr(all='raise')

images_dir = 'maps-tiff'

#mat_dir = 'annotated_map_word_polygons-20160923122008/data'
mat_dir = 'annotated_map_word_polygons-20170120041514/data'

mat_files = ['D0090-5242001.mat']
#mat_files = ['D0117-5755018.mat']
#mat_files = os.listdir(mat_dir)

# Angles we are using for rotated images
angles = np.linspace(-90, 90, 31)
# Add angles we used to data file
angles_list = [int(a) for a in angles.tolist()]

# Ground truth Python dictionary in which we will
# save the extracted polygon annotations,
# organized by approximate baseline angle
gtruth_dict = {}
gtruth_dict['data'] = {}
gtruth_dict['angles'] = angles_list

# Python dictionary containing approximate bounding boxes,
# organized by approximate baseline angle
labels_dict = {}
labels_dict['data'] = {}
labels_dict['padding'] = {}
labels_dict['angles'] = angles_list

# Bins to use for histogram
bins = np.insert(np.linspace(-90, 90, 31), 0, -180)
bin2 = np.linspace(-180, 180, 61)

# Collect stats on orientations
# Orientations range from -180 to 180 degrees
orientations = []

# Quadrant counts
q1 = 0  # x positive, y positive
q2 = 0  # x negative, y positive
q3 = 0  # x negative, y negative
q4 = 0  # x positive, y negative

print "Grouping detections by orientation...\n"

# Loop through each file
# Each file contains annotations for one map image
for mat_file in mat_files:
    # Extract name of image
    im_root = string.split(mat_file, '.')[0]
    # Initialize ground truth dictionary for image name key
    gtruth_dict['data'][im_root] = {}
    for angle in angles_list:
        gtruth_dict['data'][im_root][angle] = []

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
            # Force cast to float64 (in some rare cases, coords are read as int)
            x_coords = np.float64(p[0, :])
            y_coords = np.float64(p[1, :])
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
            # Find orientation of polygon
            x1 = x_coords[0]
            y1 = y_coords[0]
            x2 = x_coords[1]
            y2 = y_coords[1]
            orientation = np.arctan2(y2 - y1, x2 - x1) * 180.0 / np.pi
            orientations.append(orientation)
            # Update statistics on quadrant for orientation
            if x2 >= x1:
                if y2 >= y1:
                    q1 += 1
                else:
                    q4 += 1
            else:
                if y2 >= y1:
                    q2 += 1
                else:
                    q3 += 1
            # Add polygon coordinates to ground truth dictionary
            # FIX: Temporarily throw out labels with orientations in 2nd and 3rd quadrants
            if ((orientation > -93) and (orientation < 93)):
                bounding_polygon = np.vstack((x_coords, y_coords))
                closest_angle = int(angles[np.argmin(np.abs(angles - orientation))])
                gtruth_dict['data'][im_root][closest_angle].append(bounding_polygon)

    print "Finished processing %s" % mat_file

print "\nComputing approximate bounding boxes for each angle...\n"
# Loop through each file
# Each file contains annotations for one map image
for mat_file in mat_files:
    # Extract name of image
    im_root = string.split(mat_file, '.')[0]
    # Initialize labels dictionary for image name key
    labels_dict['data'][im_root] = {}
    labels_dict['padding'][im_root] = {}
    # Read image
    im_name = im_root + '.tiff'
    im_orig = cv2.imread(os.path.join(images_dir, im_name))
    # Pad image, so it doesn't get cut off when rotated
    diagonal = np.ceil(np.sqrt(np.sum(np.square(im_orig.shape))))
    rows_orig, cols_orig, _ = im_orig.shape
    row_padding = np.int_(np.ceil((diagonal - rows_orig)/2.0))
    col_padding = np.int_(np.ceil((diagonal - cols_orig)/2.0))
    rows = rows_orig + row_padding * 2
    cols = cols_orig + col_padding * 2
    labels_dict['padding'][im_root]['row_padding'] = row_padding
    labels_dict['padding'][im_root]['col_padding'] = col_padding
    #im_padded = cv2.copyMakeBorder(im_orig, row_padding, row_padding, col_padding, col_padding, cv2.BORDER_CONSTANT, value=(128, 128, 128))
    for angle in angles_list:
        # Initialize list for specific angle key
        labels_dict['data'][im_root][angle] = []
        # Check if there are any detections for this angle
        if len(gtruth_dict['data'][im_root][angle]) > 0:
            rot_mat = cv2.getRotationMatrix2D((cols / 2.0, rows / 2.0), angle, 1)
            #im_rot = cv2.warpAffine(im_padded, rot_mat, (cols, rows))
            for polygon in gtruth_dict['data'][im_root][angle]:
                x_coords = polygon[0, :]
                y_coords = polygon[1, :]
                num_coords = len(x_coords)
                # Shift coordinates
                x_coords_shift = x_coords + col_padding
                y_coords_shift = y_coords + row_padding
                # Rotate coordinates
                coord_mat = np.vstack((x_coords_shift, y_coords_shift, np.ones(num_coords)))
                rot_coords = np.dot(rot_mat, coord_mat)
                x_coords_rot = rot_coords[0,:]
                y_coords_rot = rot_coords[1,:]
                # Minimum bounding box that encloses polygon
                x_min_rot = np.min(x_coords_rot)
                x_max_rot = np.max(x_coords_rot)
                y_min_rot = np.min(y_coords_rot)
                y_max_rot = np.max(y_coords_rot)
                height_rot = y_max_rot - y_min_rot + 1
                width_rot = x_max_rot - x_min_rot + 1
                # Save bounding box data to labels_dict
                labels_dict['data'][im_root][angle].append([x_min_rot, y_min_rot, width_rot, height_rot])
                # Draw boxes
                #cv2.rectangle(im_rot, (np.float32(x_min_rot), np.float32(y_min_rot)), (np.float32(x_max_rot), np.float32(y_max_rot)), (0, 0, 255), 2)
            #im_name = im_root + '_padded_angle' + str(angle) + '.tiff'
            #cv2.imwrite(im_name, im_rot)
    print "Finished processing %s" % mat_file

print "\nQuadrant data for polygon orientations"
print "Q1 = %i, Q2 = %i, Q3 = %i, Q4 = %i\n" % (q1, q2, q3, q4)

# Save extracted annotations to json file
# Convert NumPy arrays to Python lists
for im in gtruth_dict['data']:
    for angle in angles_list:
        for idx, polygon in enumerate(gtruth_dict['data'][im][angle]):
            gtruth_dict['data'][im][angle][idx] = polygon.tolist()

gtruth_file = 'gtruth_polygons.json'
with open(gtruth_file, 'w') as f:
    json.dump(gtruth_dict, f)

print "Wrote polygon annotation data to %s" % gtruth_file

labels_file = 'labels.json'
with open(labels_file, 'w') as f:
    json.dump(labels_dict, f)

print "Wrote labels data to %s" % labels_file

orientations = np.array(orientations)
#n, bins, patches = plt.hist(orientations, bins)
