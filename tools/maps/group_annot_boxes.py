# Script to group annotations by angle for supervised training

# This scripts saves two data files
# 1) Ground truth dictionary that groups the original polygon annotations
#    by angle
# 2) Boxes dictionary that contains approximate, axis-aligned bounding boxes
#    for rotated images

# NOTE: We currently omit labels in the 2nd and 3rd quadrants

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

# Optional flag to write image files for checking boxes
CHECK_BOXES = False
# Optional flag to plot histograms
PLOT_HIST = True

images_dir = 'images'

#mat_dir = 'annotated_map_word_polygons-20160923122008/data'
mat_dir = 'annotated_map_word_polygons-20170120041514/data'

#mat_files = ['D0090-5242001.mat']
#mat_files = ['D0117-5755018.mat']
mat_files = os.listdir(mat_dir)

# Angles we are using for rotated images
angle_max = 120
angle_step = 6
angles = np.linspace(-angle_max, angle_max, (angle_max * 2) / angle_step + 1)
# Convert angles to a list of ints
# We need to do this because JSON doesn't accept NumPy data formats
# Also, we prefer to use ints rather than floats for dictionary keys
angles_list = [int(a) for a in angles.tolist()]

# Ground truth Python dictionary in which we will
# save the extracted polygon annotations,
# organized by approximate baseline angle
gtruth_dict = {}
gtruth_dict['data'] = {}
gtruth_dict['angles'] = angles_list

# Python dictionary containing approximate bounding boxes,
# organized by approximate baseline angle
boxes_dict = {}
boxes_dict['data'] = {}
boxes_dict['padding'] = {} # Height and width padding for each image
boxes_dict['angles'] = angles_list

# Collect stats on orientations
# Orientations range from -180 to 180 degrees
orientations = []

# Quadrant counts
q1 = 0  # x positive, y positive
q2 = 0  # x negative, y positive
q3 = 0  # x negative, y negative
q4 = 0  # x positive, y negative

# Collect stats on height and width of bounding boxes
heights = []
widths = []

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
            # Add polygon coordinates to ground truth dictionary under appropriate angle key
            # FIX: Temporarily throw out labels with orientations in 2nd and 3rd quadrants
            if ((orientation > -(angle_max + angle_step / 2.0)) and (orientation < angle_max + angle_step / 2.0)):
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
    # Initialize boxes dictionary for image name key
    boxes_dict['data'][im_root] = {}
    boxes_dict['padding'][im_root] = {}
    # Read image
    im_name = im_root + '.tiff'
    im_orig = cv2.imread(os.path.join(images_dir, im_name))
    # Pad image, so it doesn't get cut off when rotated
    diagonal = np.ceil(np.sqrt(np.sum(np.square(im_orig.shape))))
    height_orig, width_orig, _ = im_orig.shape
    height_padding = np.int_(np.ceil((diagonal - height_orig)/2.0))
    width_padding = np.int_(np.ceil((diagonal - width_orig)/2.0))
    height = height_orig + height_padding * 2
    width = width_orig + width_padding * 2
    # Add padding info to dictionary
    boxes_dict['padding'][im_root]['height_padding'] = height_padding
    boxes_dict['padding'][im_root]['width_padding'] = width_padding
    # Make padded version of image
    if CHECK_BOXES:
        im_padded = cv2.copyMakeBorder(im_orig, height_padding, height_padding, width_padding, width_padding, cv2.BORDER_CONSTANT, value=(128, 128, 128))
    for angle in angles_list:
        # Initialize list for specific angle key
        boxes_dict['data'][im_root][angle] = []
        # Check if there are any detections for this angle
        if len(gtruth_dict['data'][im_root][angle]) > 0:
            rot_mat = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), angle, 1)
            if CHECK_BOXES:
                im_rot = cv2.warpAffine(im_padded, rot_mat, (width, height))
            for polygon in gtruth_dict['data'][im_root][angle]:
                x_coords = polygon[0, :]
                y_coords = polygon[1, :]
                num_coords = len(x_coords)
                # Shift coordinates
                x_coords_shift = x_coords + width_padding
                y_coords_shift = y_coords + height_padding
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
                # Save height and width info
                heights.append(height_rot)
                widths.append(width_rot)
                # Save bounding box data to boxes_dict
                boxes_dict['data'][im_root][angle].append([x_min_rot, y_min_rot, width_rot, height_rot])
                # Draw boxes
                if CHECK_BOXES:
                    cv2.rectangle(im_rot, (np.float32(x_min_rot), np.float32(y_min_rot)), (np.float32(x_max_rot), np.float32(y_max_rot)), (0, 0, 255), 2)
            if CHECK_BOXES:
                im_name = im_root + '_padded_angle' + str(angle) + '.tiff'
                print "Writing image with boxes %s..." % im_name
                cv2.imwrite(im_name, im_rot)
    print "Finished processing %s" % mat_file

print "\nQuadrant data for polygon orientations"
print "Q1 = %i, Q2 = %i, Q3 = %i, Q4 = %i\n" % (q1, q2, q3, q4)

# Save annotations grouped by angle to json file
# Convert NumPy arrays to Python lists
for im in gtruth_dict['data']:
    for angle in angles_list:
        for idx, polygon in enumerate(gtruth_dict['data'][im][angle]):
            gtruth_dict['data'][im][angle][idx] = polygon.tolist()

gtruth_file = 'gtruth_grouped_by_angle.json'
with open(gtruth_file, 'w') as f:
    json.dump(gtruth_dict, f)

print "Wrote polygon annotation grouped by angle data to %s" % gtruth_file

# Save boxes grouped by angle to json file
boxes_file = 'boxes_grouped_by_angle.json'
with open(boxes_file, 'w') as f:
    json.dump(boxes_dict, f)

print "Wrote boxes grouped by angle data to %s" % boxes_file

# Convert stat lists to NumPy arrays
orientations = np.array(orientations)
heights = np.array(heights)
widths = np.array(widths)

if PLOT_HIST:
    # Bins to use for histogram
    # Bins that span -90 to 90 degrees
    bins90 = np.insert(np.linspace(-93, 93, 32), 0, -180)
    # Bins that span -180 to 180 degrees
    bins180 = np.linspace(-183, 183, 62)
    # Bins for pixels
    bins_pixel = np.arange(0,1050,50)

    plt.figure(1)
    n_orientations, bins_orientations, patches_orientations = plt.hist(orientations, bins180)
    plt.ylabel('Counts')
    plt.xlabel('Angle (degrees)')
    plt.title('Orientations')

    plt.figure(2)
    n_heights, bins_heights, patches_heights = plt.hist(heights, bins_pixel)
    plt.ylabel('Counts')
    plt.xlabel('Height (pixels)')
    plt.title('Heights')

    plt.figure(3)
    n_widths, bins_widths, patches_widths = plt.hist(widths, bins_pixel)
    plt.ylabel('Counts')
    plt.xlabel('Width (pixels)')
    plt.title('Widths')

    plt.show()
