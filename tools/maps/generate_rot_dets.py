#!/usr/bin/env python

# Script to compute detections for rotated versions of images

from __future__ import division
import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import caffe
import cv2
import os
import argparse
import string
import json

NETS = {'vgg16': ('VGG16', 'pre-trained-models/map_words_faster_rcnn.caffemodel')}

def write_detections_to_img(im, im_path, dets, threshold, color = (0, 255, 0)):
    """Draw detection boxes on image if their scores are above threshold
       Image with boxes will be written to path specified by im_path

    Args:
        im (2D array) - original image
        im_path (string) - path to write new image
        dets (2D array) - detections data
        threshold (float) - only write a detection box if its score >= threshold
        color (tuple) - RGB color for drawing detection box

    Returns: Nothing
    """

    # Loop through detections
    for i in xrange(dets.shape[0]):
        # Extract detection bounding box and score
        bbox = dets[i, :4]
        score = dets[i, -1]
        # Draw detection box if score is above threshold
        if score > threshold:
            cv2.rectangle(im, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 4)
    # Write image with detections
    cv2.imwrite(im_path, im)

def im_detect_sliding_crop(net, im, crop_h, crop_w, step):
    """ """

    imh, imw, _ = im.shape

    cls_ind = 1

    boxes = np.zeros((0, 4), dtype=np.float32)
    scores = np.zeros((0, 1), dtype=np.float32)

    y1 = 0
    while y1 < imh:
        y2 = min(y1 + crop_h, imh)
        if y2 - y1 < 25:
            y1 += step
            continue

        x1 = 0
        while x1 < imw:
            x2 = min(x1 + crop_w, imw)
            if x2 - x1 < 25:
                x1 += step
                continue

            crop_im = im[y1:y2, x1:x2, :]

            crop_scores, crop_boxes = im_detect(net, crop_im)
            crop_boxes = crop_boxes[:, 4*cls_ind:4*(cls_ind + 1)]
            crop_scores = crop_scores[:, cls_ind] + (0.01 * np.random.random() - 0.005)

            crop_boxes[:,0] += x1
            crop_boxes[:,1] += y1
            crop_boxes[:,2] += x1
            crop_boxes[:,3] += y1

            boxes = np.vstack((boxes, crop_boxes))
            scores = np.vstack((scores, crop_scores[:, np.newaxis]))

            x1 += step

        y1 += step

    return scores, boxes

def load_distinct_colors(path):
    """Load array of RGB values corresponding to distinct colors from file path

    Args:
        path (string) - file path for text file containing hexadecimal RGB colors

    Returns:
        rgb_colors (list of tuples) - each tuple in the list specifies an RGB color
    """

    # Read text file with hexadecimal RGB color values
    with open(path, 'r') as f:
        hex_colors = f.read().splitlines()

    # Convert hexadecimal values to integers
    rgb_colors = [(int(h[4:6], 16), int(h[2:4], 16), int(h[0:2], 16)) for h in hex_colors]

    return rgb_colors 

def parse_args():
    """Parse input arguments"""

    parser = argparse.ArgumentParser(description='Detect text in historical maps using Faster R-CNN')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16]',
                        choices=NETS.keys(), default='vgg16')
    parser.add_argument('--load_dets', dest='dets_file', help='Specify file for previously computed detections and skip running model')

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals
    # cfg.TEST.BBOX_REG = False

    args = parse_args()

    prototxt = 'models/map/VGG16/faster_rcnn_end2end/test.prototxt'
    caffemodel = NETS[args.demo_net][1]

    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.\nDid you run ./pre-trained-models/' 'fetch_pre_trained_model.sh?').format(caffemodel))

    if args.cpu_mode:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu_id)
        cfg.GPU_ID = args.gpu_id
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)

    print '\n\nLoaded network {:s}'.format(caffemodel)

    CONF_THRESH = 0.65
    NMS_THRESH = 0.15

    crop_w = 500
    crop_h = 500
    step = 400

    im_dir = '/scratch3/terriyu/maps'
    out_dir = '/scratch3/terriyu/working/generate/images_rawdet'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # List of image file names to process in the directory
    im_names = ['D0090-5242001.tiff']
    #im_names = ['D0006-0285025.tiff', 'D0042-1070004.tiff', 'D0042-1070013.tiff', 'D0117-5755025.tiff', 'D5005-5028100.tiff', 'D0017-1592006.tiff', 'D0042-1070005.tiff', 'D0042-1070015.tiff', 'D0117-5755033.tiff', 'D5005-5028102.tiff', 'D0041-5370006.tiff', 'D0042-1070006.tiff', 'D0079-0019007.tiff', 'D0117-5755035.tiff', 'D5005-5028149.tiff', 'D0041-5370026.tiff', 'D0042-1070007.tiff', 'D0089-5235001.tiff', 'D0117-5755036.tiff', 'D0042-1070001.tiff', 'D0042-1070009.tiff', 'D0090-5242001.tiff', 'D5005-5028052.tiff', 'D0042-1070002.tiff', 'D0042-1070010.tiff', 'D0117-5755018.tiff', 'D5005-5028054.tiff', 'D0042-1070003.tiff', 'D0042-1070012.tiff', 'D0117-5755024.tiff', 'D5005-5028097.tiff']
# Note D0042-1070015.tiff fails
    #im_names = ['D0117-5755033.tiff', 'D5005-5028102.tiff', 'D0041-5370006.tiff', 'D0042-1070006.tiff', 'D0079-0019007.tiff', 'D0117-5755035.tiff', 'D5005-5028149.tiff', 'D0041-5370026.tiff', 'D0042-1070007.tiff']
    #im_names = ['D0089-5235001.tiff', 'D0117-5755036.tiff', 'D0042-1070001.tiff', 'D0042-1070009.tiff', 'D0090-5242001.tiff', 'D5005-5028052.tiff', 'D0042-1070002.tiff', 'D0042-1070010.tiff', 'D0117-5755018.tiff', 'D5005-5028054.tiff', 'D0042-1070003.tiff', 'D0042-1070012.tiff', 'D0117-5755024.tiff', 'D5005-5028097.tiff']
    #im_names = ['D0042-1070015.tiff']

    # Array of angles to rotate the image
    #angles = [-90, -70, -50, -30, -10, 0, 10, 30, 50, 70, 90]
    #angles = [-90, 90]
    angles = np.insert(np.linspace(-90, 90, 30), 0, 0)

    # List of colors for drawing detection boxes
    #colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255), (255, 255, 0), (0, 0, 128), (0, 128, 0), (128, 0, 0), (0, 128, 128), (128, 0, 128), (128, 128, 0)]
    colors_dir = 'tools'
    colors = load_distinct_colors(os.path.join(colors_dir, 'colors_rgb_divide.txt'))

    # Loop through all image files
    for im_name in im_names:
        # Read image file
        im_orig = cv2.imread(os.path.join(im_dir, im_name))
        # Extract root of file name
        im_name_root = string.split(im_name, '.')[0]
        print "Read image %s" % im_name

        # Pad image, so it doesn't get cut off when rotated
        diagonal = np.ceil(np.sqrt(np.sum(np.square(im_orig.shape))))
        rows_orig, cols_orig, _ = im_orig.shape
        row_padding = np.int_(np.ceil((diagonal - rows_orig)/2.0))
        col_padding = np.int_(np.ceil((diagonal - cols_orig)/2.0))
        im_padded = cv2.copyMakeBorder(im_orig, row_padding, row_padding, col_padding, col_padding, cv2.BORDER_CONSTANT, value=(128, 128, 128))
        rows, cols, _ = im_padded.shape

        # Write padded image
        im_fname =  im_name_root + "_padded.tiff"
        cv2.imwrite(os.path.join(out_dir, im_fname), im_padded)

        # Make copy of padded image to draw boxes on for composite image
        im_composite = im_padded.copy()

        det_data = []
        det_counts = {}
        for idx, angle in enumerate(angles):
            # Rotate image
            rot_mat = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
            im_rot = cv2.warpAffine(im_padded, rot_mat, (cols, rows))
            # Output file name for rotated image
            im_rot_name = im_name_root + '_' + str(round(angle,1)) + 'deg.tiff'

            # # Detect all object classes and regress object bounds
            timer = Timer()
            timer.tic()
            scores, boxes = im_detect_sliding_crop(net, im_rot, crop_h, crop_w, step)
            timer.toc()
            print "Angle = %f" % angle
            print ('Detection took {:.3f}s for ' '{:d} object proposals').format(timer.total_time, boxes.shape[0])

            dets = np.hstack((boxes, scores)).astype(np.float32)
            keep = nms(dets, NMS_THRESH)
            print "Finished NMS, number of detections kept = %i" % len(keep)
            dets = dets[keep, :]

            keep = np.where(dets[:, 4] > CONF_THRESH)
            dets = dets[keep]
            det_counts[angle] = dets.shape[0]

            # Write detections on rotated image
            print "Writing detections for angle %f to file" % angle
            write_detections_to_img(im_rot, os.path.join(out_dir, im_rot_name), dets, CONF_THRESH, colors[idx])

            # Write detections on composite image for all angles
            # Save detections for composite image
            print "Rotating detections back"
            for i in xrange(dets.shape[0]):
                score = dets[i, -1]
                bbox = dets[i, :4]

                # Compute stats
                width = bbox[2] - bbox[0]
                height = bbox[3] - bbox[1] 
                aspect_ratio = width / height
                center = [bbox[0] + width / 2.0, bbox[1] + height / 2.0]

                # Add data to dictionary
                # Convert numpy types to regular Python float, so we can save dictionary to JSON later
                data_dict = {'score': float(score), 'angle': float(angle), 'color': list(colors[idx]), 'height': float(height), 'width': float(width), 'aspect_ratio': float(aspect_ratio), 'center': [float(x) for x in center], 'ul': [float(bbox[0]), float(bbox[1])], 'ur': [float(bbox[2]), float(bbox[1])], 'll': [float(bbox[0]), float(bbox[3])], 'lr': [float(bbox[2]), float(bbox[3])]}

                # Rotate box
                revrot_mat = cv2.getRotationMatrix2D((cols / 2, rows / 2), -angle, 1)
                upper_left_rot = np.float32(np.dot(revrot_mat, np.array([bbox[0], bbox[1], 1])))
                upper_right_rot = np.float32(np.dot(revrot_mat, np.array([bbox[2], bbox[1], 1])))
                lower_left_rot = np.float32(np.dot(revrot_mat, np.array([bbox[0], bbox[3], 1])))
                lower_right_rot = np.float32(np.dot(revrot_mat, np.array([bbox[2], bbox[3], 1])))
                center_rot = np.float32(np.dot(revrot_mat, center + [1]))

                # Draw bounding box
                cv2.line(im_composite, tuple(upper_left_rot), tuple(upper_right_rot), colors[idx], 4)
                cv2.line(im_composite, tuple(upper_right_rot), tuple(lower_right_rot), colors[idx], 4)
                cv2.line(im_composite, tuple(lower_right_rot), tuple(lower_left_rot), colors[idx], 4)
                cv2.line(im_composite, tuple(lower_left_rot), tuple(upper_left_rot), colors[idx], 4)

                # Add rotated values to dictionary
                data_dict.update({'center_rot': center_rot, 'ul_rot': upper_left_rot, 'ur_rot': upper_right_rot, 'll_rot': lower_left_rot, 'lr_rot': lower_right_rot})
                det_data.append(data_dict)

        # Crop composite image to remove padding
        im_composite = im_composite[row_padding:-row_padding, col_padding:-col_padding]

        im_composite_name = im_name_root + '_composite.tiff'
        cv2.imwrite(os.path.join(out_dir, im_composite_name), im_composite)

        # Convert entries in det_data to float, so we can save det_data to JSON
        for det in det_data:
            det['center_rot'] = [float(x) for x in det['center_rot']]
            det['ul_rot'] = [float(x) for x in det['ul_rot']]
            det['ur_rot'] = [float(x) for x in det['ur_rot']]
            det['ll_rot'] = [float(x) for x in det['ll_rot']]
            det['lr_rot'] = [float(x) for x in det['lr_rot']]

        # Create dictionary for detection data
        # Fill in data
        det_dict = {}
        det_dict['image_file'] = im_name
        det_dict['det_data'] = det_data
        det_dict['det_counts'] = det_counts
        det_dict['angles'] = [float(x) for x in angles]
        det_dict['colors'] = colors[:len(angles)]
        det_dict['row_padding'] = row_padding
        det_dict['col_padding'] = col_padding

        # Write data file for detections
        dets_dir = '/scratch3/terriyu/working/generate/rawdet_data'
        if not os.path.exists(dets_dir):
            os.makedirs(dets_dir)

        dets_fname = im_name_root + '_dets.json'
        with open(os.path.join(dets_dir, dets_fname), 'w') as f:
            json.dump(det_dict, f)
        print "Wrote detection data to %s" % dets_fname
