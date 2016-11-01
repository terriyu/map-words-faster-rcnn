#!/usr/bin/env python

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
from sklearn.cluster import KMeans

NETS = {'vgg16': ('VGG16', 'pre-trained-models/map_words_faster_rcnn.caffemodel')}

def vis_detections(im, title, dets, thresh):
    # im = im[:, :, (2, 1, 0)]
    for i in xrange(dets.shape[0]):
        bbox = dets[i, :4]
        score = dets[i, -1]
        if score > thresh:
            cv2.rectangle(im, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)

    cv2.imshow(title, im)
    cv2.waitKey(0)

def save_detections(im, im_name, dets, thresh, color = (0, 255, 0)):
    for i in xrange(dets.shape[0]):
        bbox = dets[i, :4]
        score = dets[i, -1]
        if score > thresh:
            cv2.rectangle(im, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 4)
    cv2.imwrite(im_name, im)

def im_detect_sliding_crop(net, im, crop_h, crop_w, step):
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

            # # check
            # cv2.imshow("im", crop_im)
            # cv2.waitKey(0)
            # print crop_im.shape
            crop_scores, crop_boxes = im_detect(net, crop_im)
            crop_boxes = crop_boxes[:, 4*cls_ind:4*(cls_ind + 1)]
            crop_scores = crop_scores[:, cls_ind] + (0.01 * np.random.random() - 0.005)

            # vis_detections(crop_im, 'crop image',
            #                np.hstack((crop_boxes,
            #                              crop_scores[:, np.newaxis])), 0.25)

            crop_boxes[:,0] += x1
            crop_boxes[:,1] += y1
            crop_boxes[:,2] += x1
            crop_boxes[:,3] += y1

            boxes = np.vstack((boxes, crop_boxes))
            scores = np.vstack((scores, crop_scores[:, np.newaxis]))

            # # print crop_boxes.shape, crop_scores.shape, boxes.shape, scores.shape
            # keep_idx = np.where(crop_scores > 0.1)
            # print len(keep_idx[0])

            # keep_idx = np.where(scores > 0.1)
            # print len(keep_idx[0])

            # vis_detections(im, 'entire image',
            #                np.hstack((boxes, scores)), 0.25)

            x1 += step

        y1 += step

    return scores, boxes

def load_distinct_colors(path):
    with open(path, 'r') as f:
        hex_colors = f.read().splitlines()

    rgb_colors = [(int(h[4:6], 16), int(h[2:4], 16), int(h[0:2], 16)) for h in hex_colors]

    return rgb_colors 

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Face Detection using Faster R-CNN')
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

    im_dir = 'images'
    out_dir = 'output'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    im_names = ['D0090-5242001.tiff']

    #angles = [-90, -70, -50, -30, -10, 0, 10, 30, 50, 70, 90]
    #angles = [50, 90]
    angles = np.insert(np.linspace(-90, 90, 20), 0, 0)

    #colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255), (255, 255, 0), (0, 0, 128), (0, 128, 0), (128, 0, 0), (0, 128, 128), (128, 0, 128), (128, 128, 0)]
    colors_dir = 'tools'
    colors = load_distinct_colors(os.path.join(colors_dir, 'colors_rgb_divide.txt'))

    for im_name in im_names:
        im_orig = cv2.imread(os.path.join(im_dir, im_name))
        diagonal = np.ceil(np.sqrt(np.sum(np.square(im_orig.shape))))
        rows_orig, cols_orig, _ = im_orig.shape
        h_padding = np.int_(np.ceil((diagonal - cols_orig)/2.0))
        v_padding = np.int_(np.ceil((diagonal - rows_orig)/2.0))

        im_padded = cv2.copyMakeBorder(im_orig, v_padding, v_padding, h_padding, h_padding, cv2.BORDER_CONSTANT, value=(128, 128, 128))
        rows, cols, _ = im_padded.shape
        im_name_root = string.split(im_name, '.')[0]

        im_fname =  im_name_root + "_padded.tiff"
        cv2.imwrite(os.path.join(out_dir, im_fname), im_padded)

        im_composite = im_padded.copy()

        det_data = []
        det_counts = {}
        for idx, angle in enumerate(angles):
            # Rotate image
            rot_mat = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
            im_rot = cv2.warpAffine(im_padded, rot_mat, (cols, rows))
            # Output file name for rotated image
            im_rot_name = im_name_root + '_' + str(angle) + 'deg.tiff'

            # # Detect all object classes and regress object bounds
            timer = Timer()
            timer.tic()
            # scores, boxes = im_detect(net, im)
            scores, boxes = im_detect_sliding_crop(net, im_rot, crop_h, crop_w, step)
            timer.toc()
            print ('Detection took {:.3f}s for ' '{:d} object proposals').format(timer.total_time, boxes.shape[0])

            dets = np.hstack((boxes, scores)).astype(np.float32)
            keep = nms(dets, NMS_THRESH)
            dets = dets[keep, :]

            keep = np.where(dets[:, 4] > CONF_THRESH)
            dets = dets[keep]
            det_counts[angle] = dets.shape[0]
            save_detections(im_rot, os.path.join(out_dir, im_rot_name), dets, CONF_THRESH, colors[idx])

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
                data_dict = {'score': float(score), 'angle': float(angle), 'height': float(height), 'width': float(width), 'aspect_ratio': float(aspect_ratio), 'center': [float(x) for x in center], 'ul': [float(bbox[0]), float(bbox[1])], 'ur': [float(bbox[2]), float(bbox[1])], 'll': [float(bbox[0]), float(bbox[3])], 'lr': [float(bbox[2]), float(bbox[3])]}

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
        im_composite = im_composite[v_padding:-v_padding, h_padding:-h_padding]

        im_composite_name = im_name_root + '_composite.tiff'
        cv2.imwrite(os.path.join(out_dir, im_composite_name), im_composite)

        # Convert entries in det_data to float, so we can save det_data to JSON
        for det in det_data:
            det['center_rot'] = [float(x) for x in det['center_rot']]
            det['ul_rot'] = [float(x) for x in det['ul_rot']]
            det['ur_rot'] = [float(x) for x in det['ur_rot']]
            det['ll_rot'] = [float(x) for x in det['ll_rot']]
            det['lr_rot'] = [float(x) for x in det['lr_rot']]

        det_dict = {}
        det_dict['det_data'] = det_data
        det_dict['det_counts'] = det_counts
        det_dict['angles'] = [float(x) for x in angles]
        det_dict['colors'] = colors[:len(angles)]

        dets_dir = 'data'
        if not os.path.exists(dets_dir):
            os.makedirs(dets_dir)

        dets_fname = im_name_root + '_dets.json'
        with open(os.path.join(dets_dir, dets_fname), 'w') as f:
            json.dump(det_dict, f)
        print "Wrote detection data to %s" % dets_fname
