#!/usr/bin/env python

# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""

import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import argparse
import time

CLASSES = ('__background__',
           'big', 'others')
COlOR = [(0,0,0),(255,0,0),(0,0,255),(255,255,0)]

seg = [[[0,0],[599,800]],[[499,0],[1099,800]],[[999,0],[1599,800]]]

CONF_THRESH = 0.3
NMS_THRESH = 0.1


NETS = {'vgg16': ('VGG16',
                  'VGG16_faster_rcnn_final.caffemodel'),
        'zf': ('ZF',
                  'ZF_faster_rcnn_final.caffemodel'),
		'hs':('hs',
                  'vgg16_faster_rcnn_iter_60000.caffemodel')}


def py_cpu_nms(dets, thresh):
    """Pure Python NMS baseline."""
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep

def save_detections(image_name, dets):
	"""Save detected bounding boxes as images."""
	im_file = os.path.join('./lib/datasets/T006B/im',image_name+'.jpg')
	im = cv2.imread(im_file)
	outpath1 = os.path.join('./lib/datasets/T006B/res',image_name+'.txt')
	fresult=open(outpath1 ,'a+')
	inds = len(dets)
	if inds == 0:
		return
	for i in range(inds):
		bbox = dets[i][:4]
		score = dets[i][4]
		cls_ind = int(dets[i][5])
		if score > CONF_THRESH:
			cv2.rectangle(im,(int(bbox[0]), int(bbox[1])),(int(bbox[2]) , int(bbox[3])),COlOR[cls_ind],5,3)
			cv2.putText(im,"%.3f" %score,(int(bbox[0]), int(bbox[1])),cv2.FONT_HERSHEY_SIMPLEX, 0.50, COlOR[cls_ind])
			fresult.write(str(bbox[0]))
			fresult.write(' ')
			fresult.write(str(bbox[1]))
			fresult.write(' ')
			fresult.write(str(bbox[2]))
			fresult.write(' ')
			fresult.write(str(bbox[3]))
			fresult.write(' ')
			fresult.write(str(score))
			fresult.write(' ')
			fresult.write(str(cls_ind))
			fresult.write('\n')
	cv2.imwrite(im_file,im)		
	fresult.close()

def detectBySeg(net,image):
	res = []
	
	timer = Timer()
	timer.tic()
	scores, boxes = im_detect(net, image)
	timer.toc()
	print ('Detection took {:.3f}s for '
		'{:d} object proposals').format(timer.total_time, boxes.shape[0])
	
	for cls_ind, cls in enumerate(CLASSES[1:]):
		cls_ind += 1 # because we skipped background
		cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
		cls_scores = scores[:, cls_ind]
		dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])).astype(np.float32)
		keep = nms(dets, NMS_THRESH)
		dets = dets[keep, :]
		inds = np.where(dets[:, -1] >= CONF_THRESH)[0]
		for i in inds:
			score = dets[i, -1]
			if score > CONF_THRESH:
				tmp = dets[i].tolist()
				tmp.append(cls_ind)
				res.append(tmp)
	return res
	
	

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=1, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16]',
                        choices=NETS.keys(), default='hs')

    args = parser.parse_args()

    return args

if __name__ == '__main__':
	cfg.TEST.HAS_RPN = True  # Use RPN for proposals
	args = parse_args()

    #prototxt = os.path.join(cfg.ROOT_DIR, 'models', 'pascal_voc','VGG_CNN_M_1024', 'faster_rcnn_end2end', 'test.prototxt')
    #caffemodel = os.path.join(cfg.ROOT_DIR, 'output','faster_rcnn_end2end', 'hs','vgg_cnn_m_1024_faster_rcnn_iter_60000.caffemodel')
	prototxt = os.path.join(cfg.ROOT_DIR, 'models', 'pascal_voc','VGG16','faster_rcnn_end2end', 'test.prototxt')
	caffemodel = os.path.join(cfg.ROOT_DIR, 'output','faster_rcnn_end2end','hs','vgg16_faster_rcnn_iter_80000.caffemodel')
	if args.cpu_mode:
		caffe.set_mode_cpu()
	else:
		caffe.set_mode_gpu()
		caffe.set_device(args.gpu_id)
		cfg.GPU_ID = args.gpu_id
		net = caffe.Net(prototxt, caffemodel, caffe.TEST)
	#print '\n\nLoaded network {:s}'.format(caffemodel)

    # Warmup on a dummy image
	im = 128 * np.ones((600, 1200, 3), dtype=np.uint8)
	for i in xrange(2):
		_, _= im_detect(net, im)

	with open('/home/dl215/workspace/whz/detection_biaoshi2/py-faster-rcnn/lib/datasets/test/test.txt','r') as images:
		im_names=images.readlines()
		
	for im_name in im_names:
		print im_name
		dets = []
		print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
		print 'Demo for data/demo/{}'.format(im_name)
		im_file = os.path.join('./lib', 'datasets','test',im_name.strip()+'.jpg')
		print im_file
		im = cv2.imread(im_file)
		for i in range(0,3):
			#seg = [[[0,0],[600,800]],[[499,0],[1099,800]],[[999,0],[1599,800]]]
			imSeg = im[seg[i][0][1]:seg[i][1][1],seg[i][0][0]:seg[i][1][0]]
			det = detectBySeg(net,imSeg)
			for list in det:
				list[0] += seg[i][0][0]
				list[2] += seg[i][0][0]
				dets.append(list)
		if len(dets) == 0:
			continue
		
		dets = np.array(dets)
		dets0 = []
		dets1 = []
		dets2 = []
		for det in dets:
			print det[5]
			if det[5] == 0:
				dets0.append(det)
			else:
				dets1.append(det)
		if len(dets0) != 0 :
			dets0 = np.array(dets0)
			keep0 = py_cpu_nms(dets0, NMS_THRESH)
			dets0 = dets0[keep0,:]
			dets0 = dets0.tolist()
			print dets0
		if len(dets1) != 0 :
			dets1 = np.array(dets1)
			keep1 = py_cpu_nms(dets1, NMS_THRESH)
			dets1 = dets1[keep1,:]
			dets1 = dets1.tolist()
			print dets1
		
		save_detections(im_name.strip(),dets0)
		save_detections(im_name.strip(),dets1)
