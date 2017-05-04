#!/usr/bin/env python

# --------------------------------------------------------
# Faster R-CNN
# Written by Jiayi Wei
# --------------------------------------------------------

"""
this python fles is for test the performance on signal detection.
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
           '01', '02', '03', '04')
COlOR = [(0,0,0),(255,0,0),(0,0,255),(255,255,0), (0, 255, 0)]

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
				  
				  
def save_image(image_name, dets):
	clas_name=['01','02','03','04']
	root_path = '/home/dl215/workspace/wei/py-faster-rcnn/lib/datasets'
	#save image with all boxes on it and record the coordination of all boxes in .txt files	
	im_file = os.path.join('./lib/datasets/test/im',image_name+'.jpg')
	im = cv2.imread(im_file)
	outpath1 = os.path.join('./lib/datasets/test/res',image_name+'.txt')
	fresult=open(outpath1 ,'a+')
	nums = len(dets)
	if nums == 0:
		return
	for i in range(nums):
		box = dets[i][:4]
		score = dets[i][4]
		cls = int(dets[i][5])
		if score > CONF_THRESH:
			cv2.rectangle(img,(int(box[0]), int(box[1])),(int(box[2]) , int(box[3])),COlOR[cls],5,3)
			cv2.putText(img,"%.3f" %score,(int(box[0]), int(box[1])),cv2.FONT_HERSHEY_SIMPLEX, 0.50, COlOR[cls])
			
			little_img_path = os.path.join(root_path, 'result', clas_name[i], image_path+'_'+str(i)+'_'+'.jpg')
			little_img = img[int(box[1]):int(box[3]), int(box[0]):int(box[2])].copy()
			cv2.imwrite(little_img_path ,little_img)
			
			txt_file.write(str(box[0]))
			txt_file.write(' ')
			txt_file.write(str(box[1]))
			txt_file.write(' ')
			txt_file.write(str(box[2]))
			txt_file.write(' ')
			txt_file.write(str(box[3]))
			txt_file.write(' ')
			txt_file.write(str(cls))
			txt_file.write('\n')
		cv2.imwrite(image_path, img)
		txt_file.close()
		
				  

def detectBySeg(net,image):
	#the detection is done here, all result are saved in "res"
	res = []
	
	timer = Timer()
	timer.tic()
	scores, boxes = im_detect(net, image)#happen here
	timer.toc()
	print ('Detection took {:.3f}s for '
		'{:d} object proposals').format(timer.total_time, boxes.shape[0])
	
	for cls_ind, cls in enumerate(CLASSES[1:]):
		cls_ind += 1 # because we skipped background
		cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]#coordination for every box
		cls_scores = scores[:, cls_ind]#score for this 
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
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16]',
                        choices=NETS.keys(), default='hs')

    args = parser.parse_args()

    return args
	
if __name__ == '__mian__':
	cfg.TEST.HAS_RPN = True  # Use RPN for proposals
	args = parse_args()
	
	prototxt = os.path.join(cfg.ROOT_DIR, 'models', 'pascal_voc','VGG16','faster_rcnn_end2end', 'test.prototxt') #prototxt
	caffemodel = os.path.join(cfg.ROOT_DIR, 'output','faster_rcnn_end2end','hs','vgg16_faster_rcnn_iter_60000.caffemodel') #caffemodel
	if args.cpu_mode:
		caffe.set_mode_cpu()
	else:
		caffe.set_mode_gpu()
		caffe.set_device(args.gpu_id)
		cfg.GPU_ID = args.gpu_id
		net = caffe.Net(prototxt, caffemodel, caffe.TEST)
	# Warmup on a dummy image
	im = 128 * np.ones((600, 1200, 3), dtype=np.uint8)
	for i in xrange(2):
		_, _= im_detect(net, im)
		
	with open('/home/dl215/workspace/wei/py-faster-rcnn/lib/datasets/test/test.txt','r'):
		im_names=images.readlines()
	#store the image as a list	
	for im_name in im_names:
		dets = []
		print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
		print 'Demo for data/demo/{}'.format(im_name)
		im_file = os.path.join('./lib', 'datasets','test',im_name.strip()+'.jpg')
		im = cv2.imread(im_file)
		for i in range(0,3):
			#seg = [[[0,0],[600,800]],[[499,0],[1099,800]],[[999,0],[1599,800]]]
			imSeg = im[seg[i][0][1]:seg[i][1][1],seg[i][0][0]:seg[i][1][0]]
			det = detectBySeg(net,imSeg)
			#×ø±ê±ä»»£»
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
		dets3 = []
		for det in dets:
			print det[5]
			if det[5] == 0:
				dets0.append(det)
			else if det[5] == 1:
				dets1.append(det)
			else if det[5] == 2:
				dets2.append(det)
			else:
				dets3.append(det)
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
		if len(dets2) != 0 :
			dets2 = np.array(dets2)
			keep2 = py_cpu_nms(dets2, NMS_THRESH)
			dets2 = dets2[keep2,:]
			dets2 = dets2.tolist()
			print dets2
		if len(dets3) != 0 :
			dets3 = np.array(dets3)
			keep3 = py_cpu_nms(dets3, NMS_THRESH)
			dets3 = dets3[keep3,:]
			dets3 = dets3.tolist()
			print dets3
	
		save_detections(im_name.strip(),dets0)
		save_detections(im_name.strip(),dets1)
		save_detections(im_name.strip(),dets2)
		save_detections(im_name.strip(),dets3)
		
		
		
		
		
		
	
