#! /usr/bin/env python

import argparse
import os
import cv2
import numpy as np
from tqdm import tqdm
from preprocessing import parse_annotation
from utils import draw_boxes
from frontend import YOLO
import json
import  time

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2"

argparser = argparse.ArgumentParser(
    description='Train and validate YOLO_v2 model on traffic sign')

argparser.add_argument(
    '-c',
    '--conf',
    help='path to configuration file')

argparser.add_argument(
    '-w',
    '--weights',
    help='path to pretrained weights')

argparser.add_argument(
    '-f',
    '--flag',
    help='0:pictures detection; 1:video detection')
def _main_(args):
 
    config_path  = args.conf
    weights_path = args.weights
    # image_path   = args.input
    flag         = int(args.flag)
    with open(config_path) as config_buffer:    
        config = json.load(config_buffer)

    ###############################
    #   Make the model 
    ###############################
    start = time.time()
    yolo = YOLO(architecture        = config['model']['architecture'],
                input_size          = config['model']['input_size'], 
                labels              = config['model']['labels'], 
                max_box_per_image   = config['model']['max_box_per_image'],
                anchors             = config['model']['anchors'])
    end = time.time()
    print "Make the model costs:%f" % (end - start)
    ###############################
    #   Load trained weights
    ###############################    

    print weights_path
    start = time.time()
    yolo.load_weights(weights_path)
    end = time.time()
    print "load weights cost :%f" % (end - start)
    ###############################
    #   Predict bounding boxes 
    ###############################

    if flag == 1:
        start = time.time()
        video_out = image_path[:-4] + '_detected' + image_path[-4:]

        video_reader = cv2.VideoCapture(image_path)

        nb_frames = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_h = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_w = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))

        video_writer = cv2.VideoWriter(video_out,
                               cv2.VideoWriter_fourcc(*'MPEG'), 
                               50.0, 
                               (frame_w, frame_h))

        for i in tqdm(range(nb_frames)):
            _, image = video_reader.read()
            
            boxes = yolo.predict(image)
            image = draw_boxes(image, boxes, config['model']['labels'])

            video_writer.write(np.uint8(image))
        video_reader.release()
        video_writer.release()
        end = time.time()
        print "predict costs :%f"%(end-start)
    if flag ==0:
        start = time.time()
        for image_path in config['test']['image']:
            #print image_path
            image = cv2.imread(image_path)
            boxes = yolo.predict(image)
            image = draw_boxes(image, boxes, config['model']['labels'])
            print len(boxes), 'boxes are found in %s'%(image_path)
            if len(boxes)>0:
                cv2.imwrite(image_path[:-4] + '_detected' + image_path[-4:], image)
        end = time.time()
        print "predict costs :%f"%(end-start)
if __name__ == '__main__':
    args = argparser.parse_args()
    _main_(args)
