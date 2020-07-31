# Example usage: 
# src/image_processing/bin/demo.py --input ../images/Zermatt_20200623-87.tif --output ../images/xxx.tif -y ../yolo-coco/
import argparse
import os
import sys
import imutils

import cv2
import numpy as np
from image_processing.io import read_image, write_image
from matplotlib import pyplot as plt

from object_detection.object_detector_yolo import ObjectDetectorYolo
from object_detection.object_detector_pyramid import ObjectDetectorPyramid

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True,
            help="path to input image")
    parser.add_argument("-o", "--output", required=True,
            help="path to output video")
    parser.add_argument("-y", "--yolo", required=True,
            help="base path to YOLO directory")
    return parser


def parse_command_line_args(args):
    parsed_args = get_parser().parse_args(args)
    return vars(parsed_args)


def main(**kwargs):


    confidence = 0.1
    suppression_threshold = 0.4
    object_detector = ObjectDetectorYolo(kwargs["yolo"], confidence, suppression_threshold)

    print("[INFO] Read image from disk...")
    image = cv2.imread(kwargs["input"])

    datatype = image.dtype
    image_8bit = list(map(np.uint8, image)) if datatype != np.uint8 else image

    object_detector_pyramid = ObjectDetectorPyramid(object_detector)
    objects = object_detector_pyramid.process_frame(image_8bit)
   
    result = image_8bit.copy()
    for o in objects:
        cv2.rectangle(result, (o[2][0],o[2][1]), (o[2][2],o[2][3]), (255,178,50), 8)
    
    cv2.imwrite("test2.jpg", result)
   

if __name__ == "__main__":
    arguments = sys.argv[1:]
    sys.exit(main(**parse_command_line_args(arguments)))
