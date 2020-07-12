import argparse
import os
import sys

import cv2
import numpy as np
from image_processing.io import read_image, write_image
from matplotlib import pyplot as plt

from object_detection.object_detector_yolo import ObjectDetectorYolo


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


def process_frame(object_detector, image):
    objects_detected = object_detector.process_frame(image)
    (object_class_ids, object_confidences, object_boxes) = object_detector.postprocess_frame(image, objects_detected)

    for (class_id, confidence, box) in zip(object_class_ids, object_confidences, object_boxes):
        if object_detector.label(class_id) in ('person','backpack'):
            left = box[0]
            top  = box[1]
            right = box[2]
            bottom = box[3]
            right = int(box[2]+20)
            bottom = int(box[3]+25)
            cv2.rectangle(image, (left,top), (right,bottom), (255,178,50),3)
            label = '%s:%.2f' % (object_detector.label(class_id), confidence)
            label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            top = max(top, label_size[1])
            cv2.rectangle(image, 
                            (left, top - round(1.5*label_size[1])), 
                            (left+round(1.5*label_size[0]), top + base_line), 
                            (255,255,255), cv2.FILLED)
            cv2.putText(image, label, (left,top), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
        
    return image

def main(**kwargs):


    confidence = 0.3
    suppression_threshold = 0.4
    object_detector = ObjectDetectorYolo(kwargs["yolo"], confidence, suppression_threshold)

    print("[INFO] Read image from disk...")
    image = cv2.imread(kwargs["input"])
    (H, W) = image.shape[:2]
    datatype = image.dtype
    image_8bit = list(map(np.uint8, image)) if datatype != np.uint8 else image

    

    # detect people mirrored in lake
    image_8bit = cv2.flip(image_8bit, 0)
    img_part = image_8bit[1730:1730+500, 5340:5340+500].copy()
    img_part_detected = process_frame(object_detector, img_part)
    #img_part_detected = cv2.flip(img_part_detected, 0)
    plt.imshow(img_part_detected[..., ::-1])
    plt.show()

    # detect people at lake (including the guy behind the hill)
    image_8bit = cv2.flip(image_8bit, 0)
    img_part = image_8bit[1840:1840+500, 5340:5340+500].copy()
    img_part_detected = process_frame(object_detector, img_part)
    plt.imshow(img_part_detected[..., ::-1])
    plt.show()
    
    # detect guy on the right border of image
    img_part = image_8bit[1600:1600+500, 5600:5600+500].copy()
    img_part_detected = process_frame(object_detector, img_part)
    plt.imshow(img_part_detected[..., ::-1])
    plt.show()
    
    # apply detector to full image
    img_part = image_8bit.copy()
    img_part_detected = process_frame(object_detector, img_part)
    plt.imshow(img_part_detected[..., ::-1])
    plt.show()

if __name__ == "__main__":
    arguments = sys.argv[1:]
    sys.exit(main(**parse_command_line_args(arguments)))
