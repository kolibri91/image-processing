# Example usage: 
# src/image_processing/bin/demo.py --input ../images/Zermatt_20200623-87.tif --output ../images/xxx.tif -y ../yolo-coco/
import argparse
import os
import sys
import time
import imutils

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
            right = int(box[2])
            bottom = int(box[3])
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

def sliding_window(image, step, window_size):
    # slide a window across the image
    for y in range(0, image.shape[0] - window_size[1], step):
        for x in range(0, image.shape[1] - window_size[0], step):
            yield (x, y, image[y:y + window_size[1], x:x + window_size[0]])
        yield(image.shape[1]-window_size[0], y, image[y:y + window_size[1], image.shape[1]-window_size[0]:image.shape[1]])

    for x in range(0, image.shape[1] - window_size[0], step):
        yield(x, image.shape[0]-window_size[1], image[image.shape[0]-window_size[1]:image.shape[0], x:x + window_size[0]])
    yield(image.shape[1]-window_size[0], image.shape[0]-window_size[1], image[image.shape[0]-window_size[1]:image.shape[0], image.shape[1]-window_size[0]:image.shape[1]])

def image_pyramid(image, scale=1.5, min_size=(224, 224)):
    # yield the original image
    yield image
    # keep looping over the image pyramid
    while True:
        # compute the dimensions of the next image in the pyramid
        w = int(image.shape[1] / scale)
        image = imutils.resize(image, width=w)
        # if the resized image does not meet the supplied minimum
        # size, then stop constructing the pyramid
        if image.shape[0] < min_size[1] or image.shape[1] < min_size[0]:
            break
        # yield the next image in the pyramid
        yield image

PYRAMID_SCALE = 2.0
WIN_STEP = 416 #312
ROI_SIZE = (1232,1232)
INPUT_SIZE = (416,416)

def main(**kwargs):


    confidence = 0.1
    suppression_threshold = 0.4
    object_detector = ObjectDetectorYolo(kwargs["yolo"], confidence, suppression_threshold)

    print("[INFO] Read image from disk...")
    image = cv2.imread(kwargs["input"])
    (H, W) = image.shape[:2]
    datatype = image.dtype
    image_8bit = list(map(np.uint8, image)) if datatype != np.uint8 else image

   
   # initialize the image pyramid
    pyramid = image_pyramid(image_8bit, scale=PYRAMID_SCALE, min_size=ROI_SIZE)
    # initialize two lists, one to hold the ROIs generated from the image
    # pyramid and sliding window, and another list used to store the
    # (x, y)-coordinates of where the ROI was in the original image
    rois = []
    locs = []
    objects = []
   
    cnt = 0
    
    # loop over the image pyramid
    for image in pyramid:
        # determine the scale factor between the *original* image
        # dimensions and the *current* layer of the pyramid
        scale = W / float(image.shape[1])
        # for each layer of the image pyramid, loop over the sliding
        # window locations
        for (x, y, roi_orig) in sliding_window(image, WIN_STEP, ROI_SIZE):
            # scale the (x, y)-coordinates of the ROI with respect to the
            # *original* image dimensions
            x = int(x * scale)
            y = int(y * scale)
            w = int(ROI_SIZE[0] * scale)
            h = int(ROI_SIZE[1] * scale)
            # take the ROI and preprocess it so we can later classify
            # the region using Keras/TensorFlow
            roi = cv2.resize(roi_orig, INPUT_SIZE)
            
    
            #roi = preprocess_input(roi)
            # update our list of ROIs and associated coordinates
            rois.append(roi)
            locs.append((x, y, x + w, y + h))   
            
            
            check_for_mirrors = False
            
            objects_detected = object_detector.process_frame(roi_orig)
            (object_class_ids, object_confidences, object_boxes) = object_detector.postprocess_frame(roi_orig, objects_detected)

            for (class_id, confidence, box) in zip(object_class_ids, object_confidences, object_boxes):
                if object_detector.label(class_id) in ('person','backpack'):
                    left = int(float(box[0]) * scale)
                    top  = int(float(box[1]) * scale)
                    right = int(float(box[2]) * scale)
                    bottom = int(float(box[3]) * scale)
                    objects.append(((x+left,y+top),(x+right,y+bottom),class_id, confidence))
                    check_for_mirrors = True
                    
            if check_for_mirrors == True:
                # flip image and reprocess it
                roi_flipped = roi_orig.copy()
                roi_flipped = cv2.flip(roi_flipped, 0)               

                objects_detected_flipped = object_detector.process_frame(roi_flipped)
                (objectflipped_class_ids, objectflipped_confidences, objectflipped_boxes) = object_detector.postprocess_frame(roi_flipped, objects_detected_flipped)
                for (flipped_class_id, flipped_confidence, flipped_box) in zip(objectflipped_class_ids, objectflipped_confidences, objectflipped_boxes):
                    if object_detector.label(flipped_class_id) in ('person','backpack'):
                        left = int(float(flipped_box[0]) * scale)
                        top  = int(float(flipped_box[1]) * scale)
                        right = int(float(flipped_box[2]) * scale)
                        bottom = int(float(flipped_box[3]) * scale)
                        objects.append(((x+left,y+h-bottom),(x+right,y+h-top),flipped_class_id, flipped_confidence))
                    
                    
            # show the visualization and current ROI
            #clone_resized = imutils.resize(clone, width=1024)
            #cv2.imshow("Visualization", clone_resized)
            ##cv2.imshow("ROI", roiOrig)
            #cv2.waitKey(1)

            #print("ROI: {} roi: {}".format(roiOrig.shape,roi.shape))
            cnt += 1
   
    print("#ROIs = {}".format(len(rois)))
   
    result = image_8bit.copy()
    for o in objects:
        cv2.rectangle(result, o[0], o[1], (255,178,50),8)
        print("Object: {} Confidence: {} Box: ({})-({})".format(o[2],o[3],o[0],o[1]))
    
    cv2.imwrite("test.jpg", result)
    #result_resized = imutils.resize(result, width=1024)
    #cv2.imshow("Visualization", result_resized)
    #cv2.waitKey(0)
            
    ## detect people mirrored in lake
    #image_8bit = cv2.flip(image_8bit, 0)
    #img_part = image_8bit[1730:1730+500, 5340:5340+500].copy()
    #img_part_detected = process_frame(object_detector, img_part)
    #plt.imshow(img_part_detected[..., ::-1])
    #plt.show()

    ## detect people at lake (including the guy behind the hill)
    #image_8bit = cv2.flip(image_8bit, 0)
    #img_part = image_8bit[1840:1840+500, 5340:5340+500].copy()
    #img_part = image_8bit[1620:1840+1232, 5340:5340+1232].copy()
    #img_part_detected = process_frame(object_detector, img_part)
    #plt.imshow(img_part_detected[..., ::-1])
    #plt.show()
    
    ## detect guy on the right border of image
    #img_part = image_8bit[1600:1600+500, 5600:5600+500].copy()
    #img_part_detected = process_frame(object_detector, img_part)
    #plt.imshow(img_part_detected[..., ::-1])
    #plt.show()
    
    ## apply detector to full image
    #img_part = image_8bit.copy()
    #img_part_detected = process_frame(object_detector, img_part)
    #plt.imshow(img_part_detected[..., ::-1])
    #plt.show()

if __name__ == "__main__":
    arguments = sys.argv[1:]
    sys.exit(main(**parse_command_line_args(arguments)))
