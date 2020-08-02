# import the necessary packages
import os
import numpy as np
import cv2
import imutils


PYRAMID_SCALE = 2.0
WIN_STEP = 416 #312
ROI_SIZE = (1232,1232)
INPUT_SIZE = (416,416)


class ObjectDetectorPyramid:
    def __init__(self, object_detector):

        # folder with yolo weights
        self.object_detector = object_detector
        

    def process_frame(self, frame_rgb):

        # initialize the image pyramid
        pyramid = self.__image_pyramid(frame_rgb, scale=PYRAMID_SCALE, min_size=ROI_SIZE)
        
        (H, W) = frame_rgb.shape[:2]
        
        # initialize two lists, one to hold the ROIs generated from the image
        # pyramid and sliding window, and another list used to store the
        # (x, y)-coordinates of where the ROI was in the original image
        rois = []
        locs = []
        objects = []
           
        # loop over the image pyramid
        for image in pyramid:
            # determine the scale factor between the *original* image
            # dimensions and the *current* layer of the pyramid
            scale = W / float(image.shape[1])
            # for each layer of the image pyramid, loop over the sliding
            # window locations
            for (x, y, roi_orig) in self.__sliding_window(image, WIN_STEP, ROI_SIZE):
                # scale the (x, y)-coordinates of the ROI with respect to the
                # *original* image dimensions
                x = int(x * scale)
                y = int(y * scale)
                w = int(ROI_SIZE[0] * scale)
                h = int(ROI_SIZE[1] * scale)
                # take the ROI and preprocess it so we can later classify
                # the region using Keras/TensorFlow
                roi = cv2.resize(roi_orig, INPUT_SIZE)
                
                # update our list of ROIs and associated coordinates
                rois.append(roi)
                locs.append((x, y, x + w, y + h))   
                
                check_for_mirrors = False
                
                (bounding_boxes, classes, confidences) =  self.__find_objects_and_return_object_informations(self.object_detector, roi_orig, scale)
            
                for (bb, class_id, confidence) in zip(bounding_boxes, classes, confidences):
                    objects.append((class_id, confidence, (x+bb[0],y+bb[1],x+bb[2],y+bb[3])))
                    check_for_mirrors = True
                        
                if check_for_mirrors == True:
                    # flip image and reprocess it
                    roi_flipped = roi_orig.copy()
                    roi_flipped = cv2.flip(roi_flipped, 0)               

                    (bounding_boxes, classes, confidences) =  self.__find_objects_and_return_object_informations(self.object_detector, roi_flipped, scale)
                    for (bb, flipped_class_id, flipped_confidence) in zip(bounding_boxes, classes, confidences):
                    
                        objects.append((flipped_class_id, flipped_confidence, (x+bb[0],y+h-bb[3],x+bb[2],y+h-bb[1])))
            
        return objects


    def process_frame_and_generate_mask(self, frame_rgb):

        objects = self.process_frame(frame_rgb)
   
        image_height,image_width,image_depth = frame_rgb.shape
        image_mask = np.zeros((image_height,image_width),dtype = np.uint8)
        for o in objects:
            cv2.rectangle(image_mask, (o[2][0],o[2][1]), (o[2][2],o[2][3]), 255, -1)
        return (image_mask, objects)
    

    def __sliding_window(self, image, step, window_size):
        xx = list(range(0, image.shape[1] - window_size[0], step))
        xx.append(image.shape[1]-window_size[0])
        yy = list(range(0, image.shape[0] - window_size[1], step))
        yy.append(image.shape[0]-window_size[1])

        # slide a window across the image
        for y in yy:
            for x in xx:
                yield (x, y, image[y:y + window_size[1], x:x + window_size[0]])


    def __image_pyramid(self, image, scale=1.5, min_size=(224, 224)):
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


    def __find_objects_and_return_object_informations(self, object_detector, frame, scaling_factor):
        objects_detected = object_detector.process_frame(frame)
        (object_class_ids, object_confidences, object_boxes) = object_detector.postprocess_frame(frame, objects_detected)

        bounding_boxes = []
        classes = []
        confidences = []
        for (class_id, confidence, box) in zip(object_class_ids, object_confidences, object_boxes):
            if object_detector.label(class_id) in ('person','backpack'):
                left = int(float(box[0]) * scaling_factor)
                top  = int(float(box[1]) * scaling_factor)
                right = int(float(box[2]) * scaling_factor)
                bottom = int(float(box[3]) * scaling_factor)
                bounding_boxes.append((left,top,right,bottom))
                classes.append(class_id)
                confidences.append(confidence)
                
        return (bounding_boxes, classes, confidences)
