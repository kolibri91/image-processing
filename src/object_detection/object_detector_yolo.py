# import the necessary packages
import os
import numpy as np
import cv2

class ObjectDetectorYolo:
    def __init__(self, yolo_folder, confidence_threshold=0.5, threshold_nonmaxima_suppression=0.4):

        # folder with yolo weights
        self.yolo_folder = yolo_folder

        self.confidence_threshold = confidence_threshold
        self.threshold_nonmaxima_suppression = threshold_nonmaxima_suppression

        self.labels = None
        self.object_detector = None
        [self.labels, self.object_detector] = self.__create_object_detector()


    def process_frame(self, frame_rgb):

        # convert frame_rgb to a blob and pass it through the object detector
        blob = self.__create_blob_from_8bit_frame(frame_rgb)

        self.object_detector.setInput(blob)

        objects_detected = self.object_detector.forward(self.__get_object_detector_outputlayer())

        return objects_detected


    def postprocess_frame(self, frame, objects_detected):
        # pylint: disable=too-many-locals
        
        (frame_height, frame_width) = frame.shape[:2]

        object_boxes = []

        (object_class_ids, object_confidences, boxes) = self.__postprocess_yolo_frame(
            objects_detected)

        for box in boxes:
            width_scale = float(frame_width / 416)
            height_scale = float(frame_height / 416)
            left = int(width_scale * box[0])
            top = int(height_scale * box[1])
            right = int(width_scale * (box[0]+box[2]))
            bottom = int(height_scale * (box[1]+box[3]))
            object_boxes.append([left, top, right, bottom])

        return (object_class_ids, object_confidences, object_boxes)


    def label(self, object_class_id):
        return self.labels[object_class_id]


    def __create_object_detector(self):
        # load the COCO class labels our YOLO model was trained on
        labels_path = os.path.sep.join([self.yolo_folder, "coco.names"])
        labels = open(labels_path).read().strip().split("\n")

        # derive the paths to the YOLO weights and model configuration
        weights_path = os.path.sep.join([self.yolo_folder, "yolov3.weights"])
        config_path = os.path.sep.join([self.yolo_folder, "yolov3.cfg"])

        # load our YOLO object detector trained on COCO dataset (80 classes)
        # and determine only the *output* layer names that we need from YOLO
        print("[INFO] loading YOLO from disk...")
        object_detector = cv2.dnn.readNetFromDarknet(config_path, weights_path) # pylint: disable=no-member

        return labels, object_detector


    def __get_object_detector_outputlayer(self):
        layers_name = self.object_detector.getLayerNames()
        # get the name of the output layers, i.e., the layers with unconnected outputs
        return [layers_name[i[0]-1] for i in self.object_detector.getUnconnectedOutLayers()]


    @classmethod
    def __create_blob_from_8bit_frame(cls, frame_rgb):

        frame_rgb_resized = cv2.resize(frame_rgb, (416, 416)) # pylint: disable=no-member
        
        # convert frame_rgb to a blob and pass it through the object detector
        blob = cv2.dnn.blobFromImage(frame_rgb_resized, 
                                     1/255, (416, 416), 
                                     [0, 0, 0], 1, crop=False) # pylint: disable=no-member
                   
        return blob


    def __postprocess_yolo_frame(self, objects_detected):
        # pylint: disable=too-many-locals
        object_class_ids = []
        object_confidences = []
        object_boxes = []

        for obj in objects_detected:
            for detection in obj:
                object_scores = detection[5:]
                object_class_id = np.argmax(object_scores)
                object_confidence = object_scores[object_class_id]
                if object_confidence > self.confidence_threshold:
                    center_x = int(detection[0]*416)
                    center_y = int(detection[1]*416)
                    width = int(detection[2]*416)
                    height = int(detection[3]*416)
                    left = int(center_x - width/2)
                    top = int(center_y - height/2)
                    object_class_ids.append(object_class_id)
                    object_confidences.append(float(object_confidence))
                    object_boxes.append([left, top, width, height])

        # maximum suppression to eliminate redundant overlapping boxes of lower object_confidences
        indices = cv2.dnn.NMSBoxes(object_boxes,
                                   object_confidences,
                                   self.confidence_threshold,
                                   self.threshold_nonmaxima_suppression) # pylint: disable=no-member
        if len(indices) > 0:
            idx = np.concatenate(indices, axis=0, out=None)
        else:
            idx = []
        return ([object_class_ids[i] for i in list(idx)], 
                [object_confidences[i] for i in list(idx)], 
                [object_boxes[i] for i in list(idx)])
