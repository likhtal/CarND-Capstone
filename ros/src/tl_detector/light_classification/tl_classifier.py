import os
import sys
import tensorflow as tf
import numpy as np
import rospy

from styx_msgs.msg import TrafficLight

NUM_CLASSES = 4
CATEGORY_INDEX = {1: {'id': 1, 'name': 'Green'}, 2: {'id': 2, 'name': 'Red'}, 3: {'id': 3, 'name': 'Yellow'}, 4: {'id': 4, 'name': 'off'}}
MIN_SCORE_THRESH = .50

class TLClassifier(object):
    def __init__(self):
        self.model_path = rospy.get_param('~model_path', False)
        self.detection_graph = tf.Graph()
        
        with self.detection_graph.as_default():
          od_graph_def = tf.GraphDef()
        
          with tf.gfile.GFile(self.model_path, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')             

        self.config = tf.ConfigProto(log_device_placement=True)
        self.config.gpu_options.per_process_gpu_memory_fraction = 0.5
        self.config.operation_timeout_in_ms = 50000 # terminate anything that don't return in 50 seconds

        self.tf_session = tf.Session(graph=self.detection_graph, config=self.config)

        # Definite input and output Tensors for detection_graph
        self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        
        # Each box represents a part of the image where a particular object was detected.
        self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')        

        print("Done initializing TL Classifier.")

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        print("start classification")
        image_np_expanded = np.expand_dims(image, axis=0)

        (boxes, scores, classes, num) = self.tf_session.run(
              [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
              feed_dict={self.image_tensor: image_np_expanded})

        boxes = np.squeeze(boxes)
        scores = np.squeeze(scores)
        classes = np.squeeze(classes).astype(np.int32)

        print("end classification")

        if scores is not None and scores[0] > MIN_SCORE_THRESH:
            class_name = CATEGORY_INDEX[classes[0]]['name']
            print((class_name, scores[0]))

            if classes[0]  == 1: 
               return TrafficLight.GREEN
            elif classes[0]  == 2: 
               return TrafficLight.RED
            elif classes[0]  == 3: 
               return TrafficLight.YELLOW

        return TrafficLight.UNKNOWN
