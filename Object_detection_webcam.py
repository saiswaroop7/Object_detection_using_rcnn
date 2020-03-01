import os
import cv2
import numpy as np
import tensorflow as tf
import sys

sys.path.append("..")

from HW5.support_files import label_map_util
from HW5.support_files import visualization_utils as vis_util

cwd = os.getcwd()

classifier_path = os.path.join(cwd,'support_files','frozen_inference_graph.pb')

label_map_path = os.path.join(cwd,'support_files','labelmap.pbtxt')

NUM_CLASSES = 1

label_map = label_map_util.load_labelmap(label_map_path)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Loading Tensorflow Model.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(classifier_path, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    
sess = tf.Session(graph=detection_graph)
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

num_detections = detection_graph.get_tensor_by_name('num_detections:0')

vidcap = cv2.VideoCapture(0)
ret = vidcap.set(3,800)
ret = vidcap.set(4,600)

while(True):

    ret, frame = vidcap.read()
    frame_expanded = np.expand_dims(frame, axis=0)

    (boxes, scores, classes) = sess.run(
        [detection_boxes, detection_scores, detection_classes],
        feed_dict={image_tensor: frame_expanded})

    vis_util.visualize_boxes_and_labels_on_image_array(
        frame,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=3,
        min_score_thresh=0.60)

    cv2.imshow('Object detector', frame)

    if cv2.waitKey(1) == ord('q'):
        break

vidcap.release()
cv2.destroyAllWindows()

