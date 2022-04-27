#Tavin Ardell
# Senior Project 2
# I have neither given nor received unauthorized aid in completing this work, nor have I used someone else's work as my own.



import os
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
from object_detection.utils import config_util
import cv2 
import numpy as np
from matplotlib import pyplot as plt
import tkinter

SCORE = 0.60

configs = config_util.get_configs_from_pipeline_file(os.path.join('libs', 'Tensorflow', 'workspace','models', 'my_ssd_mobnet', 'pipeline.config'))
detection_model = model_builder.build(model_config=configs['model'], is_training=False)
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join('libs','Tensorflow','workspace','models','my_ssd_mobnet', 'ckpt-2')).expect_partial()

configs = config_util.get_configs_from_pipeline_file(os.path.join('libs', 'Tensorflow', 'workspace','models', 'my_ssd_mobnet', 'pipeline.config'))
detection_model = model_builder.build(model_config=configs['model'], is_training=False)
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join('libs','Tensorflow','workspace','models','my_ssd_mobnet', 'ckpt-5')).expect_partial()
tf.saved_model.save(detection_model, 'libs/Tensorflow/workspace/models/my_ssd_mobnet/tfliteexport/saved_model')
os.system("python libs/Tensorflow/models/research/object_detection/export_tflite_graph_tf2.py --trained_checkpoint_dir libs/Tensorflow/workspace/models/my_ssd_mobnet --output_directory libs/Tensorflow/workspace/models/my_ssd_mobnet/tfliteexport --pipeline_config_path libs/Tensorflow/workspace/models/my_ssd_mobnet/pipeline.config")
os.system("dir -lah libs/Tensorflow/workspace/models/my_ssd_mobnet/tfliteexport/saved_model")
os.system("tflite_convert --saved_model_dir libs/Tensorflow/workspace/models/my_ssd_mobnet/tfliteexport/saved_model --output_file libs/Tensorflow/workspace/models/my_ssd_mobnet/tfliteexport/saved_model/model.tflite")
os.system("dir -lah libs/Tensorflow/workspace/models/my_ssd_mobnet/tfliteexport/saved_model/DDA_TFLite_Model.tflite") 

def detect_fn(image):
  image, shapes = detection_model.preprocess(image)
  prediction_dict = detection_model.predict(image, shapes)
  detections = detection_model.postprocess(prediction_dict, shapes)
  return detections

category_index = label_map_util.create_category_index_from_labelmap(os.path.join('libs', 'Tensorflow', 'workspace','annotations', 'label_map.pbtxt'))

IMAGE_PATH = os.path.join('libs', 'Tensorflow', 'workspace','images', 'final_tests')

import os

directory = os.fsencode(IMAGE_PATH)
count = 0

for file in os.listdir(directory): # start iterating through the files
  print(count, '/249')             # so I know how much longer the images will print for
  filename = os.fsdecode(file)      
  if filename.endswith(".bmp"):   # only read .bmp files in case any other files are in the directory by accident
    print(filename)               
    print('------------------------------------------')
    img = cv2.imread(IMAGE_PATH + '/' + filename)
    image_np = np.array(img)
    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32) # converting image to a tensor
    detections = detect_fn(input_tensor)      # attempting to detect any objects (defects) within that tensor

    num_detections = int(detections.pop('num_detections'))    # an array of all the possible defects within the image

    detections = {key: value[0, :num_detections].numpy()
                  for key, value in detections.items()}
    detections['num_detections'] = num_detections

    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    label_id_offset = 1
    image_np_with_detections = image_np.copy()

    viz_utils.visualize_boxes_and_labels_on_image_array(      # preparing the outputted image
                image_np_with_detections,               
                detections['detection_boxes'],
                detections['detection_classes']+label_id_offset,
                detections['detection_scores'],
                category_index,
                use_normalized_coordinates=True,
                max_boxes_to_draw=1,                          # I don't need to see more than 1 defects
                min_score_thresh=SCORE,                         # I want to see any defects with scores over a score
                agnostic_mode=False)


    defectCounter = 0

    for i in range(len(detections['detection_scores'])):
      if detections['detection_scores'][i] > SCORE:
        defectCounter = defectCounter + 1

    if defectCounter > 1:
      print('Multiple defects detected in part!\n')
    elif defectCounter > 0:
      print('Defect detected in part!\n')
    else:
      print('No defects found!\n')    

    for i in range(len(detections['detection_scores'])): # associating a score and a class with a label name
      if detections['detection_scores'][i] > SCORE:     
        temp = detections['detection_classes'][i]
        print(category_index[temp+1]['name'], ':', detections['detection_scores'][i]*100)

    plt.imshow(cv2.cvtColor(image_np_with_detections, cv2.COLOR_BGR2RGB))
    plt.show() # print the image
    print('\n\n')
    count = count + 1
    continue   #iterate
  else:
    continue 



