# Tavin Ardell
# Senior Projects 2
# 000754847
# I have neither given nor received unauthorized aid in completing this work, nor have I used someone else's work as my own.

from itertools import count
from logging.config import fileConfig
from pickle import GLOBAL
import re
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os
import time
from ftplib import FTP
import shutil
from math import sqrt
from PIL import Image

# Enumerated values
CAMERA_WIDTH = 1600
CAMERA_HEIGHT = 1200

SCORE = .03
ImageCounter = 1
Interpreter = tf.lite.Interpreter(model_path="model (0).tflite")


class XFTP(FTP):
    
    def upload(self, filename, callback=None):
        with open(filename, "rb") as f:
            self.storbinary("STOR " + filename, f, callback=callback)

    def download(self, filename, callback=None):
        cmd = "RETR " + filename
        if callback is None:
            t_ini = time.time()
            with open(filename, "wb") as f:
                self.retrbinary(cmd, f.write)
                t_end = time.time()
                dt = t_end - t_ini
                print(round(dt,4))
        else:

            self.retrbinary(cmd, callback)

def load_labels(path='labels.txt'):
  """Loads the labels file. Supports files with or without index numbers."""
  with open(path, 'r', encoding='utf-8') as f:
    lines = f.readlines()
    labels = {}
    for row_number, content in enumerate(lines):
      pair = re.split(r'[:\s]+', content.strip(), maxsplit=1)
      if len(pair) == 2 and pair[0].strip().isdigit():
        labels[int(pair[0])] = pair[1].strip()
      else:
        labels[row_number] = pair[0].strip()
  return labels

def set_input_tensor(interpreter, image):
  """Sets the input tensor."""
  tensor_index = interpreter.get_input_details()[0]['index']
  input_tensor = interpreter.tensor(tensor_index)()[0]
  input_tensor[:, :] = np.expand_dims(image/115, axis = 0).astype(np.float32)

def get_output_tensor(interpreter, index):
  """Returns the output tensor at the given index."""
  output_details = interpreter.get_output_details()[index]
  tensor = np.squeeze(interpreter.get_tensor(output_details['index']))
  return tensor

def detect_objects(interpreter, image, threshold):
  """Returns a list of detection results, each a dictionary of object info."""
  set_input_tensor(interpreter, image)
  interpreter.invoke()
  # Get all output details
  boxes = get_output_tensor(interpreter, 1)
  classes = get_output_tensor(interpreter, 3)
  scores = get_output_tensor(interpreter, 0)
  count = int(get_output_tensor(interpreter, 2))

  results = []
  for i in range(count):
    if scores[i] >= threshold:
      result = {
          'bounding_box': boxes[i],
          'class_id': classes[i],
          'score': scores[i]
      }
      results.append(result)
  return results

def main():

  # Different Image Directories for testing
  testingImagePath = 'test'
  queuedImagePath = 'QueuedCameraImages'
  goodImagePath = 'GoodPartImages'
  badImagePath = 'BadPartImages'
  testDirectory = os.fsencode(testingImagePath)
  implementedDirectory = os.fsencode(queuedImagePath)
  goodImageDirectory = os.fsencode(goodImagePath)
  badImageDirectory = os.fsencode(badImagePath) 
  
  #let the system know that this will be the first part
  ImageCounter = 1
  defectedPartCount = 0 
  trueDefectCount = 0
  FalseRejectCount = 0
  FalseDefectCount = 0
  fileCount = len(os.listdir(testDirectory))
  
  # Starting the model initializations
  labels = load_labels()
  Interpreter.get_tensor_details()
  Interpreter.allocate_tensors()
  _, input_height, input_width, _ = Interpreter.get_input_details()[0]['shape']
  starttotaltime = time.time()
  
  #Iterating through each file in the test directory
  for file in os.listdir(testDirectory):
    
    # Making sure the file is a .bmp image
    filename = os.fsdecode(file)
    if filename.endswith(".jpg"):
      
      # Initializing values
      starttime = time.time()
      lasttime = starttime
      lapnum = 1
      defectFlag = 0
      
      # Printing the filename of the image   
      print(ImageCounter, '/ ', fileCount)
      print(filename)
      print('-------------------------------------------')
      
      # reading and formatting the image
      img = cv2.imread(testingImagePath + '/' + filename)
      image_np = cv2.resize(img, (320, 320))
      
      #sending the model to the detection function
      detections = detect_objects(Interpreter, image_np, SCORE)
      
      # reading the results 
      for result in detections:

            # Setting the boundings for the rectangular detect label
            ymin, xmin, ymax, xmax = result['bounding_box']
            xmin = int(max(1,xmin * CAMERA_WIDTH))
            xmax = int(min(CAMERA_WIDTH, xmax * CAMERA_WIDTH))
            ymin = int(max(1, ymin * CAMERA_HEIGHT))
            ymax = int(min(CAMERA_HEIGHT, ymax * CAMERA_HEIGHT))
            
            # calculating if the error is inside of the radius of the part
            errorBounds = sqrt((xmax - 761)**2 + (-ymax + 621)**2)
            if errorBounds < 600:
              
              # if this is the first defect the model has identified, add the label and 
              if defectFlag == 0:
                
                # Adding the label and the text to the image
                cv2.rectangle(img,(xmin,ymin),(xmax,ymax),(0,191,255),3)
                cv2.putText(img,labels[int(result['class_id'])], (xmin-160, min(ymax-80, CAMERA_HEIGHT)), cv2.FONT_HERSHEY_SIMPLEX, 3,(0,191,255),4,cv2.LINE_AA)

              # if the resulting defect is greater than the desired score flag this image
              if result['score'] > SCORE:
                defectFlag += 1
                
                # if this is the first defect, print that this is a defective part
                if defectFlag == 1:
                  print('Defect detected! Please remove part')
                  print('score: '+ str(result['score']))
                  
                  # if this image has a defect and is a defective part then increment the true defect counter
                  if 'Defect' in filename:
                    trueDefectCount += 1
                    
                  # if this image has a defect and is not a defective part then increment the False reject counter
                  else:
                    FalseRejectCount += 1
                    
      # if this image does not have a defect and is a true defect then increment the false defect counter
      if 'Defect' in filename and defectFlag == 0:
        FalseDefectCount += 1
        cv2.putText(img,'WRONG', (450, 600), cv2.FONT_HERSHEY_SIMPLEX, 6,(0,0,255),15,cv2.LINE_AA)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        plt.show()
        
      # calculate and print the amount of time taken by the RPI4 for each image
      laptime = round((time.time() - lasttime), 2)
      print("Processing Time: " + str(laptime) + " seconds\n")
      
      # print the image if the algorithm has correctly identified it as a defective part
      if defectFlag > 0:
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        plt.show()
        defectedPartCount += 1
        
      # update variables
      lasttime = time.time()
      lapnum += 1
      ImageCounter = ImageCounter + 1
      
    # if the system has not selected a .bmp image, keep iterating
    else:
      continue
    
  endtime = time.time()
  totaltime = endtime - starttotaltime 
   
  print("Total Number of Defected Parts Identified: " + str(defectedPartCount))
  print("Deep Learning model accuracy: ", round(((fileCount-(FalseRejectCount + FalseDefectCount))/fileCount)*100, 2), "%")
  print("Defective part accuracy: ", round((trueDefectCount/15)*100,2), "%")
  print("False Defect rate: ", round((FalseDefectCount/(trueDefectCount+FalseDefectCount))*100, 2), "%")
  print("Processed ", fileCount, " in ", round(totaltime, 2), " seconds")
  print("Average image processing time: ", totaltime/fileCount)
    
  
  ftp = XFTP(host = '192.168.2.220', user = 'admin', passwd = '')
  ftp.cwd('CAM1/CAM1')   #change dir
  ftp.sendcmd("F_IN0")
  file_list = ftp.nlst()
  os.chdir(os.getcwd() + '\\SaveFolder')

  print("Downloading file: " + file_list[int(len(file_list))-1])

  ftp.download(file_list[int(len(file_list))-1])
  
  img = cv2.imread(file_list[int(len(file_list))-1])
    
  starttime = time.time()
  lasttime = starttime
  lapnum = 1

  image_np = cv2.resize(img, (320, 320))
  detections = detect_objects(Interpreter, image_np, SCORE)
  defectFlag = 0

  for result in detections:

      for result in detections:
        ymin, xmin, ymax, xmax = result['bounding_box']
        xmin = int(max(1,xmin * CAMERA_WIDTH))
        xmax = int(min(CAMERA_WIDTH, xmax * CAMERA_WIDTH))
        ymin = int(max(1, ymin * CAMERA_HEIGHT))
        ymax = int(min(CAMERA_HEIGHT, ymax * CAMERA_HEIGHT))
        if defectFlag == 0:
          cv2.rectangle(img,(xmin,ymin),(xmax,ymax),(0,0,255),3)
          cv2.putText(img,labels[int(result['class_id'])], (xmin+20, min(ymax, CAMERA_HEIGHT)), cv2.FONT_HERSHEY_SIMPLEX, 1.5,(0,0,255),2,cv2.LINE_AA)

        if result['score'] > SCORE:
          defectFlag += 1
          if defectFlag == 1:
            print('Defect detected! Please remove part')
            print('score: '+ str(result['score']))
        

  laptime = round((time.time() - lasttime), 2)
  print("Processing Time: " + str(laptime) + " seconds\n")
  
  plt.imshow(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
  plt.show()  
  
  if defectFlag > 0:
    os.chdir(os.getcwd() + '\\BadPartImages')
    ftp.download(file_list[ int(len(file_list))-1])
    
  if defectFlag == 0:
    os.chdir(os.getcwd() + '\\GoodPartImages')
    ftp.download(file_list[int(len(file_list))-1])
    

# Running the main program
if __name__ == "__main__":
    main()