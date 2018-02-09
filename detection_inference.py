import warnings
warnings.filterwarnings("ignore")

from keras.optimizers import Adam, SGD, Nadam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard, LearningRateScheduler
from keras.callbacks import Callback
from keras import backend as K 
from keras.models import load_model
from math import ceil 
import numpy as np 
from termcolor import colored
#from matplotlib import pyplot as plt 
from tqdm import tqdm
#from lg_model_dwc import build_model_300x300
#from lg_model_224x224 import lg_model

from keras.utils.generic_utils import CustomObjectScope

# from mn_model import mn_model
from mn_model_postprocessing import DetectionModel
# from ssd_box_encode_decode_utils import SSDBoxEncoder, decode_y, decode_y2

# training parameters
# from system_conf import gpu_conf, supress_warnings
from keras import backend as K
import scipy.misc as sm
# import json 
from keras.preprocessing import image
import matplotlib as mpl 
mpl.use('Agg')
from matplotlib import pyplot as plt
from scipy.misc import imresize

# from keras_layer_L2Normalization import L2Normalization
# from keras_layer_AnchorBoxes_tf import Anchorboxes

# model to be evaluated is kept here 
# model_path = "./models/openimages/partial_train/"
# model_name = "ssd_lg_epoch_lr_2.0_58_loss0.2154.h5"


model_path = './'
model_name = 'model.h5'

import cv2
def img_preprocess(img_path, size):

  img = cv2.imread(img_path, 1) # BGR
  img = img[:,:,::-1] # BGR->RGB

  img_resize = imresize(img, size)
  img_resize = np.expand_dims(img_resize, 0)

  return img_resize 




#!/bin/bash

OMP_NUM_THREADS=4

# supress_warnings()
# gpu_conf(gpu_id=1, 
#     load = 0.20)

# import threading
# class threadsafe_iter:
#     def __init__(self, it):
#         self.it = it
#         self.lock = threading.Lock()

#     def __iter__(self):
#         return self

#     def next(self):
#         with self.lock:
#             return self.it.next()

# def threadsafe_generator(f):
#     def g(*a, **kw):
#         return threadsafe_iter(f(*a, **kw))
#     return g


img_height =512
img_width = 512


K.clear_session()

print colored("define model", "yellow")

model = DetectionModel()

print model.summary()

print colored("model definition... done.", "green")

print colored("loading detection weights...", "yellow")

model.load_weights(model_name,  by_name= True)

# model = load_model("face_model_with_defn.h5")

print colored("Model weight loaded successfully.", "green")

print colored("now predicting...", "yellow")



img_resize = img_preprocess("./images/person.jpg", (512,512))  

y_pred = model.predict(img_resize)



np.set_printoptions(suppress=True)


def print_bb(filename, results):

  img = image.load_img(filename, target_size=(512, 512))
  img = image.img_to_array(img)


  currentAxis = plt.gca()

  # Get detections with confidence higher than 0.6.
  colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()
  color_code = min(len(results), 16)
  print colored("total number of bbs: %d" % len(results), "yellow")
  for result in results:
    # Parse the outputs.

    det_label = result[0]
    det_conf = result[1]
    det_xmin = result[2]
    det_xmax = result[3]
    det_ymin = result[4]
    det_ymax = result[5]
  
    xmin = int(det_xmin)
    ymin = int(det_ymin)
    xmax = int(det_xmax)
    ymax = int(det_ymax)

    score = det_conf
    
    plt.imshow(img / 255.)
    
    label = int(int(det_label))

    #print label
    label_name = class_names[label]
    # label_name = class_names[label]
    # print label_name 
    # print label

    display_txt = '{:0.2f}, {}'.format(score, label_name)
    
    coords = (xmin, ymin), (xmax-xmin), (ymax-ymin)
    color_code = color_code-1 
    color = colors[color_code]
    currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))
    # currentAxis.text(xmin, ymin, display_txt, bbox={'facecolor':color, 'alpha':0.5})

  plt.savefig("result.jpg")

  plt.clf()



import cv2
def img_preprocess(img_path, size):

  img = cv2.imread(img_path, 1) # BGR
  img = img[:,:,::-1] # BGR->RGB

  img_resize = imresize(img, size)
  img_resize = np.expand_dims(img_resize, 0)

  return img_resize 






