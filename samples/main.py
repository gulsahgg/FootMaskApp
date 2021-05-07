import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt

# Root directory of the project
ROOT_DIR = os.path.abspath("..\\")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
# Import COCO config
sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
import coco



# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "images")
class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()
# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)
# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']
# Load a random image from the images folder
file_names = next(os.walk(IMAGE_DIR))[2]
image = skimage.io.imread(os.path.join(IMAGE_DIR, 'withcouch.jpg'))

# Run detection
results = model.detect([image], verbose=1)

# Visualize results
r = results[0]


#Get mask
classid = r['class_ids']
mask = r['masks']
mask = mask.astype(int)
mask.shape
index=-1
import cv2 as cv
temp = cv.imread(os.path.join(IMAGE_DIR, 'withcouch.jpg'))
for k in classid:
    index+=1
    if(k==1):
        for j in range(temp.shape[2]):
            temp[:,:,j] = temp[2,2,2] * mask[:,:,index]
       
#Guo Hall Thinning
import thinning
from skimage import filters
from skimage.color import rgb2gray


Img=rgb2gray(temp)
Otsu = skimage.filters.threshold_otsu(Img)   
BW = Img > Otsu 


BW=np.uint8(temp)
BW=cv.cvtColor(BW, cv.COLOR_BGR2GRAY)
Skeleton = thinning.guo_hall_thinning(BW.copy())

#Detect points
import imutils
cnts = cv.findContours(Skeleton.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
c = max(cnts, key=cv.contourArea)
extLeft = tuple(c[c[:, :, 0].argmin()][0])
extRight = tuple(c[c[:, :, 0].argmax()][0])
extTop = tuple(c[c[:, :, 1].argmin()][0])
extBot = tuple(c[c[:, :, 1].argmax()][0])
resultImage= image.copy()
#cv.drawContours(image, [c], -1, (0, 255, 255), 2)
cv.circle(resultImage, extLeft, 8, (0, 0, 255), -1) #koyu mavi
cv.circle(resultImage, extRight, 8, (0, 255, 0), -1) #yeşil
cv.circle(resultImage, extTop, 8, (255, 0, 0), -1) #kırmızı
cv.circle(resultImage, extBot, 8, (255, 255, 0), -1) #sarı

#Draw lines

height, width, channel = resultImage.shape 
black_image = np.zeros((height, width))
cv.line(black_image,extTop,extRight,(255, 0, 0), thickness=6)
cv.line(black_image,extRight,extBot,(255, 0, 0), thickness=5)
plt.figure(figsize=(8,8))
plt.imshow(black_image)
from skimage.transform import (hough_line, hough_line_peaks)


# Compute arithmetic mean
#myResult = np.mean(black_image, axis=2)

# Perform Hough Transformation to detect lines
hspace, angles, distances = hough_line(black_image)

# Find angle
angle=[]
for _, a , distances in zip(*hough_line_peaks(hspace, angles, distances)):
    angle.append(a)

# Obtain angle for each line
angles = [a*180/np.pi for a in angle]

# Compute difference between the two lines
angle_difference = np.max(angles) - np.min(angles)
print(angle_difference)