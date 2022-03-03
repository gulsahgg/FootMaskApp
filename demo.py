#!/usr/bin/env python
# coding: utf-8

# # Mask R-CNN Demo
# 
# A quick intro to using the pre-trained model to detect and segment objects.

# In[1]:


import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt

# Root directory of the project
ROOT_DIR = os.path.abspath("../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
# Import COCO config
sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
import coco

##get_ipython().run_line_magic('matplotlib', 'inline')

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "images")


# ## Configurations
# 
# We'll be using a model trained on the MS-COCO dataset. The configurations of this model are in the ```CocoConfig``` class in ```coco.py```.
# 
# For inferencing, modify the configurations a bit to fit the task. To do so, sub-class the ```CocoConfig``` class and override the attributes you need to change.

# In[2]:


class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()


# ## Create Model and Load Trained Weights

# In[3]:


# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)


# ## Class Names
# 
# The model classifies objects and returns class IDs, which are integer value that identify each class. Some datasets assign integer values to their classes and some don't. For example, in the MS-COCO dataset, the 'person' class is 1 and 'teddy bear' is 88. The IDs are often sequential, but not always. The COCO dataset, for example, has classes associated with class IDs 70 and 72, but not 71.
# 
# To improve consistency, and to support training on data from multiple sources at the same time, our ```Dataset``` class assigns it's own sequential integer IDs to each class. For example, if you load the COCO dataset using our ```Dataset``` class, the 'person' class would get class ID = 1 (just like COCO) and the 'teddy bear' class is 78 (different from COCO). Keep that in mind when mapping class IDs to class names.
# 
# To get the list of class names, you'd load the dataset and then use the ```class_names``` property like this.
# ```
# # Load COCO dataset
# dataset = coco.CocoDataset()
# dataset.load_coco(COCO_DIR, "train")
# dataset.prepare()
# 
# # Print class names
# print(dataset.class_names)
# ```
# 
# We don't want to require you to download the COCO dataset just to run this demo, so we're including the list of class names below. The index of the class name in the list represent its ID (first class is 0, second is 1, third is 2, ...etc.)

# In[4]:


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


# Create a child process
# using os.fork() method 
#pid = os.fork()
  
  
# a Non-zero process id (pid)
# indicates the parent process 
#if pid :
      
    # Wait for the completion of
    # child process using
    # os.wait() method    
    #status = os.wait()
    #print("\nIn parent process-")
    #print("Terminated child's process id:", status[0])
    #print("Signal number that killed the child process:", status[1])
#else :
    #print("In Child process-")
    #print("Process ID:", os.getpid()) 
theFile=str(sys.argv[1])
print(str(theFile))
#myFile=str(theFile.rsplit("/",1)[1])
ff="ffmpeg -i /home/ubuntu/FootMaskApp/images/"+theFile+" -r 2 /home/ubuntu/FootMaskApp/images/frame/ffmpeg_%0d.jpeg"
os.system(ff)

# Visualize results

import cv2 as cv
import imutils
import thinning
from skimage import filters
from skimage.color import rgb2gray
import math
import itertools
from mrcnn import convFunc as cf
model.keras_model._make_predict_function()
def angleMeasure(filename):
    file_names = next(os.walk(IMAGE_DIR))[2]
    img = cv.imread(os.path.join(IMAGE_DIR+'/frame', filename))
    img = imutils.resize(img, width=1024)
    #CLAHE
    lab_img= cv.cvtColor(img, cv.COLOR_BGR2LAB)
    l, a, b = cv.split(lab_img)
    clahe = cv.createCLAHE(clipLimit=3.0, tileGridSize=(20, 20))
    clahe_img = clahe.apply(l)
    updated_lab_img2 = cv.merge((clahe_img,a,b))
    CLAHE_img = cv.cvtColor(updated_lab_img2, cv.COLOR_LAB2BGR)
    image=cv.cvtColor(CLAHE_img, cv.COLOR_BGR2RGB)
    results = model.detect([image], verbose=1)
    if not results:
       results = model.detect([img], verbose=1)
    r = results[0]
    classid = r['class_ids']
    mask = r['masks']
    box = r['rois']
    mask = mask.astype(int)
    mask.shape
    index=-1
    temp = img.copy()
    for k in classid:
        index+=1
        if(k==1):
            for j in range(temp.shape[2]):
                temp[:,:,j] = temp[2,2,2] * mask[:,:,index]
                break;
        else:
            pass
     #thinning
    temp=cv.cvtColor(temp, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(temp,(5,5),0)
    ret3,th3 = cv.threshold(blur,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
    ress=cv.bitwise_and(CLAHE_img, CLAHE_img, mask = th3)
    img_HSV = cv.cvtColor(ress, cv.COLOR_BGR2HSV)
    #skin color range for hsv color space 
    HSV_mask = cv.inRange(img_HSV, (1, 30, 30), (33, 235, 235)) 
    HSV_mask = cv.morphologyEx(HSV_mask, cv.MORPH_OPEN, np.ones((5,5), np.uint8))
    #converting from bgr to YCbCr color space
    img_YCrCb = cv.cvtColor(ress, cv.COLOR_BGR2YCrCb)
    #skin color range for hsv color space 
    YCrCb_mask = cv.inRange(img_YCrCb, (5,133,77), (235,200,150)) 
    YCrCb_mask = cv.morphologyEx(YCrCb_mask, cv.MORPH_OPEN, np.ones((5,5), np.uint8))
    #merge skin detection (YCbCr and hsv)
    global_mask=cv.bitwise_and(YCrCb_mask,HSV_mask)
    global_mask=cv.medianBlur(global_mask,3)
    global_mask = cv.morphologyEx(global_mask, cv.MORPH_OPEN, np.ones((6,6), np.uint8))
    dilation = cv.dilate(global_mask,np.ones((6,6), np.uint8),iterations = 10)
    closing = cv.morphologyEx(dilation, cv.MORPH_CLOSE, np.ones((3,3), np.uint8))
    #opening = cv.morphologyEx(img, cv.MORPH_OPEN, kernel)
    HSV_result = cv.bitwise_not(HSV_mask)
    YCrCb_result = cv.bitwise_not(YCrCb_mask)
    global_result=cv.bitwise_and(ress, ress, mask = closing)
    gg=np.uint8(global_result.copy())
    gg=cv.cvtColor(gg, cv.COLOR_BGR2GRAY)
    Skeleton = thinning.guo_hall_thinning(gg)
    input_image = Skeleton.copy().astype(np.uint8) # must be blaack and white thin network image
    eol_img = cf.find_endoflines(input_image, 0)
    # 1- Find curve Intersections
    lint_img = cf.find_line_intersection(input_image, 0)
    white_pix = np.where(lint_img != 0)
    ll,rr,tt,bb=cf.findPoints(input_image)
    extLeft,extRight,extTop,extBot=cf.findPoints(th3.copy())
    if(white_pix[0].size==0):
        extMid=extBot
    else:
        tempMid=cf.NearPoint(white_pix[1], white_pix[0], extBot)
        tempLeft=cf.NearPoint(white_pix[0], white_pix[1], ll)
        tempRight=cf.NearPoint(white_pix[0], white_pix[1], extRight)
        if(tempMid!=tempLeft and tempMid!=tempRight):
            extMid=tempMid
    x1=extLeft[0]
    y1=extLeft[1]
    x2=extMid[0]
    y2=extMid[1]
    x3=extTop[0]
    y3=extTop[1]
    if(cf.findDist(extBot,extRight)<cf.findDist(extTop,extRight) and cf.findDist(extLeft,extRight)<cf.findDist(extLeft,extBot)):
        x3=extTop[0]
        y3=extTop[1]
    if(cf.findDist(extLeft,extTop)<150):
        x3=extRight[0]
        y3=extRight[1]
    if(cf.findDist(extLeft,extTop)<cf.findDist(extTop,extRight)):
        x3=extRight[0]
        y3=extRight[1]
    angle = math.degrees(math.atan2(y3 - y2, x3 - x2) -
	                             math.atan2(y1 - y2, x1 - x2))
    if (angle<-180) :
        angle=360+angle
    elif(angle>180 and angle<360):
        angle=angle-180
    return angle
#thread
num_frames=int(os.popen("ls /home/ubuntu/FootMaskApp/images/frame | wc -l").read())
print(num_frames)
import concurrent.futures
import time
resultList=[]
start = time.time()
with concurrent.futures.ThreadPoolExecutor(max_workers = 10) as executor:
    for i in range (num_frames):
        try:
            future = executor.submit(angleMeasure, 'ffmpeg_{}.jpeg'.format(i+1))
            res=round(future.result(),2)
            if(res>0 and res<180.0):
                resultList.append(res)
        except Exception as e:
            print(e)
            pass

#parse the file name
mline=theFile.split(".")[0]
print("mline",mline)
hastaid,side,date,mtime = mline.split("_")
sol="0"
sag="Sol"
if (side == "888"):
    sag="Sağ"
    sol="1"
print(hastaid+" "+sol+" ")
dorsi=""
plantar=0.0
rom=""
print("resulList: ",resultList)
resultList = [i for i in resultList if i != 0]
#write mysql by calling mysqlex.py
if resultList:
       print("resultList: ",resultList)
       dorsi = 90-min(resultList)
       plantar = max(resultList)-90
       rom = max(resultList)-min(resultList)
else:
       print("cannot measure ROM")
       dorsi='1111'
       plantar=1111
       rom='1111'
dorsi=str(dorsi)
rom=str(rom)
os.system("sudo python3 /home/ubuntu/FootMaskApp/mysqlex.py "+rom+" "+dorsi+" "+theFile+" "+sol+" "+hastaid)
end = time.time()
os.system("sudo python3 /home/ubuntu/FootMaskApp/samples/deleteName.py")
os.system("rm /home/ubuntu/FootMaskApp/images/"+theFile)
os.system("rm /home/ubuntu/FootMaskApp/images/frame/f*")
print("time for calculation: ", end-start)

import smtplib, ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime as date

sender_email = "digitalsteps.info@gmail.com"
receiver_email = "gulsahgokhan@gmail.com"
password = "Digital12Steps."
bugun=str(date.now().strftime("%d/%m/%Y %H:%M:%S"))
message = MIMEMultipart("alternative")
message["Subject"] = "Dijital Adımlar sonuçlarınız"
message["From"] = sender_email
message["To"] = receiver_email
plant=str(plantar)
# Create the plain-text and HTML version of your message
text =(""
"Merhaba,"
+bugun+" tarihinde yapılan olcume gore sonuclariniz asagidaki gibidir."
+sag+ " Ayaginizin toplam acikligi " +rom+ " derece olarak olculmustur."
"Dorsifleksiyon acikligi " + dorsi + ", plantar fleksiyon acikligi ise " +plant+ " olarak olculmustur."
"Daha fazlasi icin web sitemizi ziyaret edin: dijitaladimlar.org"
"Saglikli gunler dileriz,"
"Dijital Adimlar Ekibi")
html =(""
"<html>"
"  <body>"
"    <p>Merhaba,<br>"
"       "+bugun+" tarihinde yapılan olcume gore sonuclariniz asagidaki gibidir: <br>"
"       "+sag+ " Ayaginizin toplam acikligi " +rom+ " derece olarak olculmustur.<br>"
"       Dorsifleksiyon acikligi " + dorsi + ", plantar fleksiyon acikligi ise " +plant+ " olarak olculmustur.<br>"
"       Daha fazlasi icin web sitemizi ziyaret edin: <br>"
"       <a href=\"http://dijitaladimlar.org\">dijitaladimlar.org</a>"
"       <br>Saglikli gunler dileriz,<br>"
"       Dijital Adimlar Ekibi<br>"
"    </p>"
"  </body>"
"</html>"
)
# Turn these into plain/html MIMEText objects
part1 = MIMEText(text, "plain")
part2 = MIMEText(html, "html")

# Add HTML/plain-text parts to MIMEMultipart message
# The email client will try to render the last part first
message.attach(part1)
message.attach(part2)
# Create secure connection with server and send email
context = ssl.create_default_context()
with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
    server.login(sender_email, password)
    server.sendmail(
        sender_email, receiver_email, message.as_string()
    )

