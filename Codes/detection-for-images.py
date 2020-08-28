import keras
print(keras.__version__)

from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color

import cv2
import os
import time
from datetime import datetime 
from tqdm import tqdm
import argparse

import json
import numpy as np

modelName = 'inference20.h5'
thresh = 0.22

modelPath = os.path.join('models', '', modelName)
model = models.load_model(modelPath, backbone_name='resnet50')

labelNames = {0: 'car', 1: 'human'}

if not os.path.exists('results/'):
    os.mkdir('results/')

dateNow = datetime.now().strftime("%Y%m%d_%H%M%S")
resultPath = os.path.join("results/", "{0}_{1}".format(dateNow, modelName))
os.mkdir(resultPath)

images_root_folder = "../test_images/"

model = models.load_model(modelPath, backbone_name='resnet50') # load retinanet model

sddImages = os.listdir(images_root_folder)
sddImages = sorted(sddImages, key=lambda name: int(name[:-4]))

##sddImages=sddImages[20:120]

resultsJson = []
index = 0

start = time.time()
for image_filename in tqdm(sddImages):
    #print("image_filename",image_filename)
    image = read_image_bgr(images_root_folder + image_filename)
    # copy to draw on
    draw = image.copy()
    draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

    # preprocess image for network
    image = preprocess_image(image)
    image, scale = resize_image(image)

    # process image
    boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
    
    boxes /= scale

    frameJson = {}

    frameJson["frame_id"] = image_filename.replace(".jpg","").replace("../test_images/","")
    frameJson["objeler"] = []
        
    # visualize detections
    for box, score, label in zip(boxes[0], scores[0], labels[0]):
        #print(box, score, label)
        # scores are sorted so we can break
        if score < thresh:
            break

        #color = label_color(label)
        colors = [[255,172,31], [161, 222, 251]]
        color = colors[label]

        b = box.astype(int)
        
        if labelNames[label] == "car":
            draw_box(draw, b, color=color)
            #print("label",labelNames[label])
            #print("score",score)
            #caption = "{} {:.3f}".format(labelNames[label], score)
            caption = "{} {:.2f}%".format(labelNames[label], score*100)
            draw_caption(draw, b, caption)

        #JSON ADD
        if labelNames[label] == "car":
            frameJson["objeler"].append({"tur": labelNames[label], "x1":int(box[0]), "y1":int(box[1]), "x2":int(box[2]), "y2":int(box[3])}) 
    
    
    resultsJson.append(frameJson)

    file_, ext = os.path.splitext(image_filename)
    imageName = file_.split('/')[-1] + ext
    outputPath = os.path.join(resultPath, imageName)
    
    draw_conv = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)
    cv2.imwrite(outputPath, draw_conv)

results = {}
results["cevaplar"] = resultsJson

#SAVE RESULT JSON
with open('{0}/results_{1}.json'.format(resultPath,dateNow), 'w') as fp:
    json.dump(resultsJson, fp)

#SAVE RESULT JSON
with open('{0}/results_{1}_toplu.json'.format(resultPath,dateNow), 'w') as fp:
    json.dump(results, fp)
