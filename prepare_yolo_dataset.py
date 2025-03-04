import os
import random
import matplotlib.pyplot as plt
import cv2
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
import random
import os
import cv2
import shutil
import tqdm
import glob
from config import DATA_FOLDER,IMAGE_PATH,ANNOTATION_PATH

train = []
with open(DATA_FOLDER+'ImageSets/Main/trainval.txt', 'r') as f:
    for line in f.readlines():
        if line[-1]=='\n':
            line = line[:-1]
        train.append(line)

test = []
with open(DATA_FOLDER+'ImageSets/Main/test.txt', 'r') as f:
    for line in f.readlines():
        if line[-1]=='\n':
            line = line[:-1]
        test.append(line)

os.mkdir('train')
os.mkdir('train/images')
os.mkdir('train/labels')

os.mkdir('test')
os.mkdir('test/images')
os.mkdir('test/labels')

train_path = 'C:/Work/GenAI/Fashion_Recom/train/'
test_path = 'C:/Work/GenAI/Fashion_Recom/test/'

print('Copying Train Data..!!')
for i in tqdm.tqdm(train):
    a = shutil.copyfile(IMAGE_PATH+i+'.jpg', train_path+'images/'+i+'.jpg')
    a = shutil.copyfile(ANNOTATION_PATH+i+'.txt', train_path+'labels/'+i+'.txt')

print('Copying Test Data..!!')
for i in tqdm.tqdm(test):
    a = shutil.copyfile(IMAGE_PATH+i+'.jpg', test_path+'images/'+i+'.jpg')
    a = shutil.copyfile(ANNOTATION_PATH+i+'.txt', test_path+'labels/'+i+'.txt')

text = """
train: C:/Work/GenAI/Fashion_Recom/train
val: C:/Work/GenAI/Fashion_Recom/test

# number of classes
nc: 10

# class names
names: ['sunglass','hat','jacket','shirt','pants','shorts','skirt','dress','bag','shoe']
"""
with open("data.yaml", 'w') as file:
    file.write(text)

