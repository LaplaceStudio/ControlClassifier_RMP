
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
import pathlib

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

# init path
crtDir=os.getcwd()
path=os.path.join(crtDir,"controls")
path=pathlib.Path(path)
labelsPath=os.path.join(crtDir,"control_labels_rmp.txt")

# Image size and batch size
batchSize=8
imgHeight=68
imgWidth=100

# Training set
trainDataset=tf.keras.preprocessing.image_dataset_from_directory(
    path,
    validation_split=0.3,
    subset="training",
    seed=1,
    image_size=(imgHeight,imgWidth),
    batch_size=batchSize)

classNames=trainDataset.class_names
np.savetxt(labelsPath,classNames,'%s',delimiter='\n')
print("Control class names:")
for s in classNames:
    print(s,end=', ')
print()
print("Control labels save to path: %s" %labelsPath)


