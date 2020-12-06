
import os
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
assert tf.__version__.startswith('2')
from tflite_model_maker import configs
from tflite_model_maker import ExportFormat
from tflite_model_maker import image_classifier
from tflite_model_maker import ImageClassifierDataLoader
from tflite_model_maker import model_spec


# init path
crtDir=os.getcwd()
path=os.path.join(crtDir,"controls")
path=pathlib.Path(path)
modelPath=os.path.join(crtDir,"ClassifierModel");
labelsPath=os.path.join(crtDir,"control_labels_rmp.txt")

# Training set
trainDataset=tf.keras.preprocessing.image_dataset_from_directory(
    path,
    validation_split=0.3,
    subset="training",
    seed=1,
    image_size=(imgHeight,imgWidth),
    batch_size=batchSize)

# Validating set
valiDataset=tf.keras.preprocessing.image_dataset_from_directory(
    path,
    validation_split=0.3,
    subset="validation",
    seed=12,
    image_size=(imgHeight,imgWidth),
    batch_size=batchSize)

# Save labels
classNames=trainDataset.class_names
print("Classes names:%s" %classNames)
np.savetxt(labelsPath,classNames,'%s',delimiter='\n')
print("Control class names:")
for s in classNames:
    print(s,end=', ')
print()
print("Control labels save to path: %s" %labelsPath)

# Load data
data=ImageClassifierDataLoader.from_folder(path)
trainData, testData=data.split(0.62)
trainData=data;

# Create model
model=image_classifier.create(trainDataset)
model.summary()

# Evaluate the model
loss, acc=model.evaluate(testData)
print("Model accuracy: %.4f" %acc)
print("Model loss: %.4f" %loss)

# Export model
model.export(modelPath)
