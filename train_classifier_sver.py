import os
import PIL
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import pathlib

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

# init path
crtDir=os.getcwd()
path=os.path.join(crtDir,"controls")
path=pathlib.Path(path)
modelPath=os.path.join(crtDir,"model");
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

# Validation set
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
print("Control labels save to path: %s" %labelsPath)

# Shuffle
AUTOTUNE=tf.data.experimental.AUTOTUNE
trainDataset=trainDataset.cache().shuffle(10).prefetch(buffer_size=AUTOTUNE)
valiDataset=valiDataset.cache().prefetch(buffer_size=AUTOTUNE)

# Create layers
numClasses=13
model=Sequential([
    layers.experimental.preprocessing.RandomRotation(0.1),
    layers.experimental.preprocessing.RandomZoom(0.06),
    layers.experimental.preprocessing.Rescaling(1./255, input_shape=(imgHeight, imgWidth, 3)),
    layers.Conv2D(4, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(8, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(96, activation='relu'),
    layers.Dense(numClasses),
])

# Compile the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Train
print("Training...")
epochs=15;
history=model.fit(
    trainDataset,
    validation_data=valiDataset,
    epochs=epochs)
print("Training completed.")

# Save model
tf.saved_model.save(model,modelPath)
print("Model save to:%s" %modelPath)





















