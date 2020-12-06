
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf
import pathlib

from datetime import datetime
from packaging import version

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
print("Control class names:")
for s in classNames:
    print(s,end=', ')
print()
print("Control labels save to path: %s" %labelsPath)

# Shuffle
AUTOTUNE=tf.data.experimental.AUTOTUNE
trainDataset=trainDataset.cache().shuffle(10).prefetch(buffer_size=AUTOTUNE)
valiDataset=valiDataset.cache().prefetch(buffer_size=AUTOTUNE)

# Create layers
numClasses=13
model=Sequential([
    layers.experimental.preprocessing.RandomRotation(0.08),
    layers.experimental.preprocessing.RandomZoom(0.06),
    layers.experimental.preprocessing.Rescaling(1./255, input_shape=(imgHeight, imgWidth, 3)),  # Rescale pixel value to [0,1]
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
model.build(input_shape=(1,68,100,3))
model.summary()

# Define the Keras TensorBoard callback.
logdir="logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir,write_graph=True)

# Train
print("Training...")
epochs=10;
history=model.fit(
    trainDataset,
    validation_data=valiDataset,
    epochs=epochs,
    callbacks=[tensorboard_callback])
print("Training completed.")

# Save model
tf.saved_model.save(model,modelPath)
print("Model save to:%s" %modelPath)



# Visualize training results
acc=history.history['accuracy']
valAcc=history.history['val_accuracy']
loss=history.history['loss']
valLoss=history.history['val_loss']
eRange=range(epochs)

plt.figure(figsize=(20,10))

plt.subplot(1, 2, 1)
plt.plot(eRange,acc,label="Training Accuracy")
plt.plot(eRange,valAcc,label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Accuracy')

plt.subplot(1, 2, 2)
plt.plot(eRange, loss, label='Training Loss')
plt.plot(eRange, valLoss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Loss')

plt.show()























