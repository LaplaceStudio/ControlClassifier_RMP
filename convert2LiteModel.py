import os
import tensorflow as tf

# Path of saved model
modelPath=os.path.join(os.getcwd(),"model");

# Convert model to tflite model
converter=tf.lite.TFLiteConverter.from_saved_model(modelPath)
liteModel=converter.convert()
print("Model converted to lite model.")

# Save converted model
with open('rmp_control_classifier.tflite','wb') as f:(
    f.write(liteModel))
print("Lite model save to model.tflite.")
