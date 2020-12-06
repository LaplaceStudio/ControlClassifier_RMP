
import tensorflow as tf
import os
import numpy as np

modelPath=os.path.join(os.getcwd(),"model")
labelsPath=os.path.join(os.getcwd(),"control_labels_rmp.txt")
model=tf.keras.models.load_model(modelPath)
labels=np.loadtxt(labelsPath,dtype=str)
print("Model loaded.")

stop=1;
while int(stop) > 0:
    cls =int(input("Input class num(1 to 13):"))
    idx =int(input("Input a control index(1 to 8):"))

    clsPath="controls_test\\{}".format(cls)
    idxPath="rmp_{}_{}.png".format(cls,idx)
    imgPath=os.path.join(os.getcwd(),clsPath,idxPath)

    img = tf.keras.preprocessing.image.load_img(imgPath)
    imgArr = tf.keras.preprocessing.image.img_to_array(img)
    imgArr = tf.expand_dims(imgArr, 0) 

    prediction = model.predict(imgArr)
    score = tf.nn.softmax(prediction)
    s=np.array(score)[0]

    print("Confidence:")
    for index in range(len(s)):
      print("%s : %.4f" %(labels[index],100*s[index]))

    stop=input("Input 0 to exit, other to continue: ")
     

