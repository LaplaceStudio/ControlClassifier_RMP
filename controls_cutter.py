import os
from PIL import Image
import imageio

# Get all images
cd = os.getcwd()
imgDir = os.path.join(cd, "RMP")
imgNames = os.listdir(imgDir)
ctrlDir = os.path.join(cd, "controls")

# Normal size of image
normWidth = 960
normHeight = 555

# Button names and position list
ctrlNames = ["VHF1", "VHF2", "VHF3", "HF1", "HF2","AM",\
    "NAV","VOR", "LS","BLANK","ADF","BFO","EXCHANGE"]
width = 100
height = 68
ctrlPos = []
ctrlPos.append((88, 224, 88+width, 224+height))
ctrlPos.append((202, 224, 202+width, 224+height))
ctrlPos.append((316, 224, 316+width, 224+height))
ctrlPos.append((88, 327, 88+width, 327+height))
ctrlPos.append((316, 327, 316+width, 327+height))
ctrlPos.append((430, 327, 430+width, 327+height))
ctrlPos.append((88, 465, 88+width, 465+height))
ctrlPos.append((202, 465, 202+width, 465+height))
ctrlPos.append((316, 465, 316+width, 465+height))
ctrlPos.append((430, 465, 430+width, 465+height))
ctrlPos.append((544, 465, 544+width, 465+height))
ctrlPos.append((658, 465, 658+width, 465+height))
ctrlPos.append((430, 82, 430+width, 82+height))

# process
i=0
for name in imgNames:
    path = os.path.join(imgDir, name)
    print("Cut image:{}".format(path))
    img = Image.open(path)
    img = img.resize((normWidth, normHeight),Image.ANTIALIAS)
    i=i+1
    c = 0
    for pos in ctrlPos:
        clsDir = os.path.join(ctrlDir, ctrlNames[c])
        if not os.path.exists(clsDir):
            os.mkdir(clsDir)
        nimg = img.crop(pos)
        npath = os.path.join(clsDir, "{}.png".format(i))
        nimg.save(npath)
        c=c+1
        print("Save a button image to:{}".format(npath))
print("Process completed.")


