import numpy as np
from PIL import Image

def getImg(imgfile):
    imgOut = Image.open(imgfile).convert("RGB")
    imgOut = np.asarray(imgOut)
    return imgOut

def imgResize(img, width, height):
    imgOut = Image.fromarray(img)
    imgOut = imgOut.resize((width, height), Image.BILINEAR)
    imgOut = np.asarray(imgOut)
    if len(imgOut.shape) == 2:
        imgOut = np.dstack((imgOut, imgOut, imgOut))
    return imgOut

def imgSave(img, dest):
    imgSave = np.clip(img, 0, 255).astype(np.uint8)
    imgSave = Image.fromarray(imgSave)
    imgSave.save(path, quality = 95)

def reverseColor(img):
    imgOut = img[:,:,::-1]
    return imgOut

def squareCrop(img):
    imgWidth = img.shape[0]
    imgHeight = img.shape[1]
    cropLength = imgWidth if imgWidth < imgHeight else imgHeight
    cropCoor = (int((imgWidth / 2) - (cropLength / 2)),
                int((imgHeight / 2) - (cropLength / 2)),
                int((imgWidth / 2) + (cropLength / 2)),
                int((imgHeight / 2) + (cropLength / 2)))
    imgOut = img[cropCoor[0]:cropCoor[2], cropCoor[1]:cropCoor[3],:]
    return imgOut

def meanOffset(img):
    imgOut = img - np.array([104.006, 116.669, 122.679], dtype = np.float32)
    return imgOut

def meanUnoffset(img):
    imgOut = img + np.array([104.006, 116.669, 122.679], dtype = np.float32)
    return imgOut
