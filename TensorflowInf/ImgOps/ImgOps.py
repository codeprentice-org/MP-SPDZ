import numpy as np
from PIL import Image

def getImg(imgfile):
    img = np.asarray(Image.open(imgfile))
    return img

def imgResize(img, width, height):
    return np.asarray(Image.fromarray(img).resize((width, height)))

def reverseColor(img):
    return img[:,:,::-1]

def squareCrop(img):
    imgWidth = img.shape[0]
    imgHeight = img.shape[1]
    cropLength = imgWidth if imgWidth < imgHeight else imgHeight
    cropCoor = (int((imgWidth / 2) - (cropLength / 2)),
                int((imgHeight / 2) - (cropLength / 2)),
                int((imgWidth / 2) + (cropLength / 2)),
                int((imgHeight / 2) + (cropLength / 2)))
    return img[cropCoor[0]:cropCoor[2], cropCoor[1]:cropCoor[3],:]

def offsetPixels(img, offset):
    return img - offset
