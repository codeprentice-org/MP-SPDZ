import numpy as np
from PIL import Image

# ----------------------------------------------------------------------
# Get numpy image array
# imgfile - path to image file
# RETURNS - numpy image array
def getImg(imgfile):
    imgOut = Image.open(imgfile).convert("RGB")
    imgOut = np.asarray(imgOut)
    return imgOut

# ----------------------------------------------------------------------
# Resize image
# img - numpy image array
# width - resize width
# height - resize height
# interpolation - interpolation algorithm for resizing
# RETURNS - resized numpy image array
def imgResize(img, width, height, interpolation = Image.BILINEAR):
    imgOut = Image.fromarray(img)
    imgOut = imgOut.resize((width, height), resample = interpolation)
    imgOut = np.asarray(imgOut)
    if len(imgOut.shape) == 2:
        imgOut = np.dstack((imgOut, imgOut, imgOut))
    return imgOut

# ----------------------------------------------------------------------
# Save image
# img - numpy image array
# dest - image destination path
# RETURNS - N/A
def imgSave(img, dest):
    imgSave = np.clip(img, 0, 255).astype(np.uint8)
    imgSave = Image.fromarray(imgSave)
    imgSave.save(path, quality = 95)

# ----------------------------------------------------------------------
# Change RGB image to BGR and vice versa
# img - numpy image array
# RETURNS - color reversed numpy image array
def reverseColor(img):
    imgOut = img[:,:,::-1]
    return imgOut

# ----------------------------------------------------------------------
# Crop image to square
# img - numpy image array
# RETURNS - cropped numpy image array
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

# ----------------------------------------------------------------------
# Offset image by Image-Net standard mean pixel
# img - numpy image array
# RETURNS - offset numpy image array
def meanOffset(img):
    imgOut = img - np.array([104.006, 116.669, 122.679], dtype = np.float32)
    return imgOut

# ----------------------------------------------------------------------
# Restore image after offsetting by Image-Net standard mean pixel
# img - numpy image array
# RETURNS - original numpy image array
def meanUnoffset(img):
    imgOut = img + np.array([104.006, 116.669, 122.679], dtype = np.float32)
    return imgOut
