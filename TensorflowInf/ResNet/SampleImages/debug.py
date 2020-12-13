import _pickle as pickle
from PIL import Image
import numpy as np
import scipy.misc
import scipy.io
import matplotlib.pyplot as plt
from statistics import median

def imread_resize(path):
    img_orig = Image.open(path).convert("RGB")
    img_orig = np.asarray(img_orig)

    img = scipy.misc.imresize(img_orig, (224, 224)).astype(np.float)
    if len(img.shape) == 2:
        # grayscale
        img = np.dstack((img,img,img))
    return img, img_orig.shape

def imsave(path, img):
    img = np.clip(np.absolute(img), 0, 255).astype(np.uint8)
    Image.fromarray(img).save(path, quality=95)

def get_dtype_np():
    return np.float32

def preprocess(image, mean_pixel):
    swap_img = np.array(image)
    img_out = np.array(swap_img)
    img_out[:, :, 0] = swap_img[:, :, 2]
    img_out[:, :, 2] = swap_img[:, :, 0]
    return img_out - mean_pixel

class bcolors:
    red = "\033[31m"
    blue = "\033[34m"
    green = "\033[32m"
    end = "\033[0m"

def getImageArray(imgfile):
    img = np.asarray(Image.open(imgfile))
    return img

def getPickledImageArray(imgfile, index = 0):
    with open(imgfile, "rb") as pklfile:
        img = pickle.load(pklfile)[0]
    return img

def comparePixels(img1, img2):
    img1_shape = img1.shape
    print(f"Image 1 shape: {img1_shape}")
    img2_shape = img2.shape
    print(f"Image 2 shape: {img2_shape}\n")
    if not np.array_equal(img1_shape, img2_shape):
        print("Error: images are of different dimensions")
        print("Quitting\n")
        return

    for i in range(img1_shape[0]):
        for j in range(img1_shape[1]):
            print(f"Displaying pixel {i} x {j}\n")
            print(bcolors.red + str(round(img1[i][j][0], 2)) + bcolors.end + "\t" + bcolors.green + str(round(img1[i][j][1], 2)) + bcolors.end + "\t" + bcolors.blue + str(round(img1[i][j][2], 2)) + bcolors.end)
            print(bcolors.red + str(round(img2[i][j][0], 2)) + bcolors.end + "\t" + bcolors.green + str(round(img2[i][j][1], 2)) + bcolors.end + "\t" + bcolors.blue + str(round(img2[i][j][2], 2)) + bcolors.end)
            inp = input("\nPressed any key to continue, 'q' to quit: ")
            if inp.lower().strip() == "q":
                print("Quitting\n")
                return

if __name__ == "__main__":
    pickledImage = getPickledImageArray("n02109961_36_enc.pkl")
    image = getImageArray("cropped_n02109961_36.JPEG")
    origImage = image
    origImage = origImage[:,:,::-1]
    sqz_mean = np.array([104.006, 116.669, 122.679], dtype=get_dtype_np())
    #image = preprocess(image, sqz_mean)
    imsave("test.JPEG", image)
    Image.fromarray(np.absolute(pickledImage).astype(np.uint8)).save("test2.JPEG")
    avg_color_per_row = np.average(image, axis=0)
    print(np.average(avg_color_per_row, axis=0))

    """
    diff_blue = []
    diff_green = []
    diff_red = []
    imgshape = image.shape
    mmm = pickledImage[:,:,::-1]
    for i in range(imgshape[0]):
        for j in range(imgshape[1]):
            diff = origImage[i][j] - mmm[i][j]
            diff_blue.append(diff[0])
            diff_green.append(diff[1])
            diff_red.append(diff[2])

    print(median(diff_blue), median(diff_green), median(diff_red))
    #plt.plot(diff_blue, color="blue")
    plt.plot(diff_green, color="green")
    #plt.plot(diff_red, color="red")
    plt.show()
    """
