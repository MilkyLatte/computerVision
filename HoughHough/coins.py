import cv2 as cv
from matplotlib import pyplot as plt
from skimage import io
import numpy as np

image1 = cv.cvtColor(cv.imread("coins1.png"), cv.COLOR_BGR2GRAY)
image2 = cv.cvtColor(cv.imread("coins2.png"), cv.COLOR_BGR2GRAY)
image3 = cv.cvtColor(cv.imread("coins3.png"), cv.COLOR_BGR2GRAY)

paddedImage1 = np.array(cv.copyMakeBorder(image1, 1, 1, 1, 1, cv.BORDER_CONSTANT, value= [0]))
paddedImage2 = cv.copyMakeBorder(image2, 1, 1, 1, 1, cv.BORDER_CONSTANT, value= [0])
paddedImage3 = cv.copyMakeBorder(image3, 1, 1, 1, 1, cv.BORDER_CONSTANT, value= [0])
imageDictionary = {
    "coins": [paddedImage1, paddedImage2, paddedImage3],
    "dx": [],
    "dy": [],
    "magnitude": [],
    "gradient": []
}
kernelx = np.array([[-1, 0, 1],
                   [-1, 0, 1],
                   [-1, 0, 1]])
kernely = np.array([[-1, -1, -1],
                   [0, 0, 0],
                   [1, 1, 1]])
#assumes image is padded and returns a convoluted non-padded image
def convolute(image, kernel):
    convolutedImage = np.zeros((image.shape[0]-1, image.shape[1]-1))
    for row in range(1, len(image)-1):
        for col in range(1, len(image[row])-1):
            section = np.array(([image[row-1][col-1], image[row-1][col], image[row-1][col+1]],
                               [image[row][col-1], image[row][col], image[row][col+1]],
                               [image[row+1][col-1], image[row+1][col], image[row+1][col+1]]))
            x = section*kernel * 1/9
            convolutedImage[row][col] = np.sum(x)
    return convolutedImage


def magnitude(dx, dy):
    finald = np.sqrt(dx**2 + dy**2)
    return(finald)


def gradient(derx, dy):
    dx = np.zeros(derx.shape)
    np.copyto(dx, derx)
    arctan = np.zeros(dx.shape)
    for row in range(len(dx)):
        for col in range(len(dx[row])):
            if dx[row][col] == 0:
                arctan[row][col] = 255
            else:
                arctan[row][col] = np.arctan(dy[row][col]/dx[row][col])
    return arctan

def sobelAdd(image, index):
    dx = convolute(image, kernelx)
    dy = convolute(image, kernely)
    cv.imwrite("dy"+str(index)+".png", dy)
    cv.imwrite("magnitude"+str(index)+".png", magnitude(dx, dy))
    cv.imwrite("gradient"+str(index)+".png", gradient(dx, dy))
    cv.imwrite("dx"+str(index)+".png", dx)




for coin in range (len(imageDictionary["coins"])):
    sobelAdd(imageDictionary["coins"][coin], coin)
