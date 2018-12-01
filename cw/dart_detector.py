import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import time
import copy
import json
import math as ma

def hough_line(image):
    angles = np.deg2rad(np.arange(-90, 90))
    im_width, im_height = image.shape
    im_diag = int(np.round(np.sqrt(im_width**2 + im_height ** 2)))
    max_ps = np.linspace(-im_diag, im_diag, im_diag * 2)

    coses = np.cos(angles)
    sins = np.sin(angles)
    n_angles = len(angles)

    houghs = np.zeros((2 * im_diag, n_angles))


    for y in range(len(image)):
        for x in range(len(image[0])):
            if (image[y][x] == 255):
                for theta in range(n_angles):
                    p = int(np.round(x * coses[theta] + y * sins[theta])) + im_diag
                    houghs[p][theta] += 1

    return houghs, angles, max_ps

def get_threshold_line(houghs):

    over_threshold = []
    preYval = 0
    preXval = 0

    for i in range(0, 1000):
        over_threshold = []
        preYval = 0
        preXval = 0
        for y in range(len(houghs)):
            for x in range(len(houghs[0])):
                if houghs[y][x] > i:
                    if preYval < y or preXval < x:
                        over_threshold.append((y, x))
                        preYval = y
                        preXval = x
        if len(over_threshold) <=  50:
            break

    return np.array(over_threshold)

def get_center_line(thresholded, image, max_ps, angles):
    lined = np.zeros(image.shape)
    for y in range(len(thresholded)):
        distance = max_ps[thresholded[y][0]]
        angle = angles[thresholded[y][1]]

        if angle != 0:
            m = -np.cos(angle) / np.sin(angle)
            b = distance / np.sin(angle)
        else:
            m = -np.cos(angle)
            b = distance

        for x in range(len(image[0])):
            y0 = int(np.round(m * x + b))

            if y0 >= 0 and y0 < len(image):
                lined[y0][x] += 1
    maximum = 0
    y1 = 0
    x1 = 0

    for y in range(len(lined[0])):
        for x in range(len(lined[1])):
            if lined[y][x] > maximum:
                maximum = lined[y][x]
                y1 = y
                x1 = x
    return (x1, y1)

def get_center_line(thresholded, image, max_ps, angles):
    lined = np.zeros(image.shape)
    for y in range(len(thresholded)):
        distance = max_ps[thresholded[y][0]]
        angle = angles[thresholded[y][1]]

        if angle != 0:
            m = -np.cos(angle) / np.sin(angle)
            b = distance / np.sin(angle)
        else:
            m = -np.cos(angle)
            b = distance

        for x in range(len(image[0])):
            y0 = int(np.round(m * x + b))

            if y0 >= 0 and y0 < len(image):
                lined[y0][x] += 1
    maximum = 0
    y1 = 0
    x1 = 0

    for y in range(len(lined[0])):
        for x in range(len(lined[1])):
            if lined[y][x] > maximum:
                maximum = lined[y][x]
                y1 = y
                x1 = x
    return (x1, y1)

def hough_circle(image, gradient, min_radius, max_radius):

    houghs = np.zeros((len(image),len(image[0]), max_radius - min_radius))
    print(len(image),len(image[0]))
    for y in range(0, len(image)):
        for x in range(0, len(image[0])):
            if image[y][x] == 255:
                 for w in range(min_radius - 1, max_radius):
                        x_weight = w * np.sin(gradient[y][x])
                        y_weight = w * np.cos(gradient[y][x])

                        x1 = int(round(abs(x - x_weight)))
                        y1 = int(round(abs(y - y_weight)))

                        if y1 >= 0 and x1 >= 0 and y1 < len(image) and x1 < len(image[0]):
                            houghs[y1][x1][w - min_radius] += 1


    return houghs

def get_threshold_circle(houghs, min_radius):
    over_threshold = []
    previous_over_threshold = []

    preXval = 0
    preYval = 0

    for j in range(1000):
        over_threshold = []
        preXval = 0
        preYval = 0
        for y in range(len(houghs)):
            for x in range(len(houghs[0])):
                for w in range(len(houghs[0][1])):
                    if houghs[y][x][w] > j:
                        if preXval < x or preYval < y:
                            over_threshold.append((x, y, w + min_radius, houghs[y][x][w]))
                            preXval = x
                            preYval = y

        if len(over_threshold) <= 150:
            if len(over_threshold) == 0:
                over_threshold = copy.deepcopy(previous_over_threshold)
            break
        else:
            previous_over_threshold = copy.deepcopy(over_threshold)

    return np.array(over_threshold)

def get_center_circle(thresholded):
    sumX = 0
    sumY = 0
    sumR = 0
    num = 0

    for i in range(len(thresholded)):
        sumX += thresholded[i][0]
        sumY += thresholded[i][1]
        sumR += thresholded[i][2]
        num += 1

    avgX = int(np.round(sumX/num))
    avgY = int(np.round(sumY/num))
    avgR = int(np.round(sumR/num))

    return (avgX, avgY, avgR)

def drewCircle(image, thresholded):
    for i in range(len(thresholded)):
        cv.circle(image, (thresholded[i][0], thresholded[i][1]), thresholded[i][2], 255, 1)

def magnitude(img):
    gradient = np.zeros(img.shape)
    src = copy.deepcopy(img)
    src = cv.GaussianBlur( src, (3,3), 0, 0);
    sobelx = cv.Sobel(src,cv.CV_64F,1,0,ksize=3)  # x
    sobely = cv.Sobel(img,cv.CV_64F,0,1,ksize=3)  # y
    for y in range(len(sobely)):
        for x in range(len(sobelx)):
            gradient[y][x] = ma.atan2(sobely[y][x], sobelx[y][x])

    return cv.addWeighted(cv.convertScaleAbs(sobelx), 0.5, cv.convertScaleAbs(sobely), 0.5, 0), gradient

def runHoughTransform(croppedTargets, ioImage, detectedTarget):
    target = {}
    # for item in croppedTargets:
    target[ioImage] = []
    for i in range(len(croppedTargets)):
        original = croppedTargets[i]
        mag, gradient = magnitude(original)
        mag_copy = np.copy(mag)
        mag_copy[mag_copy >= 50] = 255
        mag_copy[mag_copy < 50] = 0
        accumulator, angles, max_ps = hough_line(mag_copy)
        thresholded = get_threshold_line(accumulator)
        circle_accumulator = hough_circle(mag_copy, gradient, 20, len(mag_copy[0]))
        circle_thresholded = get_threshold_circle(circle_accumulator, 20)
        circle_center = get_center_circle(circle_thresholded)
        center = np.copy(original)
        cv.circle(center, (circle_center[0], circle_center[1]), 20, 255, 10)
        maximum = get_center_line(thresholded, original, max_ps, angles)
        cv.circle(center, maximum, 1, 255, 10)
        dist = np.sqrt((maximum[0] - circle_center[0]) ** 2 + (maximum[1] - circle_center[1]) ** 2)
        if dist < 20:
            # print("It's a target")
            target[ioImage].append(detectedTarget[str(ioImage)][i])

    return target



imageDictionary = {
    "image": [],
    "task1": [4, 5, 13, 14, 15],
    "grayImage": [],
    "realFaces": [],
    "realTarget": [],
    "targetDetection": []
}

imageNames = {}

for i in range(16):
    image = cv.imread("./images/dart"+str(i)+".jpg")
    imageNames["dart" + str(i) + ".jpg"] = i
    imageDictionary["image"].append(image)
    imageDictionary["grayImage"].append(cv.cvtColor(image, cv.COLOR_BGR2GRAY))


dummyImage = copy.deepcopy(imageDictionary["image"][5])

targetPath = "./dartcascade/cascade.xml"
targetCascade = cv.CascadeClassifier(targetPath)
detectedTarget = {

}

for i in range(len(imageDictionary["image"])):
    c = copy.deepcopy(imageDictionary["grayImage"][i])
    target = targetCascade.detectMultiScale(c, 1.1, 1, 0, (100,100), (250, 250))
    detectedTarget[str(i)] = []
    for (x,y,w,h) in target:
        # cv.rectangle(c,(x,y),(x+w,y+h),(0,255,0),2)
        detectedTarget[str(i)].append([x, y, w, h])
    # imageDictionary["targetDetection"].append(c)


croppedTargets= {

}

for i in range(len(detectedTarget.keys())):
    croppedTargets[str(i)] = []
    for j in range(len(detectedTarget[str(i)])):
        x, y, h, w = detectedTarget[str(i)][j]
        crop_img = imageDictionary["grayImage"][i][y:y+h, x:x+w]
        croppedTargets[str(i)].append(crop_img)

ioImage_string = ""
ioImage = 0

while 1:
    try:
        ioImage_string = str(input("Which image do you want do detect a dart board? "))
    except:
        print("try again")
        continue

    if ioImage_string in imageNames:
        break
    else:
        print("Invalid image, try again")
        continue

ioImage = imageNames[ioImage_string]

chosen_images = copy.deepcopy(croppedTargets[str(ioImage)])

dart = runHoughTransform(chosen_images, ioImage, detectedTarget)

output_image = copy.deepcopy(imageDictionary["image"][ioImage])
for item in dart:
    for i in range(len(dart[item])):
        cv.rectangle(output_image,(dart[item][i][0],dart[item][i][1]),(dart[item][i][0]+dart[item][i][2],dart[item][i][1]+dart[item][i][3]),(0,255,0),2)

cv.imwrite("detected.jpg", output_image)

print("Dart detection complete")
