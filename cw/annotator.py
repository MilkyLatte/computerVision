import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import time
import copy
from subprocess import Popen, PIPE
import json




imageDictionary = {
    "image": [],
    "task1": [4, 5, 13, 14, 15],
    "grayImage": [],
    "faceDetectedImage": []
}
cascadePath = "frontalface.xml"
cascade = cv.CascadeClassifier(cascadePath)

for i in range(16):
    image = cv.imread("./images/dart"+str(i)+".jpg")
    imageDictionary["image"].append(image)
    imageDictionary["grayImage"].append(cv.cvtColor(image, cv.COLOR_BGR2GRAY))
cv.rectangle(imageDictionary["image"][0], (125, 200), (250, 325), (0, 255, 0), 2)
cv.rectangle(imageDictionary["image"][4], (340, 100), (490, 250), (0, 255, 0), 2)

imNumber = 0

while 1:
    try:
        imNumber = int(input("what image (number) do you want to go through"))
    except:
        print("try again")
        continue
    if imNumber < 16:
        break

windowSize = 0
while 1:
    try:
        windowSize = int(input("what window size you want?"))
        break
    except:
        print("try again")
        continue
imSize = imageDictionary["image"][imNumber].shape

def loadMaps():
    with open('target.json', 'r') as f:
        a = json.load(f)
    return copy.deepcopy(a)

squares = loadMaps()
savedLocations = []
done = False
for x in range(0, imSize[1]-windowSize, 20):
    for y in range(0, imSize[0]-windowSize, 20):
        dummyImage = copy.deepcopy(imageDictionary["image"][imNumber])
        cv.rectangle(dummyImage, (x, y), (x+windowSize, y+windowSize), (0, 255, 0), 2)
        cv.imwrite("Trial.png", dummyImage)
        i = input("saveLocation?")
        if i == 'y':
            savedLocations.append((x,y,windowSize))
            if input("type done if you are done") == "done":
                done = True
                break
    if done:
        break
    if input("type done if you are done") == "done":
        break

squares[str(imNumber)] = savedLocations
with open('target.json', 'w') as f:
    a = json.dump(squares, f, separators=(',', ': '), indent=4)
