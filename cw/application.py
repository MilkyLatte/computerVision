import sys
import json
import copy
import cv2 as cv
import numpy as np



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

def hough_circle(image, gradient, min_radius, max_radius):

    houghs = np.zeros((len(image),len(image[0]), max_radius - min_radius))
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
    gradient = cv.phase(sobely, sobelx, False)

    return cv.addWeighted(cv.convertScaleAbs(sobelx), 0.5, cv.convertScaleAbs(sobely), 0.5, 0), gradient

def runHoughTransform(croppedTargets,  detectedTarget):
    target = []
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
            target.append(detectedTarget[i])
    return target


def loadJson(path):
    try:
        with open(path, 'r') as f:
            a = json.load(f)
            return copy.deepcopy(a)
    except Exception:
            print("Can't locate " + path)
            sys.exit()



def initialize():
    menuChoice = None
    while 1:
        menuChoice = input("Select one of the following options: \n 0 for testing face detection.\n 1 for new face detection \n 2 for testing dart detection \n 3 for new dart detection \n")
        try:
            menuChoice = int(menuChoice)
            if menuChoice < 4:
                break
            else:
                print("incorrect choice try again")
        except Exception:
            print("incorrect choice, try again")
    return menuChoice

def loadTestImages():
    imageDictionary = {
        "image": [],
        "task1": [4, 5, 13, 14, 15],
        "grayImage": [],
        "faceDetectedImage": [],
        "realFaces": [],
        "realTarget": [],
        "targetDetection": []
    }
    for i in range(16):
        image = cv.imread("./images/dart"+str(i)+".jpg")
        imageDictionary["image"].append(image)
        imageDictionary["grayImage"].append(cv.cvtColor(image, cv.COLOR_BGR2GRAY))
    return imageDictionary

def drawSquare(image, squares):
    for x, y, w, h in squares:
        cv.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
    cv.imwrite("detected.jpg", image)

def evaluation(real, predicted):
    numberOfFacesInImage = len(real)
    facesDetected = len(predicted)
    truePositives = 0
    falsePositives = 0
    for x, y, z in real:
        for a, b, c, _ in predicted:
            if abs(a - x) < 50 and abs(b-y) < 50:
                truePositives += 1
                break
    falsePositives = facesDetected - truePositives
    falseNegatives = numberOfFacesInImage - truePositives
    print("Number of Targets: " + str(numberOfFacesInImage))
    print("Predicted Targets: " + str(facesDetected))
    print("True Positives: " + str(truePositives))
    print("False Positives: " + str(falsePositives))
    print("False Negatives: " + str(falseNegatives) )
    return [numberOfFacesInImage, facesDetected, truePositives, falsePositives, falseNegatives]


def precision(truePositives, falsePositives):
    try:
        pre = truePositives/(truePositives + falsePositives)
        return pre
    except Exception:
        return None

def tpr(truePositives, falseNegatives):
    tp = truePositives/(truePositives + falseNegatives)
    return tp

def fullEval(ev):
    avgt = 0
    avgp = 0
    avgf = 0
    includedImg = 16
    p = precision(ev[2], ev[3])
    t = tpr(ev[2], ev[4])
    f1 = f1Score(p, t)
    print("Precision: " + str(p))
    print("TPR: " + str(t))
    print("F1: " + str(f1))


def f1Score(precision, tpr):
    try:
        return 2*((precision*tpr)/(precision + tpr))
    except Exception:
        return None


def testFaceDetection():
    try:
        cascadePath = "frontalface.xml"
    except Exception:
        print("can't locate file frontalface.xml")
        sys.exit()
    cascade = cv.CascadeClassifier(cascadePath)
    realTargetSquares = loadJson('squares.json')
    imageDictionary = loadTestImages()
    test = input("which test image would you want to test on: \n")
    detectedFaceSquares = cascade.detectMultiScale(imageDictionary["grayImage"][int(test)])

    drawSquare(imageDictionary["image"][int(test)], detectedFaceSquares)
    ev = evaluation(realTargetSquares[test], detectedFaceSquares)
    fullEval(ev)

def faceDetection():
    path = input("What is the path of the image?")
    image = cv.imread(path)
    grayImage = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    cascadePath = "frontalface.xml"
    cascade = cv.CascadeClassifier(cascadePath)
    detectedFaces = cascade.detectMultiScale(grayImage)
    drawSquare(image, detectedFaces)


def dartDetection():
    targetPath = "./dartcascade/cascade.xml"
    targetCascade = cv.CascadeClassifier(targetPath)
    path = input("What is the path of the image?")
    image = cv.imread(path)
    grayImage = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    target = targetCascade.detectMultiScale(grayImage)
    croppedTargets = []
    for i in range(len(target)):
        x, y, h, w = target[i]
        crop_img = grayImage[y:y+h, x:x+w]
        croppedTargets.append(crop_img)
    dart = runHoughTransform(croppedTargets, target)
    drawSquare(image, dart)





def testDarts():
    targetPath = "./dartcascade/cascade.xml"

    targetCascade = cv.CascadeClassifier(targetPath)
    test = input("Which test image would you want to test on \n")
    realTargetSquares = loadJson('target.json')
    imageDictionary = loadTestImages()
    target = targetCascade.detectMultiScale(imageDictionary["grayImage"][int(test)])
    croppedTargets = []
    for i in range(len(target)):
        x, y, h, w = target[i]
        crop_img = imageDictionary["grayImage"][int(test)][y:y+h, x:x+w]
        croppedTargets.append(crop_img)
    dart = runHoughTransform(croppedTargets, target)
    ev = evaluation(realTargetSquares[test], dart)
    fullEval(ev)
    drawSquare(imageDictionary["image"][int(test)], dart)




def main():
    menu = initialize()
    if menu == 0:
        testFaceDetection()
    if menu == 1:
        faceDetection()
    if menu == 2:
        testDarts()
    if menu == 3:
        dartDetection()

if __name__ == "__main__":
    main()
