    
from detectImgModel.imgHandler import ImageHandler
from detectImgModel.squareImgHandler import SquareImgHandler
import matplotlib.pyplot as plt
from shapedetector import ShapeDetector
import argparse
import imutils
import cv2
def detectSquare(imagePath):

    # load the image and resize it to a smaller factor so that
    # the shapes can be approximated better
    image = cv2.imread(imagePath)
    resizedImg = imutils.resize(image, width=300)
    ratio = image.shape[0] / float(resizedImg.shape[0])

    # convert the resized image to grayscale, blur it slightly,
    # and threshold it
    gray = cv2.cvtColor(resizedImg, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]

    # find contours in the thresholded image and initialize the
    # shape detector
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    sd = ShapeDetector()

    # loop over the contours
    for c in cnts:
        # compute the center of the contour, then detect the name of the
        # shape using only the contour
        M = cv2.moments(c)
        print(M)
        cX = int((M["m10"] / M["m00"]))
        cY = int((M["m01"] / M["m00"]))
        shape = sd.detect(c)

        # multiply the contour (x, y)-coordinates by the resize ratio,
        # then draw the contours and the name of the shape on the image
        c = c.astype("float")
        c *= ratio
        c = c.astype("int")
        cv2.drawContours(resizedImg, [c], -1, (0, 255, 0), 2)
        cv2.putText(resizedImg, shape, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX,
            0.5, (255, 255, 255), 2)

        # show the output image
        cv2.imshow("Image", resizedImg)
        cv2.waitKey(0)