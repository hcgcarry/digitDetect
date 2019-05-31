import sys
from skimage.io import imread
from skimage.morphology import skeletonize
from skimage.filters import threshold_otsu
import matplotlib.pyplot as plt
import PIL
import glob
from skimage import measure
from math import sqrt
import numpy as np
from skimage.measure import regionprops
import matplotlib.patches as patches
from skimage.transform import resize
import cv2
import math
from skimage import data,filters
import tensorflow as tf
import warnings
import heapq
from rotate import rotate
#from train.conv import readModel
#from train.threeLayerConv import readModel
from train.threeLayerConvBn import readModel
#from train.test import readModel
#from train.dnn6and8 import readModel
#from train.dnn import readModel
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.examples.tutorials.mnist import input_data
from PIL import Image
class circleList:
    def __init__(self,img):
        self.currentImg=img
        self.setProperity()
        self.circles

    def setProperity(self):
        self.imgWidth=binaryImg.shape[1]
        self.imgHeight=binaryImg.shape[0]
        self.circleMinRadius=int(self.imgWidth/60)
        self.circleMaxRadius=int(self.imgWidth/10)
        self.circleMinCenterDistance=int(self.imgWidth/10)

    def HoughCircles(self):
        self.circles= cv2.HoughCircles(self.currentImg,cv2.HOUGH_GRADIENT,1,self.circleMinCenterDistance,\
        param1=80,param2=100,minRadius=self.circleMinRadius,maxRadius=self.circleMaxRadius)

    def printCircles(self):
        #輸出檢測到圓的個數
        print(len(self.circleList[0]))

        print("-------------我是條分割線-----------------")
        #根據檢測到圓的信息，畫出每一個圓
        for circle in self.circleList[0]:
            #圓的基本信息
            #print(circle[2])
            #坐標行列(就是圓心)
            x=int(circle[0])
            y=int(circle[1])
            #半徑
            r=int(circle[2])
            #在原圖用指定顏色圈出圓，參數設定為int所以圈畫存在誤差
            img=cv2.circle(self.currentImg,(x,y),r,(0,255,255),5)
        #顯示新圖像
        plt.imshow(img,cmap='gray')
        plt.show()