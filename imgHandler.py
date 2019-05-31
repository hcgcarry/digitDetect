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
#from readCarSvcModel import rea

class ImageHandler:
    def __init__(self,path):
        self.readImg(path)
        self.setImgProperty()
        self.setOperationPeremeter()
        self.buttonList=[]

    def readImg(self,path):
        self.originImg= cv2.imread(path)
        self.currentImg=self.originImg
    def setImgProperty(self):
        ###############set property
        self.imgWidth=self.originImg.shape[1]
        self.imgHeight=self.originImg.shape[0]

        print('imgWidht',self.imgWidth,'imgHeight',self.imgHeight)
        self.buttonListColPad=0
        self.buttonListColPad=int(self.imgWidth/35)
        self.buttonListPadForDilation=0
        self.finalButtonRowPad=-int(self.imgWidth/50)

        self.multiDitHandlerLowPro=0.4
        self.multiDitHandlerArea=self.imgWidth/150
        self.buttonBiasDown=int(self.imgWidth/40)

        self.lessButtonListWidth=self.imgWidth/10
        self.lessButtonListWidth=0

        self.rowSumthreshold=0.3
        self.buttonMaxWidth=self.imgWidth/3
        self.buttonMinWidth=self.imgWidth/30
        self.circleMinRadius=int(self.imgWidth/60)
        self.circleMaxRadius=int(self.imgWidth/10)
        self.circleMinCenterDistance=int(self.imgWidth/10)
        self.digitHeight=20
    def setOperationPeremeter(self):
        self.medianTimeBeforeSobel=0
        self.medianTimeAfterSobel=0
        self.medianTimeAfterBinary=2
        self.medianTimeOfSlice=2

        self.medianSize=5

        self.finalButtonWidth=28
        self.finalImgOneDimLen=self.finalButtonWidth**2

        self.positionRange=5#直愈大愈不容易delete
        self.deleteRange=7#直愈大愈不容易delete
        self.threshold=100
        self.sobelSize=3
        self.sobelSizeDuringButtonList=3
            

    def bgr2Gray(self):
        self.currentImg= cv2.cvtColor(self.currentImg, cv2.COLOR_BGR2GRAY)
        self.grayImg=self.currentImg
    def median(self,times):
        for i in range(times):
            self.currentImg=cv2.medianBlur(self.currentImg,self.medianSize)
    def canny(self):
        self.currentImg= cv2.Canny(self.currentImg, 40, 80)
    def HoughCircles(self):
        self.circleList= cv2.HoughCircles(self.currentImg,cv2.HOUGH_GRADIENT,1,self.circleMinCenterDistance,\
        param1=80,param2=100,minRadius=self.circleMinRadius,maxRadius=self.circleMaxRadius)
        self.numOfCircles=len(self.circleList[0])
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
            img=cv2.circle(self.originImg,(x,y),r,(0,255,255),5)
        #顯示新圖像
        plt.imshow(img,cmap='gray')
        plt.show()
    def sobel(self):
        x= cv2.Sobel(self.currentImg,cv2.CV_64F,1,0,ksize=self.sobelSize)
        y= cv2.Sobel(self.currentImg,cv2.CV_64F,0,1,ksize=self.sobelSize)
        absx=cv2.convertScaleAbs(x) 
        absy=cv2.convertScaleAbs(y) 
        self.sobelImage=cv2.addWeighted(absx,1,absy,0,0)
        self.currentImg=self.sobelImage
    def binary(self):
        threshold=threshold_otsu(self.currentImg)
        ret ,self.binaryImg= cv2.threshold(self.currentImg,threshold,1,cv2.THRESH_BINARY)
        self.currentImg=self.binaryImg
    

     

    def printCurrentImg(self):
        plt.figure()
        plt.imshow(self.currentImg)
        plt.show()
        
    def cutButton(self,ax1):
        for index,circle in enumerate(self.circleList[0]):
            x=circle[0]
            y=circle[1]
            radius=circle[2]
            squareWidth=radius*sqrt(2)/2
            min_col=int(x-squareWidth)
            max_col=int(x+squareWidth)
            min_row=int(y-squareWidth)
            max_row=int(y+squareWidth)
            print(min_col,max_col,min_row,max_row)
            button=self.grayImg[min_row:max_row,min_col:max_col]
            button=cv2.resize(button,(self.finalButtonWidth,self.finalButtonWidth))
            threshold=threshold_otsu(button)
            ret , button= cv2.threshold(button,threshold,255,cv2.THRESH_BINARY)
            self.buttonList.append(button)
        
            #####plot button boundary
            ax1.imshow(self.binaryImg,cmap="gray")
            rectBorder = patches.Rectangle((min_col, min_row), max_col-min_col,\
             max_row-min_row, edgecolor="red", linewidth=2, fill=False)
            ax1.add_patch(rectBorder)
        

    def checkButtonBlackBackGround(self):
            ##判斷這張button以黑為背景還是白
        totalPixelSum=0
        for index,item in enumerate(self.buttonList):
            pixelSum=np.sum(item,axis=0)
            totalPixelSum=totalPixelSum+np.sum(pixelSum)      
        
        ###如果被僅式白的字式黑的就給他反轉
        if totalPixelSum > 255*len(self.buttonList)*self.finalImgOneDimLen/2:
            for index,button in enumerate(self.buttonList):
                button=255-button
                self.buttonList[index]=button
        

    def buttonDeleteWrongPattern(self,button):
        label_image = measure.label(button)
        regionpropsList=regionprops(label_image)
        regionCordinatesList=[]
        regionToCenterDistanceList=[]
        #一個按鍵裏面連通的region
        digitcount=0
        #####梯出不太可能式digit的根據面積還有比例
        for index,region in enumerate(regionpropsList):
            #print("regionpropsList",regionpropsList)

            min_row, min_col, max_row, max_col = region.bbox
            region_height = max_row - min_row
            region_width = max_col - min_col
            ##origin proportion
            proportion=region_height/region_width
            if self.multiDitHandlerLowPro< proportion  and self.multiDitHandlerArea< region.area:
                regionCordinates=[min_row,max_row,min_col,max_col]
                regionCordinatesList.append(regionCordinates)
                digitcount=digitcount+1
            
        #如果因為條件太嚴格所以沒抓到region 就放寬
        #print("next digit *****************8")
        if digitcount==0:
            for index ,region in enumerate(regionpropsList):

                min_row, min_col, max_row, max_col = region.bbox
                ##update row and proportion
                regionCordinates=[min_row,max_row,min_col,max_col]
                regionCordinatesList.append(regionCordinates)
        #######重regionCordinatesList裏面找出最有可能是digit的根據座標到元新的距離最短的
        minDistanceToCenterNum=0
        minDistanceToCenterIndex=0
        for index,item in enumerate(regionCordinatesList):
            distance=0
            distance=distance+(item[0]-self.finalButtonWidth/2)**2
            if distance< minDistanceToCenterNum:
                minDistanceToCenterNum=distance
                minDistanceToCenterIndex=index
        
        ######找到了最像digit的剪下來做縮放高等於20後放到全黑的button上

        #####finalButton最後呈獻的button
        finalButton=np.zeros([self.finalButtonWidth,self.finalButtonWidth])
        digitMinRow,digitMaxRow,digitMinCol,digitMaxCol=regionCordinatesList[minDistanceToCenterIndex]
        digitWidth=digitMaxCol-digitMinCol
        digitHeight=digitMaxRow-digitMinRow
        button=np.array(button)
        ##################################################從button上面才切下digit
        digit=button[digitMinRow:digitMaxRow,digitMinCol:digitMaxCol]

        #convert to Image object
        digit= Image.fromarray(digit) 

        ####將digit縮放成minus那個資料庫的digit大小
        hpercent = (digitHeight/ float(digit.size[1]))
        digitWidth= int((float(digit.size[0]) * float(hpercent)))
        if digitWidth > self.finalButtonWidth:
            digitWidth=self.finalButtonWidth

        digit= digit.resize((digitWidth, digitHeight), PIL.Image.ANTIALIAS)
        # convert back to np array
        digit=np.array(digit)
        finalButtonMinCol=14-int(digitWidth/2)
        finalButtonMaxCol=finalButtonMinCol+digitWidth
        finalButtonMinRow=14-int(digitHeight/2)
        finalButtonMaxRow=finalButtonMinRow+digitHeight
        finalButton[finalButtonMinRow:finalButtonMaxRow,finalButtonMinCol:finalButtonMaxCol]=digit
        
      

        finalButton=finalButton/255
        finalButton=np.reshape(finalButton,self.finalImgOneDimLen)
        return finalButton

    def buttonListDeleteWrongPattern(self):
        for index,button in enumerate(self.buttonList):
            self.buttonList[index]=self.buttonDeleteWrongPattern(button)


    def detectDigitNumber(self):

        detectResult=readModel(self.buttonList)
        self.showsubplot(self.buttonList,detectResult)
        plt.show()

    def showsubplot(self,images,imagesTitle=None):
        row=int(len(images)/6)+1
        #list轉換成np.array
        try:
            if isinstance(images, list):
                images=np.array(images)
                
            #如果近來的事背flatten的images
            if len(images.shape) ==2:
                imagesSize=int(sqrt(images.shape[1]))
                images=np.resize(images,(images.shape[0],imagesSize,imagesSize))
        except:
            pass
        if imagesTitle is None:
            imagesTitle=list(range(len(images)))
        
        plt.figure(figsize=(30,30))

        for index in range(len(images)):
            plt.subplot(row,6,index+1)
            plt.imshow(images[index],cmap="gray")
            
            plt.title(imagesTitle[index])
            #ticks

        

