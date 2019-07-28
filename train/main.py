from skimage.io import imread
from skimage.morphology import skeletonize
from skimage.filters import threshold_otsu
import matplotlib.pyplot as plt
import PIL
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
#from train.conv import readModel
#from train.threeLayerConv import readModel
#from train.threeLayerConvBn import readModel
from train.test import readModel
#from train.dnn6and8 import readModel
#from train.dnn import readModel
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.examples.tutorials.mnist import input_data
from PIL import Image
from binary import cv2
#from readCarSvcModel import readModel 
#from sklearndnn import readModel

#from sklearndnn import readModel
##################parameter

medianTimeBeforeSobel=0
medianTimeAfterSobel=0
medianTimeAfterBinary=2
medianTimeOfSlice=0
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
batch_xs, batch_ys = mnist.train.next_batch(10)

medianSizeBeforeSobel=5
medianSizeAfterSobel=5
medianSizeBeforeBinary=5

finalImgWidth=28
finalImgOneDimLen=finalImgWidth**2

positionRange=5#直愈大愈不容易delete
deleteRange=7#直愈大愈不容易delete
threshold=100
deleteBigRange=12
sobelSize=3
thresholdRange=5
imgIndex=15

affineOffset=5

class Button:
    def __init__(self):
        self.digitImgList=[]
        self.digitListCordinates=[]
        self.digitListPredict=[]
        self.originButton=0
    
def rotate(image, angle, center=None, scale=1.0):
    # 获取图像尺寸
    (h, w) = image.shape[:2]
 
    # 若未指定旋转中心，则将图像中心设为旋转中心
    if center is None:
        center = (w / 2, h / 2)
 
    # 执行旋转
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))
 
    # 返回旋转后的图像
    return rotated


def showsubplot(images,imagesTitle=None):
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



def showhistogram(images,imagesTitle=None):
    if imagesTitle is None:
        imagesTitle=list(range(len(images)))
    plt.figure(figsize=(30,30))
    for index in range(len(images)):
        hist = cv2.calcHist([images[index]],[0],None,[256],[0.0,255.0])
        plt.subplot(1,len(images),index+1)
        plt.plot(hist)
        plt.title(imagesTitle[index])
    plt.show()

#############################################################################begin
for programIndex in range(6,imgIndex):
    #################initial
    affineItemList=[]
    ###########################
    originImg = cv2.imread("./evpicture/test/ujpeg{}.jpeg".format(programIndex))
    #originImg = cv2.imread("./evpicture/test/images (2).jpeg")
    #originImg = cv2.imread("./evpicture/home.jpg")
    #originImg = cv2.imread("./evpicture/test/jpg{}.jpg".format(programIndex))
    imgWidth=originImg.shape[1]
    imgHeight=originImg.shape[0]

    buttonPadding=int(imgWidth/40)
    buttonPadding=0
    rowSumthreshold=0.4
    #buttonMaxWidth=imgWidth;buttonMinWidth=0
    buttonMaxWidth=imgWidth/3;buttonMinWidth=imgWidth/30
    #originImg = cv2.imread("./evpicture/home.jpg".format(imgIndex))
    #################################################parameter end

    #img = cv2.imread("./evpicture/1.jpg",0)


    img= cv2.cvtColor(originImg, cv2.COLOR_BGR2GRAY)
    imgBeforeMedian=img
    ############median before sobel
    for i in range(medianTimeBeforeSobel):
        img=cv2.medianBlur(img,medianSizeBeforeSobel)


    ######################邊緣化sobel

    x= cv2.Sobel(img,cv2.CV_64F,1,0,ksize=sobelSize)
    y= cv2.Sobel(img,cv2.CV_64F,0,1,ksize=sobelSize)
    sobel=cv2.convertScaleAbs(y) 
    """
    absy=cv2.convertScaleAbs(y) 
    absx=cv2.convertScaleAbs(x)
    sobel=cv2.addWeighted(absx,0.5,absy,0.5,0)
    plt.imshow(sobel,cmap="gray")
    plt.show()
    """
    x= cv2.Sobel(img,cv2.CV_64F,1,0,ksize=sobelSize)
    y= cv2.Sobel(img,cv2.CV_64F,0,1,ksize=sobelSize)
    absy=cv2.convertScaleAbs(y) 
    absx=cv2.convertScaleAbs(x)
    #就是有計算xy方向sobel的
    xySobel=cv2.addWeighted(absx,0.5,absy,0.5,0)
    #################################縱向計算最量的地方
    rowSum=np.sum(sobel,axis=0)
    rowSumMean=np.mean(rowSum)
    rowSum=rowSum-rowSumMean
    """
    plt.figure()
    plt.plot(np.linspace(0,imgWidth,imgWidth),rowSum)
    plt.show()
    """
    rowSumMax=np.max(rowSum)
    class buttonList:
        sobelImages=[]
        def __init__(self,leftCol,rightCol):
            self.leftCol=leftCol
            self.rightCol=rightCol
            self.button_objects_cordinates=[]
            

    buttonListArray=[]
    leftCol=0
    rightCol=0
    count=0
    #########連續亮點判斷

    for index in range(imgWidth):
        if rowSum[index] > rowSumMax*rowSumthreshold:
            if leftCol==0:
                leftCol=index
                count=count+1
            elif index==leftCol+count:
                count=count+1
                rightCol=index
        else:
            if rightCol!=0 and buttonMinWidth<=rightCol-leftCol<=buttonMaxWidth:###右邊然後下降
                buttonListArray.append(buttonList(leftCol-buttonPadding,rightCol+buttonPadding))
                print("#############append",leftCol,rightCol)
            count=0;rightCol=0;leftCol=0;

    buttonListArrayLen=len(buttonListArray)
    print(buttonListArrayLen)
    #################################################將buttonlist裏面每個物件的left right 拿去擷取原本img再做sobel 最後subplot出
    #&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    #fig, (ax1) = plt.subplots(1)
    #ax1.imshow(img, cmap="gray");

    showImages=[]
    button_like_objects=[]
    imagesTitle=[]
    testimage=[]

    subplotLen=4
    #對buttonlist 做操作(目標取出一個一個button
    for index in range(buttonListArrayLen):
        #slice
        item=img[:,buttonListArray[index].leftCol:buttonListArray[index].rightCol]
        #sobel
        x=cv2.Sobel(item,cv2.CV_64F,1,0,ksize=sobelSize)
        #y=cv2.Sobel(item,cv2.CV_64F,0,1,ksize=sobelSize)
        absx=cv2.convertScaleAbs(x) 
        itemSobel=cv2.convertScaleAbs(x) 
        #absy=cv2.convertScaleAbs(y) 
        #itemSobel=cv2.addWeighted(absx,1,absy,0,0)
        showImages.append(itemSobel)
        #median filter
        for i in range(medianTimeOfSlice):
            itemSobel=cv2.medianBlur(itemSobel,medianSizeBeforeBinary)
        #binary
        threshold=threshold_otsu(itemSobel)
        ret , itemThreshold= cv2.threshold(itemSobel,threshold,1,cv2.THRESH_BINARY)

        #erosion and dilation
        kernelWidth=int(item.shape[1]/1)
        kernel = np.ones((2,kernelWidth),np.uint8)
        #opening
        dilation1 = cv2.dilate(itemThreshold,kernel,iterations=1)
        erosion1 = cv2.erode(dilation1,kernel,iterations=1)
        #closing
        erosion2 = cv2.erode(erosion1,kernel,iterations=1)
        ######################insert
        showImages.append(itemSobel)
        showImages.append(itemThreshold)
        showImages.append(dilation1)
        showImages.append(erosion1)
        showImages.append(erosion2)
        #################3######title
        title="left={},right={}".format(buttonListArray[index].leftCol,buttonListArray[index].rightCol)
        imagesTitle.append(title)
        #title after median
        title="median times={}".format(medianTimeOfSlice)
        imagesTitle.append(title)
        #title after binary
        title="threshold={}".format(threshold)
        imagesTitle.append(title)
        #opening
        title="dilation1"
        imagesTitle.append(title)
        title="erosion1"
        imagesTitle.append(title)
        title="erosion2"
        imagesTitle.append(title)
        #ticks
        #plt.title(title)
        #plt.subplot(1,buttonListArrayLen+subplotLen,2*index+2)
        #plt.imshow(itemsobel,cmap="gray")

        ####################################連通
        label_image = measure.label(erosion2)
            #plate dimension
        plateDimension=(buttonMinWidth,buttonMaxWidth,buttonMinWidth,buttonMaxWidth)

        min_height,max_height,min_width,max_width=plateDimension


        for region in regionprops(label_image):
            #widthBias=int(region_width/8);heightBias=int(region_height/8)
            widthBias=0;heightBias=0

            min_row, min_col, max_row, max_col = region.bbox
            region_height = max_row - min_row
            min_col=buttonListArray[index].leftCol+widthBias
            max_col=buttonListArray[index].rightCol-widthBias
            region_width = max_col - min_col
            ##origin proportion
            proportion=region_width/region_height
            ##update row and proportion

            #if region_height >= min_height and region_height <= max_height and region_width >= min_width and region_width <= max_width and region_width > region_height:
            if 0.6<proportion<3.0:
            #if True:
                if proportion>1:
                    min_row=max_row-max_col+min_col+heightBias; 
                if min_row<0:
                    min_row=0
                max_row=max_row
                proportion=region_width/region_height
                button_like_objects.append(img[min_row:max_row,
                                          min_col:max_col])
                #buttonListArray[index].button_objects_cordinates.append((min_row,max_row))
                cordinate=[min_row,max_row]

                buttonListArray[index].button_objects_cordinates.append(cordinate)
                # let"s draw a red rectangle over those regions
    #####################進接過慮按鍵
    gapList=[]

    for index,buttonList in enumerate(buttonListArray):
        button_cordinates_list=buttonListArray[index].button_objects_cordinates
        for buttonIndex in range(len(button_cordinates_list)-1):
           gap=button_cordinates_list[buttonIndex+1][1] - button_cordinates_list[buttonIndex][1]
           gapList.append(gap)
           
    gapList.sort()
    """
    ####顯示上面append的那些變化過程再加上原本的圖像
    plt.figure("img")
    plt.imshow(img,cmap="gray")
    plt.xticks(np.linspace(0,imgWidth,6))

    showsubplot(showImages,imagesTitle)


    plt.show()
    """
    gapCoverCount=[]

    for index,item in enumerate(gapList):
        count=0
        maxGap=item+positionRange
        minGap=item-positionRange
        for index2,item2 in enumerate(gapList):
            if minGap<item2<maxGap:
                count=count+1
        gapCoverCount.append(count)
                
    ####找出gapCoverCount陣列裏面質最高的前兩個index
    biggestIndex,secondBigIndex=heapq.nlargest(2,range(len( gapCoverCount)),key=gapCoverCount.__getitem__)
    ####找出gapCoverConut裏面最高的兩個直
    biggest,secondBig=heapq.nlargest(2,gapCoverCount)
    if gapList[biggestIndex] > gapList[secondBigIndex]:
        ##給gap比較大的一個減小的直 因為通常比較小的gap才是對的
        if biggest-1 >= secondBig or gapList[biggestIndex]-positionRange < gapList[secondBigIndex]:
            standardGap=gapList[biggestIndex]
        else:
            standardGap=gapList[secondBigIndex]
    else:
        standardGap=gapList[biggestIndex]


    #standardGap=gapList[int(len(gapList)/2)]
    print("############################next img")

    #########################################################過濾掉數字鍵的上面or下面那些
    finalButtonList=[]
    noThresholdButtonList=[]

    for index,buttonList in enumerate(buttonListArray):
        print(index,"list##############")
        button_cordinates_list=buttonListArray[index].button_objects_cordinates
        print("button_cordinates_list",button_cordinates_list)
        rightButtonList=button_cordinates_list.copy()
        for buttonIndex in range(len(button_cordinates_list)):
            #if final button
            if buttonIndex==len(button_cordinates_list)-1:
                if button_cordinates_list[buttonIndex][1]-button_cordinates_list[buttonIndex-1][1]>standardGap+deleteRange or\
                        button_cordinates_list[buttonIndex][1]-button_cordinates_list[buttonIndex-1][1]<standardGap-deleteRange:
                    try: 
                        rightButtonList.remove(button_cordinates_list[buttonIndex])
                        print("delete final button",button_cordinates_list[buttonIndex])
                    except:
                        pass

            else:
                nextButtonRow= button_cordinates_list[buttonIndex+1][1]
                currentButtonRow=button_cordinates_list[buttonIndex][1]
                gap=nextButtonRow-currentButtonRow
                # if first button
                try:
                    check=1
                    button_cordinates_list[buttonIndex+2]
                except:
                    check=0
                    ##delete first button
                if buttonIndex==0:
                    if check==1 and  (gap<standardGap-deleteRange or gap>standardGap+deleteRange):
                        #rightButtonList.remove(button_cordinates_list[buttonIndex])
                        print("delete first button",button_cordinates_list[buttonIndex])
                    #if check==1 and  (gap<standardGap-deleteRange or gap>standardGap+deleteRange )and standardGap-positionRange <  button_cordinates_list[buttonIndex+2][1]-nextButtonRow <standardGap+positionRange:
                        rightButtonList.remove(button_cordinates_list[buttonIndex])
                        #print("delete first button",button_cordinates_list[buttonIndex])
                ###進階的刪除(可刪除上下多個脫序的情況,但是不一定可以刪掉)
                """
                else: 
                    ##刪除錯誤的前幾個按鍵 錯誤的式上面的
                    if check==1 and (standardGap+deleteRange<gap or gap<standardGap-deleteRange) and\
                        standardGap-positionRange <  button_cordinates_list[buttonIndex+2][1]-nextButtonRow <standardGap+positionRange and\
                        ((int(gap/standardGap)*standardGap+standardGap)-gap >positionRange or\
                        gap-int(gap/standardGap)*standardGap > positionRange ):
                        if buttonIndex+1 < len(button_cordinates_list)/2:
                            for wrongButtonIndex in  range(buttonIndex+1):
                                try:
                                    rightButtonList.remove(button_cordinates_list[wrongButtonIndex])
                                    print("delete upper button {}".format(wrongButtonIndex),button_cordinates_list[wrongButtonIndex])
                                except:
                                    pass
                        ##刪除錯誤的後幾個按鍵 錯誤的式下面的 
                    try:
                        check=1
                        button_cordinates_list[buttonIndex-1]
                    except:
                        check=0
                        ##刪除錯誤的後幾個按鍵 錯誤的式下面的 
                    else:
                        if check==1 and (standardGap+deleteRange<gap or gap<standardGap-deleteRange) and\
                            standardGap-positionRange <  currentButtonRow-button_cordinates_list[buttonIndex-1][1] <standardGap+positionRange and \
                            ((int(gap/standardGap)*standardGap+standardGap)-gap >positionRange \
                            or gap-int(gap/standardGap)*standardGap > positionRange) :
                            if buttonIndex+1 > len(button_cordinates_list)/2:
                                for wrongButtonIndex in  range(buttonIndex+1,len(button_cordinates_list)):
                                    try:
                                        rightButtonList.remove(button_cordinates_list[wrongButtonIndex+1])
                                        print("delete down button {}".format(wrongButtonIndex),button_cordinates_list[wrongButtonIndex])
                                    except:
                                        pass
                """


        rightButtonListLen=len(rightButtonList)
        count=0
        wrongButtonList=[]
        ######處理再一個standard gap 裏面有多個button 的情況
        buttonIndex=0


        while buttonIndex < rightButtonListLen:
            checkGap=rightButtonList[buttonIndex][1]+standardGap+positionRange
                #掃描一個standardGap裏面有幾個button
            for checkButtonIndex in range(buttonIndex+1,rightButtonListLen):
                if rightButtonList[checkButtonIndex][1] <= checkGap:
                    count=count+1
            if count>1:
                for index in range(1,count):
                    wrongButtonList.append(rightButtonList[buttonIndex+index])
                
                buttonIndex=buttonIndex+count
            #如果少描到0個或是一個還是要讓他繼續
            else:
                buttonIndex=buttonIndex+1
            count=0
                

        #delete wrongButton
        for item in wrongButtonList:
            rightButtonList.remove(item)


            
    ######################每個確定是按鍵的處理
        buttonListArray[index].button_objects_cordinates=rightButtonList


        for buttonIndex,button in enumerate(rightButtonList):
            max_col=buttonListArray[index].rightCol
            min_col=buttonListArray[index].leftCol
            min_row,max_row=rightButtonList[buttonIndex]
            item=img[min_row:max_row,min_col:max_col]
            item=cv2.resize(item,(finalImgWidth,finalImgWidth))
            threshold=threshold_otsu(item)
            ret , item= cv2.threshold(item,threshold,255,cv2.THRESH_BINARY)
            ################affine img
            min_row=min_row-affineOffset;max_row=max_row+affineOffset
            min_col=min_col-affineOffset;max_col=max_col+affineOffset
            ###如果超出範圍
            """
            if min_row < 0:
                min_row=0
            if max_row > imgHeight:
                max_row = imgHeight 
            if min_col < 0:
                min_col=0
            if max_col > imgWidth:
                max_col=imgWidth
            affineItem=xySobel[min_row:max_row,min_col:max_col]

            affineItem=np.array(affineItem)
            print("affineitemshape",min_row,max_row,min_col,max_col)
            #affineItem=cv2.resize(affineItem,(finalImgWidth,finalImgWidth))
            threshold=threshold_otsu(affineItem)
            ret , affineItem= cv2.threshold(affineItem,threshold,255,cv2.THRESH_BINARY)
            affineItemList.append(affineItem)
            """

            finalButtonList.append(item)
            rectBorder = patches.Rectangle((min_col, min_row), max_col-min_col, max_row-min_row, edgecolor="red", linewidth=2, fill=False)
            #&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
            #ax1.add_patch(rectBorder)
    #######################確定的按鍵裡面的處理 (將數字和背景不要的干擾分開)
    #showsubplot(affineItemList)
    print("@@@@@@@")
    print("finalbuttonlist pixel max value",np.max(finalButtonList))
    print("finalButtonList shape",finalButtonList[0].shape)
    totalPixelSum=0
    
    ##判斷這張button以黑為背景還是白
    for index,item in enumerate(finalButtonList):
        pixelSum=np.sum(item,axis=0)
        totalPixelSum=totalPixelSum+np.sum(pixelSum)
    
    oneButtonOneDigitList=[]
    ################察看一下mnist裡面的digit屬性

    """
    test=[]
    for index , item in enumerate(batch_xs):
        item=np.reshape(item,(finalImgWidth,finalImgWidth))
        threshold=threshold_otsu(item)
        ret , itemThreshold= cv2.threshold(item,threshold,255,cv2.THRESH_BINARY)
        test.append(itemThreshold)
    
    finalButtonList=test
    """
    ###############
    titleList=[]
    fbuttonList=[]
    #fbuttonlist 是 fButton的陣列 他主要功能式儲存一個button有多個digit的情況
    hasInvertToWhite=0 

    #如果背景式白色的話舊把他轉成黑色字體轉成白色
    for index, item in enumerate(finalButtonList):
        
        if totalPixelSum > 255*len(finalButtonList)*finalImgOneDimLen/2:
            #除與255是因為uint8 -1 的話位元一齣變成255 所以要處以255讓他變成1
            item=255-item
            print("conver to white digit")
            hasInvertToWhite=1
            


        label_image = measure.label(item)
        plateDimension=(buttonMinWidth,buttonMaxWidth,buttonMinWidth,buttonMaxWidth)
        min_height,max_height,min_width,max_width=plateDimension
        regionCordinatesList=[]
        regionpropsList=regionprops(label_image)

        #一個按鍵裏面連通的region
        digitcount=0
        for index2 ,region in enumerate(regionpropsList):
            #widthBias=int(region_width/8);heightBias=int(region_height/8)
            widthBias=0;heightBias=0

            min_row, min_col, max_row, max_col = region.bbox
            region_height = max_row - min_row
            region_width = max_col - min_col
            ##origin proportion
            proportion=region_height/region_width
            ##update row and proportion
            if 0.6< proportion  and 70 < region.area:
                regionCordinates=[min_row,max_row,min_col,max_col]
                regionCordinatesList.append(regionCordinates)
                digitcount=digitcount+1
            
        #如果因為條件太嚴格所以沒抓到region 就放寬
        if digitcount==0:
            for index2 ,region in enumerate(regionpropsList):
                #widthBias=int(region_width/8);heightBias=int(region_height/8)
                widthBias=0;heightBias=0

                min_row, min_col, max_row, max_col = region.bbox
                region_height = max_row - min_row
                region_width = max_col - min_col
                ##origin proportion
                proportion=region_height/region_width
                ##update row and proportion
                regionCordinates=[min_row,max_row,min_col,max_col]
                regionCordinatesList.append(regionCordinates)

        regionToCenterDistanceList=[]
        for index2 ,item2 in enumerate(regionCordinatesList):
            distance=0
            MiddlePoint=[]
            MiddlePoint.append([item2[0],(item2[2]+item2[3])/2])
            MiddlePoint.append([item2[1],(item2[2]+item2[3])/2])
            MiddlePoint.append([(item2[0]+item2[1])/2,item2[2]])
            MiddlePoint.append([(item2[0]+item2[1])/2,item2[3]])
            for item3 in MiddlePoint:
                for item4 in item3:
                    item4=item4-14
                    distance=distance+item4**2
            regionToCenterDistanceList.append(distance)

        ####找出gapCoverCount陣列裏面質最高的index
        try:
            smallestIndex,second=heapq.nsmallest(2,range(len( regionToCenterDistanceList)),key=regionToCenterDistanceList.__getitem__)
            smallestIndex=smallestIndex[0]
            second=second[0]
        except:
            smallestIndex=heapq.nsmallest(1,range(len( regionToCenterDistanceList)),key=regionToCenterDistanceList.__getitem__)
            smallestIndex=smallestIndex[0]
            second=-1
        fbutton=Button()

        if second!=-1 and regionCordinatesList[second][0] +positionRange> regionCordinatesList[smallestIndex][0] > regionCordinatesList[second][0] - positionRange and\
            regionCordinatesList[second][1] +positionRange> regionCordinatesList[smallestIndex][1] > regionCordinatesList[second][1] - positionRange:
            if regionCordinatesList[smallestIndex][2] < regionCordinatesList[second][2]:
                fbutton.digitListCordinates.append(regionCordinatesList[smallestIndex])
                fbutton.digitListCordinates.append(regionCordinatesList[second])
            else:
                fbutton.digitListCordinates.append(regionCordinatesList[second])
                fbutton.digitListCordinates.append(regionCordinatesList[smallestIndex])
        else:
            fbutton.digitListCordinates.append(regionCordinatesList[smallestIndex])
            
        ##################################決定要傳進去被擷取出最後的digit的按鈕
        
        
        

        fbutton.originButton=item
        print("item.shape",item.shape)
        fbuttonList.append(fbutton)
    finalButtonList=[]

    #loop fbutton 
    for fbuttonIndex,fbutton in enumerate(fbuttonList):
        print("fbuttonindex",fbuttonIndex)
        item=fbuttonList[fbuttonIndex].originButton
        ############################################這邊拿fbutton的digitListCordiantes來做for回圈是因為fbuttonList[fbutytonIndex]
        #############################只是單一個按鍵而已，但是digitListCordinates裏面存了一個按鍵裏面多個digit的座標
        for digitIndex,digit in enumerate(fbuttonList[fbuttonIndex].digitListCordinates):
            print("digitIndex",digitIndex)
            digitCordinates=fbuttonList[fbuttonIndex].digitListCordinates[digitIndex]

            newImg=np.zeros([finalImgWidth,finalImgWidth])
            digitMinRow,digitMaxRow,digitMinCol,digitMaxCol=digitCordinates
            item=np.array(item)
            ##從這邊主角變成newimg
            digitWidth=digitMaxCol-digitMinCol
            digitHeight=digitMaxRow-digitMinRow
            ##################################################img 是 裁剪下來要的那個digit
            
            img=item[digitMinRow:digitMaxRow,digitMinCol:digitMaxCol]

            #convert to Image object
            img = Image.fromarray(img)

            digitHeight= 20

            hpercent = (digitHeight/ float(img.size[1]))
            digitWidth= int((float(img.size[0]) * float(hpercent)))
            if digitWidth > 28:
                digitWidth=28
            img = img.resize((digitWidth, digitHeight), PIL.Image.ANTIALIAS)
            # convert back to np array
            img=np.array(img)
            newImgMinCol=14-int(digitWidth/2)
            newImgMaxCol=newImgMinCol+digitWidth
            newImgMinRow=14-int(digitHeight/2)
            newImgMaxRow=newImgMinRow+digitHeight


            newImg[newImgMinRow:newImgMaxRow,newImgMinCol:newImgMaxCol]=img
            
            digitArea=np.sum(newImg,axis=1)
            digitArea=np.sum(digitArea)
            print("digit area",digitArea)
            print("prportion",digitHeight/digitWidth)

            """


            #erosion
            
            #kernelWidth=2
            #kernelVertical = np.ones((1,kernelWidth),np.uint8)
            #kernelHorizational = np.ones((kernelWidth,1),np.uint8)
            #opening
            #newImg= cv2.erode(newImg,kernelVertical,iterations=1)
            #newImg= cv2.erode(newImg,kernelHorizational,iterations=1)
            """
            """
            skeletonizeImg=skeletonize(newImg)
            newImg=skeletonizeImg
            """
            

            newImg=newImg/255
            newImg=np.reshape(newImg,finalImgOneDimLen)
            fbuttonList[fbuttonIndex].digitImgList.append(newImg)
            print("newImg.shape",newImg.shape)
            finalButtonList.append(newImg)
            region=int((digitMaxRow-digitMinRow)*(digitMaxCol-digitMinCol))
            proportion=(digitMaxRow-digitMinRow)/(digitMaxCol-digitMinCol)


    ####################################
    """fbuttonlist 是 fButton的陣列 他主要功能式儲存一個button有多個digit的情況

    mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
    batch_xs, batch_ys = mnist.train.next_batch(10)

    """
    """
    result=readModel(batch_xs)
    showsubplot(batch_xs,predict)
    """

    result=readModel(finalButtonList)
    showsubplot(finalButtonList,result)
    #showsubplot(finalButtonList)
    plt.show()
