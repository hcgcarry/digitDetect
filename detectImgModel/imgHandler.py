from detectImgModel.__init__ import *
from shapedetector import ShapeDetector

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
    def sobel(self,xWeight,yWeight):
        x= cv2.Sobel(self.currentImg,cv2.CV_64F,1,0,ksize=self.sobelSize)
        y= cv2.Sobel(self.currentImg,cv2.CV_64F,0,1,ksize=self.sobelSize)
        absx=cv2.convertScaleAbs(x) 
        absy=cv2.convertScaleAbs(y) 
        self.sobelImage=cv2.addWeighted(absx,xWeight,absy,yWeight,0)
        self.currentImg=self.sobelImage

    def binary(self):
        threshold=threshold_otsu(self.currentImg)
        _,self.binaryImg= cv2.threshold(self.currentImg,threshold,1,cv2.THRESH_BINARY)
        self.currentImg=self.binaryImg
    

     

    def printCurrentImg(self):
        plt.figure()
        plt.imshow(self.currentImg)
        plt.show()
        
   

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
        button=np.array(button)
        ##################################################從button上面才切下digit
        digit=button[digitMinRow:digitMaxRow,digitMinCol:digitMaxCol]

        #convert to Image object
        digit= Image.fromarray(digit) 

        ####將digit縮放成minus那個資料庫的digit大小
        hpercent = (self.digitHeight/ float(digit.size[1]))
        digitWidth= int((float(digit.size[0]) * float(hpercent)))
        if digitWidth > self.finalButtonWidth:
            digitWidth=self.finalButtonWidth

        digit= digit.resize((digitWidth,self.digitHeight), PIL.Image.ANTIALIAS)
        # convert back to np array
        digit=np.array(digit)
        finalButtonMinCol=14-int(digitWidth/2)
        finalButtonMaxCol=finalButtonMinCol+digitWidth
        finalButtonMinRow=14-int(self.digitHeight/2)
        finalButtonMaxRow=finalButtonMinRow+self.digitHeight
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
    def showCurrentImg(self):
        images=[]
        images.append(self.currentImg)
        self.showsubplot(images)

    def showGray_Soble_binary_Img(self):
        images=[]
        images.append(self.grayImg)
        images.append(self.sobelImage)
        images.append(self.binaryImg)
        self.showsubplot(images)

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
        plt.show()

    def detectContours(self):
        ###########################不好用
        # find contours in the thresholded image and initialize the
        # shape detector
        
        resizedImg = imutils.resize(self.currentImg, width=300)
        #ratio = .shape[0] / float(resizedImg.shape[0])
        print(resizedImg.shape[0])

        #######get contour
        cnts = cv2.findContours(resizedImg.copy(), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        #######

        ############detect shape and show on image
        sd = ShapeDetector()
        '''
        for index,c in enumerate(cnts):
            print("index:",index)
            print("len:",len(c))
        '''
        # loop over the contours
        for c in cnts:
            print("c:",c)
            # compute the center of the contour, then detect the name of the
            # shape using only the contour
            M = cv2.moments(c)
            print(M)
            if M["m00"] ==0:
                continue
            cX = int((M["m10"] / M["m00"]))
            cY = int((M["m01"] / M["m00"]))
            shape = sd.detect(c)

            # multiply the contour (x, y)-coordinates by the resize ratio,
            # then draw the contours and the name of the shape on the image
            c = c.astype("float")
            #c *= ratio
            c = c.astype("int")
            cv2.drawContours(resizedImg, [c], -1, (0, 255, 0), 2)
            #cv2.putText(resizedImg, shape, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            # show the output image
            plt.imshow(resizedImg)
            #cv2.imshow("Image", resizedImg)
            #cv2.waitKey(0)


        plt.show()
            

