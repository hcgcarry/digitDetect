from detectImgModel.imgHandler import ImageHandler
from detectImgModel.__init__ import *
class CircleImgHandler(ImageHandler):

  
    def setImgProperty(self):
        super(CircleImgHandler,self).setImgProperty()
        self.circleMinRadius=int(self.imgWidth/40)
        self.circleMaxRadius=int(self.imgWidth/10)
        self.circleMinCenterDistance=int(self.imgWidth/10)
        self.cutButtonColBias=10
        self.cutButtonRowBias=10
  
  

    def HoughCircles(self):
        #調整parameter2可以調整視為元的難易度
        self.circleList= cv2.HoughCircles(self.currentImg,cv2.HOUGH_GRADIENT,1,self.circleMinCenterDistance,\
        param1=80,param2=90,minRadius=self.circleMinRadius,maxRadius=self.circleMaxRadius)
        if self.circleList is None:
            return -1
        else:
            self.numOfCircles=len(self.circleList[0])
            return self.numOfCircles
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
            min_col,max_col,min_row,max_row=self.cutButtonWidthAndHeightBias(min_col,max_col,min_row,max_row)
            print(min_col,max_col,min_row,max_row)
            button=self.grayImg[min_row:max_row,min_col:max_col]
            button=cv2.resize(button,(self.finalButtonWidth,self.finalButtonWidth))
            threshold=threshold_otsu(button)
            _, button= cv2.threshold(button,threshold,255,cv2.THRESH_BINARY)
            self.buttonList.append(button)
        
            #####plot button boundary
            ax1.imshow(self.binaryImg,cmap="gray")
            rectBorder = patches.Rectangle((min_col, min_row), max_col-min_col,\
             max_row-min_row, edgecolor="red", linewidth=2, fill=False)
            ax1.add_patch(rectBorder)
        
        
    def cutButtonWidthAndHeightBias(self,min_col,max_col,min_row,max_row):
        min_col=min_col+self.cutButtonColBias
        max_col=max_col-self.cutButtonColBias
        min_row=min_row+self.cutButtonRowBias
        max_row=max_row-self.cutButtonRowBias
        return min_col,max_col,min_row,max_row

    ##減小切的範圍

   

        

