from detectImgModel.imgHandler import ImageHandler
from detectImgModel.__init__ import *

class SquareImgHandler(ImageHandler):

  
    def setImgProperty(self):
        super(SquareImgHandler,self).setImgProperty()
        self.squareMinCenterDistance=int(self.imgWidth/10)
  
  

    def HoughSquares(self):
        self.squareList= cv2.HoughSquares(self.currentImg,cv2.HOUGH_GRADIENT,1,self.squareMinCenterDistance,\
        param1=80,param2=100,minRadius=self.squareMinRadius,maxRadius=self.squareMaxRadius)
        self.numOfSquares=len(self.squareList[0])
    def printSquares(self):
        #輸出檢測到圓的個數
        print(len(self.squareList[0]))

        print("-------------我是條分割線-----------------")
        #根據檢測到圓的信息，畫出每一個圓
        for square in self.squareList[0]:
            #圓的基本信息
            #print(square[2])
            #坐標行列(就是圓心)
            x=int(square[0])
            y=int(square[1])
            #半徑
            r=int(square[2])
            #在原圖用指定顏色圈出圓，參數設定為int所以圈畫存在誤差
            img=cv2.circle(self.originImg,(x,y),r,(0,255,255),5)
        #顯示新圖像
        plt.imshow(img,cmap='gray')
        plt.show()
    

     

        
    def cutButton(self,ax1):
        for index,square in enumerate(self.squareList[0]):
            x=square[0]
            y=square[1]
            radius=square[2]
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
        # find contours in the thresholded image and initialize the
        
        

   

        

