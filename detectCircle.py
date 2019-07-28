from detectImgModel.imgHandler import ImageHandler
from detectImgModel.circleImgHandler import CircleImgHandler
import matplotlib.pyplot as plt
def detectCircle(imagePath):
    medianTimes=2
    Img=CircleImgHandler(imagePath)
    Img.bgr2Gray()
    #plt.imshow(Img.currentImg,cmap='gray')
    #plt.show()
    Img.median(medianTimes)
    numOfCircles=Img.HoughCircles()
    if numOfCircles < 2:
        print("there have no circle ,assume square button")
        return -1
    else:
        #Img.printCircles()
        Img.sobel(1,1)
        Img.binary()
        fig, (ax1) = plt.subplots(1)
        Img.cutButton(ax1)
        plt.show()
        Img.checkButtonBlackBackGround()
        Img.buttonListDeleteWrongPattern()
        Img.detectDigitNumber()