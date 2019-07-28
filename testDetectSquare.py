from detectImgModel.imgHandler import ImageHandler
from detectImgModel.squareImgHandler import SquareImgHandler
import matplotlib.pyplot as plt
def testDetectSquare(imagePath):
    medianTimes=2
    showImages=[]
    Img=SquareImgHandler(imagePath)
    Img.bgr2Gray()
    Img.median(medianTimes)
    showImages.append(Img.currentImg)
    Img.sobel(1,1)
    showImages.append(Img.currentImg)
    #Img.canny()
    Img.binary()
    showImages.append(Img.currentImg)
    Img.showsubplot(showImages)
    Img.detectContours()
    #Img.showGray_Soble_binary_Img()
    '''
    Img.HoughCircles()
    #Img.printCircles()
    Img.sobel()
    Img.binary()
    fig, (ax1) = plt.subplots(1)
    Img.cutButton(ax1)
    plt.show()
    Img.checkButtonBlackBackGround()
    Img.buttonListDeleteWrongPattern()
    Img.detectDigitNumber()
    '''