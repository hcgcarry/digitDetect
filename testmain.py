import glob
from detectImgModel.imgHandler import ImageHandler
from detectImgModel.squareImgHandler import SquareImgHandler
from detectImgModel.circleImgHandler import CircleImgHandler
from detectSquare import detectSquare
from detectCircle import detectCircle
from testDetectSquare import testDetectSquare
from old.generalDetectButton import generalDetectButton
#############################################################
picture_path='evpicture/total/all'
#picture_path='evpicture/circle/test'
############################################################

for imageName in glob.glob('{}/*.jpg'.format(picture_path)):
    #detectSquare(imagePath)
    #testDetectSquare(imagePath)
    print("imageName",imageName)
    status=detectCircle(imageName)
    if status==-1:
        generalDetectButton(imageName)
        
        


    
    
