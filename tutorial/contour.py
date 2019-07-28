import numpy as np
import cv2

img = cv2.imread("./evpicture/erosion.jpg")
imgray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret,thresh = cv2.threshold(imgray,127,255,0)
image ,contours,hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
#绘制独立轮廓，如第四个轮廓
#imag = cv2.drawContour(img,contours,-1,(0,255,0),3)
#但是大多数时候，下面方法更有用
height=int(img.shape[1]/5)
width=int(img.shape[0]/5)
cv2.drawContours(img,contours,-1,(0,255,0),3)
img=cv2.resize(img,(height,width))
thresh=cv2.resize(thresh,(height,width))

cv2.imshow("thresh",thresh)
cv2.imshow("imag",img)
cv2.waitKey(0)
cv2.destroyAllWindows()
cnt=contours[0]
M=cv2.moments(cnt)
print(M)
