import numpy as np
import cv2
  
img= cv2.imread('test3.png')

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret, binary = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)

_, contours, hier = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

cv2.drawContours(img,contours,-1,(0,0,255),3)

cnt = contours[1]
print ("there are " + str(len(cnt)) + " points in contours[0]")
print (cnt)
 
cnt = contours[0]
print ("there are " + str(len(cnt)) + " points in contours[1]")
print (cnt)

approx = cv2.approxPolyDP(cnt,30,True)
print ("after approx, there are " + str(len(approx)) + " points")
print (approx)
cv2.drawContours(img,[approx],0,(0,255,0),-1)
 
cv2.imshow("img", img)
cv2.waitKey(0)
