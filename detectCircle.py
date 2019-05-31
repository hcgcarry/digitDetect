import  cv2
#載入並顯示圖片
img=cv2.imread("1.jpg")
cv2.imshow("1",img)
#降噪（模糊處理用來減少瑕疵點）
result = cv2.blur(img, (5,5))
cv2.imshow("2",result)
#灰度化,就是去色（類似老式照片）
gray=cv2.cvtColor(result,cv2.COLOR_BGR2GRAY)
cv2.imshow("3",gray)

#param1的具體實現，用於邊緣檢測    
canny = cv2.Canny(img, 40, 80)   
cv2.imshow("4", canny)  


#霍夫變換圓檢測
circles= cv2.HoughCircles(gray,cv2.HOUGH_GRADIENT,1,50,param1=80,param2=30,minRadius=15,maxRadius=20)
#輸出返回值，方便查看類型
print(circles)
'''

#輸出檢測到圓的個數
print(len(circles[0]))

print("-------------我是條分割線-----------------")
#根據檢測到圓的信息，畫出每一個圓
for circle in circles[0]:
    #圓的基本信息
    print(circle[2])
    #坐標行列(就是圓心)
    x=int(circle[0])
    y=int(circle[1])
    #半徑
    r=int(circle[2])
    #在原圖用指定顏色圈出圓，參數設定為int所以圈畫存在誤差
    img=cv2.circle(img,(x,y),r,(0,0,255),1,8,0)
#顯示新圖像
cv2.imshow("5",img)

#按任意鍵退出
'''
cv2.waitKey(0)
cv2.destroyAllWindows()