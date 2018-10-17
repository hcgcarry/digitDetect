import numpy as np
import matplotlib.pyplot as plt
from math import sqrt

def showsubplot(images,imagesTitle=None):

    row=int(len(images)/6)+1
    #list轉換成np.array
    try:
        if isinstance(images, list):
            images=np.array(images)
            
        #如果近來的事背flatten的images
        if len(images[0].shape) ==1:
            imagesSize=int(sqrt(images.shape[1]))
            images=np.reshape(images,(images.shape[0],imagesSize,imagesSize))
    except:
        print('reshape has something wrong')
    if imagesTitle is None:
        imagesTitle=list(range(len(images)))
    
    plt.figure(figsize=(30,30))

    for index in range(len(images)):
        plt.subplot(row,6,index+1)
        plt.imshow(images[index],cmap='gray')
        
        plt.title(imagesTitle[index])
        #ticks

