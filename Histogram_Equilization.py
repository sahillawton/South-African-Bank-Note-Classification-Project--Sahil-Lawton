# import Opencv
#Use this easy method to perform histogram equilization
import cv2

# import Numpy 
import numpy as np

# read a image using imread 
img = cv2.imread(r'C:\Users\Sahil\Desktop\BankNotesDataset\train\ten\IMG_20200604_143353.jpg', 0)

equ = cv2.equalizeHist(img)


res = np.hstack((img, equ))

cv2.imwrite(r'C:\Users\Sahil\Documents\Adobe\EquilizedImag23e.jpg', equ)