#This code is is the implementation of contrast limiting adaptive histogram equilization.
#Contrast limiting AHE allows for a greater degree of contrast and definition in our output
#Image in comparision to normal histogram equilization
import numpy as np
import cv2 as cv
img = cv.imread(r'C:\Users\Sahil\Desktop\BankNotesDataset\train\ten\IMG_20200604_143353.jpg',0)
# create a CLAHE object (Arguments are optional).
clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
cl1 = clahe.apply(img)
cv.imwrite(r'C:\Users\Sahil\Documents\Adobe\EquilizedImag24e.jpg',cl1)