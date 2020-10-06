import os
import matplotlib.pyplot as plt
from Median_Filter import median_filter, add_gaussian_noise, rgb2gray
from Wiener_Filter import blur, gaussian_kernel, wiener_filter
import os
import numpy as np
from numpy.fft import fft2, ifft2
from scipy.signal import gaussian, convolve2d
import matplotlib.pyplot as plt
import cv2

#
directory = r'C:\Users\Sahil\Desktop\Backup Bank Note Folders\ToBeProcessedDataset\train\ten'

image_paths = []
for entry in os.scandir(directory):
    if (entry.path.endswith(".jpg")
            or entry.path.endswith(".png")) and entry.is_file():
        image_paths.append(entry.path)
        #print(li.pop())

count = 0
for path in image_paths:
    # read a image using imread
    img = cv2.imread(path, 0)

    print(path)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl1 = clahe.apply(img)
    cv2.imwrite(r'C:\Users\Sahil\Desktop\Backup Bank Note Folders\ToBeProcessedDataset\train\ten\image'+str(count)+'.jpg', equ)
    count+=1

