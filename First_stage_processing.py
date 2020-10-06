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

#This method allows for the creation of Wiener filtered images in a specific folder with a specific name
#The method below is replicated across all bank note denomanations to populate our training, validation
#and testing set
#Please note that the contrast limiting adaptive histogram equilization is done separately in the Second_stage_processing .py
#file due to the demand that doing both operations simultaneously had on computational resources

directory = r'C:\Users\Sahil\Desktop\ToBeProcessedDataset\validation\ten'

image_paths = []
for entry in os.scandir(directory):
    if (entry.path.endswith(".jpg")
            or entry.path.endswith(".png")) and entry.is_file():
        image_paths.append(entry.path)
        #print(li.pop())

count = 0
for path in image_paths:
    # Load image and convert it to gray scale
    file_name = os.path.join(path)
    img = rgb2gray(plt.imread(file_name))
    # Apply Wiener Filter
    filtered_image = wiener_filter(img, gaussian_kernel(9), K=0.5)
    cv2.imwrite(r'C:\Users\Sahil\Desktop\ToBeProcessedDataset\validation\ten\firststagepreprocessed_image' + str(count) + '.jpg', filtered_image)
    count+=1


