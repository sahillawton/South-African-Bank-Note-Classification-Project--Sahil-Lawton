import tkinter as tk
from tkinter import *
import time
from tkinter import filedialog
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from tkinter import *
from PIL import ImageTk,Image

#This is the main file that brings together the models, and
#image preprocessing for our final project

root = tk.Tk()
# dimensions of our images
img_width, img_height = 150, 150

# load the model we saved
model = load_model('banknoteauthdentest.h5')
link =""


canvas1 = tk.Canvas(root, width=800, height=600, relief='raised')
canvas1.pack()

label1 = tk.Label(root, text='Please click the button to input an image')
label1.config(font=('helvetica', 14))
canvas1.create_window(400, 175, window=label1)


def processImage():




    input = filedialog.askopenfile(initialdir="/")
    print(input.name)
    img = image.load_img(input.name
                         ,
                         target_size=(img_width, img_height))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    images = np.vstack([x])
    start_time = time.time()
    classes = model.predict_classes(images, batch_size=10)
    print(classes)

    # predicting multiple images at once
    img = image.load_img(input.name
                         ,
                         target_size=(img_width, img_height))
    y = image.img_to_array(img)
    y = np.expand_dims(y, axis=0)

    # pass the list of multiple images np.vstack()
    images = np.vstack([x, y])
    classes = model.predict_classes(images, batch_size=10)

    # print the classes, the images belong to
    print(classes)
    print(classes[0])

    prediction = 'cant process'
    if classes[0] == 0:
        prediction = 'real fifty'
    elif classes[0] == 1:
        prediction = 'fake fifty'
    elif classes[0] == 2:
        prediction = 'real hundred'
    elif classes[0] == 3:
        prediction = 'fake hundred'
    elif classes[0] == 4:
        prediction = 'real ten'
    elif classes[0] == 5:
        prediction = 'fake ten'
    elif classes[0] == 6:
        prediction = 'real twenty'
    elif classes[0] == 7:
        prediction = 'fake twenty'
    elif classes[0] == 8:
        prediction = 'real two hundred'
    elif classes[0] == 9:
        prediction = 'fake two hundred'
    print(prediction)

    img = ImageTk.PhotoImage(Image.open(input.name))
    output = "This is a "+prediction+ " rand note. The time to process was "+str(time.time()-start_time)+ " seconds"
    label4 = tk.Label(root, text=output, font=('helvetica', 10, 'bold'))
    canvas1.create_window(400, 360, window=label4)


button1 = tk.Button(text='Input Image', command=processImage, bg='brown', fg='white',
                    font=('helvetica', 9, 'bold'))
canvas1.create_window(400, 330, window=button1)

root.mainloop()

