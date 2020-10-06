# #Please note that the code in this .py file was run in Jupyter notebook
#Select all and type control + / to un-comment this section
# #for the creation of a model, hence the inline imports. This code will
# #only run in Jupyter notebook
# #It is put into the same folder as the other .py files for the convenience
# #of searching for the code
# import numpy as np
# import pandas as pd
# %matplotlib inline
# import matplotlib as mpl
# import matplotlib.pyplot as plt
# import os
#
# #The keras library allows for an easier building of the
# #Neural network model
# import tensorflow as tf
# from tensorflow import keras
#
# #The training directories are located on my PC. Please change it according to your dataset
# train_dir = r'C:\Users\Sahil\Desktop\Tries\ToBeProcessedDataset\train'
# validation_dir = r'C:\Users\Sahil\Desktop\Tries\ToBeProcessedDataset\validation'
# test_dir = r'C:\Users\Sahil\Desktop\Tries\ToBeProcessedDataset\test'
#
# #The ImageDataGenerator library
# #allows for the augmenting of data
# #so that there isnt an
# #overfitting of of the training data to the validation
# #data
# #Some steps include
# #Data Preprocessing
# #Read the picture files
# #Decode the JPEG content to RGB grid of pixels
# #Convert these into floating point tensors
# #Rescale the pixel values(between 0 and 255) to the [0,1] interval
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# #height shift in the range 20 percent
# train_datagen = ImageDataGenerator(
# rescale = 1./255,
# rotation_range = 40,
# width_shift_range = 0.2,
# height_shift_range = 0.2,
# shear_range = 0.2,
# zoom_range = 0.2,
# horizontal_flip = True)
# test_datagen = ImageDataGenerator(rescale = 1./ 255)
# train_generator = train_datagen.flow_from_directory(
# train_dir,
# target_size=(150,150),
# batch_size = 20,
# class_mode = 'categorical')
# validation_generator = test_datagen.flow_from_directory(
# validation_dir,
# target_size =(150, 150),
# batch_size=2,
# class_mode = 'categorical')
# #link: https://keras.io/preprocessing/image/ for further reading
#
# #Our model uses a VGG16 transfer learning model which won the ILSVRC competion
# #We initially used a custom built CNN architecture,
# #however, the VGG16 architecture showed far higher accuracies than the
# #custom built architecture
# #We only use the convolutional base
# #as we prefer to use our own classification layer
# #The input image shape is allowed to be 150*150 due to the
# #150 input neurons
# from tensorflow.keras.applications import VGG16
# conv_base = VGG16(weights='imagenet',
# include_top= False,
# input_shape =(150,150,3))
#
# #This method gives an indication of the
# #look of the nn structure, and
# #the arrangement of hyperparameter
# conv_base.summary()
#
# from tensorflow.keras import layers
# from tensorflow.keras import models
#
# #We use the VGG16 base in a
# #sequential model for the base, and
# #add a flattening layer for the dense
# #layers to be compatible with the conv base input
#
# model = models.Sequential()
#
# model.add(conv_base)
# model.add(layers.Flatten())
# model.add(layers.Dense(256, activation='relu'))
# model.add(layers.Dense(256, activation='relu'))
# model.add(layers.Dense(256, activation='relu'))
# model.add(layers.Dense(256, activation='relu'))
# model.add(layers.Dense(10, activation='softmax'))
#
# model.summary()
#
#
# #We use the categorical crossentropy in optimizing our model
# #The acc and AUC metrics represent accuracy and area under curve
# from tensorflow.keras import optimizers
# model.compile(loss='categorical_crossentropy',
# optimizer= optimizers.RMSprop(lr= 2e-5),
# metrics = ['acc', tf.keras.metrics.AUC()])
#
# #can also use save best here, if you dont want to save all epochs
# checkpoint_cb = keras.callbacks.ModelCheckpoint("CNN_Project_Model-{epoch:02d}.h5")
#
#
# #This code starts the training of the network
# history = model.fit_generator(
# train_generator,
# steps_per_epoch=6,
# epochs =500,
# validation_data = validation_generator,
# validation_steps = 6,
# callbacks=[checkpoint_cb])
#
# #This code graphically represents the accuracy,
# #validation accuracy, validation loss etc,
# #as the model gets trained.
# #we see how the training data fits the validation
# #data and the accuracy increasing as time moves forward
# pd.DataFrame(history.history).plot(figsize = (8,5))
# plt.grid(True)
# plt.gca().set_ylim(0,1)
# plt.show()
#
#
# hist_df = pd.DataFrame(history.history)
# #save history variable into a csv file
# hist_csv_file = 'history.csv'
# with open(hist_csv_file, mode='w') as f:
#     hist_df.to_csv(f)
#
#
# #We were only calculating accuracies on validation set
# #lets see how it performs on test set
# #test_datagen is same object for validation. Reshapping data from 0 to 255 to 0 to 1
# test_generator = test_datagen.flow_from_directory(
# test_dir,
# target_size= (150,150),
# batch_size=2,
# class_mode = 'categorical')
#
#
# model.evaluate_generator(test_generator, steps = 2)
#
#
# #We save the trained network as an .h5
# #file here, to allow for the model to be used in other applications
# model.save('banknoteauthdentest.h5')
#
# from tensorflow.keras.preprocessing import image
# from tkinter import *
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing import image
# import numpy as np
# root = Tk()
#
# e = Entry(root, width =50, borderwidth =5)
# e.pack()
#
# # dimensions of our images    -----   are these then grayscale (black and white)?
# img_width, img_height = 150, 150
#
# # load the model we saved
# model = load_model('banknoteauthdentest.h5')
# link =""
#
# def myClick():
#     link = e.get()
#     link = link.replace('\\','/')
#     # predicting images
#     img = image.load_img(link
#         ,
#         target_size=(img_width, img_height))
#     x = image.img_to_array(img)
#     x = np.expand_dims(x, axis=0)
#
#     images = np.vstack([x])
#     classes = model.predict_classes(images, batch_size=10)
#     print(classes)
#
#     # predicting multiple images at once
#     img = image.load_img(link
#          ,
#         target_size=(img_width, img_height))
#     y = image.img_to_array(img)
#     y = np.expand_dims(y, axis=0)
#
#     # pass the list of multiple images np.vstack()
#     images = np.vstack([x, y])
#     classes = model.predict_classes(images, batch_size=10)
#
#     # print the classes, the images belong to
#     print(classes)
#     print(classes[0])
#
#     prediction = 'cant process'
#     if classes[0] == 0:
#         prediction = 'fifty'
#     elif classes[0] == 1:
#         prediction = 'fake fifty'
#     elif classes[0] == 2:
#         prediction = 'hundred'
#     elif classes[0] == 3:
#         prediction = 'fake hundred'
#     elif classes[0] == 4:
#         prediction = 'ten'
#     elif classes[0] == 5:
#         prediction = 'fake ten'
#     elif classes[0] == 6:
#         prediction = 'twenty'
#     elif classes[0] == 7:
#         prediction = 'fake twenty'
#     elif classes[0] == 8:
#         prediction = 'two hundred'
#     elif classes[0] == 9:
#         prediction = 'fake two hundred'
#     print(prediction)
#
#
#     myLabel = Label(root, text = prediction)
#     myLabel.pack()
#
#
#
#
# myButton = Button(root, text = "Process", command= myClick)
# myButton.pack()
#
# root.mainloop()