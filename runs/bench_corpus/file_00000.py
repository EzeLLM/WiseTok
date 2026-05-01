# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 12:53:34 2019

@author: Internship010
"""

import tensorflow as tf

#Importing Model and layers

from keras.models import Sequential

from keras.layers import Conv2D

from keras.layers import MaxPooling2D

from keras.layers import Flatten

from keras.layers import Dense

#from keras.models import model_from_yaml

Classifier = Sequential()

#Layering

Classifier.add(Conv2D(32,(3,3),input_shape=(64,64,3),activation="relu")) #(Output dim,kernel size,stride)

Classifier.add(MaxPooling2D(pool_size=(2,2)))

Classifier.add(Flatten())

Classifier.add(Dense(activation='relu',units=128))

Classifier.add(Dense(activation='sigmoid',units=1))#Sigmoid is used as activation for getting probability since we need probability of the image being with a dog or a cat



#Compiling the Whole Network

Classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])


from keras.preprocessing.image import ImageDataGenerator

#Generating Object for trainingset

train_datagen = ImageDataGenerator(rescale=1./255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)

#Generating Object for testingset

test_datagen = ImageDataGenerator(rescale=1./255)

#Generating

train_set = train_datagen.flow_from_directory('training_set',target_size=(64,64),batch_size=32,class_mode='binary')

test_set = test_datagen.flow_from_directory('test_set',target_size=(64,64),batch_size=32,class_mode='binary')

Classifier.fit_generator(train_set,steps_per_epoch=20,epochs=2,validation_data=test_set,validation_steps=20)

import numpy as np

from keras.preprocessing import image


test_image = image.load_img('men1.jpg',target_size=(64,64))

test_image = image.img_to_array(test_image)

test_image = np.expand_dims(test_image,axis=0)



result = Classifier.predict(test_image)
print(result[0][0])
