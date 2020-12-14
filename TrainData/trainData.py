import numpy as np
import cv2
import tensorflow as tf
import time
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout

features = np.load('../LoadData/features.npy')
labels = np.load('../LoadData/labels.npy')

# # Checking if data has been loaded perfectly
# print("Total features accessed : ", len(features))
# print("Total labels accessed : ", len(labels))
# print("Shape of image : ", features[0].shape)# # Checking if data has been loaded perfectly
# print("Total features accessed : ", len(features))
# print("Total labels accessed : ", len(labels))
# print("Shape of image : ", features[0].shape)

f_train, f_test, l_train, l_test = train_test_split(features, labels, test_size=0.2, random_state=0)

# print("Shape of f_train : ", f_train.shape)
# print("Shape of l_train : ", l_train.shape)
# print("Shape of f_test : ", f_test.shape)
# print("Shape of l_test : ", l_test.shape)

l_train = to_categorical(l_train, 43)
l_test = to_categorical(l_test, 43)

model = Sequential()

model.add(Conv2D(filters=64, kernel_size=(3,3),activation='relu', input_shape= (50, 50, 3)))
model.add(MaxPool2D(pool_size=(2, 2)))

#model.add(Dropout(rate=0.25))

model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))
model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Dropout(rate=0.25))

model.add(Flatten())
model.add(Dense(512, activation='relu'))

model.add(Dropout(rate=0.25))
model.add(Dense(43, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(f_train, l_train, batch_size=32, epochs=3, validation_data=(f_test, l_test))

val_loss,val_acc=model.evaluate(f_test,l_test)
    
model.save("../TrainedModels/L={}-A={}.model".format(int(val_loss*100), int(val_acc*100)))