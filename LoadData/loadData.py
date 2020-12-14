import numpy as np
import cv2
import os

features = []
labels = []
classes = 43

for i in range(classes):
    path = os.path.join('../TrainImages',str(i))
    images = os.listdir(path)
    for imgName in images:
        try:
            img_path = os.path.join(path,imgName)
            image = cv2.imread(img_path)
            image = cv2.resize(image,(50,50))
            image = np.array(image)
            features.append(image)
            labels.append(i)
        except Exception as e:
            print(e)
            
# print("Image in list : ", features[100])
# print("Present folder of image in list : ", labels[3000])
features = np.array(features)
labels = np.array(labels)
# print("Total features accessed : ", len(features))
# print("Total labels accessed : ", len(labels))
# print("Shape of image : ", features[0].shape)

np.save('features', features)
np.save('labels', labels)