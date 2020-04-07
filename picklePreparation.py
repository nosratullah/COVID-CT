import pickle
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random

dataDir = 'COVID-CT-master/Images-processed/'
categories = ['CT_COVID', 'CT_NonCOVID']

training_data = []
img_size = 224

def create_training_data():
    img_size = 224
    for category in categories:
        path = os.path.join(dataDir, category)
        class_num = categories.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (img_size, img_size))
                training_data.append([new_array, class_num])
            except Exception as e:
                pass

create_training_data()

random.shuffle(training_data)

X = []
Y = []

for features, label in training_data:
    X.append(features)
    Y.append(label)

X = np.array(X).reshape(-1, img_size, img_size, 1)
Y = np.array(Y)

pickle_out = open("{}COVID_NonCOVID.pickle".format(dataDir), 'wb')
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("{}COVID_lables.pickle".format(dataDir), 'wb')
pickle.dump(Y, pickle_out)
pickle_out.close()
