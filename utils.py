from config import TRAINING_SIZE, TRAIN_PATH, SAVED_MODEL_PATH
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from skimage.filters import threshold_otsu


def getTrainingData(train_path):
    '''This Function Reads and pre process the image
    for model training'''
    gender = ["male", "female"]
    training_data = []
    for x in gender:
        num_gender = gender.index(x)
        for i in os.listdir(train_path + str(x)[:1000]):
            data = cv2.imread(train_path + x + "/" + str(i))
            data_1 = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)
            data_2 = cv2.resize(data_1, (50, 50))
            training_data.append([data_2, num_gender])
    print("Finished Loading")
    main_data = pd.DataFrame(training_data)
    main_data["Gender"] = main_data[1]
    main_data["Images"] = main_data[0]
    del main_data[0]
    del main_data[1]
    X_clean = []
    for image in main_data["Images"]:
        threshold_value = threshold_otsu(image)
        binary_image = image > threshold_value
        X_clean.append(binary_image.flatten())
    return X_clean, main_data
