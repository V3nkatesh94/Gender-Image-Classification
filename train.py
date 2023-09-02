from config import TRAIN_PATH, saveModelPath
from utils import getTrainingData
import pickle
import numpy as np
import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report

x, X = getTrainingData(TRAIN_PATH)
x_train, x_test, y_train, y_test = train_test_split(
    x, X["Gender"], test_size=0.2, random_state=42
)
model = RandomForestClassifier()

# model training
model.fit(x_train, y_train)
with open(saveModelPath + "RandomForest_500.pkl", "wb") as files:
    pickle.dump(model, files)
# # prediction from test data set
y_pred = model.predict(x_test)

f1_score = f1_score(y_test, y_pred, average="weighted")
print("f1 score: ", f1_score)
print(classification_report(y_test, y_pred))
