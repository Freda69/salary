# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 04:49:23 2019

@author: User
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import style
style.use('ggplot')

dataset = pd.read_csv('salary.csv')
X =  dataset.iloc[500: , :].values
y = dataset.iloc[: , :1].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X =LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
onehotencoder = OneHotEncoder(categorical_features=[0])

X = onehotencoder.fit_transform(X).toarray()
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 0)

y_pred = predict(regressor, data.frame(Level = 1))

