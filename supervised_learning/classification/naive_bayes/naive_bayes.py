# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 03:03:35 2019

@author: User
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
from matplotlib import style
style.use('ggplot')

dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, [2,3]].values
y = dataset.iloc[:, 4].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1 , X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() -1, stop = X_)
                      np.arange(start = X_set[:, 1].min() -1, stop = X )
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T))
             alpha = 0.775, cmap = ListedCollormap(('red' , 'green'))
plt.xlim(X1.min(), X1.max)