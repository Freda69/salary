# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 03:03:23 2019

@author: User
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import style
style.use('ggplot')
# 
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X,y)

y_pred = regressor.predict([[6.5]])

X_grid = np.arange(min(X),max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X,y, color = 'green')
plt.plot(X_grid, regressor.predict(X_grid), color = 'pink')
plt.title('Truth or Bluff (Decision Tree Regression)')
plt.xlabel('Positive level')
plt.ylabel('Salary')
plt.show()

