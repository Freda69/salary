# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 07:01:45 2019

@author: User
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
from matplotlib import style
style.use('ggplot')

dataset = pd.read_csv('Social_Network_Qds.csv')
X = dataset.iloc[:, [2,3]].values
y = dataset.iloc[:, []]