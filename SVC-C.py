# -*- coding: utf-8 -*-
"""
Created on Tue Nov 11 15:48:36 2014

@author: 
"""

from numpy import *
from matplotlib.pyplot import *
from sklearn.svm import SVC

x = random.uniform(0, 1, (100, 2))
t = (x[:, 1] > x[:, 0]**2) | ((x[:, 0]-1)**2 + x[:, 1]**2 < 0.1)

clf = SVC(C=100)
clf.fit(x, t)

X, Y = meshgrid(linspace(0, 1), linspace(0, 1))
Z = clf.predict(c_[X.ravel(), Y.ravel()])
Z = Z.reshape(X.shape)

scatter(x[:, 0], x[:, 1], c=t, s=50)
pcolor(X, Y, Z, alpha=0.3)
show()