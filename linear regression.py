'''
This program implements Linear Regression on iris dataset in Python.
Author : Ankit Sharma
Indian Statistical Institute, Kolkata
E-mail: 27ankitsharma@gmail.com
'''

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

data = pd.read_csv("iris.csv",usecols=[0,1])
seplen = data['sepalLength']
sepwid = data['sepalWidth']

plt.plot(sepwid,seplen,'o')
plt.xlabel("SepalWidth")
plt.ylabel("SepalLength")
y = np.array(seplen)
x = np.array(sepwid)
cov = np.cov(x,y)
cov_xy = cov[0][1]
var_x = cov[0][0]
b = cov_xy / var_x

meanx = np.mean(x)
meany = np.mean(y)
a = meany - b * meanx 

plt.title("Linear Regression",color="r")
plt.plot(x,b*x+a)
plt.show()
