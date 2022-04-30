# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 14:03:30 2022

@author: Kyle
"""

import pandas as pd
from sklearn.linear_model import LinearRegression as LR
from sklearn.preprocessing import PolynomialFeatures as PF
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_excel('Electricty Consumption.xlsx', sheet_name = 'Day_1')
df2 = pd.read_excel('Electricty Consumption.xlsx', sheet_name = 'Day_2')

x = df['Index']
x2 = df2['Index']
y = df['KWh']
y2 = df2['KWh']

x_train = x.values.reshape(-1,1)
x_pred = x2.values.reshape(-1,1)
y_train = y.values.reshape(-1,1)
y2_train = y.values.reshape(-1,1)

Poly = PF(5)
x_poly = Poly.fit_transform(x_train)
x_poly_2 = Poly.fit_transform(x_pred)
reg2 = LR()
reg2.fit(x_poly,y_train)

print('poly5Const = ', reg2.intercept_)
print('poly5Coef = ', reg2.coef_)
y_pred2 = reg2.predict(x_poly_2)
 
err = y2_train - y_pred2
errList = err.tolist()
squared_error = err**2

total_squared_error = np.sum(squared_error)
print('Total squared error of prediction = ', total_squared_error)

plt.scatter(x2, y2)
plt.title("day 2 actual")
plt.xlabel("index")
plt.ylabel("load kWh")

plt.scatter(x2,y_pred2)
plt.title("day 2 actual vs. polynomial regression order 5 prediction")
plt.xlabel("index")
plt.ylabel("load (kWh)")


fig, ax = plt.subplots()
ax.hist(errList)
ax.set_title('absolute error')
ax.set(xlabel='abs. error(actual load - predicted load)', ylabel='ocurrences')