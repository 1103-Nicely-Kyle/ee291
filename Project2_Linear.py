# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 14:47:28 2022

@author: knicely
"""

import pandas as pd
from sklearn.linear_model import LinearRegression as LR
import matplotlib.pyplot as plt

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


reg = LR()
reg.fit(x_train,y_train)
print('linearConst = ',reg.intercept_)
print('linearCoeff = ', reg.coef_)

y_pred = reg.predict(x_pred)

plt.scatter(x2, y2)
plt.title("day 2 actual")
plt.xlabel("index")
plt.ylabel("load kWh")

plt.scatter(x2, y_pred)
plt.title("day 2 actual vs. linear regression prediction")
plt.xlabel("index")
plt.ylabel("load (kWh)")

err = y2_train - y_pred
errList = err.tolist()


fig, ax = plt.subplots()
ax.hist(errList)
ax.set_title('absolute error')
ax.set(xlabel='abs. error(actual load - predicted load)', ylabel='ocurrences')