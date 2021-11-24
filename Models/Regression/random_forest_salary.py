"""

 *** Random Forest ***

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

df = pd.read_csv("../../Datasets/Position_Salaries.csv")

""" Step 1) Pre-processing """
df = pd.read_csv("../../Datasets/Position_Salaries.csv")

""" Step 2) Modelling """
x = df.iloc[:, 1:-1].values
y = df.iloc[:, -1].values

ruby = RandomForestRegressor(n_estimators=1000, random_state=69)
ruby.fit(x, y)
pred = ruby.predict([[10]])
print(pred)

""" Step 3) Visualisation and Evaluation """
x_grid = np.arange(min(x), max(x), 0.1)
x_grid = x_grid.reshape(len(x_grid), 1)
plt.scatter(x, y, color='red', marker='X')
plt.plot(x_grid, ruby.predict(x_grid), color='blue')
plt.title('Random Forest Regression')
plt.xlabel('Position')
plt.ylabel('Salary')
plt.show()
