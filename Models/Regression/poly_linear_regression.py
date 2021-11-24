import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# Read dataset in.
df = pd.read_csv("../../Datasets/Position_Salaries.csv")

# Visualise data.
# plt.scatter(df.Level, df.Salary)
# plt.plot(df.Level, df.Salary)
# plt.show()

""" Step 1 Pre-processing """
# print(df.info())
# print(df.describe())

""" Step 2) Modelling """
x = df.iloc[:, 1:-1].values
y = df.iloc[:, -1].values

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=1)

penny = PolynomialFeatures(6)
x_poly = penny.fit_transform(x_train)

lin_reg = LinearRegression()
lin_reg.fit(x_poly, y_train)

poly_test_x = penny.transform(x_test)
pred = lin_reg.predict(poly_test_x)

print(pred)
print(y_test)

x_grid = np.arange(min(x), max(x), 0.1)  # for smooth looks arrange(start, stop, range)
x_grid = x_grid.reshape((len(x_grid), 1))  # for smooth looks: return/reshape the smoothness to x_grid
plt.scatter(x, y, color='red')
plt.plot(x_grid, lin_reg.predict(penny.fit_transform(x_grid)), color='blue')
# without smoothness plt.plot(x_poly, y_pred2, color='blue')
plt.title('Polynomial Linear Regression')
plt.xlabel('Levels')
plt.ylabel('Salary')
plt.show()

print(mean_absolute_error(y_test, pred))
