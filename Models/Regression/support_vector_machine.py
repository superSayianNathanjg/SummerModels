"""

*** Support Vector Machine ***

Model:
- Fit_transform = used for training.
- Fit = used for testing, y will not know values from training.
Support Vector Machines
- Only works with linearly separable datasets.
- Hyperplane => "Decision plane"
- https://monkeylearn.com/blog/introduction-to-support-vector-machines-svm/
- Support vectors => Are only points that interact with hyperplane.

- If dataset is not separable linearly, then you cannot use SVM, but what if you changed dimensions of graph?
- Maybe it's possible on a 3d plane?
- Hyperplane intersects

- For linear regression, you have to squish polynomial datasets to linear.
- Using Support Vector Machine to escape to higher dimensions to use polynomial datasets.
- Kernel choices: Find the optimal kernel:
    "Try Gaussian  Radial Basic Function (RBF), if not working talk to Data Scientist for them to evaluate
    other kernels."
- For now, remember Gaussian Radian Basic Function (RBF) with SVM.
- Key points to remember


"""

import pandas as pd

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler

""" Step 1) Pre-processing """
df = pd.read_csv("../../Datasets/Position_Salaries.csv")

""" Step 2) Modelling """
x = df.iloc[:, 1:-1].values
y = df.iloc[:, -1].values

# 1: SVR is not explicit equation
# 2: Both feature value (Salaries is 1-10, other is 10,000 to 100,000 etc) is different (10 vs 1000000), so MUST apply
# feature scaling dependent and independent variables
# *** even a equation is  implicit equation like the multiple linear regression, MUST follow the second condition

# plt.scatter(x, y)
# plt.plot(x, y)
# plt.show()

selena_x = StandardScaler()
x = selena_x.fit_transform(x)  # Fit transform on above x.

selena_y = StandardScaler()
# y = selena_y.fit_transform(y)  # Fit transform on above y.
""" Type(y) = numpy array. So you can use reshape """
y = y.reshape(len(y), 1)  # Changed from 1d to 2d array. Both x and y match dimensions.
y = selena_y.fit_transform(y)
# print(y)

from sklearn.svm import SVR  # Support Vector Regression (SVR),  Support Vector Classification (SVC).

svr = SVR(kernel='rbf')  # Kernel used = Gaussian Radian Basic Function (RBF)
svr.fit(x, y)

"""
 Because all of dataset was used in training, you cannot use test set for evaluation. 
 So instead, manually predict. => svr.predict[[4,5]]] 
"""
pred = svr.predict(selena_x.transform([[9.5]]))

""" Step 3) Visualisation """
pred = pred.reshape(-1, 1)
pred = selena_y.inverse_transform(pred)  # Result is incorrect, need to inverse result.
print(pred)
