import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

#  Using dataset https://www.kaggle.com/ashok4kaggle/salary-prediction-with-sklearn
df = pd.read_csv("../Datasets/Salary_Data.csv")

""" Step 1) Pre-processing """
# Prints top 3 rows and x columns.
# print(df.head(3))

# print(df.isnull().sum())  # Alternate way to check for null values.
# print(df.info())  # Check if any null values.
# print(df.describe())  # Mean/std/min/max etc of both variables.


""" Step 2) Modelling/Machine learning """
# plt.plot(df.YearsExperience, df.Salary)  # Optional to use plot.
# plt.scatter(df.YearsExperience, df.Salary)  #  Scatterplot used because two variables are both numerical.
# plt.show()


x = df.iloc[:, :-1].values  # Return years experience. Code == All rows except last one.
y = df.iloc[:, -1].values   # Return salary.

# Split dataset.
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=69)

"""

Not using standard scalar, as only 2 columns Years experience and salary. Because only need x = years experience 
then you do not need to use scalar. If there was more than one column for x, and they were different ranges, then 
you need to do feature scaling.  
# If you were using different labels/categories such as salary and height, where height is cm and salary is just 
using numbers, you have to use standardScaler to standardise data. 

"""

from sklearn.linear_model import LinearRegression

rhianna = LinearRegression()  # Create model.
rhianna.fit(x_train, y_train)
pred = rhianna.predict(x_test)

pred = rhianna.predict([[20]])  # Predicting salary based on 20 years experience. 2D array, [*[*values*]*]

""" Visualisation """
print(pred)

# print(x_test[0], pred[0], y_test[0])
# plt.scatter(x_test, y_test, color='r')
# plt.scatter(x_test, pred)
# plt.plot(x_test, pred)
# plt.show()

""" Evaluation - from sklearn.metrics import mean_absolute_error """
# print(mean_absolute_error(y_test, pred))  # Average amount that is incorrect for predictions. (+-) 5845 etc.

"""
 Questions for regression model. 
 When using regression models: 
 What is the difference between mean absolute error and mean squared error?
 
 Mean absolute error: 
 
 Mean squared error:  
 
 
 """
