"""

*** Logistic Regression

    If data (target features) are binary you can use logistic regression. Otherwise, don't use it (most of the time).
    Dataset used is social network ads.
    Features:
        - Age
        - Salary
        - Purchased (This is the binary target feature)

    Given input age and salary, place into logistic regression model, and determine if they will purchase.

 Correlation values:
    - Between -1 and 1.
    - Positive correlation => With bigger height, bigger weight.
        - 0.7 > Strongly positive.
    - Negative correlation => With higher price, less purchases.
        - From -0.7 > strongly negative.

 IMPORTANT:
    - Age and estimated salary values have different rages.
    Need to do feature scaling so they use the same range.
"""
import math

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


def val(tp, fn, fp, tn):
    from math import sqrt
    # --- Recall (senstivity) and Specificity
    recall = tp / (tp + fn)
    specificity = tn / (tn + fp)

    # --- Precision
    precision = tp / (tp + fp)

    # --- Accuracy
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    # --- F-1 Score
    a = 1 / precision
    b = 1 / recall
    f1 = 2 / (a + b)

    # --- mcc
    x = (tp * tn) - (fp * fn)
    y = sqrt((tp + fp)*(tp + fn)*(tn + fp) * (tn + fn))
    mcc = x / y

    # --- Youdens J statistic
    j = (recall + specificity) - 1  # J = recall + specificity -1
    return recall, specificity, accuracy, mcc, f1, j


# Import dataset
df = pd.read_csv("../Datasets/Social_Network_Ads.csv")

""" Step1) Pre-processing """
# print(df.head())  # check for null values.
# print(df.describe())  # Describe mean, std dev, etc.
# print(df.corr())  # Correlation.


""" Step 2) Modelling """
x = df.iloc[:, : -1].values  # Age, estimated salary
y = df.iloc[:, -1].values  # Purchased (Target feature)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=69)

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)  # DO NOT USE FIT.TRANSFORM as you don't want test data to know train values.

from sklearn.linear_model import LogisticRegression

linda = LogisticRegression()
linda.fit(x_train, y_train)
pred = linda.predict(sc.transform([[47, 80000]]))

""" Step 3) Visualisation """
print(pred)

""" Step 4) Evaluation """
from sklearn.metrics import confusion_matrix
tn, fp, fn, tp = confusion_matrix(y_true=y_test, y_pred=linda.predict(x_test)).ravel()
recall, specificity, accuracy, mcc, f1, j = val(tp, fn, fp, tn)
print('Recall is %.2f' % recall)
print('Specificity is %.2f' % specificity)
print('Accuracy is %.2f' % accuracy)
print('F-1 is %.2f' % f1)
print('MCC is %.2f' % mcc)
print('Youdens J statistic is %.2f' % j)

