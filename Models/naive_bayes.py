"""
Naive Bayes:
    - Bayes rules

    Law of Probability ( Defective example with factory )
    Bayes Rules ( Find specific factory defective rates )


Check week 4 stat1070 lecture.

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from Helpers import val as v
from sklearn.metrics import confusion_matrix
df = pd.read_csv("../Datasets/Social_Network_Ads.csv")

""" Step 1) """
x = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=42)
sc = StandardScaler()

x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)  # DO NOT USE FIT.TRANSFORM as you don't want test data to know train values.


""" Step 2) Modelling """
gal_gadot = GaussianNB()
gal_gadot.fit(x_train, y_train)


""" Step 3) Evaluation and Visualisation """
pred = gal_gadot.predict(sc.transform([[30, 78000]]))  # For prediction with custom values, do this way.
tn, fp, fn, tp = confusion_matrix(y_true=y_test, y_pred=gal_gadot.predict(x_test)).ravel()  # Test set analysis.
recall, specificity, accuracy, mcc, f1, j = v.val(tp, fn, fp, tn)
print('Recall is %.2f' % recall)
print('Specificity is %.2f' % specificity)
print('Accuracy is %.2f' % accuracy)
print('F-1 is %.2f' % f1)
print('MCC is %.2f' % mcc)
print('Youdens J statistic is %.2f' % j)

cn = confusion_matrix(y_true=y_test, y_pred=gal_gadot.predict(x_test))
print(cn)  # # Print confusion matrix
print(pred)  # Print custom prediction.