"""
 Support Vector Machine - Classification Version

"""
import pandas as pd

# Import dataset
from sklearn.metrics import confusion_matrix

df = pd.read_csv("../Datasets/Social_Network_Ads.csv")

""" Step1) Pre-processing """
# print(df.head())  # check for null values.
# print(df.describe())  # Describe mean, std dev, etc.
# print(df.corr())  # Correlation.


""" Step 2) Modelling """
x = df.iloc[:, : -1].values  # Age, estimated salary
y = df.iloc[:, -1].values  # Purchased (Target feature)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=42)

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)  # DO NOT USE FIT.TRANSFORM as you don't want test data to know train values.

from sklearn.svm import SVC

selena_williams = SVC(kernel="rbf")
selena_williams.fit(x_train, y_train)


""" Step 3) Evaluation """
# pred = selena_williams.predict(sc.transform([[30, 80000]]))
import Helpers.val as v
tn, fp, fn, tp = confusion_matrix(y_true=y_test, y_pred=selena_williams.predict(x_test)).ravel()  # Test set analysis.
recall, specificity, accuracy, mcc, f1, j = v.val(tp, fn, fp, tn)
print('Recall is %.2f' % recall)
print('Specificity is %.2f' % specificity)
print('Accuracy is %.2f' % accuracy)
print('F-1 is %.2f' % f1)
print('MCC is %.2f' % mcc)
print('Youdens J statistic is %.2f' % j)

cn = confusion_matrix(y_true=y_test, y_pred=selena_williams.predict(x_test))
print(cn)  # # Print confusion matrix
# print(pred)  # Print custom prediction.


""" Evaluation Results 
Kernel choice: RBF => recall .96 and Linear = .68 
RBF false negatives = 1, linear = 9 false negatives. 


Question: How can you determine which kernel is best? 
    - Model and create all models and compare results to determine best kernel. 
"""