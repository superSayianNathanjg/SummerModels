"""
    Decision Tree - Classification Version

    Decision tree doesn't need to be binary.
    Multiple tree and unbalanced trees are okay.
    Implementation is similar to Naive bayes
"""

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from Helpers import val as v
import numpy as np
# Import dataset.
df = pd.read_csv("../../Datasets/Social_Network_Ads.csv")

""" Step 1) Pre-processing """
# None required.

""" Step 2) Modelling """
x = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=42)
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)  # DO NOT USE FIT.TRANSFORM as you don't want test data to know train values.

# Geni is for impurity,
from sklearn.tree import DecisionTreeClassifier
# classifier = DecisionTreeClassifier(criterion='entropy', random_state=42)
# classifier.fit(x_train, x_test)
classifier = DecisionTreeClassifier(criterion='entropy', random_state=0)
classifier.fit(x_train, y_train)

# from sklearn.naive_bayes import GaussianNB
# greta = GaussianNB()
# greta.fit(x_train, y_train)

""" Step 3) Evaluation and Visualisation """
print('The prediction is %.2f' % classifier.predict(sc.transform([[30, 87000]])))
if int(classifier.predict(sc.transform([[30, 87000]])) == 1):
    print("This person will buy the product")
else:
    print("This person will NOT buy the product")
# pred = classifier.predict(sc.transform([[30, 78000]]))  # For prediction with custom values, do this way.
# tn, fp, fn, tp = confusion_matrix(y_true=y_test, y_pred=classifier.predict(x_test)).ravel()  # Test set analysis.
# recall, specificity, accuracy, mcc, f1, j = v.val(tp, fn, fp, tn)

 # Train set visualization
    x_set, y_set = sc.inverse_transform(x_train), y_train
    X1, X2 = np.meshgrid(np.arange(start=x_set[:, 0].min() - 10, stop=x_set[:, 0].max() + 10, step=1),
                         np.arange(start=x_set[:, 1].min() - 1000, stop=x_set[:, 1].max() + 1000, step=1))
    plt.contourf(X1, X2, classifier.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
                 alpha=0.75, cmap=ListedColormap(('red', 'green')))
    plt.xlim(X1.min(), X1.max())
    plt.ylim(X2.min(), X2.max())
    for i, j in enumerate(np.unique(y_set)):
        plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1], c=ListedColormap(('red', 'green'))(i), label=j)
    plt.title('Decision Tree Classification with Entropy (Training set)')
    plt.xlabel('Age')
    plt.ylabel('Estimated Salary')
    plt.legend()
    plt.show()
