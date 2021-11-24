"""

 ***  Decision Trees   ***

When using Support Vector Machine (SVM), hard to determine which dimension to use.
So, how to separate targets?
_
_  x x   -      x
_  x x   -      x
_ ---------------------
_        -   x x
_        -  x x
_______________________

* Multiple lines placed between values (X). This is modelled using a decision tree.
* Use decision tree logic to define values.
* Then train model on decision tree.
* Use model to predict new values.

- Decision tree values are invariant, such that doesn't require scaling. Logically separated so doesn't require scaling.
- Regardless of 2d, 3d or other dimensions, positions are relevant and will remain the same. => No feature scaling.
"""

import pandas as pd
from matplotlib import pyplot as plt  # To show plots you need this for EVERYTHING.
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor  # Import Decision Tree
from sklearn import tree

""" Step 1) Dataset and Pre-processing """
df = pd.read_csv("../../Datasets/Position_Salaries.csv")

""" Step 2) Modelling """
x = df.iloc[:, 1:-1].values
y = df.iloc[:, -1].values

taylor = DecisionTreeRegressor(random_state=42)
swift = taylor.fit(x, y)  # Doesn't require further pre-processing (scaling etc)

# ALWAYS - For predicting, use 2D arrays => [ [<prediction value>] ]
""" Step 3) Visualising  """
pred = swift.predict([[6.5]])
plt.figure(figsize=(50, 10))  # Resize figure.
tree.plot_tree(swift, feature_names="Level")

plt.show()

""" Step 4) Evaluations """
# Because we did NOT split dataset into train/test, we can not perform mean absolute error checks.
# If we did split, we could obtain mean absolute error, and then compare with mean absolute error from SVG.
# Comparing the two mean absolute error values is a good way to compare model performance.

"""
 Questions: 
 - Q1: What is the difference between stochastic and non stochastic tree?
 - Q2: What is the difference between Random Forest and Bagging? 
    Random Forest: A collection and combination of decision trees, which uses average voting (or other voting types) to
        determine values. You can change weight of decision trees. 
    
    Bagging: You can select which type of voting you want to use. 1) Weighted voting, 2) Average voting, 3) Stacking.
        Combination of models (ex = SVM, Decision Tree) which are run parallel, results are then evaluated using voting
        to determine the best one. 
      
"""