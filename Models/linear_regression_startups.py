import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

df = pd.read_csv("../Datasets/50_Startups.csv")

""" Step 1) Pre-processing """
pd.set_option('display.max_columns', len(df.columns))
# print(df.head(4))
# print(df.isnull().sum())
# print(df.info())  #  Not null.
# print(df.describe())  # Determine std deviation etc.

# Modify dataset to change state values from String to numeric.
# Determine how many unique values in state.
# print(df["State"].unique())  # 3 unique {NY, Cali, Flo}
states = {"New York": 0, "California": 1, "Florida": 2}
df.State = [states[x] for x in df.State]

""" Step 2) Modelling/Machine learning """
# print(df.info())
x = df.iloc[:, :1].values  # R&D Spend.
y = df.iloc[:, -1].values  # Profit.

# Split into train/test data.
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=42)

susan = LinearRegression()  # Create model.
susan.fit(x_train, y_train)  # Fit training values to model.
pred = susan.predict(x_test)

plt.plot(x_test, pred)
plt.scatter(x_test, pred, color='r')
plt.show()

""" Step 3) Evaluation """
# plt.plot(df)
# plt.show()

print(mean_absolute_error(y_test, pred))  # Average amount that is incorrect for predictions. (+-) 5845 etc.
