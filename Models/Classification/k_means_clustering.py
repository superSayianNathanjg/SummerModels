"""
K Means Clustering

- WCSS: To find optimal cluster number.
    * Look for elbow shape. When less and less changes with increased number of K.

"""
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np

# Import dataset.
df = pd.read_csv("../../Datasets/Mall_Customers.csv")

""" Step1) Pre-processing """
# print(df.head())  # check for null values.
# print(df.describe())  # Describe mean, std dev, etc.
# print(df.corr())  # Correlation.

gender = {"Male": 0, "Female": 1}
df.Genre = [gender[Item] for Item in df.Genre]

""" Step 2) Modelling """
cluster = df.iloc[:, [2, 3, 4]].values  # values for every column except first one.
# cluster = df.iloc[:, [3, 4]].values  # values for every column except first and second one.  #Drop gender/age
# jUST FOR WCSS.
wcss = []

# Found that k = 5.
# for itr in range(1, 6):
#     k = KMeans(n_clusters=itr, init='k-means++', random_state=42)
#     k.fit(cluster)
#     wcss.append(k.inertia_)
# plt.plot(range(1, 6), wcss)
# plt.xlabel('Clustes')
# plt.show()

kate_upton = KMeans(n_clusters=5, init='k-means++', random_state=0)
pred = kate_upton.fit_predict(cluster)
print(pred)

""" Step 3) Evaluation and Visualisation """
from mpl_toolkits.mplot3d import Axes3D

# Visualization the clusters
fig2 = plt.figure()
ax2 = fig2.add_subplot(111, projection='3d')
row = 0
item = np.asarray(df['Genre'])
item2 = np.asarray(pred)
for i, j in zip(item, item2):
    if i == 0:  # male
        if j == 0:
            x1, y1, z1 = df.at[row, 'Age'], df.at[row, 'Annual Income (k$)'], df.at[row, 'Spending Score (1-100)']
            ax2.scatter(xs=x1, ys=y1, zs=z1, marker='o', c='b')
        elif j == 1:
            x2, y2, z2 = df.at[row, 'Age'], df.at[row, 'Annual Income (k$)'], df.at[row, 'Spending Score (1-100)']
            ax2.scatter(xs=x2, ys=y2, zs=z2, marker='o', c='r')
        elif j == 2:
            x3, y3, z3 = df.at[row, 'Age'], df.at[row, 'Annual Income (k$)'], df.at[row, 'Spending Score (1-100)']
            ax2.scatter(xs=x3, ys=y3, zs=z3, marker='o', c='pink')
        elif j == 3:
            x4, y4, z4 = df.at[row, 'Age'], df.at[row, 'Annual Income (k$)'], df.at[row, 'Spending Score (1-100)']
            ax2.scatter(xs=x4, ys=y4, zs=z4, marker='o', c='y')
        elif j == 4:
            x5, y5, z5 = df.at[row, 'Age'], df.at[row, 'Annual Income (k$)'], df.at[row, 'Spending Score (1-100)']
            ax2.scatter(xs=x5, ys=y5, zs=z5, marker='o', c='g')
    elif i == 1:  # female
        if j == 0:
            x1, y1, z1 = df.at[row, 'Age'], df.at[row, 'Annual Income (k$)'], df.at[row, 'Spending Score (1-100)']
            ax2.scatter(xs=x1, ys=y1, zs=z1, marker='x', c='b')
        elif j == 1:
            x2, y2, z2 = df.at[row, 'Age'], df.at[row, 'Annual Income (k$)'], df.at[row, 'Spending Score (1-100)']
            ax2.scatter(xs=x2, ys=y2, zs=z2, marker='x', c='r')
        elif j == 2:
            x3, y3, z3 = df.at[row, 'Age'], df.at[row, 'Annual Income (k$)'], df.at[row, 'Spending Score (1-100)']
            ax2.scatter(xs=x3, ys=y3, zs=z3, marker='x', c='pink')
        elif j == 3:
            x4, y4, z4 = df.at[row, 'Age'], df.at[row, 'Annual Income (k$)'], df.at[row, 'Spending Score (1-100)']
            ax2.scatter(xs=x4, ys=y4, zs=z4, marker='x', c='y')
        elif j == 4:
            x5, y5, z5 = df.at[row, 'Age'], df.at[row, 'Annual Income (k$)'], df.at[row, 'Spending Score (1-100)']
            ax2.scatter(xs=x5, ys=y5, zs=z5, marker='x', c='g')
    row = row + 1
ax2.set_xlabel('Age')
ax2.set_ylabel('Annual Income (k$)')
ax2.set_zlabel('Spending Score (1-100)')
plt.show()
