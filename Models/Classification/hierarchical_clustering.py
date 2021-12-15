"""


Hierarchical Cluster

2 types:
    - Agglomerative (Bottom up)
    - Divisive (Top down)

    Bottom to top (agglomerative) is more efficient.
    Building clusters from individual clusters to bigger ones.

    Top to bottom is less efficient as you start with one cluster, then deconstruct.

    Why do you need to use a dendogram?
    - To find suitable number of clusters.
    Always find the longest distance.

"""
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


# Import Dataset.
df = pd.read_csv("../../Datasets/Mall_Customers.csv")

""" Step 1) Pre-processing """
# print(df.head())
# print(df.describe())  # Describe mean, std dev, etc.
# print(df.corr())  # Correlation.

gender = {"Male": 0, "Female": 1}
df.Genre = [gender[Item] for Item in df.Genre]

""" Step 2) Modelling """
cluster = df.iloc[:, [2, 3, 4]].values  # values for every column except first one.

""" To find optimal clusters """
from scipy.cluster import hierarchy  # to determine optimal cluster number for dendogram
dandogram = hierarchy.dendrogram(hierarchy.linkage(cluster, method="ward"))
plt.show()

""" Results show ~3 is optimal """

from sklearn.cluster import AgglomerativeClustering

hermione = AgglomerativeClustering(n_clusters=3, affinity="euclidean", linkage="ward")
pred = hermione.fit_predict(cluster)
# print(pred)

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


