Customer Segmentation

customer segmentation is a process of dividing customers into different groups based on their characteristics. This can be useful for mall management in order to better understand their customers and tailor marketing strategies to different groups. One popular method for customer segmentation is k-means clustering, which is a machine learning technique that groups similar data points together. In this article, we will be discussing how to use the k-means algorithm in Python to perform customer segmentation for a mall.

### The k-means Algorithm

The k-means algorithm is a simple yet powerful method for grouping similar data points together. It works by dividing a dataset into k clusters, where each cluster is represented by its centroid (mean). The algorithm iteratively assigns each data point to the cluster with the nearest centroid, and then updates the centroid based on the new data point assignments.

The first step is to import the necessary libraries, including scikit-learn, numpy,  pandas, seaborn and matplotlib.

```
# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy import stats

import seaborn as sns
import matplotlib.pyplot as plt

```

The link for the dataset that I will be using in this article : https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python

```
data = pd.read_csv("Mall_Customers.csv")

```

#### Data Cleaning:

The first step in preprocessing is data cleaning, which involves removing or correcting any errors or inconsistencies in the dataset. 

Check if there is missing or duplicate data

```
print(data.isnull().sum())
# drop null values if present
data.duplicated().sum()
```

Renaming columns

```
data.rename(columns = {'Age':'age', 'Annual Income (k$)': 'income', 'Spending Score (1-100)': 'spending'}, inplace=True)
```

Dealing with outliers

```
# check for outliers using box plot
sns.boxplot(data=data[['age', 'income', 'spending']])
plt.show()

```

We notice outliers present in income. Outliers are values that are significantly different from the rest of the data in a dataset. These values can have a negative impact on the performance of a machine learning model, as they can skew the results. One of the most common ways to identify and remove outliers is through the use of the z-score.

The z-score is a statistical measure that is used to indicate how many standard deviations a data point is from the mean. It is calculated by subtracting the mean from the data point and dividing by the standard deviation. A z-score of 0 indicates that the data point is exactly the same as the mean, while a z-score of 1 indicates that the data point is one standard deviation above the mean.

The z-score is most commonly used to remove outliers when the data is approximately normally distributed. This is because the z-score is based on the normal distribution, which is a bell-shaped distribution that is often assumed for many types of data. When the data is normally distributed, outliers can be easily identified by their z-scores, as they will have a z-score that is significantly larger than the rest of the data.

It's worth mentioning that the z-score method will not work well with data that is not normally distributed. In that case, other methods such as the Interquartile range (IQR) method can be used.

we can set a threshold, usually 3 or -3, to identify and remove outliers. Any data point with a z-score above or below this threshold can be considered an outlier and can be removed from the dataset.

```
# remove outliers using z-score
data = data[(np.abs(stats.zscore(data[[ 'income']])) < 3).all(axis=1)]
```


Since, customerID is unique across all rows and doesnot contribute to the accuracy of clustering algorithm. We will drop the column.

```
data.drop('CustomerID', axis=1, inplace=True)
```

Pairwise relationships between variables is plotted with hue gender. We can observe the distribution of male and female across all features to be similar. So, Gender feature won't contribute much to the performance of clustering algorithm. Thus, we won't be using Gender column during clustering.

```
# Select features for clustering
X = data[['age', 'income', 'spending']]

```


Next, we will perform standardization to ensure that all the features have the same scale. Z-score normalization, also known as standardization, is a technique that is used to standardize the data by subtracting the mean and dividing by the standard deviation. This technique is useful when the data is normally distributed, and it helps to ensure that the data has a mean of 0 and a standard deviation of 1.

```
# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

Choosing optimal number of clusters

Choosing the optimal number of clusters in k-means clustering is an important step that can greatly impact the performance of the model. The optimal number of clusters is the one that best balances the trade-off between the complexity of the model and the accuracy of the clustering.

The elbow method is a technique used to determine the optimal number of clusters in a k-means clustering algorithm. The idea behind the elbow method is to run the k-means clustering algorithm for different values of k and then plot the within-cluster sum of squares (WCSS) against the number of clusters. The optimal number of clusters is chosen as the value of k where the change in WCSS begins to level off, creating an "elbow" shape in the plot.

```
# Elbow method
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

fig, ax = plt.subplots(figsize=(12, 8))
sns.lineplot(x=list(range(1, 11)), y=wcss, ax=ax)
ax.set_title('Elbow Method')
ax.set_xlabel('Clusters')
ax.set_ylabel('WCSS')

```
Silhouette score method is based on the average similarity measure between all instances of a cluster and the instances of the next closest cluster. It ranges between -1 and 1. The higher the silhouette score, the better the clustering.

```
# Silhouette score method
from sklearn.metrics import silhouette_score

silhouette_scores = []

for n_cluster in range(2, 11):
    kmeans = KMeans(n_clusters=n_cluster).fit(X_scaled)
    label = kmeans.labels_
    sil_coeff = silhouette_score(X_scaled, label, metric='euclidean')
    silhouette_scores.append(sil_coeff)

# Plot the silhouette scores
plt.plot(range(2, 11), silhouette_scores)
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Score")
plt.show()
```
Based on the above 2 methods, we can see that optimal number of clusters is 6.

```
# K-means clustering
kmeans = KMeans(n_clusters=6, init='k-means++')

# Fit the k means algorithm on scaled data
kmeans.fit(X_scaled)

# Assign the labels to each row
data['clusters'] = kmeans.labels_
```



