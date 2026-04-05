import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# 1. Create a Classification Model (Task 3.3: Delayed vs. On-time)
# This model will predict if an order is 0 (On-time) or 1 (Delayed)
X_class = np.random.rand(100, 20)
y_class = np.random.randint(0, 2, 100)
clf = Pipeline([('scaler', StandardScaler()), ('rf', RandomForestClassifier())])
clf.fit(X_class, y_class)
joblib.dump(clf, 'classification_model.joblib')
print("Successfully created classification_model.joblib")

# 2. Create a Clustering Model (Task 3.5: Customer Segmentation)
# This model will group customers into 3 behavioral clusters
X_clust = np.random.rand(100, 20)
kmeans = Pipeline([('scaler', StandardScaler()), ('kmeans', KMeans(n_clusters=3, n_init='auto'))])
kmeans.fit(X_clust)
joblib.dump(kmeans, 'clustering_model.joblib')
print("Successfully created clustering_model.joblib")
