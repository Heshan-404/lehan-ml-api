import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# 1. Classification Model A (Task 3.3a: Delivery Status)
X = np.random.rand(100, 20)
y_delivery = np.random.randint(0, 4, 100) # 4 statuses: Delivered, Shipped, Pending, Returned
clf_delivery = Pipeline([('scaler', StandardScaler()), ('rf', RandomForestClassifier())])
clf_delivery.fit(X, y_delivery)
joblib.dump(clf_delivery, 'delivery_status_model.joblib')
print("Created delivery_status_model.joblib")

# 2. Classification Model B (Task 3.3b: Customer Segment - Supervised)
y_segment = np.random.randint(0, 3, 100) # 3 types: New, Returning, VIP
clf_segment = Pipeline([('scaler', StandardScaler()), ('rf', RandomForestClassifier())])
clf_segment.fit(X, y_segment)
joblib.dump(clf_segment, 'customer_segment_model.joblib')
print("Created customer_segment_model.joblib")

# 3. Clustering Model (Task 3.5: Behavioral Clustering - Unsupervised)
kmeans = Pipeline([('scaler', StandardScaler()), ('kmeans', KMeans(n_clusters=3, n_init='auto'))])
kmeans.fit(X)
joblib.dump(kmeans, 'behavioral_clustering_model.joblib')
print("Created behavioral_clustering_model.joblib")
