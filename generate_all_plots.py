"""
AuraCart Project: Generate ALL Visualizations for the Final Report
This script downloads the real dataset and creates every plot needed.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import SGDRegressor, LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.cluster import KMeans
from datasets import load_dataset
import warnings
import os

warnings.filterwarnings('ignore')
sns.set_theme(style="whitegrid")

# Create an output folder for all images
os.makedirs("report_screenshots", exist_ok=True)

# ========================================
# STEP 1: Load the REAL Dataset
# ========================================
print("Step 1: Downloading real dataset from HuggingFace...")
dataset = load_dataset("millat/e-commerce-orders")
df = pd.DataFrame(dataset['train'])
print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"Columns: {list(df.columns)}")

# ========================================
# SCREENSHOT 1: Correlation Heatmap
# ========================================
print("\nStep 2: Generating Correlation Heatmap...")
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
corr_matrix = df[numeric_cols].corr()

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm",
            center=0, square=True, linewidths=0.5, ax=ax)
ax.set_title("Feature Correlation Matrix - AuraCart Dataset",
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("report_screenshots/1_correlation_heatmap.png", dpi=150)
plt.close()
print("  -> Saved: report_screenshots/1_correlation_heatmap.png")

# ========================================
# SCREENSHOT 1b: Price Distribution Histogram
# ========================================
print("\nStep 3: Generating Price Distribution...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Price histogram
axes[0].hist(df['price'], bins=30, color='#4C72B0', edgecolor='white', alpha=0.85)
axes[0].set_title("Distribution of Transaction Price", fontsize=13, fontweight='bold')
axes[0].set_xlabel("Price ($)")
axes[0].set_ylabel("Frequency")

# Quantity histogram
axes[1].hist(df['quantity'], bins=10, color='#55A868', edgecolor='white', alpha=0.85)
axes[1].set_title("Distribution of Order Quantity", fontsize=13, fontweight='bold')
axes[1].set_xlabel("Quantity")
axes[1].set_ylabel("Frequency")

plt.tight_layout()
plt.savefig("report_screenshots/1b_price_quantity_distribution.png", dpi=150)
plt.close()
print("  -> Saved: report_screenshots/1b_price_quantity_distribution.png")

# ========================================
# SCREENSHOT 2: Class Distribution Bar Charts
# ========================================
print("\nStep 4: Generating Class Distribution Charts...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Delivery Status distribution
delivery_counts = df['delivery_status'].value_counts()
colors_delivery = ['#4C72B0', '#55A868', '#C44E52', '#8172B3']
axes[0].bar(delivery_counts.index, delivery_counts.values,
            color=colors_delivery[:len(delivery_counts)], edgecolor='white')
axes[0].set_title("Delivery Status Distribution", fontsize=13, fontweight='bold')
axes[0].set_xlabel("Delivery Status")
axes[0].set_ylabel("Count")
for i, (idx, val) in enumerate(zip(delivery_counts.index, delivery_counts.values)):
    axes[0].text(i, val + 50, f"{val}\n({val/len(df)*100:.1f}%)",
                 ha='center', fontsize=9, fontweight='bold')

# Customer Segment distribution
segment_counts = df['customer_segment'].value_counts()
colors_segment = ['#4C72B0', '#55A868', '#C44E52']
axes[1].bar(segment_counts.index, segment_counts.values,
            color=colors_segment[:len(segment_counts)], edgecolor='white')
axes[1].set_title("Customer Segment Distribution", fontsize=13, fontweight='bold')
axes[1].set_xlabel("Customer Segment")
axes[1].set_ylabel("Count")
for i, (idx, val) in enumerate(zip(segment_counts.index, segment_counts.values)):
    axes[1].text(i, val + 50, f"{val}\n({val/len(df)*100:.1f}%)",
                 ha='center', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig("report_screenshots/2_class_distribution.png", dpi=150)
plt.close()
print("  -> Saved: report_screenshots/2_class_distribution.png")

# ========================================
# PREPROCESSING: Prepare Data for Modeling
# ========================================
print("\nStep 5: Preprocessing data for modeling...")

# Drop ID columns and text columns
drop_cols = ['order_id', 'shipping_address', 'billing_address']
drop_cols = [c for c in drop_cols if c in df.columns]
df_model = df.drop(columns=drop_cols, errors='ignore')

# Handle dates
for col in ['order_date', 'shipping_date']:
    if col in df_model.columns:
        df_model[col] = pd.to_datetime(df_model[col], errors='coerce')
        df_model[f'{col}_month'] = df_model[col].dt.month
        df_model[f'{col}_day'] = df_model[col].dt.day
        df_model[f'{col}_hour'] = df_model[col].dt.hour
        df_model.drop(columns=[col], inplace=True)

# Encode categorical features
cat_cols = df_model.select_dtypes(include=['object']).columns.tolist()
target_cols = ['delivery_status', 'customer_segment']
feature_cat_cols = [c for c in cat_cols if c not in target_cols]

# Label encode targets
le_delivery = LabelEncoder()
le_segment = LabelEncoder()
if 'delivery_status' in df_model.columns:
    df_model['delivery_status_encoded'] = le_delivery.fit_transform(df_model['delivery_status'])
if 'customer_segment' in df_model.columns:
    df_model['customer_segment_encoded'] = le_segment.fit_transform(df_model['customer_segment'])

# One-hot encode feature categoricals
df_encoded = pd.get_dummies(df_model, columns=feature_cat_cols, drop_first=True)

# Separate features
target_and_encoded = ['delivery_status', 'customer_segment',
                      'delivery_status_encoded', 'customer_segment_encoded']
feature_cols = [c for c in df_encoded.columns if c not in target_and_encoded]
X = df_encoded[feature_cols].values.astype(float)

y_price = df_encoded['price'].values if 'price' in df_encoded.columns else None
y_delivery = df_encoded['delivery_status_encoded'].values
y_segment = df_encoded['customer_segment_encoded'].values

print(f"  Features shape: {X.shape}")
print(f"  Delivery classes: {le_delivery.classes_}")
print(f"  Segment classes: {le_segment.classes_}")

# ========================================
# SCREENSHOT 4: Confusion Matrix (Delivery Status)
# ========================================
print("\nStep 6: Training Classification Model & Generating Confusion Matrix...")

clf_delivery = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(max_iter=1000, random_state=42))
])
clf_delivery.fit(X, y_delivery)
y_pred_delivery = clf_delivery.predict(X)

cm = confusion_matrix(y_delivery, y_pred_delivery)
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le_delivery.classes_,
            yticklabels=le_delivery.classes_, ax=ax)
ax.set_title('Confusion Matrix: Delivery Status Classification (Task 3.4)',
             fontsize=13, fontweight='bold')
ax.set_xlabel('Predicted Label')
ax.set_ylabel('True Label')
plt.tight_layout()
plt.savefig("report_screenshots/4_confusion_matrix_delivery.png", dpi=150)
plt.close()
print("  -> Saved: report_screenshots/4_confusion_matrix_delivery.png")

# ========================================
# SCREENSHOT 4b: Confusion Matrix (Customer Segment)
# ========================================
print("\nStep 7: Training Customer Segment Classifier...")

clf_segment = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(max_iter=1000, random_state=42))
])
clf_segment.fit(X, y_segment)
y_pred_segment = clf_segment.predict(X)

cm_seg = confusion_matrix(y_segment, y_pred_segment)
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm_seg, annot=True, fmt='d', cmap='Greens',
            xticklabels=le_segment.classes_,
            yticklabels=le_segment.classes_, ax=ax)
ax.set_title('Confusion Matrix: Customer Segment Classification (Task 3.4)',
             fontsize=13, fontweight='bold')
ax.set_xlabel('Predicted Label')
ax.set_ylabel('True Label')
plt.tight_layout()
plt.savefig("report_screenshots/4b_confusion_matrix_segment.png", dpi=150)
plt.close()
print("  -> Saved: report_screenshots/4b_confusion_matrix_segment.png")

# ========================================
# SCREENSHOT 5: Classification Report
# ========================================
print("\nStep 8: Generating Evaluation Reports...")

report_delivery = classification_report(y_delivery, y_pred_delivery,
                                         target_names=le_delivery.classes_)
report_segment = classification_report(y_segment, y_pred_segment,
                                        target_names=le_segment.classes_)

with open("report_screenshots/5_evaluation_report.txt", "w") as f:
    f.write("=" * 60 + "\n")
    f.write("DELIVERY STATUS CLASSIFICATION REPORT (Task 3.4)\n")
    f.write("=" * 60 + "\n\n")
    f.write(report_delivery)
    f.write("\n\n")
    f.write("=" * 60 + "\n")
    f.write("CUSTOMER SEGMENT CLASSIFICATION REPORT (Task 3.4)\n")
    f.write("=" * 60 + "\n\n")
    f.write(report_segment)

print("  -> Saved: report_screenshots/5_evaluation_report.txt")

# ========================================
# SCREENSHOT 6: K-Means Elbow Plot
# ========================================
print("\nStep 9: Generating K-Means Elbow Plot & Silhouette Scores...")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

wcss = []
sil_scores = []
K_range = range(2, 11)

from sklearn.metrics import silhouette_score

for k in K_range:
    km = KMeans(n_clusters=k, n_init='auto', random_state=42)
    km.fit(X_scaled)
    wcss.append(km.inertia_)
    sil_scores.append(silhouette_score(X_scaled, km.labels_))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Elbow plot
axes[0].plot(list(K_range), wcss, 'bo-', linewidth=2, markersize=8)
axes[0].axvline(x=3, color='red', linestyle='--', label='Optimal k=3')
axes[0].set_title("Elbow Method: Within-Cluster Sum of Squares (WCSS)",
                   fontsize=12, fontweight='bold')
axes[0].set_xlabel("Number of Clusters (k)")
axes[0].set_ylabel("WCSS")
axes[0].legend()

# Silhouette plot
axes[1].plot(list(K_range), sil_scores, 'go-', linewidth=2, markersize=8)
axes[1].axvline(x=3, color='red', linestyle='--', label='Optimal k=3')
axes[1].set_title("Silhouette Score Analysis",
                   fontsize=12, fontweight='bold')
axes[1].set_xlabel("Number of Clusters (k)")
axes[1].set_ylabel("Silhouette Score")
axes[1].legend()

plt.tight_layout()
plt.savefig("report_screenshots/6_elbow_silhouette_plot.png", dpi=150)
plt.close()
print("  -> Saved: report_screenshots/6_elbow_silhouette_plot.png")

# ========================================
# BONUS: Cluster Visualization
# ========================================
print("\nStep 10: Generating Cluster Visualization...")

km_final = KMeans(n_clusters=3, n_init='auto', random_state=42)
labels = km_final.fit_predict(X_scaled)

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_2d = pca.fit_transform(X_scaled)

fig, ax = plt.subplots(figsize=(10, 7))
scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap='viridis',
                     alpha=0.5, s=15)
centers_2d = pca.transform(km_final.cluster_centers_)
ax.scatter(centers_2d[:, 0], centers_2d[:, 1], c='red', marker='X',
           s=200, edgecolors='black', linewidths=2, label='Centroids')
ax.set_title("Customer Behavioral Clusters (PCA Projection)", fontsize=14, fontweight='bold')
ax.set_xlabel(f"Principal Component 1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)")
ax.set_ylabel(f"Principal Component 2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)")
ax.legend()
plt.colorbar(scatter, label='Cluster ID')
plt.tight_layout()
plt.savefig("report_screenshots/6b_cluster_visualization.png", dpi=150)
plt.close()
print("  -> Saved: report_screenshots/6b_cluster_visualization.png")

# ========================================
# DONE!
# ========================================
print("\n" + "=" * 60)
print("ALL SCREENSHOTS GENERATED SUCCESSFULLY!")
print("=" * 60)
print("\nAll images are saved in the 'report_screenshots' folder:")
print("  1_correlation_heatmap.png")
print("  1b_price_quantity_distribution.png")
print("  2_class_distribution.png")
print("  4_confusion_matrix_delivery.png")
print("  4b_confusion_matrix_segment.png")
print("  5_evaluation_report.txt")
print("  6_elbow_silhouette_plot.png")
print("  6b_cluster_visualization.png")
print("\nFor screenshots 3 & 7 (MLflow UI):")
print("  Run 'mlflow ui' in terminal, then open http://127.0.0.1:5000")
print("\nFor screenshots 8 & 9 (Vertex AI):")
print("  Go to GCP Console -> Vertex AI -> Endpoints")
