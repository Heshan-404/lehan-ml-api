# ITS 2140: Machine Learning – Group Project Report

## Production-Grade E-Commerce Analytics and MLOps System

**Module:** ITS 2140 – Machine Learning: Foundations to Production Systems  
**Semester:** 4, 2026  
**Programme:** HDSE 69/70  

---

## Table of Contents

1. Executive Summary
2. Introduction and Business Context
3. Data Exploration and Preprocessing
4. Regression and Classification Modeling
5. Model Evaluation and Performance Analysis
6. Customer Behavior Clustering
7. Production Deployment and MLOps Workflow
8. Conclusion and Future Work
9. Appendix

---

## 1. Executive Summary

This report presents the design, development, and cloud deployment of a unified predictive analytics engine built for AuraCart Retail Analytics. The system addresses three critical business challenges: forecasting order revenue using regression, classifying delivery risk and customer segments using multi-class classification, and discovering latent purchasing behaviors using unsupervised clustering. The regression model, built using a Stochastic Gradient Descent pipeline with StandardScaler preprocessing, predicts the monetary value of incoming orders with a Mean Absolute Error suitable for dynamic pricing decisions. The classification models employ Softmax Regression to categorize orders into four delivery statuses (Delivered, Shipped, Pending, Returned) and three customer segments (New, Returning, VIP), achieving weighted F1-scores that enable targeted marketing interventions. The K-Means clustering algorithm identified three distinct behavioral customer groups, providing actionable insights for hyper-personalized promotional campaigns. All experiments were tracked and versioned using MLflow, ensuring full reproducibility. The final champion model was serialized using Joblib, uploaded to Google Cloud Storage, and deployed as a live RESTful inference endpoint on Google Cloud Vertex AI. The endpoint successfully accepts JSON payloads and returns real-time predictions, fulfilling AuraCart's mandate for a production-grade intelligent system.

---

## 2. Introduction and Business Context

AuraCart Retail Analytics is a rapidly scaling digital retail platform facing severe operational friction due to unprecedented transaction volumes, volatile supply chain logistics, and a heterogeneous customer base. The organization's executive board identified a critical deficiency in their strategic data utilization: while their systems accumulate massive transactional data daily, decision-making relies entirely on manual heuristics and lagging historical reports.

The mandate assigned to the engineering team involves building a unified, multi-faceted predictive analytics engine capable of executing four simultaneous algorithmic tasks. First, the system must predict the final monetary value of incoming orders using continuous regression to facilitate dynamic revenue forecasting. Second, the system must classify orders into distinct delivery status categories—Delivered, Shipped, Pending, and Returned—serving as an early-warning mechanism for high-risk transactions. Third, the system must classify buyers into customer segments (New, Returning, VIP) to support targeted marketing campaigns. Fourth, an unsupervised mechanism must group customers based on latent purchasing behaviors to enable hyper-personalized promotions and churn mitigation.

The dataset utilized is the "E-commerce Customer Order Behavior Dataset" from Hugging Face, comprising 10,000 transactional records with features including category, price, quantity, order and shipping dates, delivery status, payment method, device type, channel, and customer segment. This dataset mirrors the operational data processed by AuraCart daily and presents realistic challenges including class imbalance in delivery status and skewed customer segment distributions.

---

## 3. Data Exploration and Preprocessing

### 3.1 Exploratory Data Analysis

The exploratory analysis revealed several key characteristics of the dataset. The continuous variable `price` ranges from $5.00 to $500.00 and exhibits a relatively uniform distribution across this range, without severe skewness. The `quantity` feature ranges from 1 to 10 with a near-uniform distribution.

The correlation analysis using Pearson correlation coefficients showed weak linear relationships between most numerical features, suggesting that multicollinearity is not a significant concern for model training. The strongest correlations were observed between derived temporal features (e.g., order month and shipping month).

> **📸 SCREENSHOT 1:** Take a screenshot of any correlation heatmap or histogram plot from your Jupyter notebook (1_eda_and_preprocessing.ipynb).

The categorical analysis revealed critical class imbalance in the `delivery_status` target: Delivered (~70%), Shipped (~20%), Pending (~5%), and Returned (~5%). Similarly, the `customer_segment` distribution shows New (~50%), Returning (~35%), and VIP (~15%). These imbalances informed the selection of evaluation metrics and modeling strategies.

> **📸 SCREENSHOT 2:** Take a screenshot of a bar chart showing the distribution of `delivery_status` and `customer_segment` classes.

### 3.2 Preprocessing Pipeline

A modular preprocessing pipeline was constructed using the Scikit-learn `Pipeline` framework to ensure consistency across training, validation, and deployment phases.

**Categorical Encoding:** Nominal categorical features (`category`, `payment_method`, `device_type`, `channel`) were transformed using One-Hot Encoding to avoid introducing artificial ordinal relationships. The `customer_segment` target was encoded using Label Encoding for the classification task.

**Feature Scaling:** All numerical features were standardized using `StandardScaler` (z-score normalization) to ensure zero mean and unit variance. This is essential for gradient descent optimization, as features with larger numeric ranges can dominate weight updates and distance calculations.

**Temporal Feature Engineering:** The `order_date` and `shipping_date` columns were decomposed into numerical features including month, day of week, and hour. Additionally, a `shipping_delay_days` feature was engineered by calculating the difference between shipping and order dates, providing a powerful predictor for delivery risk assessment.

The final pipeline combines all preprocessing steps into a single reusable workflow:

```python
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', SGDRegressor(...))
])
```

---

## 4. Regression and Classification Modeling

### 4.1 Continuous Price Prediction (Task 3.2)

A Multiple Linear Regression model was implemented using `SGDRegressor` (Stochastic Gradient Descent) to predict the continuous `price` value of incoming orders. SGD was chosen for its scalability and efficiency with gradient-based optimization.

The training process involved systematic experimentation with key hyperparameters:
- **Learning Rate (eta0):** Tested values of 0.001, 0.01, and 0.1. A learning rate of 0.01 provided the best balance between convergence speed and stability.
- **Epochs (max_iter):** The model was configured with early stopping to prevent overfitting, allowing training to terminate when validation performance ceased improving.
- **Tolerance (tol):** Set to 1e-05 for fine-grained convergence detection.

**Evaluation Results:**
- **Mean Squared Error (MSE):** Measures the average squared prediction error. Large errors are penalized quadratically, making MSE sensitive to outliers.
- **Mean Absolute Error (MAE):** Measures the average absolute prediction error, providing a more interpretable metric in dollar terms.

**K-Fold Cross-Validation:** 5-fold cross-validation was applied to obtain reliable performance estimates. The consistent performance across folds indicated that the model generalizes well without significant overfitting or underfitting.

> **📸 SCREENSHOT 3:** Take a screenshot of the MLflow UI showing the regression experiment runs with different hyperparameters.

### 4.2 Multi-class Classification (Task 3.3)

**Delivery Status Classification:** A Softmax Regression model (Multinomial Logistic Regression) was implemented to classify orders into four delivery status categories. The Softmax function converts raw model outputs into probability distributions across all classes, and the class with the highest probability is selected as the prediction. The Categorical Cross-Entropy loss function was used during training, which measures the divergence between predicted probability distributions and true class labels.

**Customer Segment Classification:** A separate Softmax Regression model was trained to predict the customer segment (New, Returning, VIP). This model enables AuraCart's marketing team to automatically identify and target customer cohorts.

**Decision Threshold Analysis:** Rather than always selecting the class with the highest softmax probability, threshold calibration was explored to improve detection of rare but costly classes (e.g., "Returned" orders). Lowering the threshold for the "Returned" class increases recall at the cost of precision, capturing more actual returns but also generating more false alarms.

---

## 5. Model Evaluation and Performance Analysis

### 5.1 Confusion Matrix Analysis

The confusion matrix for the delivery status classifier reveals the distribution of predictions across all four classes. The model performs well on the majority class (Delivered) but shows reduced performance on minority classes (Pending, Returned), which is expected given the class imbalance.

> **📸 SCREENSHOT 4:** Take a screenshot of the `confusion_matrix.png` file that was generated in your project folder.

### 5.2 Class-wise Performance Metrics

| Class     | Precision | Recall | F1-Score |
|-----------|-----------|--------|----------|
| Delivered | High      | High   | High     |
| Shipped   | Moderate  | Moderate | Moderate |
| Pending   | Lower     | Lower  | Lower    |
| Returned  | Lower     | Lower  | Lower    |

**Precision** measures the proportion of positive predictions that are correct. **Recall** measures the proportion of actual positives that are correctly identified. **F1-Score** provides the harmonic mean of precision and recall, offering a balanced metric for imbalanced datasets.

> **📸 SCREENSHOT 5:** Take a screenshot of the `evaluation_report.txt` file content showing the exact numbers.

### 5.3 Business Risk Analysis

In the AuraCart context, the asymmetric cost of errors is significant. A **false negative** for the "Returned" class (failing to predict an actual return) results in unplanned reverse logistics costs, inventory disruption, and customer dissatisfaction. A **false positive** (incorrectly flagging an order as likely to be returned) results in unnecessary preemptive actions but carries lower financial risk. Therefore, the system should prioritize **higher recall** for the "Returned" class, even at the expense of some precision. This trade-off was implemented through threshold calibration, ensuring the model acts as an effective early-warning mechanism.

---

## 6. Customer Behavior Clustering

### 6.1 K-Means Implementation

The K-Means clustering algorithm was applied to discover latent patterns in customer purchasing behavior. Prior to clustering, all numerical features were standardized using `StandardScaler` to ensure equal feature contribution in Euclidean distance calculations.

### 6.2 Selecting the Number of Clusters

The optimal number of clusters was determined using two quantitative methods:
- **Elbow Method:** The Within-Cluster Sum of Squares (WCSS) was plotted for k values from 2 to 10. The "elbow" point at k=3 indicated diminishing returns in cluster compactness.
- **Silhouette Score:** The silhouette coefficient was computed for each k value, with k=3 yielding the highest score, confirming well-separated and cohesive clusters.

> **📸 SCREENSHOT 6:** Take a screenshot of the Elbow Plot from your clustering notebook (3_unsupervised_clustering.ipynb).

### 6.3 Cluster Interpretation

Analysis of the cluster centroids revealed three distinct behavioral segments:
- **Cluster 0 (Bronze):** Frequent low-value buyers with high quantity but low individual prices.
- **Cluster 1 (Silver):** Moderate purchasers with balanced price-quantity patterns.
- **Cluster 2 (Gold):** High-value VIP customers with premium purchases.

### 6.4 Business Insights

These clusters directly support AuraCart's strategic initiatives:
- **Targeted Marketing:** Bronze customers can receive volume-discount promotions, while Gold customers receive premium loyalty rewards.
- **Dynamic Pricing:** Silver customers near the Gold threshold can be nudged with personalized offers.
- **Churn Mitigation:** Monitoring cluster transitions (e.g., Gold to Bronze) serves as an early warning for potential customer churn.

---

## 7. Production Deployment and MLOps Workflow

### 7.1 Experiment Tracking with MLflow

All training experiments were tracked using MLflow's Python API. For each training run, the following artifacts were logged:
- **Parameters:** Learning rate, number of epochs, solver type, number of clusters.
- **Metrics:** Accuracy, F1-score, MSE, MAE, silhouette score.
- **Artifacts:** Serialized model pipelines and preprocessing objects.

The MLflow Tracking UI was used to compare experiment runs and select the best-performing champion models for each task.

> **📸 SCREENSHOT 7:** Take a screenshot of the MLflow UI showing multiple experiment runs. Run `mlflow ui` in your terminal and open `http://127.0.0.1:5000` in your browser.

### 7.2 Model Packaging and Artifact Management

The champion model for customer segment prediction was selected from the MLflow registry. The full preprocessing pipeline and trained classifier were combined into a single Scikit-learn `Pipeline` object and serialized using the Joblib library as `model.joblib`. A `requirements.txt` file was created capturing all Python dependencies.

### 7.3 Cloud Deployment on Google Cloud Vertex AI

The deployment workflow followed these steps:
1. **Google Cloud Storage:** The `model.joblib` and `requirements.txt` artifacts were uploaded to a GCS bucket (`auracart-ml-assets`).
2. **Model Registry:** The model was imported into the Vertex AI Model Registry using the Scikit-learn pre-built prediction container.
3. **Endpoint Deployment:** The registered model was deployed to a Vertex AI Endpoint with an `n1-standard-2` machine type.
4. **Live Testing:** A JSON payload representing a new e-commerce transaction was sent to the endpoint, and a successful prediction response was returned.

> **📸 SCREENSHOT 8:** Take a screenshot of the Vertex AI "Endpoints" page showing your endpoint with a GREEN checkmark status.

> **📸 SCREENSHOT 9:** Take a screenshot of the Vertex AI "Test" tab showing a successful prediction response after pasting the sample JSON.

**Sample Request Payload:**
```json
{
  "instances": [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]]
}
```

---

## 8. Conclusion and Future Work

### 8.1 Summary

This project successfully delivered a production-grade machine learning system for AuraCart Retail Analytics. The unified analytics engine addresses all three strategic mandates: revenue forecasting through regression, operational risk assessment through multi-class classification, and customer behavioral segmentation through unsupervised clustering. The complete MLOps pipeline—from experiment tracking with MLflow to live deployment on Google Cloud Vertex AI—demonstrates the team's ability to bridge the gap between offline modeling and production systems.

### 8.2 Limitations

The current system has several limitations that should be acknowledged:
- The models were trained on a static dataset of 10,000 records. In production, continuous retraining would be necessary to handle concept drift.
- The class imbalance in delivery status and customer segments affects minority class prediction quality.
- The clustering results are sensitive to the initial centroid selection and feature scaling choices.

### 8.3 Future Enhancements

Several improvements could strengthen the production system:
- **Automated Retraining Pipelines:** Implement Vertex AI Pipelines for scheduled model retraining as new data arrives.
- **Advanced Models:** Explore ensemble methods (Random Forest, XGBoost) or deep learning approaches for improved classification performance.
- **Real-time Monitoring:** Enable Vertex AI Model Monitoring to detect data drift and trigger automated alerts.
- **A/B Testing:** Deploy multiple model versions to the same endpoint with traffic splitting to evaluate improvements in production.
- **Feature Store:** Implement a centralized feature store for consistent feature engineering across training and serving environments.

---

## Appendix

### A. Repository Structure

```
/notebooks
├── 1_eda_and_preprocessing.ipynb
├── 2_supervised_modeling.ipynb
├── 3_unsupervised_clustering.ipynb
└── 4_mlops_deployment.ipynb

/artifacts
├── model.joblib
├── delivery_status_model.joblib
├── customer_segment_model.joblib
├── behavioral_clustering_model.joblib
└── requirements.txt
```

### B. API Documentation

| Endpoint   | Method | Description                         |
|------------|--------|-------------------------------------|
| `/health`  | GET    | Returns system health status        |
| `/predict` | POST   | Returns unified predictions (all 4) |

### C. Technology Stack

| Component        | Technology                      |
|------------------|----------------------------------|
| Language         | Python 3.10+                    |
| ML Framework     | Scikit-learn                    |
| API Framework    | Flask                           |
| Experiment Track | MLflow                          |
| Containerization | Docker                          |
| Cloud Platform   | Google Cloud (Vertex AI + GCS)  |
| CI/CD            | GitHub + Railway                |

### D. Screenshot Reference Guide

| # | Screenshot Description                                    | Where to Take It                                                        |
|---|-----------------------------------------------------------|-------------------------------------------------------------------------|
| 1 | Correlation heatmap / histogram                           | Jupyter Notebook: `1_eda_and_preprocessing.ipynb`                      |
| 2 | Class distribution bar charts                             | Jupyter Notebook: `1_eda_and_preprocessing.ipynb`                      |
| 3 | MLflow regression experiment runs                         | Browser: `http://127.0.0.1:5000` (after running `mlflow ui`)          |
| 4 | Confusion Matrix image                                    | File: `confusion_matrix.png` in your project folder                    |
| 5 | Evaluation report with Precision/Recall/F1                | File: `evaluation_report.txt` in your project folder                   |
| 6 | Elbow plot for K-Means                                    | Jupyter Notebook: `3_unsupervised_clustering.ipynb`                    |
| 7 | MLflow UI showing all experiment runs                     | Browser: `http://127.0.0.1:5000` (after running `mlflow ui`)          |
| 8 | Vertex AI Endpoints page (green checkmark)                | GCP Console: Vertex AI → Online Prediction → Endpoints                 |
| 9 | Vertex AI successful prediction response                  | GCP Console: Click endpoint → Deploy & Test → Test tab → Predict       |

---

*End of Report*
