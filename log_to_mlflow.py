import mlflow
import joblib
import os

# Set up MLflow
mlflow.set_experiment("AuraCart_Retail_Analytics")

def log_model_to_mlflow(model_path, task_name):
    print(f"Logging {task_name} to MLflow...")
    with mlflow.start_run(run_name=task_name):
        model = joblib.load(model_path)
        
        # Log parameters (example)
        mlflow.log_params({
            "model_type": "Pipeline",
            "task": task_name,
            "system": "AuraCart Unified v1.2"
        })
        
        # Log the model artifact itself
        mlflow.sklearn.log_model(model, "model")
        print(f"Successfully logged {task_name} to MLflow.")

if __name__ == "__main__":
    # Log all 3 models to fulfill Task 4.1
    log_model_to_mlflow("model.joblib", "Revenue_Forecasting_Regression")
    log_model_to_mlflow("classification_model.joblib", "Order_Risk_Classification")
    log_model_to_mlflow("clustering_model.joblib", "Customer_Behavioral_Clustering")
    
    print("\n✅ Evidence created! You can now run 'mlflow ui' in your terminal to see the dashboard.")
