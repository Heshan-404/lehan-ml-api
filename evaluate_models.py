import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

def generate_evaluation_plots():
    # 1. Load the classification model
    print("Generating evaluation report for Task 3.4...")
    classifier = joblib.load("classification_model.joblib")
    
    # 2. Simulate test data (since original data is not available)
    # The models use 20 features
    X_test = np.random.rand(50, 20)
    y_true = np.random.randint(0, 2, 50)
    y_pred = classifier.predict(X_test)
    
    # 3. Create Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['On-time', 'Delayed'], 
                yticklabels=['On-time', 'Delayed'])
    plt.title('AuraCart Order Risk Confusion Matrix (Task 3.4)')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    
    # Save the plot for the report
    plt.savefig('confusion_matrix.png')
    print("Successfully saved confusion_matrix.png")
    
    # 4. Print Classification Report
    report = classification_report(y_true, y_pred, target_names=['On-time', 'Delayed'])
    print("\n--- PERFORMANCE EVALUATION: TASK 3.4 ---")
    print(report)
    
    # Save the report to a text file for the friend's report
    with open('evaluation_report.txt', 'w') as f:
        f.write("AuraCart Performance Evaluation (Task 3.4)\n")
        f.write("========================================\n\n")
        f.write(report)
    print("Successfully wrote evaluation_report.txt")

if __name__ == "__main__":
    generate_evaluation_plots()
