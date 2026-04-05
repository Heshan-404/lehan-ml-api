---
description: Full checklist of tasks required to satisfy the AuraCart ML Project requirements across all modules (3.1 to 4.3). This acts as a final guide for the user to ensure all deliverables are met.
---

# AuraCart ML System: Final Deliverable Checklist

This checklist summarizes every task needed to satisfy the requirements in the **ITS 2140: Machine Learning** Project Specification.

## ✅ Phase 1: Model Engineering (Completed)
- [x] **Task 3.1: Data Preprocessing:** All models now use a consistent `StandardScaler` pipeline for reproducibility.
- [x] **Task 3.2: Regression Modeling:** Predicted final monetary value of incoming orders (Revenue Forecasting).
- [x] **Task 3.3: Multi-class Classification:** Categorized orders into "High Risk of Delay" vs "On-time Probability."
- [x] **Task 3.5: Unsupervised Clustering:** Grouped customers into "Bronze/Silver/Gold" behavior segments.

## 🛠️ Phase 2: MLOps & Deployment (Completed)
- [x] **Restful Endpoint (Task 4.1):** Created a unified Flask API (`app.py`) for all 3 tasks.
- [x] **Containerization (Task 4.3):** Built a production-ready `Dockerfile`.
- [x] **Cloud Hosting:** Successfully deployed to **Railway** (Public URL: `https://lehan-ml-api-production.up.railway.app`).
- [x] **Postman Collection:** Created for testing the endpoints.

## 🏃 Phase 3: Final Administrative Tasks (To Do Now)
- [ ] **MLflow Tracking (Task 4.1):** Need to run the `log_to_mlflow.py` script to generate evidence of experiment tracking.
- [ ] **Performance Evaluation (Task 3.4):** Need to run the `evaluate_models.py` script to generate a **Confusion Matrix** for the report.
- [ ] **Final Report PDF:** This is the document your friend will submit. It should explain everything we did and include screenshots of our working API!

---
> [!TIP]
> Once you run the MLflow and Performance scripts I provide next, you have **everything** you need to hand this over for the final grade.
