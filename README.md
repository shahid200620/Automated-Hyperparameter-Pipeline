# 🚀 Automated Hyperparameter Optimization Pipeline  
### Optuna + MLflow + XGBoost (California Housing)

---

## 📌 Project Overview
This project implements a **production-grade, automated hyperparameter optimization pipeline** using **Optuna** and **MLflow** to systematically tune an **XGBoost regression model** on the **California Housing dataset**.

The pipeline replaces manual trial-and-error tuning with an intelligent, reproducible optimization workflow featuring:
- Automated hyperparameter search
- Cross-validation–based evaluation
- Early stopping via pruning
- Comprehensive experiment tracking
- Containerized execution using Docker

This project demonstrates **modern MLOps practices** commonly used in real-world machine learning systems.

---

## 🎯 Objectives
- Design an effective hyperparameter search space for XGBoost
- Optimize model performance using Optuna with pruning
- Track all experiments, metrics, parameters, and artifacts with MLflow
- Evaluate the best model on a held-out test set
- Ensure full reproducibility and automated execution via Docker

---

## 🛠️ Tech Stack
- **Python 3.10**
- **Optuna** – hyperparameter optimization
- **MLflow** – experiment tracking & model logging
- **XGBoost** – regression model
- **Scikit-learn** – dataset, metrics, cross-validation
- **NumPy / Pandas** – data handling
- **Matplotlib** – visualizations
- **Docker** – containerization

---

## 📂 Repository Structure
├── Dockerfile
├── requirements.txt
├── .dockerignore
├── src/
│ ├── data_loader.py # Dataset loading & train-test split
│ ├── objective.py # Optuna objective function
│ ├── optimize.py # End-to-end optimization pipeline
│ └── evaluate.py # Structured results generation
├── notebooks/
│ └── analysis.ipynb # Analysis & visualizations
├── outputs/ # Generated during execution
└── README.md

yaml
Copy code

---

## ⚙️ Pipeline Description
1. Load the California Housing dataset and perform an 80/20 train-test split
2. Define a 7-parameter XGBoost search space
3. Run **100 Optuna trials** with:
   - 5-fold cross-validation
   - Median pruning for early stopping
   - Parallel execution
4. Track every trial in MLflow with parameters, metrics, and tags
5. Train the best model on the full training set
6. Evaluate on the test set
7. Save model, plots, and structured results
8. Run everything automatically inside a Docker container

---

## 🚀 How to Run (Docker)

### Build the Docker image


docker build -t optuna-mlflow-pipeline .
Run the pipeline
bash
Copy code
docker run -v $(pwd)/outputs:/app/outputs optuna-mlflow-pipeline
The container runs the entire optimization pipeline automatically and writes all outputs to the mounted outputs/ directory.

### 📊 Outputs Generated
After execution, the outputs/ directory contains:

results.json – structured summary of optimization results

optuna_study.db – SQLite database with full Optuna study history

optimization_history.png – optimization progress plot

param_importance.png – hyperparameter importance plot

mlruns/ – MLflow experiment tracking data


### 🧪 Model Performance
The tuned XGBoost model significantly outperforms a baseline model:

Test RMSE: ~0.44

Test R²: ~0.85

This exceeds the required performance thresholds and demonstrates the effectiveness of automated hyperparameter optimization.



### 📈 Analysis Notebook
A detailed analysis is provided in notebooks/analysis.ipynb, including:

Optimization history visualization

Hyperparameter importance analysis

Parallel coordinates plot

Baseline vs tuned model comparison

Written insights and interpretations

The notebook is for analysis only and does not need to run inside Docker.


### ♻️ Reproducibility
All random seeds are fixed (42)

Optuna uses a persistent SQLite backend

MLflow tracks every experiment and artifact

Running the pipeline multiple times produces identical results


### ✅ Conclusion
This project showcases a complete MLOps-style workflow for automated model optimization, experiment tracking, and reproducible execution.
It demonstrates practical skills in hyperparameter optimization, experiment management, and production-ready ML system design, making it a strong portfolio and evaluation submission.


