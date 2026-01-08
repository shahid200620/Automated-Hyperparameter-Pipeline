import os
import random
import time
import json

import numpy as np
import mlflow
import optuna

from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from sklearn.metrics import mean_squared_error

from src.data_loader import load_and_split_data
from src.objective import objective


def main():
    start_time = time.time()

    # -----------------------------
    # Reproducibility
    # -----------------------------
    random.seed(42)
    np.random.seed(42)

    # -----------------------------
    # MLflow setup
    # -----------------------------
    mlflow.set_experiment("optuna-xgboost-optimization")

    # -----------------------------
    # Load data
    # -----------------------------
    X_train, X_test, y_train, y_test = load_and_split_data()

    # -----------------------------
    # Optuna study setup
    # -----------------------------
    study = optuna.create_study(
        study_name="xgboost-housing-optimization",
        direction="maximize",
        sampler=TPESampler(seed=42),
        pruner=MedianPruner(
            n_startup_trials=10,
            n_warmup_steps=5
        ),
        storage="sqlite:///optuna_study.db",
        load_if_exists=True
    )

    # -----------------------------
    # Optimization loop
    # -----------------------------
    def wrapped_objective(trial):
        with mlflow.start_run(run_name=f"trial_{trial.number}"):
            try:
                score = objective(trial, X_train, y_train)

                cv_mse = -score
                cv_rmse = np.sqrt(cv_mse)

                # Log hyperparameters
                mlflow.log_params(trial.params)
                mlflow.log_param("trial_number", trial.number)

                # Log metrics
                mlflow.log_metric("cv_mse", cv_mse)
                mlflow.log_metric("cv_rmse", cv_rmse)

                # Tag trial state
                mlflow.set_tag("trial_state", "COMPLETE")

                return score

            except optuna.TrialPruned:
                mlflow.set_tag("trial_state", "PRUNED")
                raise

            except Exception:
                mlflow.set_tag("trial_state", "FAIL")
                raise

    study.optimize(
        wrapped_objective,
        n_trials=100,
        n_jobs=2
    )

    print("Optimization completed.")
    print("Best CV RMSE:", np.sqrt(-study.best_value))


if __name__ == "__main__":
    main()
