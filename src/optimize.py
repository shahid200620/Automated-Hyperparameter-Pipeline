import os
import random
import time
import json
from src.evaluate import generate_results

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

        # -----------------------------
    # Train best model on full training data
    # -----------------------------
    best_params = study.best_params

    # best_model = optuna.integration.XGBoostPruningCallback  # dummy reference to ensure optuna-xgboost compatibility
    from xgboost import XGBRegressor

    model = XGBRegressor(
        **best_params,
        random_state=42,
        n_jobs=1,
        objective="reg:squarederror",
        verbosity=0
    )

    model.fit(X_train, y_train)

    # -----------------------------
    # Evaluate on test set
    # -----------------------------
    y_pred = model.predict(X_test)

    test_mse = mean_squared_error(y_test, y_pred)
    test_rmse = np.sqrt(test_mse)
    test_r2 = model.score(X_test, y_test)

    print("Test RMSE:", test_rmse)
    print("Test R2:", test_r2)

    # -----------------------------
    # MLflow final run for best model
    # -----------------------------
    with mlflow.start_run(run_name="best_model_run"):
        mlflow.set_tag("best_model", "true")

        mlflow.log_params(best_params)

        mlflow.log_metric("test_mse", test_mse)
        mlflow.log_metric("test_rmse", test_rmse)
        mlflow.log_metric("test_r2", test_r2)

        mlflow.xgboost.log_model(
            model,
            artifact_path="model"
        )

        # -----------------------------
    # Generate Optuna visualizations
    # -----------------------------
    from optuna.visualization import (
        plot_optimization_history,
        plot_param_importances
    )

    import matplotlib.pyplot as plt

    os.makedirs("outputs", exist_ok=True)

    # Optimization history plot
    fig1 = plot_optimization_history(study)
    fig1.write_image("outputs/optimization_history.png")

    # Parameter importance plot
    fig2 = plot_param_importances(study)
    fig2.write_image("outputs/param_importance.png")

    # Log plots as MLflow artifacts
    with mlflow.start_run(run_name="optuna_visualizations"):
        mlflow.log_artifact("outputs/optimization_history.png")
        mlflow.log_artifact("outputs/param_importance.png")

        # -----------------------------
    # Generate results.json
    # -----------------------------
    generate_results(
        study,
        X_train,
        X_test,
        y_train,
        y_test,
        start_time
    )





if __name__ == "__main__":
    main()
