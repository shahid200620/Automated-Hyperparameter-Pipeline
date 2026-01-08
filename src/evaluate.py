import json
import os
import time

import numpy as np
import optuna
from sklearn.metrics import mean_squared_error

from src.data_loader import load_and_split_data
from xgboost import XGBRegressor


def generate_results(study, X_train, X_test, y_train, y_test, start_time):
    os.makedirs("outputs", exist_ok=True)

    # Best CV RMSE
    best_cv_rmse = np.sqrt(-study.best_value)

    # Train best model
    best_params = study.best_params

    model = XGBRegressor(
        **best_params,
        random_state=42,
        n_jobs=1,
        objective="reg:squarederror",
        verbosity=0
    )

    model.fit(X_train, y_train)

    # Test evaluation
    y_pred = model.predict(X_test)
    test_mse = mean_squared_error(y_test, y_pred)
    test_rmse = np.sqrt(test_mse)
    test_r2 = model.score(X_test, y_test)

    # Trial stats
    n_trials_completed = len(
        [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    )
    n_trials_pruned = len(
        [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    )

    results = {
        "n_trials_completed": n_trials_completed,
        "n_trials_pruned": n_trials_pruned,
        "best_cv_rmse": best_cv_rmse,
        "test_rmse": test_rmse,
        "test_r2": test_r2,
        "best_params": best_params,
        "optimization_time_seconds": time.time() - start_time
    }

    with open("outputs/results.json", "w") as f:
        json.dump(results, f, indent=4)
