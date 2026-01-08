import os
import random
import time

import numpy as np
import mlflow
import optuna

from src.data_loader import load_and_split_data
from src.objective import objective


def main():
    # -----------------------------
    # Reproducibility
    # -----------------------------
    random.seed(42)
    np.random.seed(42)

    # -----------------------------
    # MLflow setup
    # -----------------------------
    mlflow.set_experiment("optuna-xgboost-optimization")

    print("MLflow experiment set: optuna-xgboost-optimization")


if __name__ == "__main__":
    main()
