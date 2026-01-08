from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split


def load_and_split_data(test_size=0.2, random_state=42):
    """
    Load California Housing dataset and split into train and test sets.

    Parameters:
    test_size (float): Proportion of dataset to include in test split
    random_state (int): Random seed for reproducibility

    Returns:
    X_train, X_test, y_train, y_test
    """
    data = fetch_california_housing(as_frame=True)
    X = data.data
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state
    )

    return X_train, X_test, y_train, y_test
