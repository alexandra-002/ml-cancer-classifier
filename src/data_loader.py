import pandas as pd
from sklearn.datasets import load_breast_cancer


def load_data():
    """
    Load breast cancer dataset.

    Returns:
        X: Features
        y: Labels
    """

    dataset = load_breast_cancer()

    X = pd.DataFrame(
        dataset.data,
        columns=dataset.feature_names
    )

    y = pd.Series(dataset.target)

    return X, y