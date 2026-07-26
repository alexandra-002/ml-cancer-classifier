from sklearn.linear_model import LogisticRegression


def train_model(X_train, y_train):
    """
    Train a logistic regression classifier.

    Returns:
        trained model
    """

    model = LogisticRegression(
        max_iter=10000,
        random_state=42
    )

    model.fit(
        X_train,
        y_train
    )

    return model