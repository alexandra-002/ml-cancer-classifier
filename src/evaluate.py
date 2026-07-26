from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


def evaluate_model(model, X_test, y_test):
    """
    Evaluate classification model performance.
    """

    predictions = model.predict(X_test)

    results = {
        "accuracy": accuracy_score(
            y_test,
            predictions
        ),

        "precision": precision_score(
            y_test,
            predictions
        ),

        "recall": recall_score(
            y_test,
            predictions
        ),

        "f1_score": f1_score(
            y_test,
            predictions
        ),

        "confusion_matrix": confusion_matrix(
            y_test,
            predictions
        )
    }

    return results