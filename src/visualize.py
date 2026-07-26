import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay


def plot_confusion_matrix(model, X_test, y_test, model_name):
    """
    Save confusion matrix visualization.
    """

    predictions = model.predict(X_test)

    display = ConfusionMatrixDisplay.from_predictions(
        y_test,
        predictions
    )

    plt.title(f"{model_name} Confusion Matrix")

    plt.savefig(
        f"images/{model_name}_confusion_matrix.png",
        bbox_inches="tight"
    )

    plt.close()


def plot_model_comparison(csv_file):
    """
    Create bar chart comparing model performance.
    """

    data = pd.read_csv(csv_file)

    metrics = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1_Score"
    ]

    data.set_index("Model")[metrics].plot(
        kind="bar"
    )

    plt.title(
        "Model Performance Comparison"
    )

    plt.ylabel(
        "Score"
    )

    plt.ylim(
        0,
        1
    )

    plt.tight_layout()

    plt.savefig(
        "images/model_comparison.png"
    )

    plt.close()


def plot_feature_importance(model, feature_names):
    """
    Plot Random Forest feature importance.
    """

    importance = pd.Series(
        model.feature_importances_,
        index=feature_names
    )

    importance = importance.sort_values(
        ascending=False
    ).head(10)

    importance.plot(
        kind="bar"
    )

    plt.title(
        "Top 10 Important Features"
    )

    plt.ylabel(
        "Importance"
    )

    plt.tight_layout()

    plt.savefig(
        "images/feature_importance.png"
    )

    plt.close()