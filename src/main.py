from data_loader import load_data
from preprocessing import preprocess_data
from train import train_model
from evaluate import evaluate_model
from save_results import save_metrics


def main():

    X, y = load_data()

    X_train, X_test, y_train, y_test = preprocess_data(
        X,
        y
    )

    model = train_model(
        X_train,
        y_train
    )

    results = evaluate_model(
        model,
        X_test,
        y_test
    )

    save_metrics(results)

    print("Model Performance")
    print("-----------------")

    print(
        "Accuracy:",
        round(results["accuracy"], 3)
    )

    print(
        "Precision:",
        round(results["precision"], 3)
    )

    print(
        "Recall:",
        round(results["recall"], 3)
    )

    print(
        "F1 Score:",
        round(results["f1_score"], 3)
    )

    print("\nConfusion Matrix:")
    print(results["confusion_matrix"])


if __name__ == "__main__":
    main()