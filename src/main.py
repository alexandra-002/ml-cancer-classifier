from data_loader import load_data
from preprocessing import split_data, scale_data
from train import (
    train_logistic_regression,
    train_random_forest
)
from evaluate import evaluate_model
from save_results import save_metrics
from visualize import (
    plot_confusion_matrix,
    plot_model_comparison,
    plot_feature_importance
)


def main():

    X, y = load_data()

    X_train, X_test, y_train, y_test = split_data(
        X,
        y
    )

    X_train, X_test = scale_data(
        X_train,
        X_test
    )


    logistic_model = train_logistic_regression(
        X_train,
        y_train
    )

    random_forest_model = train_random_forest(
        X_train,
        y_train
    )


    logistic_results = evaluate_model(
        logistic_model,
        X_test,
        y_test
    )

    random_forest_results = evaluate_model(
        random_forest_model,
        X_test,
        y_test
    )


    print("Logistic Regression")
    print(logistic_results)

    print("\nRandom Forest")
    print(random_forest_results)


    save_metrics(
        logistic_results,
        "logistic_regression"
    )

    save_metrics(
        random_forest_results,
        "random_forest"
    )

    plot_confusion_matrix(
        logistic_model,
        X_test,
        y_test,
        "logistic_regression"
    )


    plot_confusion_matrix(
        random_forest_model,
        X_test,
        y_test,
        "random_forest"
    )


    plot_model_comparison(
        "results/model_metrics.csv"
    )


    plot_feature_importance(
        random_forest_model,
        X.columns
    )


if __name__ == "__main__":
    main()