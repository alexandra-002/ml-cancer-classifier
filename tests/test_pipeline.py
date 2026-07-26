import sys
from pathlib import Path

sys.path.append(
    str(Path(__file__).parent.parent)
)

from src.data_loader import load_data
from src.preprocessing import split_data, scale_data
from src.train import train_logistic_regression, train_random_forest
from src.evaluate import evaluate_model


def test_dataset_loads():

    X, y = load_data()

    assert X.shape[0] > 0
    assert X.shape[1] == 30
    assert len(y) > 0


def test_train_test_split():

    X, y = load_data()

    X_train, X_test, y_train, y_test = split_data(
        X,
        y
    )

    assert len(X_train) > 0
    assert len(X_test) > 0
    assert len(y_train) > 0
    assert len(y_test) > 0


def test_scaling():

    X, y = load_data()

    X_train, X_test, y_train, y_test = split_data(
        X,
        y
    )

    X_train_scaled, X_test_scaled = scale_data(
        X_train,
        X_test
    )

    assert X_train_scaled.shape == X_train.shape
    assert X_test_scaled.shape == X_test.shape


def test_logistic_regression_training():

    X, y = load_data()

    X_train, X_test, y_train, y_test = split_data(
        X,
        y
    )

    X_train, X_test = scale_data(
        X_train,
        X_test
    )

    model = train_logistic_regression(
        X_train,
        y_train
    )

    predictions = model.predict(X_test)

    assert len(predictions) == len(y_test)


def test_random_forest_training():

    X, y = load_data()

    X_train, X_test, y_train, y_test = split_data(
        X,
        y
    )

    X_train, X_test = scale_data(
        X_train,
        X_test
    )

    model = train_random_forest(
        X_train,
        y_train
    )

    predictions = model.predict(X_test)

    assert len(predictions) == len(y_test)


def test_model_evaluation():

    X, y = load_data()

    X_train, X_test, y_train, y_test = split_data(
        X,
        y
    )

    X_train, X_test = scale_data(
        X_train,
        X_test
    )

    model = train_logistic_regression(
        X_train,
        y_train
    )

    results = evaluate_model(
        model,
        X_test,
        y_test
    )

    assert "accuracy" in results
    assert "precision" in results
    assert "recall" in results
    assert "f1_score" in results
    assert "confusion_matrix" in results