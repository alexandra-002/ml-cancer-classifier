from data_loader import load_data
from preprocessing import preprocess_data


def main():

    X, y = load_data()

    X_train, X_test, y_train, y_test = preprocess_data(X, y)

    print("Training data shape:")
    print(X_train.shape)

    print("\nTesting data shape:")
    print(X_test.shape)


if __name__ == "__main__":
    main()