from data_loader import load_data


X, y = load_data()

print(X.head())

print("\nDataset size:")
print(X.shape)

print("\nLabels:")
print(y.value_counts())