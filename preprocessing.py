import pandas as pd
import numpy as np

# Categorical feature mapping
ORIGIN_LOCATIONS = {
    'Torgersen': 0,
    'Biscoe': 1,
    'Dream': 2
}

# Five input features
FEATURES = (
    'CulmenLength',
    'CulmenDepth',
    'FlipperLength',
    'OriginLocation',
    'BodyMass'
)

# Three output classes
SPECIES_MAPPING = {
    'Adelie': 0,
    'Chinstrap': 1,
    'Gentoo': 2
}

def load_csv(file):
    return pd.read_csv(file)

def split_data(X, y, train_per_class=30):
    np.random.seed(42)
    train_indices = []
    test_indices = []

    for label in (0, 1, 2):
        indices = np.flatnonzero(y == label)

        np.random.shuffle(indices)
        train_indices.extend(indices[:train_per_class])
        test_indices.extend(indices[train_per_class:])

    np.random.shuffle(train_indices)
    np.random.shuffle(test_indices)

    return (
        X.iloc[train_indices],
        X.iloc[test_indices],
        y.iloc[train_indices],
        y.iloc[test_indices]
    )

def fill_group(s):
    valid = s.dropna().to_numpy(dtype=float)
    return s.fillna(np.mean(valid))

def preprocess_data(df):
    #Seperate the target column
    X = df.drop("Species", axis=1)
    y = df["Species"]

    #Encode originLocation Column
    X['OriginLocation'] = X['OriginLocation'].map(ORIGIN_LOCATIONS)
    #X = pd.get_dummies(X, columns=['OriginLocation'], dtype=int)

    #Encode species Column
    y = y.map(SPECIES_MAPPING)

    #Split the dataset
    X_train, X_test, y_train, y_test = split_data(X, y, 30)

    #Fill missing values in numerical columns
    for col in X_train.columns:
        if col != 'OriginLocation':
            X_filled = X.groupby('OriginLocation')[col].transform(fill_group)
            X_train[col] = X_train[col].fillna(X_filled.loc[X_train.index])
            X_test[col]  = X_test[col].fillna(X_filled.loc[X_test.index])

    #Convert the data from dataframe to arrays
    X_train = X_train.to_numpy(dtype=float)
    X_test = X_test.to_numpy(dtype=float)

    #Normalization
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    std[std == 0] = 1
    X_train_scaled = (X_train - mean) / std
    X_test_scaled = (X_test - mean) / std

    #One hot encoding
    y_train_int = y_train.to_numpy(dtype=int)
    y_test_int = y_test.to_numpy(dtype=int)
    y_train_encoded = np.eye(3)[y_train_int]
    y_test_encoded = np.eye(3)[y_test_int]

    return {
        "X_train": X_train,
        "X_test": X_test,
        "X_train_scaled": X_train_scaled,
        "X_test_scaled": X_test_scaled,
        "y_train": y_train_int,
        "y_test": y_test_int,
        "y_train_encoded": y_train_encoded,
        "y_test_encoded": y_test_encoded,
        "mean": mean,
        "std": std
    }