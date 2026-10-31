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

    rng = np.random.default_rng(42)

    train_indices = []
    test_indices = []

    for label in (0, 1, 2):

        indices = np.flatnonzero(y == label)

        if len(indices) != 50:
            raise ValueError(
                f"Each class must have 50 samples, "
                f"but class {label} has {len(indices)}."
            )

        rng.shuffle(indices)

        train_indices.extend(indices[:train_per_class])
        test_indices.extend(indices[train_per_class:])

    rng.shuffle(train_indices)
    rng.shuffle(test_indices)

    return (
        X[train_indices],
        X[test_indices],
        y[train_indices],
        y[test_indices]
    )

def preprocess_data(df):

    # Validate dataset
    if len(df) != 150:
        raise ValueError("Dataset must contain exactly 150 samples.")

    if df['Species'].isna().any():
        raise ValueError("Species column contains missing values.")

    if set(df['Species'].unique()) != set(SPECIES_MAPPING.keys()):
        raise ValueError("Dataset must contain the three expected species.")

    if not (df['Species'].value_counts() == 50).all():
        raise ValueError("Each species must contain exactly 50 samples.")

    # Select features and target
    working_df = df[list(FEATURES) + ['Species']].copy()

    # Encode OriginLocation
    values = working_df['OriginLocation'].astype('string').str.strip()

    mapping = {
        key.casefold(): value
        for key, value in ORIGIN_LOCATIONS.items()
    }

    encoded_mapping = {
        '0': 0,
        '1': 1,
        '2': 2
    }

    combined_mapping = {**mapping, **encoded_mapping}

    values = values.str.casefold()

    unknown = values.notna() & ~values.isin(combined_mapping)

    if unknown.any():
        raise ValueError(
            f"Unknown OriginLocation values: "
            f"{values[unknown].unique().tolist()}"
        )

    working_df['OriginLocation'] = values.map(combined_mapping)

    # Convert remaining features to numeric
    for feature in FEATURES:
        working_df[feature] = pd.to_numeric(
            working_df[feature], errors='coerce'
        )

    # Encode species
    y = working_df['Species'].map(SPECIES_MAPPING).to_numpy()

    # Extract input features
    X = working_df[list(FEATURES)].to_numpy(dtype=float)

    # Split data
    X_train, X_test, y_train, y_test = split_data(
        X, y, train_per_class=30
    )

    # Handle missing values using training data only
    for col, feature in enumerate(FEATURES):

        valid = X_train[
            np.isfinite(X_train[:, col]), col
        ]

        if len(valid) == 0:
            raise ValueError(f"No valid values for {feature}.")

        if feature == 'OriginLocation':
            values, counts = np.unique(
                valid, return_counts=True
            )
            replacement = values[np.argmax(counts)]
        else:
            replacement = valid.mean()

        X_train[:, col] = np.where(
            np.isfinite(X_train[:, col]),
            X_train[:, col],
            replacement
        )

        X_test[:, col] = np.where(
            np.isfinite(X_test[:, col]),
            X_test[:, col],
            replacement
        )

    # Normalize using training data only
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)

    std[std == 0] = 1.0

    X_train_scaled = (X_train - mean) / std
    X_test_scaled = (X_test - mean) / std

    # One-hot encoding
    y_train_encoded = np.eye(3)[y_train]

    return {
        "X_train": X_train,
        "X_test": X_test,
        "X_train_scaled": X_train_scaled,
        "X_test_scaled": X_test_scaled,
        "y_train": y_train,
        "y_test": y_test,
        "y_train_encoded": y_train_encoded,
        "mean": mean,
        "std": std
    }