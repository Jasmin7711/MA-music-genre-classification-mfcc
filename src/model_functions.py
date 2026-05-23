# src/model_functions.py

import pandas as pd
from sklearn.preprocessing import StandardScaler


def prepare_features_labels(
    df_train,
    df_val,
    df_test,
    target_col="genre",
    normalize_features=True,
    reduction_factor=1.0,
    random_state=42
):
    """
    Prepare feature matrices and labels for classical machine learning models.
    """
    df_train = df_train[df_train[target_col].notna()].copy()
    df_val = df_val[df_val[target_col].notna()].copy()
    df_test = df_test[df_test[target_col].notna()].copy()

    if reduction_factor < 1.0:
        df_train = (
            df_train
            .groupby(target_col, group_keys=False)
            .apply(lambda x: x.sample(frac=reduction_factor, random_state=random_state))
            .reset_index(drop=True)
        )

        print(
            f"Training dataset reduced to {len(df_train)} samples "
            f"({reduction_factor * 100:.1f}%)"
        )

    meta_cols = [
        "track_id",
        "sample",
        "segment",
        "genre",
        "track_genre_top",
        "split",
        "file_name",
        "is_augmented"
    ]

    feature_cols = [
        c for c in df_train.columns
        if c not in meta_cols
    ]

    if normalize_features:
        scaler = StandardScaler()

        X_train = scaler.fit_transform(df_train[feature_cols])
        X_val = scaler.transform(df_val[feature_cols])
        X_test = scaler.transform(df_test[feature_cols])
    else:
        X_train = df_train[feature_cols].to_numpy()
        X_val = df_val[feature_cols].to_numpy()
        X_test = df_test[feature_cols].to_numpy()

    y_train = df_train[target_col]
    y_val = df_val[target_col]
    y_test = df_test[target_col]

    return X_train, y_train, X_val, y_val, X_test, y_test