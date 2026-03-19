import numpy as np
from sklearn.preprocessing import MinMaxScaler

def preprocess_data(df):
    # Drop missing values
    df = df.dropna()

    # IQR-based outlier removal
    for col in df.columns:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        df = df[(df[col] >= lower) & (df[col] <= upper)]

    # Normalize features except popularity
    scaler = MinMaxScaler()
    feature_cols = df.columns.drop('popularity')
    df[feature_cols] = scaler.fit_transform(df[feature_cols])

    return df
