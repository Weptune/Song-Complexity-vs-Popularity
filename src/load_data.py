import pandas as pd

def load_dataset(path):
    df = pd.read_csv(path)

    features = [
        'danceability',
        'energy',
        'loudness',
        'speechiness',
        'acousticness',
        'instrumentalness',
        'liveness',
        'valence',
        'tempo',
        'duration_ms',
        'popularity'
    ]

    df = df[features]
    return df
