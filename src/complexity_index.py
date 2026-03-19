import pandas as pd

def add_complexity_dimensions(df):
    """
    Adds multi-dimensional musical complexity measures:
    - rhythmic_complexity
    - dynamic_complexity
    - structural_complexity
    - musical_complexity (overall)
    """

    # Rhythmic complexity
    rhythmic_features = [
        'tempo',
        'danceability',
        'speechiness'
    ]
    df['rhythmic_complexity'] = df[rhythmic_features].mean(axis=1)

    # Dynamic complexity
    dynamic_features = [
        'energy',
        'loudness',
        'liveness'
    ]
    df['dynamic_complexity'] = df[dynamic_features].mean(axis=1)

    # Structural / timbral complexity
    structural_features = [
        'instrumentalness',
        'acousticness',
        'duration_ms'
    ]
    df['structural_complexity'] = df[structural_features].mean(axis=1)

    # Overall musical complexity (equal weighting)
    df['musical_complexity'] = (
        df['rhythmic_complexity'] +
        df['dynamic_complexity'] +
        df['structural_complexity']
    ) / 3

    return df
