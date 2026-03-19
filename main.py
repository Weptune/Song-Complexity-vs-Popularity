from src.load_data import load_dataset
from src.preprocess import preprocess_data
from src.complexity_index import add_complexity_dimensions
from src.clustering import perform_clustering
from src.classification import classify_songs
from src.association_rules import mine_rules

df = load_dataset("data/spotify_audio_features.csv")
df = preprocess_data(df)
df = add_complexity_dimensions(df)
df = perform_clustering(df)
classify_songs(df)
mine_rules(df)

print("Pipeline executed successfully.")
