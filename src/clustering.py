from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def perform_clustering(df, k=3):
    X = df[['musical_complexity', 'popularity']]

    kmeans = KMeans(n_clusters=k, random_state=42)
    df['cluster'] = kmeans.fit_predict(X)

    plt.scatter(
        df['musical_complexity'],
        df['popularity'],
        c=df['cluster']
    )
    plt.xlabel("Musical Complexity")
    plt.ylabel("Popularity")
    plt.title("Clustering of Songs")
    plt.savefig("results/plots/clusters.png")
    plt.close()

    return df
