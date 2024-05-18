import pandas as pd
import numpy as np


class KMeansClustering:
    def __init__(self, k, max_iterations=100):
        self.k = k
        self.max_iterations = max_iterations

    def fit(self, X):
        # Initialize centroids randomly
        self.centroids = X[np.random.choice(X.shape[0], self.k, replace=False)]

        for _ in range(self.max_iterations):
            # Assign points to the nearest centroid
            distances = self._calculate_distances(X, self.centroids)
            labels = np.argmin(distances, axis=1)

            # Update centroids
            new_centroids = np.array(
                [X[labels == i].mean(axis=0) for i in range(self.k)]
            )

            # Check for convergence (It checks if the centroids have changed significantly, If they haven't the algorithm terminates early.)
            if np.allclose(self.centroids, new_centroids):
                break

            self.centroids = new_centroids

        # Calculate distances to centroids
        distances = self._calculate_distances(X, self.centroids)
        self.distances = np.min(distances, axis=1)
        return labels

    def _calculate_distances(self, X, centroids):
        distances = np.sqrt(np.sum((X[:, np.newaxis] - centroids) ** 2, axis=2))
        return distances


def read_data(file_path):
    data = pd.read_csv(file_path)
    X = data[["IMDB Rating"]].values
    return X, data


def main():
    file_path = "imdb_top_2000_movies.csv"  # Update with your file path
    k = int(input("Enter number of clusters (K): "))

    X, data = read_data(file_path)
    kmeans = KMeansClustering(k)
    labels = kmeans.fit(X)

    # Output clusters
    clusters = {i: [] for i in range(k)}
    for i, label in enumerate(labels):
        clusters[label].append(data.iloc[i]["Movie Name"])

    for i, cluster in clusters.items():
        print(f"Cluster {i+1}:\n")
        print("\n".join(cluster))
        print()

    # Detecting outliers
    outliers = data[kmeans.distances > np.percentile(kmeans.distances, 95)]
    print("Outliers:\n", outliers)


if __name__ == "__main__":
    main()
