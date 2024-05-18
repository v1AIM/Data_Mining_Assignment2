import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
import pandas as pd
from clustering import KMeansClustering
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


def read_data(file_path, percentage):
    data = pd.read_csv(file_path)
    num_rows = int(len(data) * (percentage / 100))
    data = data.head(num_rows)  # Read only a percentage of data
    X = data[["IMDB Rating"]].values
    return X, data


def process_data():
    file_path = file_path_var.get()
    percentage = float(percentage_var.get())
    k = int(k_var.get())

    X, data = read_data(file_path, percentage)

    # Detecting outliers and removing them using IQR
    q1 = data["IMDB Rating"].quantile(0.25)
    q3 = data["IMDB Rating"].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers_indices = data[
        (data["IMDB Rating"] < lower_bound) | (data["IMDB Rating"] > upper_bound)
    ].index
    outliers = data.iloc[outliers_indices]

    # Removing outliers from data
    data = data.drop(outliers_indices)

    # Print lower and upper bounds
    print(f"Lower bound: {lower_bound}")
    print(f"Upper bound: {upper_bound}")

    # Now perform clustering on data without outliers
    kmeans_no_outliers = KMeansClustering(k)
    labels_no_outliers = kmeans_no_outliers.fit(data[["IMDB Rating"]].values)

    # Output clusters
    clusters = {i: [] for i in range(k)}
    for i, label in enumerate(labels_no_outliers):
        clusters[label].append((f"Cluster {label+1}", data.iloc[i]["Movie Name"]))

    # Populate tabbed interface with tables for each cluster
    for i in range(k):
        cluster_movies = [movie[1] for movie in clusters[i]]
        populate_table(cluster_movies, f"Cluster {i+1}")

    # Populate table for outliers
    populate_table(outliers["Movie Name"], "Outliers")


################################################################################


# def process_data():
#     file_path = file_path_var.get()
#     percentage = float(percentage_var.get())
#     k = int(k_var.get())

#     X, data = read_data(file_path, percentage)

#     # Calculate Z-scores
#     z_scores = (data["IMDB Rating"] - data["IMDB Rating"].mean()) / data[
#         "IMDB Rating"
#     ].std()

#     # Define threshold for outliers (e.g., Z-score greater than 3 or less than -3)
#     threshold = 3

#     # Detect outliers
#     outliers_indices = np.where(np.abs(z_scores) > threshold)[0]
#     outliers = data.iloc[outliers_indices]

#     # Removing outliers from data
#     data = data.drop(outliers_indices)

#     # Print threshold
#     print(f"Z-score threshold: {threshold}")

#     # Now perform clustering on data without outliers
#     kmeans_no_outliers = KMeansClustering(k)
#     labels_no_outliers = kmeans_no_outliers.fit(data[["IMDB Rating"]].values)

#     # Output clusters
#     clusters = {i: [] for i in range(k)}
#     for i, label in enumerate(labels_no_outliers):
#         clusters[label].append((f"Cluster {label+1}", data.iloc[i]["IMDB Rating"]))

#     # Populate tabbed interface with tables for each cluster
#     for i in range(k):
#         cluster_movies = [movie[1] for movie in clusters[i]]
#         populate_table(cluster_movies, f"Cluster {i+1}")

#     # Populate table for outliers
#     populate_table(outliers["IMDB Rating"], "Outliers")


def populate_table(data, tab_title):
    frame = ttk.Frame(tab_control)
    tab_control.add(frame, text=tab_title)

    tree = ttk.Treeview(frame, columns=("Movie Name",), show="headings")
    tree.heading("Movie Name", text="Movie Name")
    tree.grid(row=0, column=0, padx=10, pady=10)

    for item in data:
        tree.insert("", "end", values=(item,))


def browse_file():
    file_path = filedialog.askopenfilename(
        initialdir="/",
        title="Select file",
        filetypes=(("CSV files", "*.csv"), ("All files", "*.*")),
    )
    file_path_var.set(file_path)


# Create GUI
root = tk.Tk()
root.title("K-Means Clustering")

file_path_var = tk.StringVar()
percentage_var = tk.StringVar()
k_var = tk.StringVar()

file_label = tk.Label(root, text="Select CSV file:")
file_label.grid(row=0, column=0, padx=10, pady=10)

file_entry = tk.Entry(root, textvariable=file_path_var)
file_entry.grid(row=0, column=1, padx=10, pady=10)

browse_button = tk.Button(root, text="Browse", command=browse_file)
browse_button.grid(row=0, column=2, padx=10, pady=10)

percentage_label = tk.Label(root, text="Enter percentage of data to read:")
percentage_label.grid(row=1, column=0, padx=10, pady=10)

percentage_entry = tk.Entry(root, textvariable=percentage_var)
percentage_entry.grid(row=1, column=1, padx=10, pady=10)

k_label = tk.Label(root, text="Enter number of clusters (K):")
k_label.grid(row=2, column=0, padx=10, pady=10)

k_entry = tk.Entry(root, textvariable=k_var)
k_entry.grid(row=2, column=1, padx=10, pady=10)

process_button = tk.Button(root, text="Process", command=process_data)
process_button.grid(row=3, column=0, columnspan=3, padx=10, pady=10)

# Create tabbed interface for displaying tables
tab_control = ttk.Notebook(root)
tab_control.grid(row=4, column=0, columnspan=3, padx=10, pady=10)

root.mainloop()
