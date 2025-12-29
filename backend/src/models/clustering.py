import pandas as pd 
from sklearn.cluster import KMeans 

class CareerCluster:
    def __init__(self, n_clusters=10):
        "K-Menas clustering model"
        self.n_clusters = n_clusters
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
        self.labels = None

    def fit_predict(self, vectors):
        "Fit the K-Means model and predict cluster labels"
        print(f"Clustering careers into {self.n_clusters} groups...")
        self.labels = self.kmeans.fit_predict(vectors)

        return self.labels