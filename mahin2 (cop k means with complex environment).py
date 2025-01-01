import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

employee_data = {
    "Alice Johnson": [8, 6, 3],
    "Bob Smith": [7, 7, 4],
    "Carol White": [3, 8, 9],
    "David Brown": [4, 9, 7],
    "Eve Black": [6, 5, 6],
    "Frank Miller": [5, 7, 8],
    "Grace Lee": [9, 2, 4],
    "Hannah Turner": [6, 6, 5],
    "Isaac Harris": [4, 8, 7],
    "Jack Walker": [7, 3, 6],
    "Lily Green": [8, 5, 7],
    "Michael Scott": [6, 7, 5],
    "Nancy Adams": [4, 6, 8],
    "Oscar Roberts": [5, 9, 6],
    "Paul Carter": [7, 8, 4]
}

employees = list(employee_data.keys())
data = np.array(list(employee_data.values()))
n = len(employees)

name_to_index = {name: idx for idx, name in enumerate(employees)}
index_to_name = {idx: name for name, idx in name_to_index.items()}

def apply_kmeans(X, k):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    return kmeans.labels_

k = 4  # Number of teams
assignments = apply_kmeans(data, k)

def visualize_clusters(data, assignments, index_to_name):
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(data)

    plt.figure(figsize=(12, 10))
    for i, cluster_id in enumerate(assignments):
        plt.scatter(reduced_data[i, 0], reduced_data[i, 1], 
                    c=f"C{cluster_id}", label=f"Team {cluster_id + 1}", s=100)

    for i, name in enumerate(index_to_name.values()):
        plt.annotate(name, (reduced_data[i, 0], reduced_data[i, 1]), fontsize=10)

    plt.title("K-Means: Employee Team Assignment", fontsize=16)
    plt.xlabel("Principal Component 1", fontsize=14)
    plt.ylabel("Principal Component 2", fontsize=14)
    plt.legend(loc="upper right", fontsize=12)
    plt.grid(True)
    plt.show()

visualize_clusters(data, assignments, index_to_name)
