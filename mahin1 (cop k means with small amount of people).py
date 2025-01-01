import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import normalize

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

must_link = [
    ("Alice Johnson", "Bob Smith"),
    ("Carol White", "David Brown"),
    ("Frank Miller", "Grace Lee"),
    ("Hannah Turner", "Isaac Harris"),
    ("Lily Green", "Michael Scott")
]

cannot_link = [
    ("Eve Black", "Jack Walker"),
    ("Frank Miller", "Jack Walker"),
    ("Alice Johnson", "Carol White"),
    ("Nancy Adams", "Oscar Roberts"),
    ("Paul Carter", "Grace Lee")
]

employees = list(employee_data.keys())
data = np.array(list(employee_data.values()))
n = len(employees)

name_to_index = {name: idx for idx, name in enumerate(employees)}
index_to_name = {idx: name for name, idx in name_to_index.items()}

constraints = np.zeros((n, n))
for a, b in must_link:
    i, j = name_to_index[a], name_to_index[b]
    constraints[i, j] = 1
    constraints[j, i] = 1
for a, b in cannot_link:
    i, j = name_to_index[a], name_to_index[b]
    constraints[i, j] = -1
    constraints[j, i] = -1

def cop_kmeans(X, k, constraints, max_iter=100):
    np.random.seed(42)
    n_samples, n_features = X.shape

    centroids = X[np.random.choice(range(n_samples), k, replace=False)]

    assignments = np.full(n_samples, -1)
    for iteration in range(max_iter):
        distances = euclidean_distances(X, centroids)
        new_assignments = np.full(n_samples, -1)

        for i in range(n_samples):
            sorted_indices = np.argsort(distances[i])
            for c in sorted_indices:
                valid = True
                for j in range(n_samples):
                    if constraints[i, j] == 1 and new_assignments[j] != -1 and new_assignments[j] != c:
                        valid = False
                    if constraints[i, j] == -1 and new_assignments[j] != -1 and new_assignments[j] == c:
                        valid = False
                if valid:
                    new_assignments[i] = c
                    break

        if np.all(assignments == new_assignments):
            break
        assignments = new_assignments

        for c in range(k):
            cluster_points = X[assignments == c]
            if len(cluster_points) > 0:
                centroids[c] = np.mean(cluster_points, axis=0)

    return assignments

k = 4  
assignments = cop_kmeans(data, k, constraints)

def visualize_clusters(data, assignments, index_to_name):
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(data)

    plt.figure(figsize=(12, 10))
    for i, cluster_id in enumerate(assignments):
        plt.scatter(reduced_data[i, 0], reduced_data[i, 1], 
                    c=f"C{cluster_id}", label=f"Team {cluster_id + 1}", s=100)

    for i, name in enumerate(index_to_name.values()):
        plt.annotate(name, (reduced_data[i, 0], reduced_data[i, 1]), fontsize=10)

    plt.title("COP-KMeans: Employee Team Assignment", fontsize=16)
    plt.xlabel("Principal Component 1", fontsize=14)
    plt.ylabel("Principal Component 2", fontsize=14)
    plt.legend(loc="upper right", fontsize=12)
    plt.grid(True)
    plt.show()

visualize_clusters(data, assignments, index_to_name)
