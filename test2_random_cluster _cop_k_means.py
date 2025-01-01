import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist


np.random.seed(42)
cluster_1 = np.random.normal(loc=[2, 2], scale=0.8, size=(50, 2))
cluster_2 = np.random.normal(loc=[8, 8], scale=0.8, size=(50, 2))
cluster_3 = np.random.normal(loc=[5, 1], scale=0.8, size=(50, 2))
X = np.vstack([cluster_1, cluster_2, cluster_3])


k = 3


must_link = (0, 1)  
must_link = (1, 55)
cannot_link = (2, 3)  


np.random.seed(42)
centers = X[np.random.choice(X.shape[0], k, replace=False), :]

max_iter = 100


colors = ['r', 'g', 'b']
constraint_color = 'y'

plt.figure(figsize=(10, 8))

for iteration in range(max_iter):

    distances = cdist(X, centers, 'euclidean')
    cluster_assignments = np.argmin(distances, axis=1)


    if cluster_assignments[must_link[0]] != cluster_assignments[must_link[1]]:
        cluster_assignments[must_link[1]] = cluster_assignments[must_link[0]]


    if cluster_assignments[cannot_link[0]] == cluster_assignments[cannot_link[1]]:

        for cluster_idx in range(k-1, -1, -1):
            if cluster_idx != cluster_assignments[cannot_link[0]]:
                cluster_assignments[cannot_link[1]] = cluster_idx
                break


    new_centers = np.array([X[cluster_assignments == i].mean(axis=0) if np.any(cluster_assignments == i) else centers[i]
                            for i in range(k)])


    plt.clf()
    for i in range(k):
        points_in_cluster = X[cluster_assignments == i]
        plt.scatter(points_in_cluster[:, 0], points_in_cluster[:, 1], color=colors[i], label=f'Cluster {i}', s=50)
        plt.scatter(centers[i, 0], centers[i, 1], color=colors[i], edgecolor='k', marker='x', s=200, label=f'Center {i}')


    for idx, (x, y) in enumerate(X):
        plt.text(x, y, str(idx), fontsize=8, color='black', ha='center', va='center')


    plt.plot([X[must_link[0], 0], X[must_link[1], 0]], 
             [X[must_link[0], 1], X[must_link[1], 1]], 
             color=constraint_color, linestyle='--', linewidth=2, label="Must-Link Constraint")


    if cluster_assignments[cannot_link[0]] == cluster_assignments[cannot_link[1]]:
        plt.plot([X[cannot_link[0], 0], X[cannot_link[1], 0]], 
                 [X[cannot_link[0], 1], X[cannot_link[1], 1]], 
                 color='black', linestyle=':', linewidth=2, label="Cannot-Link Violation")

    plt.title(f"Iteration {iteration + 1}")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    plt.pause(1)  


    if np.allclose(centers, new_centers, atol=1e-4):
        break
    centers = new_centers

plt.show()


print("Final Cluster Centers:\n", centers)
print("Final Cluster Assignments:\n", cluster_assignments)
