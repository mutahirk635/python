
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from K_Mean_Algorithem import K_Mean

# Define centroids and cluster standard deviation
centroids = [(2, 2), (8, 3)]

cluster_std = [1, 1]

# Generate sample data
X, Y = make_blobs(n_samples=500, centers=centroids, random_state=2, n_features=2, cluster_std=cluster_std)

# Initialize and use the K_Mean class
km = K_Mean(n_cluster=2,max_iteration=100)
Y= km.Predict(X)

# # Separate datasets based on true cluster labels
# dataset_1 = X[y == 0]
# dataset_2 = X[y == 1]
#
# # Plot each dataset with different styles
# plt.scatter(dataset_1[:, 0], dataset_1[:, 1], color='red', s=50, label='Dataset 1', marker='o')  # Style for Dataset 1
# plt.scatter(dataset_2[:, 0], dataset_2[:, 1], color='blue', s=70, label='Dataset 2', marker='^')  # Style for Dataset 2
#
# # Add legend and labels
# plt.legend()
# plt.title("Visually Differentiated Datasets")
# plt.xlabel("Feature 1")
# plt.ylabel("Feature 2")
# plt.show()