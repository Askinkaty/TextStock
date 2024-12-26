import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap


def plot_embeddings_pca(embeddings, labels=None, n_components=2, title="Embeddings PCA"):
    """
    Args:
        embeddings (np.array):  A numpy array where each row is an embedding.
        labels (list or np.array, optional): Labels corresponding to each embedding for coloring.
        n_components (int): Number of principal components to reduce to (2 or 3).
        title (str): The title of the plot.
    """
    pca = PCA(n_components=n_components)
    reduced_embeddings = pca.fit_transform(embeddings)

    plt.figure(figsize=(10, 8))
    if n_components == 2:
        if labels is not None:
            sns.scatterplot(x=reduced_embeddings[:, 0], y=reduced_embeddings[:, 1], hue=labels, palette='viridis')
        else:
            plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1])
        plt.xlabel(f"Principal Component 1 ({pca.explained_variance_ratio_[0]:.2f})")
        plt.ylabel(f"Principal Component 2 ({pca.explained_variance_ratio_[1]:.2f})")
    elif n_components == 3:
        ax = plt.figure().add_subplot(projection='3d')
        if labels is not None:
            scatter = ax.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], reduced_embeddings[:, 2], c=labels, cmap='viridis')
            plt.colorbar(scatter, label="Labels")
        else:
            ax.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], reduced_embeddings[:, 2])
        ax.set_xlabel(f"Principal Component 1 ({pca.explained_variance_ratio_[0]:.2f})")
        ax.set_ylabel(f"Principal Component 2 ({pca.explained_variance_ratio_[1]:.2f})")
        ax.set_zlabel(f"Principal Component 3 ({pca.explained_variance_ratio_[2]:.2f})")
    else:
        raise ValueError("n_components must be 2 or 3 for plotting.")

    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_embeddings_tsne(embeddings, labels=None, n_components=2, perplexity=30, n_iter=300, title="Embeddings t-SNE"):
    """
    Args:
        embeddings (np.array): A numpy array where each row is an embedding.
        labels (list or np.array, optional): Labels corresponding to each embedding for coloring.
        n_components (int): Number of dimensions to reduce to (2 or 3).
        perplexity (int): Perplexity parameter for t-SNE.
        n_iter (int): Number of iterations for t-SNE.
        title (str): The title of the plot.
    """
    tsne = TSNE(n_components=n_components, perplexity=perplexity, n_iter=n_iter, random_state=42)
    reduced_embeddings = tsne.fit_transform(embeddings)

    plt.figure(figsize=(10, 8))
    if n_components == 2:
        if labels is not None:
            sns.scatterplot(x=reduced_embeddings[:, 0], y=reduced_embeddings[:, 1], hue=labels, palette='viridis')
        else:
            plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1])
        plt.xlabel("t-SNE Dimension 1")
        plt.ylabel("t-SNE Dimension 2")
    elif n_components == 3:
        ax = plt.figure().add_subplot(projection='3d')
        if labels is not None:
            scatter = ax.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], reduced_embeddings[:, 2], c=labels, cmap='viridis')
            plt.colorbar(scatter, label="Labels")
        else:
            ax.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], reduced_embeddings[:, 2])
        ax.set_xlabel("t-SNE Dimension 1")
        ax.set_ylabel("t-SNE Dimension 2")
        ax.set_zlabel("t-SNE Dimension 3")
    else:
        raise ValueError("n_components must be 2 or 3 for plotting.")

    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_embeddings_umap(embeddings, labels=None, n_components=2, n_neighbors=10, min_dist=0.1, title="Embeddings UMAP"):
    """
    Args:
        embeddings (np.array): A numpy array where each row is an embedding.
        labels (list or np.array, optional): Labels corresponding to each embedding for coloring.
        n_components (int): Number of dimensions to reduce to (2 or 3).
        n_neighbors (int): The size of local neighborhood used for manifold approximation.
        min_dist (float): The effective minimum distance between embedded points.
        title (str): The title of the plot.
    """
    reducer = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors, min_dist=min_dist, random_state=42)
    reduced_embeddings = reducer.fit_transform(embeddings)

    plt.figure(figsize=(10, 8))
    if n_components == 2:
        if labels is not None:
            sns.scatterplot(x=reduced_embeddings[:, 0], y=reduced_embeddings[:, 1], hue=labels, palette='viridis')
        else:
            plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1])
        plt.xlabel("UMAP Dimension 1")
        plt.ylabel("UMAP Dimension 2")
    elif n_components == 3:
        ax = plt.figure().add_subplot(projection='3d')
        if labels is not None:
            scatter = ax.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], reduced_embeddings[:, 2], c=labels, cmap='viridis')
            plt.colorbar(scatter, label="Labels")
        else:
            ax.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], reduced_embeddings[:, 2])
        ax.set_xlabel("UMAP Dimension 1")
        ax.set_ylabel("UMAP Dimension 2")
        ax.set_zlabel("UMAP Dimension 3")
    else:
        raise ValueError("n_components must be 2 or 3 for plotting.")

    plt.title(title)
    plt.tight_layout()
    plt.show()