import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from scipy.spatial import distance_matrix, cKDTree
from scipy.stats import pearsonr

from scipy.spatial import procrustes

# Preprocesssing functions
def compute_pseudo_distance(interaction_matrix):
    """
    Compute a pseudo-distance matrix (invert and normalize) from the interaction matrix.

    Parameters:
    interaction_matrix (csr_matrix): Tally matrix.

    Returns:
    csr_matrix: Pseudo-distance matrix.
    """
    max_value = interaction_matrix.max()
    interaction_matrix_normalized = interaction_matrix / max_value
    ones_matrix = csr_matrix(np.ones(interaction_matrix.shape))
    pseudo_dist_matrix = ones_matrix - interaction_matrix_normalized
    # pseudo_dist_matrix = StandardScaler().fit(pseudo_dist_matrix, with_mean=False)

    return pseudo_dist_matrix


def prune_interactions(interaction_matrix, min_interactions):
    """
    Prune interactions with fewer than min_interactions.

    Parameters:
    interaction_matrix (csr_matrix): Tally matrix.
    min_interactions (int): Minimum number of tallies required to keep an interaction.

    Returns:
    csr_matrix: Pruned interaction matrix.
    """
    pruned_matrix = interaction_matrix.copy()
    pruned_matrix.data[pruned_matrix.data < min_interactions] = 0
    pruned_matrix.eliminate_zeros()
    return pruned_matrix


# Reconstruction functions
def umap_driver(
    pseudo_dist_matrix,
    n_neighbors=5,
    min_dist=0.5,
    n_epochs=1000,
    metric="euclidean",
    verbose=False,
):
    """
    Run UMAP on the pseudo-distance matrix.
    """
    import umap

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=42,
        verbose=verbose,
        n_epochs=n_epochs,
        metric=metric,
    )
    interaction_umap = reducer.fit_transform(pseudo_dist_matrix)
    return interaction_umap


def densmap_driver(pseudo_dist_matrix, n_neighbors=5, min_dist=0.5, dens_lambda=0.1):
    """
    Run DensMAP on the pseudo-distance matrix.
    """
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=42,
        verbose=True,
        densmap=True,
        dens_lambda=dens_lambda,
    )
    interaction_umap = reducer.fit_transform(pseudo_dist_matrix)
    return interaction_umap


def phate_driver(pseudo_distance_matrix, random_state=42):
    """
    Run PHATE on the pseudo-distance matrix.
    """
    import phate

    phate_op = phate.PHATE(random_state=random_state)
    interaction_phate = phate_op.fit_transform(pseudo_distance_matrix)
    return interaction_phate


def tsne_driver(pseudo_distance_matrix, perplexity=1000):
    """
    Run t-SNE on the pseudo-distance matrix.
    """
    from sklearn.manifold import TSNE

    interaction_tsne = TSNE(
        n_components=2, learning_rate="auto", init="random", perplexity=perplexity
    ).fit_transform(pseudo_distance_matrix)
    return interaction_tsne


# Quality/fidelity assessment
def localreconstruction_KNN(dataset, k):
    """
    Evaluate local reconstruction using KNN.
    """
    groundtruth_x = dataset["x"]
    groundtruth_y = dataset["y"]
    umap_x = dataset["umap_1"]
    umap_y = dataset["umap_2"]

    groundtruth_points = np.column_stack((groundtruth_x, groundtruth_y))
    umap_points = np.column_stack((umap_x, umap_y))

    groundtruth_tree = cKDTree(groundtruth_points)
    umap_tree = cKDTree(umap_points)
    error_list = []

    error = 0
    for i in range(len(groundtruth_points)):

        _, groundtruth_idx = groundtruth_tree.query(groundtruth_points[i], k + 1)
        groundtruth_idx = groundtruth_idx[1:]  # Exclude the point itself

        _, umap_idx = umap_tree.query(umap_points[i], k + 1)
        umap_idx = umap_idx[1:]
        # Calculate intersection of neighbor sets
        intersection_count = len(set(groundtruth_idx) & set(umap_idx))
        # print("Intersection for point", i, ":", intersection_count)

        error += intersection_count
        error_list.append(intersection_count / k)

    tot_error = error / (k * len(groundtruth_points))

    return (tot_error, error_list)


# compares pairwise_distance matrices from interaction_umap using pearson coefficient, REQUIRES FULL DISTANCE MATRIX
def globalreconstruction(interaction_umap, dist_matrix):
    """
    Evaluate global reconstruction using Pearson correlation.
    """
    inferred_dist_matrix = distance_matrix(interaction_umap, interaction_umap, p=2)

    normalized_dist_matrix = dist_matrix / np.max(dist_matrix)
    normalized_inferred_matrix = inferred_dist_matrix / np.max(inferred_dist_matrix)

    # Flatten matrices to compute Pearson correlation
    from scipy.sparse import issparse

    # Check if the matrix is sparse
    if issparse(normalized_dist_matrix):
        flat_dist = normalized_dist_matrix.toarray().flatten()
    else:
        flat_dist = normalized_dist_matrix.flatten()  # Already dense

    if issparse(normalized_inferred_matrix):
        flat_inferred = normalized_inferred_matrix.toarray().flatten()
    else:
        flat_inferred = normalized_inferred_matrix.flatten()  # Already dense

    flat_inferred = normalized_inferred_matrix.flatten()

    correlation, _ = pearsonr(flat_dist, flat_inferred)

    return correlation


def frobeniusglobalreconstruction(interaction_umap, dist_matrix):
    """
    Evaluate global reconstruction using Frobenius norm.
    """
    inferred_dist_matrix = distance_matrix(interaction_umap, interaction_umap, p=2)
    print(inferred_dist_matrix.shape)
    normalized_dist_matrix = dist_matrix / np.max(dist_matrix)
    normalized_inferred_matrix = inferred_dist_matrix / np.max(inferred_dist_matrix)
    difference = normalized_dist_matrix - normalized_inferred_matrix
    frobenius_norm = np.linalg.norm(difference, "fro")
    return frobenius_norm


def best_knn(shape, params_dict, dist_matrix, driver, k):
    """
    Find the best parameters based on KNN similarity.
    """
    knn_error_list = []

    for keys, params in params_dict.items():
        print(params)
        interaction_matrix = simulate_concatemerization(
            bc_map=shape, dist_matrix=dist_matrix, params=params
        )
        pseudo_distance_matrix = build_pseudo_distance(interaction_matrix)
        interaction_umap = driver(pseudo_distance_matrix)

        # Ensure we are not modifying a copy of a DataFrame slice
        shape = shape.copy()
        shape["umap_1"] = pd.DataFrame(interaction_umap)[0]
        shape["umap_2"] = pd.DataFrame(interaction_umap)[1]

        knn_error, error_list = localreconstruction_KNN(shape, k)
        shape["error"] = pd.Series(error_list)
        knn_error_list.append(knn_error)

    max_index = np.argmax(knn_error_list)
    # Return the parameters associated with the index
    best_params = list(params_dict.values())[max_index]
    print("Best KNN", knn_error_list[max_index])
    return best_params


def best_frob(shape, params_dict, dist_matrix, driver):
    """
    Find the best parameters based on Frobenius norm.
    """
    frob_list = []

    for keys, params in params_dict.items():
        print(params)
        interaction_matrix = simulate_concatemerization(
            bc_map=shape, dist_matrix=dist_matrix, params=params
        )
        pseudo_distance_matrix = build_pseudo_distance(interaction_matrix)
        interaction_umap = driver(pseudo_distance_matrix)

        # Ensure we are not modifying a copy of a DataFrame slice
        shape = shape.copy()
        shape["umap_1"] = pd.DataFrame(interaction_umap)[0]
        shape["umap_2"] = pd.DataFrame(interaction_umap)[1]

        frob = frobeniusglobalreconstruction(interaction_umap, dist_matrix)
        frob_list.append(frob)

    min_index = np.argmin(frob)

    # Return the parameters associated with the index
    best_params = list(params_dict.values())[min_index]
    return best_params


def alignment(dataframe):
    """ 
    Aligns the UMAP coordinates to the original x and y coordinates using Procrustes analysis.
    """
    #normalize UMAP and scale
    dataframe["umap_1"] = (dataframe["umap_1"] - dataframe["umap_1"].min()) / (dataframe["umap_1"].max() - dataframe["umap_1"].min())
    dataframe["umap_2"] = (dataframe["umap_2"] - dataframe["umap_2"].min()) / (dataframe["umap_2"].max() - dataframe["umap_2"].min())
    max_x = dataframe["umap_1"].max()
    max_y = dataframe["umap_2"].max()
    dataframe["umap_1"] = dataframe["umap_1"] * max_x
    dataframe["umap_2"] = dataframe["umap_2"] * max_y

    # Perform Procrustes analysis
    xy = dataframe[["x", "y"]].to_numpy()
    scaledumap = dataframe[["umap_1", "umap_2"]].to_numpy()
    m1, m2, disparity = procrustes(xy, scaledumap)
    dataframe["umap_1"] = m2[:, 0]
    dataframe["umap_2"] = m2[:, 1]

    return dataframe
