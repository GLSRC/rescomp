# -*- coding: utf-8 -*-
""" Measures for the calculation of generalized local neighborhoods """

import numpy as np
# import scipy.sparse.csgraph
import sklearn.cluster
# import scipy.cluster.hierarchy


# TODO: add synonym dict for cluster method
# TODO this should use the LocESN internal functions
# TODO: Comments, docstring, ...
def find_local_neighborhoods(locality_matrix, neighbors, cores=1, cluster_method="hacky_loc_neighbors",
                             cluster_linkage="average"):
    input_dim = locality_matrix.shape[1]

    if input_dim % cores == 0:
        nr_nbhds = input_dim // cores
    elif cluster_method == "hacky_loc_neighbors":
        raise Exception(
            "for cluster_methods %s, the input_dim %d must be an integer multiple of the number of "
            "cores %d" % (cluster_method, input_dim, cores))
    else:
        nr_nbhds = input_dim // cores + 1

    # Find core labels
    if cores == 1 or cluster_method == "hacky_loc_neighbors":
        core_labels = []
        for nbhd_index in range(nr_nbhds):
            core_labels += [nbhd_index] * cores

        core_labels = np.array(core_labels)

    elif cluster_method == "agglomerative":
        # "With affinity='precomputed', the input matrix is interpreted as a matrix of distances between observations"
        #  https://stackoverflow.com/questions/47321133/sklearn-hierarchical-agglomerative-clustering-using-similarity-matrix
        #  Hence, we need to change the locality matrix to a matrix of distances
        # meh = np.amax(locality_matrix)
        distance_matrix = -locality_matrix + np.amax(locality_matrix)

        # # NOTE: Calculating, and plotting, the whole tree/dendogram only works
        # #  if the model has a distances_ parameter, which
        # #  AgglomerativeClustering only returns in recent scipy versions and
        # #  with n_clusters=None, and distance_threshold not None
        # model = sklearn.cluster.AgglomerativeClustering(
        #     affinity='precomputed', n_clusters=None,
        #     linkage=cluster_linkage, compute_full_tree=True,
        #     distance_threshold=0)
        # model = model.fit(distance_matrix)
        # core_labels = model.labels_
        #
        # # # import scipy.cluster.hierarchy
        # # import scipy.spatial.distance as ssd
        # # distance_vector_condensed = ssd.squareform(distance_matrix + distance_matrix.T)
        # # linkage = scipy.cluster.hierarchy.linkage(distance_vector_condensed)
        # # dendro = scipy.cluster.hierarchy.dendrogram(linkage)
        # # plt.show()
        # #
        # # # https://scikit-learn.org/stable/auto_examples/cluster/plot_agglomerative_dendrogram.html
        # #
        # # # plt.title('Hierarchical Clustering Dendrogram')
        # # # plot the top three levels of the dendrogram
        # # linkage_matrix = scipy.cluster.hierarchy.linkage(locality_matrix)
        # # scipy.cluster.hierarchy.dendrogram(linkage_matrix)
        #
        # def plot_dendrogram(model, **kwargs):
        #     # Create linkage matrix and then plot the dendrogram
        #
        #     # create the counts of samples under each node
        #     counts = np.zeros(model.children_.shape[0])
        #     n_samples = len(model.labels_)
        #     for i, merge in enumerate(model.children_):
        #         current_count = 0
        #         for child_idx in merge:
        #             if child_idx < n_samples:
        #                 current_count += 1  # leaf node
        #             else:
        #                 current_count += counts[child_idx - n_samples]
        #         counts[i] = current_count
        #
        #     linkage_matrix = np.column_stack([model.children_, model.distances_,
        #                                       counts]).astype(float)
        #
        #     # Plot the corresponding dendrogram
        #     scipy.cluster.hierarchy.dendrogram(linkage_matrix, **kwargs)
        #
        # matplotlib.use("macosx")
        # plot_dendrogram(model)
        # plt.show()

        # model = sklearn.cluster.AgglomerativeClustering(
        #     affinity='precomputed', n_clusters=20,
        #     linkage=cluster_linkage, compute_full_tree=True,
        #     distance_threshold=None)
        model = sklearn.cluster.AgglomerativeClustering(
            affinity='precomputed', n_clusters=nr_nbhds,
            linkage=cluster_linkage, compute_full_tree=True)

        model = model.fit(distance_matrix)
        core_labels = model.labels_

    else:
        raise Exception("cluster_method %s not implemented" % cluster_method)

    neighborhood_matrix = np.zeros(shape=(nr_nbhds, input_dim))

    if cluster_linkage == "average":
        # put the cores into the neighborhood_matrix

        # TODO this should use the LocESN internal functions
        # NOTE This edits the neighborhood_matrix using the slice nbhd
        for nbhd_index in range(nr_nbhds):
            nbhd = neighborhood_matrix[nbhd_index]
            np.sum(locality_matrix[core_labels == nbhd_index], axis=0, out=nbhd)

            cutoff_for_furtherst_nb = np.sort(nbhd)[-(neighbors + cores)]

            # assign 2 for core, 1 for neighbor, 0 else in a kinda bad way,
            #  because I couldn't figure out a smarter one
            nbhd[core_labels == nbhd_index] = np.inf
            nbhd[nbhd < cutoff_for_furtherst_nb] = - np.inf
            nbhd[np.isfinite(nbhd)] = 1
            nbhd[nbhd == np.inf] = 2
            nbhd[nbhd == - np.inf] = 0

    else:
        raise Exception("cluster_linkage %s not implemented" % cluster_linkage)

    neighborhood_matrix = neighborhood_matrix.astype(int)
    return neighborhood_matrix


pass  # TODO: Needed?


# def generate_locality_mst(locality_matrix):
#     # the 0.1 is added as an edge weight of 0 is counted as no connection in the scipy mst algorithm
#     anti_locality_matrix = 1 - locality_matrix + 0.1
#     return scipy.sparse.csgraph.minimum_spanning_tree(anti_locality_matrix).toarray()


# from https://stackoverflow.com/questions/20491028/optimal-way-to-compute-pairwise-mutual-information-using-numpy
def shan_entropy(c):
    c_normalized = c / float(np.sum(c))
    c_normalized = c_normalized[np.nonzero(c_normalized)]
    H = -sum(c_normalized * np.log2(c_normalized))
    return H


def nmi_ts(x, y, bins=None):
    # # number of bins Sqrt(n/5) recommended from https://stats.stackexchange.com/questions/179674/number-of-bins-when-computing-mutual-information
    # alex + christoph used Sqrt(n/4) though, so that's what I'm using here too
    # if bins == None: bins = int(np.sqrt(min(len(x), len(y))/4))

    # # BS normalization, but kinda reasonable results
    # # https://stackoverflow.com/questions/20491028/optimal-way-to-compute-pairwise-mutual-information-using-numpy
    # # normalizing by dividing through np.log(10)= figured out by plugging in the same time series twice. It's wrong though
    # c_xy = np.histogram2d(x, y, bins)[0]
    # mi = sklearn.metrics.mutual_info_score(None, None, contingency=c_xy)
    # nmi = mi / np.log(10)

    # # total BS
    # https: // stackoverflow.com / questions / 20491028 / optimal - way - to - compute - pairwise - mutual - information - using - numpy
    # c_x = np.histogram(x, bins=bins)[0]
    # c_y = np.histogram(y, bins=bins)[0]
    # nmi = sklearn.metrics.normalized_mutual_info_score(c_x, c_y, average_method='arithmetic')

    # # literally no results at all
    # https: // elife - asu.github.io / PyInform / starting.html
    # nmi = pyinform.mutualinfo.mutual_info(x, y, local=False)

    # # not normalizable it seems, but reasonable results as well
    # # the entropy fct. only seems to work for at least 2 dim data
    # # https://gist.github.com/GaelVaroquaux/ead9898bd3c973c40429#file-mutual_info-py
    # mi = mutual_info_stolen.mutual_information_2d(x, y, normalized=False)
    # nmi = mi
    # # nmi = mi/np.sqrt(mutual_info_stolen.entropy(x) * mutual_info_stolen.entropy(y))

    # Far from perfect, but at least I know what's going on, as it's what Alex
    # and Christoph used
    # TODO Seems to fail for degenerate inputs, like all ones
    # https://stackoverflow.com/questions/20491028/optimal-way-to-compute-pairwise-mutual-information-using-numpy
    c_xy = np.histogram2d(x, y, bins)[0]
    c_x = np.histogram(x, bins)[0]
    c_y = np.histogram(y, bins)[0]
    h_x = shan_entropy(c_x)
    h_y = shan_entropy(c_y)
    h_xy = shan_entropy(c_xy)
    mi = h_x + h_y - h_xy
    # nmi = mi
    nmi = mi / np.sqrt(h_x * h_y)

    return nmi


# TODO: Normalize the time series?
def nmi_loc(matrix, bins=None, rowvar=False):
    """
    Calculate the paiwise normalized mutual information for a set of time series
    Args:
        matrix (np.array): 2dim matrix with different time series on the columns/rows
        bins (int): number of bins to divide the data in for estimating PDFs
        rowvar (bool): True if each row is a time series
            False if each column is a time series (our typical case)
            Variable named to be consistent with np.corrcoef()

    Returns: Normalized mutual information matrix

    """

    if bins == None: bins = int(np.sqrt(matrix.shape[0] / 4))
    if rowvar == True: matrix = matrix.T

    n_ts = matrix.shape[1]
    nmi_matrix = np.zeros((n_ts, n_ts))

    for ts1_index in range(n_ts):
        for ts2_index in range(n_ts):
            if ts1_index == ts2_index:
                nmi_matrix[ts1_index, ts2_index] = 1
            elif ts1_index < ts2_index:
                nmi_matrix[ts1_index, ts2_index] = \
                    nmi_ts(matrix[:, ts1_index], matrix[:, ts2_index], bins=bins)
            else:
                nmi_matrix[ts1_index, ts2_index] = \
                    nmi_matrix[ts2_index, ts1_index]

    return nmi_matrix


def sn_loc(matrix, rowvar=False):
    if rowvar == True: matrix = matrix.T

    input_dim = matrix.shape[1]

    pass  # --- Just use the spatial local states pathak neighborhood

    loc_inverse_dist_matrix = np.zeros((input_dim, input_dim))

    # NOTE loc neighborhood doesn't have a fallback for when there are 2 distances in a row that are both exactly at the cutoff, but the nr neighbors is odd. This tie breaker solves this, by always choosing the element with the smaller index.
    tie_breaker = 1e-8

    for row in range(input_dim):
        for i in range(input_dim):
            # loc_inverse_dist_matrix[row, i] = 1 - abs((i - row)/input_dim)
            loc_inverse_dist_matrix[row, i] = \
                input_dim - min(
                    abs(i + tie_breaker - row),
                    abs(i + tie_breaker - (row + input_dim)),
                    abs(i + tie_breaker - (row - input_dim))
                )

    return loc_inverse_dist_matrix


def cc_loc(matrix, rowvar=False):
    # this is the cross correlation coefficient from the lecture
    corr_matrix = np.corrcoef(matrix, rowvar=rowvar)
    corr_loc_matrix = np.abs(corr_matrix)
    return corr_loc_matrix
