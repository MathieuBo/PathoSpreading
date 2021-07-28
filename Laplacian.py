import numpy as np


def get_laplacian(adj_mat, expression_data=None, return_in_degree=False):
    """ Compute the Degree Matrix from adjacency matrix and then compute the Laplacian Matrix.
    ---
    Inputs:
        adj_mat: adjacency matrix created  in processed_files, DataFrame
        expression_data (default: None) : expression data created in processed, DataFrame
        return_in_degree (default: False): Boolean to return in degree Laplacian matrix
    ---
    Outputs:
        laplacian: (out degree) laplacian matrix, array.
        (optional): in_degree_laplacian: in degree laplacian matrix
    """

    # Conversion to numpy
    adj_mat = adj_mat.values

    # Calculation of laplacian
    np.fill_diagonal(adj_mat, 0)

    # Extracting the max eigenvalue from our matrix and scaling the whole matrix
    if expression_data is None:
        adj_mat = adj_mat / np.max(np.real(np.linalg.eig(adj_mat)[0]))
    else:

        expression_data = expression_data.values[:, 0]
        diag_exp_data = np.diag(expression_data)

        adj_mat_exp_data = np.dot(diag_exp_data, adj_mat)  # Synuclein weighted matrix
        adj_mat = adj_mat_exp_data / np.max(np.real(np.linalg.eig(adj_mat_exp_data)[0]))

    # Computation of in- and out-degree
    out_degree = adj_mat.sum(axis=1)  # To sum over rows
    in_degree = adj_mat.sum(axis=0)  # To sum over columns

    # Out-degree Laplacian matrix
    laplacian = np.diag(out_degree) - adj_mat

    # In-degree Laplacian matrix
    in_degree_laplacian = np.diag(in_degree) - adj_mat

    if return_in_degree is True:
        return laplacian, in_degree_laplacian
    else:
        return laplacian
