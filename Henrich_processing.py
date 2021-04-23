import pandas as pd

def process_pathdata(exp_data, connectivity_ipsi, connectivity_contra, condition):

    print("Using the Data set from Henrich et al., 2020")
    # Reordering the data

    ipsi = " ipsi"
    contra = " kontra"
    col_idx_ipsi = [i for i in exp_data.columns if ipsi in i]
    col_idx_contra = [i for i in exp_data.columns if contra in i]
    data_ipsi = exp_data[col_idx_ipsi]
    data_contra = exp_data[col_idx_contra]

    # Renaming the data with attributed "i" or "c" and no parenthesis
    col_idx_ipsi = ["i"+i.split(ipsi)[0] for i in col_idx_ipsi]
    col_idx_contra = ["c"+i.split(contra)[0] for i in col_idx_contra]

    data_ipsi.columns = col_idx_ipsi
    data_contra.columns = col_idx_contra

    #Creation of conn_names & Using the connectivity order in our pathology data
    conn_names = [i for i in connectivity_contra.columns]
    conn_names_ipsi = ["i"+i for i in conn_names]
    conn_names_contra = ["c"+i for i in conn_names]

    data_ipsi = data_ipsi[conn_names_ipsi]
    data_contra = data_contra[conn_names_contra]

    # Creation of path_data: As a column matrix (similar to grp_mean in Henderson code)

    path_data = pd.concat([exp_data[exp_data.columns[0:2]], data_ipsi, data_contra], axis=1)
    path_data = path_data[path_data[path_data.columns[1]] == condition]
    array = [path_data.columns[i] for i in range(0, len(path_data.columns)) if i != 1]
    path_data_filtered = path_data[array]
    path_data_transposed = path_data_filtered.transpose()
    path_data_transposed.columns = path_data_transposed.iloc[0]
    path_data = path_data_transposed.drop(path_data_transposed.index[[0]])
    print("Pathology dataset filtered and computed for the condition:", condition)

    # Creation of ROInames
    ROInames = [i for i in path_data.index]

    # Creation of the connectivity Laplacian

    W = pd.concat([pd.concat([connectivity_ipsi, connectivity_contra], axis=1),
                   pd.concat([connectivity_contra, connectivity_ipsi], axis=1)], axis=0)

    # Checking if the matrix was tiled properly
    if (((W.iloc[0:57, 0:57] != W.iloc[0:57, 0:57]).sum()).sum() > 0):  # Summing over columns and then rows
        print("!!! Adjacency matrix: failed concatenation !!!")  # If False the double sum equals 0
    else:
        print('Adjacency matrix: successful concatenation')

    return W, path_data, conn_names, ROInames