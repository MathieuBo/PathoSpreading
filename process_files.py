import pandas as pd
import numpy as np

def process_pathdata(exp_data, connectivity_ipsi, connectivity_contra, nb_region=None):

    """
    Function that processes the quantified alpha-synuclein data and the connectivity maps (ipsilateral and controlateral)
    Chosing a number of regions will return segmented part of W and Path_data which first data elements will be the
    values of the seed CPu.
    Possibility to implement an argument seed to consider which element is the seed and subsequently include it
    when nb_regions non-null
    ---
    Inputs:
        exp_data --> Table that contains the values of the quantified alpha-synuclein pathology measured at different
        timepoints and with different conditions.
        connectivity_ipsi/contra --> Connectivity tables, the index and the columns are the regions of interest


    ---
    Outputs:
        W --> adjacency matrix shaped in a DataFrame
        path_data --> experimental DataFrame that contains the alpha-syn quantification
        conn_names --> List of brain region names
        ROI_names --> ordered list of iRegions and then cRegions
    """
    conn_names = [i.split(' (')[0] for i in connectivity_contra.columns]
        # Extract the name of the brain regions but not the subregions
    path_names = exp_data.columns[2::]

    c_path_names_contra = [path_names[i] for i in range(0, len(path_names)) if
                           path_names[i][0] == "c"]  # Extraction of the names of the regions starting by a "c"
    path_names_contra = [i[1:] for i in c_path_names_contra]  # Same without the first letter (to fit with conn.names)

    i_path_names_ipsi = [path_names[i] for i in range(0, len(path_names)) if path_names[i][0] == "i"]
    path_names_ipsi = [i[1:] for i in i_path_names_ipsi]

    # Order path and connectivity by the same name
    path_data_ipsi = exp_data.loc[:, i_path_names_ipsi]  # Creates a DataFrame with reorganized columns
    path_data_contra = exp_data.loc[:, c_path_names_contra]

    ordered_matched_ipsi = []
    for i in range(0, len(conn_names)):  # len(conn_names) = 58
        for k in range(0, len(path_names_ipsi)):  # len(path_names_ipsi) =
            if path_names_ipsi[k] == conn_names[i]:
                ordered_matched_ipsi.append(k)
                # Returns a list containing the ranks of path_names_ipsi that fits conn.names

    ordered_matched_contra = []
    for i in range(0, len(conn_names)):
        for k in range(0, len(path_names_contra)):
            if path_names_contra[k] == conn_names[i]:
                ordered_matched_contra.append(k)
                # Returns a list containing the ranks of path_names_contra that fits conn.names

    # Creation of path_data
    if nb_region==None:
        path_data = pd.concat([exp_data.loc[:, exp_data.columns[0:2]._index_data],
                           path_data_ipsi.loc[:, path_data_ipsi.columns._index_data[ordered_matched_ipsi]],
                           path_data_contra.loc[:, path_data_contra.columns._index_data[ordered_matched_contra]]],
                          axis=1)

        path_data = path_data.rename(columns={"MBSC Region": "Conditions"})  # Renaming a column
    else:
        path_data = pd.concat([exp_data.loc[:, exp_data.columns[0:2]._index_data],
                               path_data_ipsi.loc[:, path_data_ipsi.columns._index_data[ordered_matched_ipsi]],
                               path_data_contra.loc[:, path_data_contra.columns._index_data[ordered_matched_contra]]],
                              axis=1)
        path_data = path_data.rename(columns={"MBSC Region": "Conditions"})  # Renaming a column
        path_data_i = path_data[['Time post-injection (months)','Conditions','iCPu'] +
                                [i for i in path_data_ipsi.loc[:, path_data_ipsi.columns._index_data[ordered_matched_ipsi]] if i not in ['Time post-injection (months)', 'Conditions', 'iCPu']]]
        path_data_c= path_data[['Time post-injection (months)','Conditions','cCPu'] +
                               [i for i in path_data_contra.loc[:, path_data_contra.columns._index_data[ordered_matched_contra]] if i not in ['Time post-injection (months)', 'Conditions', 'cCPu']]]
        path_data = pd.concat([path_data_i.iloc[:, 0:nb_region+2], path_data_c.iloc[:, 2:nb_region+2]], axis=1)
    #Reorganizing so that the seed "iCPu" is the first column


    # tile matrix such that sources are rows, columns are targets (Oh et al. 2014 Fig 4)
    connectivity_ipsi.columns = conn_names  # Sets the names of the columns and the index to be the same using the list conn_names
    connectivity_ipsi.index = conn_names  #

    connectivity_contra.columns = conn_names
    connectivity_contra.index = conn_names

    if nb_region == None:
        W = pd.concat([pd.concat([connectivity_ipsi, connectivity_contra], axis=1),
                   pd.concat([connectivity_contra, connectivity_ipsi], axis=1)], axis=0)
    else:
        # Reindexing with Seed as first column and Row
        connectivity_ipsi = connectivity_ipsi[["CPu"] + [c for c in connectivity_ipsi if c not in ['CPu']]]
        connectivity_ipsi = connectivity_ipsi.reindex(
            ["CPu"] + [c for c in connectivity_ipsi.index if c not in ["CPu"]])

        connectivity_contra = connectivity_contra[["CPu"] + [c for c in connectivity_contra if c not in ['CPu']]]
        connectivity_contra = connectivity_contra.reindex(
            ["CPu"] + [c for c in connectivity_contra.index if c not in ["CPu"]])

        # Sectioning the connectivity matrix
        sect_conn_ipsi = pd.DataFrame(connectivity_ipsi.values[0:nb_region, 0:nb_region], index=connectivity_ipsi.index[0:nb_region],
                                          columns=connectivity_ipsi.columns[0:nb_region])
        sect_conn_contra = pd.DataFrame(connectivity_contra.values[0:nb_region, 0:nb_region], index=connectivity_contra.index[0:nb_region],
                                            columns=connectivity_contra.columns[0:nb_region])
        W = pd.concat([pd.concat([sect_conn_ipsi, sect_conn_contra], axis=1),
                           pd.concat([sect_conn_contra, sect_conn_ipsi], axis=1)], axis=0)

    # Checking if the matrix was tiled properly
    if (((W.iloc[0:57, 0:57] != W.iloc[0:57, 0:57]).sum()).sum() > 0):  # Summing over columns and then rows
        print("!!! Adjacency matrix: failed concatenation !!!")  # If False the double sum equals 0
    else:
        print('Adjacency matrix: successful concatenation')

    # retain indices to reorder like original data variable for plotting on mouse brains
    ROInames = ["i" + i for i in conn_names] + ["c" + i for i in conn_names]
            # List of ROI w/ first the contro regions and then the ipsi regions.

    orig_order = []
    for i in range(0, len(ROInames)):  # Reordering according to ROInames
        for k in range(0, len(exp_data.columns._index_data) - 2):
            if ROInames[k] == exp_data.columns._index_data[2::][i]:
                orig_order.append(k)  # List containing the index of the original data

    return W, path_data, conn_names, ROInames


def process_roi_coord(coor, roi_names):
    """ Function to process roi coordinates table. Reindex the coordinate table.
    ---
    Inputs:
        coor --> ROIcoords.csv table
        roi_names --> ROInames created in process_pathdata
    ---
    Outputs:
        coor --> input matrix with sorted rows
    """

    coor.rename(columns={'Unnamed: 0': 'ROI'}, inplace=True)
    idx = []
    for i in range(0, len(roi_names)):  # Reordering according to ROInames
        for k in range(0, len(coor['ROI'])):
            if roi_names[i] == coor['ROI'][k]:
                idx.append(k)
    coor = coor.loc[idx, :]
    # From the new index created we reorganize the table by index.

    return coor


def process_gene_expression_data(expression, roi_names):
    """ Function to process gene expression data
    ---
    Inputs:
        expression --> Pandas DataFrame containing the gene expression data per region
        roi_names --> ROInames created in process_pathdata. List of ordered ROInames (first iROI then cROI)
    ---
    Outputs:
        Ordered expression data as pandas Dataframe. Panda DataFrame
    """

    expression_ordered = []
    for i in range(0, len(roi_names)):  # Reordering according to ROInames
        for k in range(0, len(expression.index)):
            if roi_names[i] == expression.index[k]:
                expression_ordered.append(expression.index[k])

    return expression.loc[expression_ordered, :]  # Reordered Snca expression


def mean_pathology(timepoints, path_data):
    """
    Process experimental data to return mean per group and per timepoint
    ---
    Inputs:
        timepoints: list of experimental timepoints
        path_data: path_data created in process_path_data. Contains the pathology data quantified.
    ---
    Outputs:
        grp_mean: Dataframe with mean pathology per group, timepoints and regions
    """


    mice = []
    for time in timepoints:  # Creation of a list of 3 panda dataframe. These 3 df correspond to the 3 tp
        l = path_data[path_data['Time post-injection (months)'] == time][path_data['Conditions'] == 'NTG'][
            path_data.columns[2::]]
        l = l.reset_index(drop=True)
        # Reset the index, drop = True is used to remove the old index as it is by default added as a column
        mice.append(l)  # list of Dataframe

    grp_mean = []
    for i in np.arange(len(timepoints)):
        grp_mean.append(mice[i].mean())  # Careful, now the mean are display as columns and not as rows anymore
    grp_mean = pd.concat([grp_mean[i] for i in np.arange(len(timepoints))], axis=1)
    grp_mean.columns = ["MPI {}".format(i) for i in timepoints]

    return grp_mean