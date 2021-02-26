import pandas as pd
import numpy as np
from tqdm import tqdm

import process_files
import Laplacian
import fitfunctions

# np.random.seed(0o1)

def initialization_stability(exp_data, connectivity_ipsi,connectivity_contra, nb_region):
    """
    Function that first processes the quantified alpha-synuclein data and the connectivity maps (ipsilateral and controlateral)
   and slice them after random sampling. The seed of injection is moved to be the first element of path_data and W.
    ---
    Inputs:
        exp_data --> Table that contains the values of the quantified alpha-synuclein pathology measured at different
        timepoints and with different conditions.
        connectivity_ipsi/contra --> Connectivity tables, the index and the columns are the regions of interest
        nb_regions --> Number of regions to select
        timepopint --> List of the different timepoints
        c_rng --> Constant to tune the time scale


    ---
    Outputs:
        W --> adjacency matrix shaped in a DataFrame randomly selected using ROInames. Number of rows = nb_regions
        path_data --> experimental DataFrame that contains the alpha-syn quantification randomly.
                    Number of rows = nb_regions+2
        ROI_names --> ordered list of iRegions and then cRegions randomly selected. """
    #Randomizing ROINames
    conn_names = [i.split(' (')[0] for i in connectivity_contra.columns]

    conn_names_wX_seed = [i for i in conn_names if i not in ['CPu']]
    i_seq_rand = ['iCPu'] + ["i" + k for k in np.random.choice(conn_names_wX_seed, size=nb_region-1, replace=False)]
    c_seq_rand = ["c" + i[1::] for i in i_seq_rand]
    ROInames = i_seq_rand + c_seq_rand

    #Randomizing Pathology using ROInames
    path_data = pd.concat([exp_data.loc[:, exp_data.columns[0:2]._index_data],exp_data[ROInames]], axis=1)
    path_data = path_data.rename(columns={"MBSC Region": "Conditions"})
    #Randomizing Adjacency using ROInames
        #Setting the names of the columns and the index to be the same using the list conn_names
    connectivity_ipsi.columns = conn_names
    connectivity_ipsi.index = conn_names

    connectivity_contra.columns = conn_names
    connectivity_contra.index = conn_names

        # Seed as first row and column
    connectivity_ipsi = connectivity_ipsi[["CPu"] + [c for c in connectivity_ipsi if c not in ['CPu']]]
    connectivity_ipsi = connectivity_ipsi.reindex(
        ["CPu"] + [c for c in connectivity_ipsi.index if c not in ["CPu"]])

    connectivity_contra = connectivity_contra[["CPu"] + [c for c in connectivity_contra if c not in ['CPu']]]
    connectivity_contra = connectivity_contra.reindex(
        ["CPu"] + [c for c in connectivity_contra.index if c not in ["CPu"]])

        # Slicing the Adjacency using ROInames
    connectivity_names =[i[1::] for i in i_seq_rand]
    connectivity_contra = connectivity_contra.loc[connectivity_names,connectivity_names]
    connectivity_ipsi = connectivity_ipsi.loc[connectivity_names,connectivity_names]
        # Adjacency
    W = pd.concat([pd.concat([connectivity_ipsi, connectivity_contra], axis=1),
                   pd.concat([connectivity_contra, connectivity_ipsi], axis=1)], axis=0)

    return W, path_data, ROInames

def stability(exp_data, connectivity_ipsi, connectivity_contra, nb_region, timepoints, c_rng):
    """
    Compute the correlation coefficient obtained from 0 to nb_regions data used. Three minimal regions required to
    compute the correlation coefficients for each MPI.
    ---
    Inputs:
    exp_data: DataFrame that contains the alpha-synuclein pathology quantified - csv
    connectivity_ipsi: Dataframe containing the ipsilateral connectivity data - csv
    connectivity_contra: Dataframe containing the controlateral connectivity data -csv
    nb_region: Number of regions to use for the computation of the correlation coefficient - integer
    timepoints: list of the different timepoints
    c_rng:
    ---
    Outputs:
    model_stability: DataFrame that contains the number of regions used and the subsequent values of correlation
    for each MPI. Each columns has a length of nb_region-3
    """

    # How many regions to get a stable r?
    rep = [] #Number of iterations
    c_stab = []
    r_stab = []
    for regions in tqdm(range(1, nb_region+1)):

        W, path_data, ROInames = initialization_stability(exp_data=exp_data, connectivity_ipsi=connectivity_ipsi,
                                                          connectivity_contra=connectivity_contra, nb_region=regions)

        if regions > 3:

            #while sum(W.iloc[0, :]) == 0:
                # print("Loading a Connectivity matrix with at least one non-null value in CPu")
                # W, path_data, ROInames = initialization_stability(exp_data=exp_data,
                #                                                   connectivity_ipsi=connectivity_ipsi,
                #                                                   connectivity_contra=connectivity_contra,
                #                                                   nb_region=regions)

            L = Laplacian.get_laplacian(adj_mat=W, expression_data=None, return_in_degree=False)
                    # Need a minimum of 2 values to have a valid Laplacian
            grp_mean = process_files.mean_pathology(timepoints=timepoints, path_data=path_data)

            c, r = fitfunctions.c_fit(log_path=np.log10(grp_mean), L_out=L, tp=timepoints, seed='iCPu', c_rng=c_rng,
                                        roi_names=ROInames)
            #List of the c and r values
            c_stab.append(c)
            r_stab.append(r)
            rep.append(regions) # List that cumulates the number of regions
        else:
             print("")

    r_stab_df = pd.DataFrame(r_stab, columns=["MPI1", "MPI3", "MPI6"])
    regions = pd.DataFrame(rep, columns=["Number Regions"])
    model_stability = pd.concat([regions, r_stab_df], axis=1)
    return model_stability







###################
import pandas as pd
import numpy as np
import process_files
import Laplacian

connectivity_ipsi = pd.read_csv("./Data83018/connectivity_ipsi.csv", index_col=0)
connectivity_contra = pd.read_csv("./Data83018/connectivity_contra.csv", index_col=0)
exp_data = pd.read_csv("./Data83018/data.csv", header=0)
c_rng = np.linspace(start=0.01, stop=10, num=100)
timepoints =[1,3,6]

W, path_data, conn_names, ROInames = process_files.process_pathdata(exp_data=exp_data,
                                                                                        connectivity_contra=connectivity_contra,
                                                                                        connectivity_ipsi=connectivity_ipsi)
ind_grp = process_files.ind_pathology(timepoints=timepoints, path_data=path_data)
l_out = Laplacian.get_laplacian(adj_mat=W)

c_fit_ani = fitfunctions.c_fit_individual(ind_patho=ind_grp, L_out=l_out, tp=timepoints, seed="iCPu", c_rng=c_rng, roi_names=ROInames)






def stability_individual(exp_data, connectivity_ipsi, connectivity_contra, nb_region, timepoints, c_rng):
    """
    Compute the correlation coefficient obtained from 0 to nb_regions data used for each animals.
    Three minimal regions required to create a Laplacian
    ---
    Inputs:
    exp_data: DataFrame that contains the alpha-synuclein pathology quantified - csv
    connectivity_ipsi: Dataframe containing the ipsilateral connectivity data - csv
    connectivity_contra: Dataframe containing the controlateral connectivity data -csv
    nb_region: Number of regions to use for the computation of the correlation coefficient - integer
    timepoints: list of the different timepoints
    c_rng:
    ---
    Outputs:
    model_stability: DataFrame that contains the number of regions used and the subsequent values of correlation
    for each MPI. Each columns has a length of nb_region-3
    """

    # How many regions to get a stable r?
    rep = [] #Number of iterations
    #c_stab = []
    r_stab = []
    for regions in tqdm(range(1, nb_region+1)):

        W, path_data, ROInames = initialization_stability(exp_data=exp_data, connectivity_ipsi=connectivity_ipsi,
                                                          connectivity_contra=connectivity_contra, nb_region=regions)

        if regions > 3:

            #while sum(W.iloc[0, :]) == 0:
                # print("Loading a Connectivity matrix with at least one non-null value in CPu")
                # W, path_data, ROInames = initialization_stability(exp_data=exp_data,
                #                                                   connectivity_ipsi=connectivity_ipsi,
                #                                                   connectivity_contra=connectivity_contra,
                #                                                   nb_region=regions)

            L = Laplacian.get_laplacian(adj_mat=W, expression_data=None, return_in_degree=False)
                    # Need a minimum of 2 values to have a valid Laplacian
            print("L is -------------------------", L)
            ind_grp = process_files.ind_pathology(timepoints=timepoints, path_data=path_data)
            print("IND GRP ----------------", ind_grp)
            c_fit_ani = fitfunctions.c_fit_individual(ind_patho=ind_grp, L_out=L, tp=timepoints, seed="iCPU", c_rng=c_rng,
                                          roi_names=ROInames)
            print("c_fit_indi -------------------", c_fit_ani)
            #List of the c and r values
            #c_stab.append(c_fit_ani)
        #     r_stab.append(c_fit_ani.loc["r"])
        #     rep.append(regions) # List that cumulates the number of regions
        # else:
        #      print("")

    # r_stab_df = pd.DataFrame(r_stab, columns=["MPI1", "MPI3", "MPI6"])
    # regions = pd.DataFrame(rep, columns=["Number Regions"])
    # model_stability = pd.concat([regions, r_stab_df], axis=1)
    #return model_stability
    return c_fit_ani
