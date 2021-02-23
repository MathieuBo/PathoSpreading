import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

import process_files
import Laplacian
import fitfunctions

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
    for regions in tqdm(range(0,nb_region+1)):
        W, path_data, conn_names, ROInames = process_files.process_pathdata(exp_data=exp_data,
                                                                                                   connectivity_ipsi=connectivity_ipsi,
                                                                                                   connectivity_contra=connectivity_contra,
                                                                                                   nb_region=regions)
        # For each iterations ROInames is reorganized
        ipsi_sec = ['iCPu'] + ["i" + i for i in conn_names if i not in ['CPu']]
        ipsi_sec = ipsi_sec[0:regions]
        contra_sec = ['cCPu'] + ["c" + i for i in conn_names if i not in ['CPu']]
        contra_sec = contra_sec[0:regions]
        ROInames = ipsi_sec+contra_sec

        # print("For",regions,"regions")
        # print("--------------------------------------------------------------------------")
        # print("W is", W.head(),"\n")
        # print("path is ", path_data.head(),"\n")
        # #print("conn_names is ", conn_names, "\n")
        # print(" ROInames  ", ROInames, "\n", len(ROInames))
        if regions >3:
            L = Laplacian.get_laplacian(adj_mat=W, expression_data=None, return_in_degree=False) #Need min 2 values for a valid Laplacian
            #print("Len L", len(L))
            grp_mean = process_files.mean_pathology(timepoints=timepoints, path_data=path_data)
            c, r = fitfunctions.c_fit(log_path=np.log10(grp_mean), L_out=L, tp=timepoints, seed='iCPu', c_rng=c_rng,
                                        roi_names=ROInames)
            # List of the c and r values
            c_stab.append(c)
            r_stab.append(r)
            rep.append(regions) # List that cumulates the number of regions
        else:
             print("")

    r_stab_df = pd.DataFrame(r_stab, columns=["MPI1", "MPI3", "MPI6"])
    regions = pd.DataFrame(rep,columns=["Number Regions"])
    model_stability = pd.concat([regions, r_stab_df], axis=1)
    return model_stability

