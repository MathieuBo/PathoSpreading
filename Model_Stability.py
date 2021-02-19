import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import process_files
import Laplacian
import fitfunctions
from tqdm import tqdm

from scipy.stats import pearsonr, linregress, percentileofscore

# DataFrame with header, pS129-alpha-syn quantified in each brain region
exp_data = pd.read_csv("./Data83018/data.csv", header=0)
connectivity_ipsi = pd.read_csv("./Data83018/connectivity_ipsi.csv", index_col=0)
connectivity_contra = pd.read_csv("./Data83018/connectivity_contra.csv", index_col=0)

# Processing the Data
timepoints = [1, 3, 6]
c_rng = np.linspace(start=0.01, stop=10, num=100)

# W, path_data, conn_names, orig_order, n_regions, ROInames = process_files.process_pathdata(exp_data=exp_data,
#                                                                                            connectivity_ipsi=connectivity_ipsi,
#                                                                                            connectivity_contra=connectivity_contra,
#                                                                                            section=None)
# How many regions to get a stable r?
c_stab = []
r_stab = []

for regions in tqdm(range(0,15)):
    W, path_data, conn_names, orig_order, n_regions, ROInames = process_files.process_pathdata(exp_data=exp_data,
                                                                                               connectivity_ipsi=connectivity_ipsi,
                                                                                               connectivity_contra=connectivity_contra,
                                                                                               nb_region=regions)
    ipsi_sec = ['iCPu'] + ["i" + i for i in conn_names if i not in ['CPu']]
    ipsi_sec = ipsi_sec[0:regions]
    contra_sec = ['cCPu'] + ["c" + i for i in conn_names if i not in ['CPu']]
    contra_sec = contra_sec[0:regions]
    ROInames = ipsi_sec+contra_sec# ROInames reorganized with CPu first

    print("For",regions,"regions")
    print("--------------------------------------------------------------------------")
    print("W is", W.head(),"\n")
    print("path is ", path_data.head(),"\n")

    #grp_mean = process_files.mean_pathology(timepoints=timepoints, path_data=path_data)
    if regions >3:
        L = Laplacian.get_laplacian(adj_mat=W, expression_data=None, return_in_degree=False) #Need min 2 values for a valid Laplacian
        print("Len L", len(L))
        grp_mean = process_files.mean_pathology(timepoints=timepoints, path_data=path_data)
        c, r = fitfunctions.c_fit(log_path=np.log10(grp_mean), L_out=L, tp=timepoints, seed='iCPu', c_rng=c_rng,
                                    roi_names=ROInames)
        c_stab.append(c)
        # r_stab.append(r)
    # else:
    #     print("")
# for i in tqdm(range(5,len(path_data.values[0])+1)):
#     path_stab = pd.DataFrame(path_data.values[:, 0:i], columns=path_data.columns[0:i])
#     W_stab = pd.DataFrame(W.values[0:i-2, 0:i-2], index=W.index[0:i-2], columns=W.columns[0:i-2])
#     grp_mean_stab = process_files.mean_pathology(timepoints=timepoints, path_data=path_stab)
#     Lap = Laplacian.get_laplacian(adj_mat=W, expression_data=None, return_in_degree=False)
#     Lap_stab = Lap[0:i, 0:i]
#

# test = pd.DataFrame(np.array([["0.25","0.33","1","0.7","0.8"],["0.32","0.1","2","0.89","0.41"],["0","0.15","3","0.76","0.43"],["0.12","0.56","4","0.1","0"],["0.14","0.21","5","0.5","0.78"]]), index=["i1","i2","i3","i4","i5"], columns=["c1","c2","c3","c4","c5"])
# test_c = test[["c3"]+[c for c in test if c not in ['c3']]]
# test_r = test_c.reindex(["i3"]+[c for c in test_c.index if c not in ['i3']])