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
best_c = 1.64





# Shuffle of the adjacency matrix W
W, path_data, conn_names, orig_order, n_regions, ROInames = process_files.process_pathdata(exp_data=exp_data,
                                                                                           connectivity_ipsi=connectivity_ipsi,
                                                                                           connectivity_contra=connectivity_contra)
grp_mean = process_files.mean_pathology(timepoints=timepoints, path_data=path_data)
c_adj = []
r_adj = []
# for i in tqdm(range(0, 2)):
#     np.random.shuffle(W.values)
#     random_Lap = Laplacian.get_laplacian(adj_mat=W, expression_data=None, return_in_degree=False)
#     c_rand, r_rand = fitfunctions.c_fit(log_path=np.log10(grp_mean), L_out=random_Lap, tp=timepoints, seed='iCPu', c_rng=c_rng,
#                        roi_names=ROInames)
#     c_adj.append(c_rand)
#     r_adj.append(r_rand)
percentile = percentileofscore(c_adj, best_c)
print('{r} seed is the {p}th percentile\n'.format(r='iCPu', p=percentile))





# Shuffle of the pathology matrix path_data
W, path_data, conn_names, orig_order, n_regions, ROInames = process_files.process_pathdata(exp_data=exp_data,
                                                                                           connectivity_ipsi=connectivity_ipsi,
                                                                                          connectivity_contra=connectivity_contra)
c_path = []
r_path=[]
# for i in tqdm(range(0,2)):
#     path_val = path_data.iloc[:,2::].values
#     np.random.shuffle(path_val)
#     path_data.iloc[:, 2::] = path_val
#     grp_mean = process_files.mean_pathology(timepoints=timepoints, path_data=path_data)
#     Lap = Laplacian.get_laplacian(adj_mat=W, expression_data=None, return_in_degree=False)
#     c_rand, r_rand = fitfunctions.c_fit(log_path=np.log10(grp_mean), L_out=Lap, tp=timepoints, seed='iCPu', c_rng=c_rng,
#                                 roi_names=ROInames)
#     c_path.append(c_rand)
#     r_path.append(r_rand)
percentile = percentileofscore(c_path, best_c)
print('{r} seed is the {p}th percentile\n'.format(r='iCPu', p=percentile))





# Shuffle of the grp_mean (all timepoint data together)
W, path_data, conn_names, orig_order, n_regions, ROInames = process_files.process_pathdata(exp_data=exp_data,
                                                                                           connectivity_ipsi=connectivity_ipsi,
                                                                                          connectivity_contra=connectivity_contra)
c_path = []
r_path=[]
# for i in tqdm(range(0,2)):
#     grp_mean = process_files.mean_pathology(timepoints=timepoints, path_data=path_data)
#     np.random.shuffle(grp_mean.values)
#     Lap = Laplacian.get_laplacian(adj_mat=W, expression_data=None, return_in_degree=False)
#     c_rand, r_rand = fitfunctions.c_fit(log_path=np.log10(grp_mean), L_out=Lap, tp=timepoints, seed='iCPu', c_rng=c_rng,
#                                 roi_names=ROInames)
#     c_path.append(c_rand)
#     r_path.append(r_rand)
percentile = percentileofscore(c_path, best_c)
print('{r} seed is the {p}th percentile\n'.format(r='iCPu', p=percentile))






# Shuffle of the grp_mean (Shuffle of the different timepoint)
W, path_data, conn_names, orig_order, n_regions, ROInames = process_files.process_pathdata(exp_data=exp_data,
                                                                                           connectivity_ipsi=connectivity_ipsi,
                                                                                          connectivity_contra=connectivity_contra)
c_path = []
r_path=[]
# for i in tqdm(range(0,2)):
#     grp_mean = process_files.mean_pathology(timepoints=timepoints, path_data=path_data)
#     for time in range(0, len(timepoints)):
#         np.random.shuffle(grp_mean.values[:, time]) # Shuffling data of a same timepoint
#     Lap = Laplacian.get_laplacian(adj_mat=W, expression_data=None, return_in_degree=False)
#     c_rand, r_rand = fitfunctions.c_fit(log_path=np.log10(grp_mean), L_out=Lap, tp=timepoints, seed='iCPu', c_rng=c_rng,
#                                 roi_names=ROInames)
#     c_path.append(c_rand)
#     r_path.append(r_rand)
percentile = percentileofscore(c_path, best_c)
print('{r} seed is the {p}th percentile\n'.format(r='iCPu', p=percentile))





#Computing the covariance of the grp_mean between timepoints
W, path_data, conn_names, orig_order, n_regions, ROInames = process_files.process_pathdata(exp_data=exp_data,
                                                                                           connectivity_ipsi=connectivity_ipsi,
                                                                                          connectivity_contra=connectivity_contra)
grp_mean = process_files.mean_pathology(timepoints=timepoints, path_data=path_data)
#grp_mean = np.transpose(grp_mean)
grp_mean_cov = np.cov(grp_mean.values, bias=True)
fig = plt.figure(figsize=(16, 18))
sns.heatmap(grp_mean_cov, annot=False, cmap='hot', fmt='g', xticklabels=ROInames, yticklabels=ROInames)
plt.show()
