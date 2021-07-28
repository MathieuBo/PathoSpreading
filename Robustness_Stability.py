import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import percentileofscore
from tqdm import tqdm

import process_files
import Laplacian
import fitfunctions


# Computing the robustness of the model
def random_robustness(self, exp_data, timepoints, best_c, best_r, file_format, RandomSeed=False, RandomAdja=False, RandomPath=False, suffix=""):
    # Random Seed
    if RandomSeed:
        c_values = []
        r_val = []
        print("Loading of Random Seeding test:")
        for region in tqdm(self.ROInames):
            local_c, local_r_val = fitfunctions.c_fit(log_path=np.log10(self.grp_mean),
                                                      L_out=self.l_out,
                                                      tp=timepoints,
                                                      seed=region,
                                                      c_rng=self.c_rng,
                                                      roi_names=self.ROInames)

            c_values.append(local_c)
            r_val.append(local_r_val)

        percentile = percentileofscore(c_values, best_c)

        print('Random Initiation of the Pathology: {r} seed is the {p:.2f}th percentile\n'.format(r=self.seed, p=percentile))
        print("Plotting the {r} Fit versus Fits of random seed regions\n ".format(r=self.seed))

        RndSeed = pd.DataFrame(r_val, columns=["MPI1", "MPI3", "MPI6"])  # should be same as grp.mean
        cmap= plt.get_cmap('tab10')

        plt.figure(figsize=(4.2, 3), constrained_layout=True)
        sns.swarmplot(data=RndSeed, size=5, zorder=0, alpha=.5, color='gray', edgecolor='darkgray')

        for time in range(0, len(timepoints)):
            plt.plot(time, best_r[time], "d", color=cmap(1), markersize=6)

        plt.ylabel("Pearson's $r$", fontsize=16)
        plt.ylim(-.5, .8)
        plt.yticks([-.4, 0, .4, .8])
        plt.xlabel(' ')
        plt.savefig("{}/{}/plots/Model_Robustness/Random_Seed.{}".format(self.output_path, suffix, file_format), dpi=300)

    else:
        print("Robustness- Random seeding ignored")
    # Random Adjacency Matrix
    if RandomAdja:
        c_adj = []
        r_adj = []
        print("Loading of Random Shuffling test (Adjacency matrix):")
        for i in tqdm(range(0, 150), desc='Regions'):  # Adding an input for the iteration?
            np.random.shuffle(self.W.values)
            random_Lap = Laplacian.get_laplacian(adj_mat=self.W, expression_data=None, return_in_degree=False)
            c_rand, r_rand = fitfunctions.c_fit(log_path=np.log10(self.grp_mean),
                                                L_out=random_Lap,
                                                tp=timepoints,
                                                seed='iCPu',
                                                c_rng=self.c_rng,
                                                roi_names=self.ROInames)
            c_adj.append(c_rand)
            r_adj.append(r_rand)

        print("Plotting the adjacency matrix Fit versus Fits of random adjacency matrices\n")
        RndAdj = pd.DataFrame(r_adj, columns=["MPI1", "MPI3", "MPI6"])

        cmap= plt.get_cmap('tab10')
        plt.figure(figsize=(4.2, 3), constrained_layout=True)
        sns.swarmplot(data=RndAdj, size=3.5, zorder=0, alpha=.5, color='gray', edgecolor='darkgray')
        for time in range(0, len(timepoints)):
            plt.plot(time, best_r[time], "d", color=cmap(1), markersize=6)

        plt.ylabel("Pearson's $r$", fontsize=16)
        plt.ylim(-.5, .8)
        plt.yticks([-.4, 0, .4, .8])
        plt.xlabel(' ')

        plt.savefig("{}/{}/plots/Model_Robustness/Random_Adja{}.{}".format(self.output_path, suffix, suffix, file_format), dpi=300)

    # Random Pathology Mean Matrix
    if RandomPath:
        W, path_data, conn_names, ROInames = process_files.process_pathdata(
            exp_data=exp_data,
            connectivity_contra=self.connectivity_contra,
            connectivity_ipsi=self.connectivity_ipsi)
        c_path = []
        r_path = []
        print("Loading of Random Pathology Mean test:")
        for i in tqdm(range(0, 150)):
            grp_mean = process_files.mean_pathology(timepoints=timepoints, path_data=path_data)
            for time in range(0, len(timepoints)):
                np.random.shuffle(grp_mean.values[:, time])  # Shuffling data of a same timepoint
            Lap = Laplacian.get_laplacian(adj_mat=W, expression_data=None, return_in_degree=False)
            c_rand, r_rand = fitfunctions.c_fit(log_path=np.log10(grp_mean), L_out=Lap, tp=timepoints, seed='iCPu',
                                                c_rng=self.c_rng,
                                                roi_names=ROInames)
            c_path.append(c_rand)
            r_path.append(r_rand)

        print("Plotting the non-shuffled pathology Fit versus shuffled pathology fits \n")
        RndPath = pd.DataFrame(r_path, columns=["MPI1", "MPI3", "MPI6"])

        cmap= plt.get_cmap('tab10')
        plt.figure(figsize=(4.2, 3), constrained_layout=True)
        sns.swarmplot(data=RndPath, size=3.5, zorder=0, alpha=.5, color='gray', edgecolor='darkgray')
        for time in range(0, len(timepoints)):
            plt.plot(time, best_r[time], "d", color=cmap(1), markersize=6)

        plt.ylabel("Pearson's $r$", fontsize=16)
        plt.ylim(-.5, .8)
        plt.yticks([-.4, 0, .4, .8])
        plt.xlabel(' ')
        plt.savefig("{}/{}/plots/Model_Robustness/Random_Patho{}.{}".format(self.output_path, suffix, suffix, file_format), dpi=300)

