import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

np.seterr(divide='ignore') #Hide Runtime warning regarding log(0) = -inf

from tqdm import tqdm

from scipy.stats import pearsonr, linregress, percentileofscore
from statsmodels.stats.multitest import multipletests

import process_files
import Laplacian
import fitfunctions


class DataManager(object):

    def __init__(self, exp_data, synuclein, timepoints, seed='iCPu'):

        self.seed = seed

        # Connectivity tables
        self.connectivity_ipsi = pd.read_csv("./Data83018/connectivity_ipsi.csv", index_col=0)

        self.connectivity_contra = pd.read_csv("./Data83018/connectivity_contra.csv", index_col=0)

        self.W, self.path_data, self.conn_names, self.orig_order, self.n_regions, self.ROInames = process_files.process_pathdata(exp_data=exp_data,
                                                                                                   connectivity_contra=self.connectivity_contra,
                                                                                                   connectivity_ipsi=self.connectivity_ipsi)

        self.coor = pd.read_csv("./Data83018/ROIcoords.csv")

        self.coor_ordered = process_files.process_roi_coord(coor=self.coor, roi_names=self.ROInames)

        self.synuclein = process_files.process_gene_expression_data(expression=synuclein, roi_names=self.ROInames)

        self.timepoints = timepoints

        self.l_out = None

        self.grp_mean = process_files.mean_pathology(timepoints=timepoints, path_data=self.path_data)

        self.c_rng = np.linspace(start=0.01, stop=10, num=100)

        print('Data Manager initialized\n')

    def compute_graph(self, use_expression_values=False, expression_data=None):

        if use_expression_values is False:
            self.l_out = Laplacian.get_laplacian(adj_mat=self.W)

        else:
            self.l_out= Laplacian.get_laplacian(adj_mat=self.W, expression_data=self.synuclein)

        print('Graph computed - Laplacian matrix created\n')

    def find_best_c(self):

        c_Grp, r = fitfunctions.c_fit(log_path=np.log10(self.grp_mean),
                                        L_out=self.l_out,
                                        tp=timepoints,
                                        seed=self.seed,
                                        c_rng=self.c_rng,
                                        roi_names=self.ROInames)

        return c_Grp, r

    def model_robustness(self, best_c, best_r, RandomSeed=False, RandomAdja=False, RandomPath=False):
    # Random Seed
        if RandomSeed == True:
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

            print('{r} seed is the {p}th percentile\n'.format(r=self.seed, p=percentile))
            print("Plotting the CPu Fit versus Fits of random seed regions... ")
            RndSeed = pd.DataFrame(r_val, columns=["MPI1", "MPI3", "MPI6"]) # should be same as grp.mean
            sns.swarmplot(data=RndSeed, size=4, zorder=0)
            for time in range(0,len(timepoints)):
                plt.plot(time, best_r[time], "o", color="red", markersize=3)
            plt.title("CPu seed VS random region seed")
            plt.ylabel("Fit(r)")
            plt.legend()
            plt.show()
        else:
            print("Robustness- Random seeding ignored")
    #Random Adjacency Matrix
        if RandomAdja == True:
            c_adj = []
            r_adj = []
            print("Loading of Random Shuffling test (Adjacency matrix):")
            for i in tqdm(range(0, 150)):# Adding an input for the iteration?
                np.random.shuffle(self.W.values) #CAREFUL => Does it modify W somewhere else outside the function?
                random_Lap = Laplacian.get_laplacian(adj_mat=self.W, expression_data=None, return_in_degree=False)
                c_rand, r_rand = fitfunctions.c_fit(log_path=np.log10(self.grp_mean),
                                                    L_out=random_Lap,
                                                    tp=timepoints,
                                                    seed='iCPu',
                                                    c_rng=self.c_rng,
                                                    roi_names=self.ROInames)
                c_adj.append(c_rand)
                r_adj.append(r_rand)

            percentile = percentileofscore(c_adj, best_c)

            print('{r} seed is the {p}th percentile\n'.format(r='iCPu', p=percentile))
            print("Plotting the adjacency matrix Fit versus Fits of random adjacency matrices... ")
            RndAdj = pd.DataFrame(r_adj, columns=["MPI1", "MPI3", "MPI6"])
            sns.swarmplot(data=RndAdj, size=4, zorder=0)
            for time in range(0,len(timepoints)):
                plt.plot(time, best_r[time], "o", color="red", markersize=3)
            plt.title("Adjacency Fit VS random Adjacency shuffle fits")
            plt.ylabel("Fit(r)")
            plt.legend()
            plt.show()
        else:
            print("Robustness- Random Shuffle of Adjacency matrix ignored")
    # Random Pathology Mean Matrix
        if RandomPath == True:
            self.W, self.path_data, self.conn_names, self.orig_order, self.n_regions, self.ROInames = process_files.process_pathdata(
                exp_data=exp_data,
                connectivity_contra=self.connectivity_contra,
                connectivity_ipsi=self.connectivity_ipsi) # To check; Reimporting path_data to make sure to have a version that did not get modified
            c_path = []
            r_path = []
            print("Loading of Random Pathology Mean test:")
            for i in tqdm(range(0, 150)):
                grp_mean = process_files.mean_pathology(timepoints=timepoints, path_data=self.path_data)
                for time in range(0, len(timepoints)):
                    np.random.shuffle(grp_mean.values[:, time])  # Shuffling data of a same timepoint
                Lap = Laplacian.get_laplacian(adj_mat=self.W, expression_data=None, return_in_degree=False)
                c_rand, r_rand = fitfunctions.c_fit(log_path=np.log10(grp_mean), L_out=Lap, tp=timepoints, seed='iCPu',
                                                    c_rng=self.c_rng,
                                                    roi_names=self.ROInames)
                c_path.append(c_rand)
                r_path.append(r_rand)
            percentile = percentileofscore(c_path, best_c)
            print('{r} seed is the {p}th percentile\n'.format(r='iCPu', p=percentile))
            print("Plotting the non-shuffled pathology Fit versus shuffled pathology fits ")
            RndPath = pd.DataFrame(r_path, columns=["MPI1", "MPI3", "MPI6"])
            sns.swarmplot(data=RndPath, size=4, zorder=0)
            for time in range(0,len(timepoints)):
                plt.plot(time, best_r[time], "o", color="red", markersize=3)
            plt.title("Pathology mean/timepoint fit VS shuffled pathology mean/timepoints fits")
            plt.ylabel("Fit(r)")
            plt.legend()
            plt.show()
        else:
            print("Robustness- Random Shuffle of Pathology Mean matrix ignored")



    def predict_pathology(self, c_Grp, seeding_region=None, suffix=''):

        if seeding_region is None:
            Xo = fitfunctions.make_Xo(ROI=self.seed, ROInames=self.ROInames)
        else:
            Xo = fitfunctions.make_Xo(ROI=seeding_region, ROInames=self.ROInames)

        Xt_Grp = [fitfunctions.predict_Lout(self.l_out, Xo, c_Grp, i) for i in timepoints]

        data_to_export = pd.DataFrame(np.transpose(Xt_Grp), columns=['MPI{}'.format(i) for i in timepoints])
        data_to_export['regions'] = self.ROInames
        data_to_export.to_csv('../output/predicted_pathology{}.csv'.format(suffix))

        return Xt_Grp

    def compute_vulnerability(self, Xt_Grp, c_Grp, suffix=''):

        vulnerability = pd.DataFrame(0, columns=["MPI 1", "MPI 3", "MPI 6"], index=self.grp_mean.index)

        stats_df = []
        masks = dict()
        for M in range(0, len(timepoints)): # M iterates according to the number of timepoint
            Df = pd.DataFrame({"experimental_data": np.log10(self.grp_mean.iloc[:, M]).values,
                               "ndm_data": np.log10(Xt_Grp[M])},
                              index=self.grp_mean.index)  # Runtime Warning
            # exclude regions with 0 pathology at each time point for purposes of computing fit
            mask = (Df["experimental_data"] != -np.inf) & (Df['ndm_data'] != -np.inf) & (Df['ndm_data'] != np.nan)

            masks["MPI %s" % timepoints[M]] = mask
            Df = Df[mask]

            cor = {"MPI": "%s" % (M),
                   "Pearson r": pearsonr(Df["experimental_data"], Df["ndm_data"])[0],
                   "p_value": pearsonr(Df["experimental_data"], Df["ndm_data"])[1]}

            stats_df.append(cor)

            print('----------------------------')
            print("Month Post Injection %s" % timepoints[M])
            print("Number of Regions used: ", Df.shape[0])
            print("Pearson correlation coefficient", cor['Pearson r'])
            print('Pvalue (non corrected)', cor['p_value'])
            print('----------------------------\n')

            slope, intercept, r_value, p_value, std_err = linregress(x=Df['ndm_data'], y=Df['experimental_data'])
            Df['linreg_data'] = slope * Df['ndm_data'] + intercept
            Df['residual'] = Df['experimental_data'] - Df['linreg_data']

            Df.to_csv('../output/model_output_MPI{}{}.csv'.format(timepoints[M], suffix))

        stats_df = pd.DataFrame(stats_df)
        # Boneferroni method for correction of pvalues
        _, stats_df['adj_p_value'], _, _ = multipletests(stats_df['p_value'], method="bonferroni")

        stats_df.to_csv('../output/stats.csv')


if __name__ == '__main__':

    # DataFrame with header, pS129-alpha-syn quantified in each brain region
    exp_data = pd.read_csv("./Data83018/data.csv", header=0)

    synuclein = pd.read_csv("./Data83018/SncaExpression.csv", index_col=0, header=None)

    timepoints = [1, 3, 6]

    #Load data and computation Laplacian matrices
    dm = DataManager(exp_data=exp_data, synuclein=synuclein, timepoints=timepoints, seed='iCPu')
    dm.compute_graph()

    #Use experimental data to fit the model and find the best c (scaling parameter) and best r
    c, r = dm.find_best_c()

    #Test robustness of the model by comparing c values for fit in other brain regions
    #dm.model_robustness(best_c=c, best_r=r, RandomSeed=True, RandomAdja=True, RandomPath=True)
    # If all True takes 20-25 min to run the code

    #Predict pathology
    predicted_pathology = dm.predict_pathology(c_Grp=c)
    predicted_pathology_seeding_sn = dm.predict_pathology(c_Grp=c, seeding_region='iSN', suffix='_seedSN')

    #Compare model-based prediction with observed data to extrapolate region vulnerability
    dm.compute_vulnerability(Xt_Grp=predicted_pathology, c_Grp=c)