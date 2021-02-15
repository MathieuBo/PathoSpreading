import os
import pandas as pd
import numpy as np

np.seterr(divide='ignore') #Hide Runtime warning regarding log(0) = -inf

from tqdm import tqdm

import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import pearsonr, linregress, percentileofscore
from statsmodels.stats.multitest import multipletests

import process_files
import Laplacian
import fitfunctions

#basedir = "C:\\Users\\thoma\\Documents\\M1_Neurasmus\\NeuroBIM_M1\\Internship\\GitRepo\\PathoSpreading"
#os.chdir(basedir)  # Sets the wd
#opdir = "asyndiffusion3"

#try:
#    os.mkdir(opdir)
#except WindowsError as error: # Prevent the algorithm to stop if the folder is already created. For Mac users need to replace by OSError.
#    print(error)

############################
####### Process Data #######
############################

### Load files for Windows
#exp_data = pd.read_csv("Data83018\\data.csv", header=0)
#connectivity_ipsi = pd.read_csv("Data83018\\connectivity_ipsi.csv", index_col=0) # Connectivity table
#connectivity_contra = pd.read_csv("Data83018\\connectivity_contra.csv", index_col=0)
#coor = pd.read_csv("Data83018\\ROIcoords.csv")
#synuclein = pd.read_csv("Data83018\\SncaExpression.csv", index_col = 0, header= None)


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

        c_Grp = fitfunctions.c_fit(log_path=np.log10(self.grp_mean),
                                        L_out=self.l_out,
                                        tp=timepoints,
                                        seed=self.seed,
                                        c_rng=self.c_rng,
                                        roi_names=self.ROInames)

        return c_Grp

    def random_seeds(self, best_c):

        c_values = []
        for region in tqdm(self.ROInames):
            local_c = fitfunctions.c_fit(log_path=np.log10(self.grp_mean),
                               L_out=self.l_out,
                               tp=timepoints,
                               seed=region,
                               c_rng=self.c_rng,
                               roi_names=self.ROInames)

            c_values.append(local_c)

        percentile = percentileofscore(c_values, best_c)
        print('{r} seed is the {p}th percentile\n'.format(r=self.seed, p=percentile))

    def compute_vulnerability(self, c_Grp):

        vulnerability = pd.DataFrame(0, columns=["MPI 1", "MPI 3", "MPI 6"], index=self.grp_mean.index)

        Xo = fitfunctions.make_Xo(ROI=self.seed, ROInames=self.ROInames)
        Xt_Grp = [fitfunctions.predict_Lout(self.l_out, Xo, c_Grp, i) for i in timepoints]

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
            print('Pvalue (non corrected', cor['p_value'])
            print('----------------------------\n')

            slope, intercept, r_value, p_value, std_err = linregress(x=Df['ndm_data'], y=Df['experimental_data'])
            Df['linreg_data'] = slope * Df['ndm_data'] + intercept
            Df['residual'] = Df['experimental_data'] - Df['linreg_data']

            Df.to_csv('../output/model_output_MPI{}.csv'.format(timepoints[M]))

        stats_df = pd.DataFrame(stats_df)
        # Boneferroni method for correction of pvalues
        _, stats_df['adj_p_value'], _, _ = multipletests(stats_df['p_value'], method="bonferroni")

        stats_df.to_csv('../output/stats.csv')


if __name__ == '__main__':

    # DataFrame with header, pS129-alpha-syn quantified in each brain region
    exp_data = pd.read_csv("./Data83018/data.csv", header=0)

    synuclein = pd.read_csv("./Data83018/SncaExpression.csv", index_col=0, header=None)

    timepoints = [1, 3, 6]

    dm = DataManager(exp_data=exp_data, synuclein=synuclein, timepoints=timepoints, seed='iCPu')
    dm.compute_graph()

    c = dm.find_best_c()

    #dm.random_seeds(best_c=c)

    dm.compute_vulnerability(c_Grp=c)

