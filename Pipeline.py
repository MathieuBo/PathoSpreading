import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from scipy.stats import pearsonr, linregress
from statsmodels.stats.multitest import multipletests

np.seterr(divide='ignore')  # Hide Runtime warning regarding log(0) = -inf

import process_files
import Laplacian
import fitfunctions
import Robustness_Stability
import summative_model


class DataManager(object):
    # Initialization
    def __init__(self, exp_data, synuclein, timepoints, seed, use_expression_values=None):

        self.seed = seed

        # Connectivity tables
        self.connectivity_ipsi = pd.read_csv("./Data83018/connectivity_ipsi.csv", index_col=0)

        self.connectivity_contra = pd.read_csv("./Data83018/connectivity_contra.csv", index_col=0)

        self.W, self.path_data, self.conn_names, self.ROInames = process_files.process_pathdata(exp_data=exp_data,
                                                                                                connectivity_contra=self.connectivity_contra,
                                                                                                connectivity_ipsi=self.connectivity_ipsi)

        self.synuclein = process_files.process_gene_expression_data(expression=synuclein, roi_names=self.ROInames)

        self.timepoints = timepoints

        self.l_out = None

        self.grp_mean = process_files.mean_pathology(timepoints=timepoints, path_data=self.path_data)

        self.c_rng = np.linspace(start=0.01, stop=100, num=1000)

        self.use_expression_values = use_expression_values

        print('Data Manager initialized\n')

    # Main Functions
    def compute_graph(self):

        if self.use_expression_values:

            self.l_out = Laplacian.get_laplacian(adj_mat=self.W, expression_data=self.synuclein)
            print('Using a syn expression values')

        else:

            self.l_out = Laplacian.get_laplacian(adj_mat=self.W)

        print('Graph computed - Laplacian matrix created\n')

    def find_best_c_and_r(self):

        c_Grp, r = fitfunctions.c_fit(log_path=np.log10(self.grp_mean),
                                      L_out=self.l_out,
                                      tp=timepoints,
                                      seed=self.seed,
                                      c_rng=self.c_rng,
                                      roi_names=self.ROInames)

        return c_Grp, r

    def predict_pathology(self, c_Grp):

        # Initialization
        if self.use_expression_values:
            gene_exp = '_SNCA'
            suffix = "_{}{}".format(self.seed, gene_exp)
        else:
            suffix = "_{}".format(self.seed)

        Xo = fitfunctions.make_Xo(ROI=self.seed, ROInames=self.ROInames)

        print("suffix is ", suffix)

        Xt_Grp = [fitfunctions.predict_Lout(self.l_out, Xo, c_Grp, i) for i in timepoints]

        data_to_export = pd.DataFrame(np.transpose(Xt_Grp), columns=['MPI{}'.format(i) for i in timepoints])
        data_to_export['regions'] = self.ROInames
        data_to_export.to_csv('../output/predicted_pathology{}.csv'.format(suffix))

        return Xt_Grp

    def predict_iterative(self):
        summative_model.predict_pathology_iter(self=self, timepoints=timepoints)

    def compute_outputs_and_graphs(self, Xt_Grp):

        # Initialization
        if self.use_expression_values:
            gene_exp = '_SNCA'
            suffix = "_{}{}".format(self.seed, gene_exp)
        else:
            suffix = "_{}".format(self.seed)
        try:
            os.mkdir('../plots/Density_vs_residuals/')
        except WindowsError:  # For Mac users need to replace by OSError.
            print("")

        stats_df = []
        masks = dict()
        print('---------------------------------------------------')
        print('----------------NON-ITERATIVE MODEL----------------')
        print('---------------------------------------------------\n')
        for M in range(0, len(timepoints)):
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

            print('---------------------------------------------------')
            print("Month Post Injection %s" % timepoints[M])
            print("Number of Regions used: ", Df.shape[0])
            print("Pearson correlation coefficient", cor['Pearson r'])
            print('Pvalue (non corrected)', cor['p_value'])
            print('---------------------------------------------------\n')

            slope, intercept, r_value, p_value, std_err = linregress(x=Df['ndm_data'], y=Df['experimental_data'])
            Df['linreg_data'] = slope * Df['ndm_data'] + intercept
            Df['residual'] = Df['experimental_data'] - Df['linreg_data']
            # Saving the data as csv
            Df.to_csv('../output/model_output_MPI{}{}.csv'.format(timepoints[M], suffix))

        # Saving the lollipop plots
        for time in timepoints:
            mpi = pd.read_csv('../output/model_output_MPI{}{}.csv'.format(time, suffix))
            mpi = mpi.rename(columns={'Unnamed: 0': 'region'})
            plt.figure(figsize=(7.4,5.3))

            plt.rc('xtick', labelsize=15)
            plt.rc('ytick', labelsize=15)
            plt.tight_layout()

            plt.vlines(mpi["ndm_data"], mpi['linreg_data'], mpi['linreg_data'] + mpi['residual'] - 0.04,
                       lw=0.8, color='blue', linestyles="dotted", label="Residual")
            sns.regplot(x=mpi["ndm_data"], y=mpi["experimental_data"], data=mpi,
                        scatter_kws={'s': 40, 'facecolor': 'blue'})
            plt.xlabel("Log(Predicted)", fontsize=18)
            plt.ylabel("Log(Path)", fontsize=18)
            #plt.title("Month Post Injection {} - Conditions{}".format(time, suffix), fontsize=19)
            plt.legend()

            plt.savefig('../plots/Predicted_VS_Path_MPI{}{}.png'.format(time, suffix), dpi=300)
            plt.savefig('../plots/Predicted_VS_Path_MPI{}{}.pdf'.format(time, suffix), dpi=300)

            plt.show()
        # Saving the density Vs Residual plots
        plt.figure()
        for time in timepoints:
            mpi = pd.read_csv('../output/model_output_MPI{}{}.csv'.format(time, suffix))
            mpi = mpi.rename(columns={'Unnamed: 0': 'region'})
            sns.kdeplot(x='residual', data=mpi, label='{} MPI'.format(time))
            plt.title("Density(residual) - Conditions{}".format(suffix))
            plt.legend(title='Timepoints')

        plt.savefig('../plots/Density_vs_residuals/density_VS_residual{}.png'.format(suffix), dpi=300)
        plt.savefig('../plots/Density_vs_residuals/density_VS_residual{}.png'.format(suffix), dpi=300)
        plt.show()

        stats_df = pd.DataFrame(stats_df)
        # Boneferroni method for correction of pvalues
        _, stats_df['adj_p_value'], _, _ = multipletests(stats_df['p_value'], method="bonferroni")

        stats_df.to_csv('../output/stats{}.csv'.format(suffix))

    def predicted_heatmap(self, Drop_Seed=False):

        # Initialization
        if self.use_expression_values:
            gene_exp = '_SNCA'
            suffix = "_{}{}".format(self.seed, gene_exp)
        else:
            suffix = "_{}".format(self.seed)
        try:
            os.mkdir('../plots/Heatmap_Predictions/')
        except WindowsError:  # For Mac users need to replace by OSError.
            print("")

        patho = pd.read_csv('../output/predicted_pathology{}.csv'.format(suffix))

        if Drop_Seed == False:
            plt.figure(figsize=(10, 25))
            norm_path = patho[['MPI1', 'MPI3', 'MPI6']].values
            norm_path = (norm_path - np.min(norm_path)) / (np.max(norm_path) - np.min(norm_path))
            plt.imshow(norm_path, cmap='BuPu', aspect=.1)
            plt.yticks(np.arange(patho.shape[0]), patho['regions'])
            plt.xticks(np.arange(patho.columns[1:4].shape[0]), patho.columns[1:4])
            plt.title("Predictions(MPI) - Conditions{}".format(suffix))
            plt.colorbar()
            plt.savefig("../plots/Heatmap_Predictions/Predictions_{}.png".format(suffix), dpi=300)
            plt.savefig("../plots/Heatmap_Predictions/Predictions_{}.pdf".format(suffix), dpi=300)
            plt.show()
        else:
            for i in patho.index:
                if patho.loc[i, "regions"] == self.seed:
                    patho = patho.drop(i)
                    print(self.seed, "dropped.")
            plt.figure(figsize=(10, 25))
            norm_path = patho[['MPI1', 'MPI3', 'MPI6']].values
            norm_path = (norm_path - np.min(norm_path)) / (np.max(norm_path) - np.min(norm_path))
            plt.imshow(norm_path, cmap='BuPu', aspect=.1)
            plt.yticks(np.arange(patho.shape[0]), patho['regions'])
            plt.xticks(np.arange(patho.columns[1:4].shape[0]), patho.columns[1:4])
            plt.title("Predictions(MPI) - Conditions{} - {} excluded".format(suffix, self.seed))
            plt.colorbar()
            plt.savefig("../plots/Heatmap_Predictions/Predictions_{}_excluded{}.pdf".format(suffix, self.seed), dpi=300)
            plt.savefig("../plots/Heatmap_Predictions/Predictions_{}_excluded{}.png".format(suffix, self.seed), dpi=300)
            plt.show()
        return patho

    def plotting_r_function_of_c(self):
        # Initialization
        if self.use_expression_values:
            gene_exp = '_SNCA'
            suffix = "_{}{}".format(self.seed, gene_exp)
        else:
            suffix = "_{}".format(self.seed)
        try:
            os.mkdir('../plots/Fits(c)/')
        except WindowsError:  # For Mac users need to replace by OSError.
            print("")

        c_r_table = fitfunctions.extract_c_and_r(log_path=np.log10(self.grp_mean),
                                                 L_out=self.l_out,
                                                 tp=timepoints,
                                                 seed=self.seed,
                                                 c_rng=self.c_rng,
                                                 roi_names=self.ROInames)
        for t in timepoints:
            plt.figure()
            sns.scatterplot(x=c_r_table[str(t), "c"], y=c_r_table[str(t), "r"])
            plt.xlabel('c')
            plt.ylabel('Best Fit(r)')
            plt.title('MPI {} - Conditions{}'.format(t, suffix))
            plt.savefig('../plots/Fits(c)/plot_r_(c)_MPI{}{}.png'.format(t, suffix), dpi=300)
            plt.show()
        return c_r_table

    # Controls

    def model_robustness(self, exp_data, timepoints, best_c, best_r, RandomSeed=False,
                         RandomAdja=False, RandomPath=False):
        if self.use_expression_values:
            gene_exp = '_SNCA'
            suffix = "_{}{}".format(self.seed, gene_exp)
        else:
            suffix = "_{}".format(self.seed)

        try:
            os.mkdir('../plots/Model_Robustness/')
        except WindowsError:  # For Mac users need to replace by OSError.
            print("")

        Robustness_Stability.random_robustness(self, best_c=best_c, best_r=best_r, exp_data=exp_data,
                                               timepoints=timepoints,
                                               RandomSeed=RandomSeed, RandomAdja=RandomAdja, RandomPath=RandomPath,
                                               suffix=suffix)

    # Stability

    def compute_stability(self, Sliding_Window=None):
        # Initialization
        if self.use_expression_values:
            gene_exp = '_SNCA'
            suffix = "_{}{}".format(self.seed, gene_exp)
        else:
            suffix = "_{}".format(self.seed)
        try:
            os.mkdir('../plots/Stab_Grp/')
        except WindowsError:  # For Mac users need to replace by OSError.
            print("")
        try:
            os.mkdir('../plots/Stab_Grp_Sliding/')
        except WindowsError:  # For Mac users need to replace by OSError.
            print("")
        _, fit = dm.find_best_c_and_r()
        Robustness_Stability.stability(exp_data=exp_data, connectivity_ipsi=self.connectivity_ipsi,
                                       connectivity_contra=self.connectivity_contra, nb_region=nb_region,
                                       timepoints=timepoints, c_rng=self.c_rng, r=fit, Sliding_Window=Sliding_Window,
                                       suffix=suffix, seed=seed)


    def plot_individual_stability(self, exp_data, Sliding_Window=None):
        # Initialization

        if self.use_expression_values:
            gene_exp = '_SNCA'
            suffix = "_{}{}".format(self.seed, gene_exp)
        else:
            suffix = "_{}".format(self.seed)

        try:
            os.mkdir('../plots/Stab_Ind/')
        except WindowsError:  # For Mac users need to replace by OSError.
            print("")

        try:
            os.mkdir('../plots/Stab_Ind_Sliding/')
        except WindowsError:  # For Mac users need to replace by OSError.
            print("")
        _, fit = dm.find_best_c_and_r()
        Robustness_Stability.stability_individual(exp_data=exp_data, connectivity_ipsi=dm.connectivity_ipsi,
                                                  connectivity_contra=dm.connectivity_contra, nb_region=nb_region,
                                                  timepoints=timepoints, c_rng=dm.c_rng, suffix=suffix, r=fit, seed=seed,
                                                  Sliding_Window= Sliding_Window)

    # Iterative model
    def extract_c_r_iter(self):
        # Initialization
        if self.use_expression_values:
            gene_exp = '_SNCA'
            suffix = "_{}{}".format(self.seed, gene_exp)
        else:
            suffix = "_{}".format(self.seed)
        try:
            os.mkdir('../plots/Fits(c)/')
        except WindowsError:  # For Mac users need to replace by OSError.
            print("")

        c_r_table = summative_model.extract_c_and_r_iter(log_path=np.log10(self.grp_mean),
                                                 L_out=self.l_out,
                                                 tp=timepoints,
                                                 seed=self.seed,
                                                 c_rng=self.c_rng,
                                                 roi_names=self.ROInames)
        return c_r_table
if __name__ == '__main__':
    # DataFrame with header, pS129-alpha-syn quantified in each brain region

    exp_data = pd.read_csv("./Data83018/data.csv", header=0)

    synuclein = pd.read_csv("./Data83018/SncaExpression.csv", index_col=0,
                            header=None)  # Gene Expression ==> Generalization

    timepoints = [1, 3, 6]

    seed = "iCPu"

    use_expression_values = False

    nb_region = 58

# Load data

    dm = DataManager(exp_data=exp_data,
                     synuclein=synuclein,
                     timepoints=timepoints,
                     seed=seed,
                     use_expression_values=use_expression_values)

# Main Functions
    # Computation of the Laplacian

    dm.compute_graph()

    # Use experimental data to fit the model and find the best c (scaling parameter) and best r

    c, r = dm.find_best_c_and_r()

    # Predict pathology

    predicted_pathology = dm.predict_pathology(c_Grp=c)

    # Predict pathology with the iterative model
    #dm.predict_iterative()

# Main outputs of the model; Output_table - Predicted_Vs_Pathology - Density_Vs_Residuals

    dm.compute_outputs_and_graphs(Xt_Grp=predicted_pathology)

    # Predicted heatmap

    #dm.predicted_heatmap(Drop_Seed=True)

    # Plotting r(c) for each MPI

    #c_r_table = dm.plotting_r_function_of_c()

# Controls
    # Robustness of the model - If ALL TRUE then takes 25 min to run the code

    # dm.model_robustness(best_c=c, best_r=r,
    #                     exp_data=exp_data, timepoints=timepoints,
    #                     RandomSeed=True, RandomAdja=True, RandomPath=True)

# Stability
    # Stability of the model --> All animals included

    #dm.compute_stability(Sliding_Window=None)

    # Stability of the model --> Individuals

    #dm.plot_individual_stability(exp_data=exp_data, Sliding_Window=4)
