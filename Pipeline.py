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
import Robustness_Stability


class DataManager(object):

    def __init__(self, exp_data, synuclein, timepoints, seed='iCPu', use_expression_values=None):

        self.seed = seed

        # Connectivity tables
        self.connectivity_ipsi = pd.read_csv("./Data83018/connectivity_ipsi.csv", index_col=0)

        self.connectivity_contra = pd.read_csv("./Data83018/connectivity_contra.csv", index_col=0)

        self.W, self.path_data, self.conn_names, self.ROInames = process_files.process_pathdata(exp_data=exp_data,
                                                                                                   connectivity_contra=self.connectivity_contra,
                                                                                                   connectivity_ipsi=self.connectivity_ipsi)

        self.coor = pd.read_csv("./Data83018/ROIcoords.csv")

        self.coor_ordered = process_files.process_roi_coord(coor=self.coor, roi_names=self.ROInames)

        self.synuclein = process_files.process_gene_expression_data(expression=synuclein, roi_names=self.ROInames)

        self.timepoints = timepoints

        self.l_out = None

        self.grp_mean = process_files.mean_pathology(timepoints=timepoints, path_data=self.path_data)

        self.c_rng = np.linspace(start=0.01, stop=10, num=100)

        self.use_expression_values = use_expression_values

        print('Data Manager initialized\n')

    def compute_graph(self):

        if self.use_expression_values:

            self.l_out = Laplacian.get_laplacian(adj_mat=self.W, expression_data=self.synuclein)
            print('Using a syn expression values')
        else:

            self.l_out = Laplacian.get_laplacian(adj_mat=self.W)

        print('Graph computed - Laplacian matrix created\n')

    def find_best_c(self):

        c_Grp, r = fitfunctions.c_fit(log_path=np.log10(self.grp_mean),
                                        L_out=self.l_out,
                                        tp=timepoints,
                                        seed=self.seed,
                                        c_rng=self.c_rng,
                                        roi_names=self.ROInames)

        return c_Grp, r

    def find_c_r(self):
        c_r_table = fitfunctions.extract_c_and_r(log_path=np.log10(self.grp_mean),
                                        L_out=self.l_out,
                                        tp=timepoints,
                                        seed=self.seed,
                                        c_rng=self.c_rng,
                                        roi_names=self.ROInames)
        return c_r_table


    def model_robustness(self, exp_data, timepoints, best_c, best_r, RandomSeed=False, RandomAdja=False, RandomPath=False, suffix=""):
        Robustness_Stability.random_robustness(self, best_c=best_c, best_r=best_r, exp_data=exp_data, timepoints=timepoints,
                                          RandomSeed=RandomSeed, RandomAdja=RandomAdja, RandomPath=RandomPath, suffix=suffix)




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
            # Saving the data as csv
            Df.to_csv('../output/model_output_MPI{}{}.csv'.format(timepoints[M], suffix))

        # Saving the lollipop plots
        for time in timepoints:
            mpi = pd.read_csv('../output/model_output_MPI{}{}.csv'.format(time, suffix))
            mpi = mpi.rename(columns={'Unnamed: 0': 'region'})
            plt.figure()
            plt.vlines(mpi["ndm_data"], mpi['linreg_data'], mpi['linreg_data'] + mpi['residual'] - 0.04,
                       lw=0.8, color='blue', linestyles="dotted", label="Residual")
            sns.regplot(mpi["ndm_data"], mpi["experimental_data"], data=mpi,
                        scatter_kws={'s': 40, 'facecolor': 'blue'})
            plt.xlabel("Log(Predicted)")
            plt.ylabel("Log(Path)")
            plt.title("Month Post Injection {}".format(time))
            plt.legend()

            plt.savefig('../plots/{}Predicted_VS_Path_MPI{}.png'.format(suffix, time), dpi=300)
            plt.savefig('../plots/{}Predicted_VS_Path_MPI{}.pdf'.format(suffix, time), dpi=300)

            plt.show()
        # Saving the density Vs Residual plots
        plt.figure()
        for time in timepoints:
            mpi = pd.read_csv('../output/model_output_MPI{}{}.csv'.format(time, suffix))
            mpi = mpi.rename(columns={'Unnamed: 0': 'region'})
            sns.kdeplot(x='residual', data=mpi, label='{} MPI'.format(time))
            plt.title("Density(residual)")
            plt.legend(title='Timepoints')
        plt.savefig('../plots/Density_vs_residuals/density_VS_residual{}.png'.format(suffix), dpi=300)
        plt.savefig('../plots/Density_vs_residuals/density_VS_residual{}.png'.format(suffix), dpi=300)
        plt.show()

        stats_df = pd.DataFrame(stats_df)
        # Boneferroni method for correction of pvalues
        _, stats_df['adj_p_value'], _, _ = multipletests(stats_df['p_value'], method="bonferroni")

        stats_df.to_csv('../output/stats{}.csv'.format(suffix))

    def predicted_heatmap(self, Drop_Seed=False, suffix=""):
        patho = pd.read_csv('../output/predicted_pathology{}.csv'.format(suffix))

        if Drop_Seed == False:
            plt.figure(figsize=(10, 25))
            norm_path = patho[['MPI1', 'MPI3', 'MPI6']].values
            norm_path = (norm_path - np.min(norm_path)) / (np.max(norm_path) - np.min(norm_path))
            plt.imshow(norm_path, cmap='BuPu', aspect=.1)
            plt.yticks(np.arange(patho.shape[0]), patho['regions'])
            plt.xticks(np.arange(patho.columns[1:4].shape[0]), patho.columns[1:4])
            plt.title("Vulnerability")
            plt.colorbar()
            plt.savefig("../plots/Heatmap_Predictions/Predictions_SN_included{}.png".format(suffix), dpi =300)
            plt.savefig("../plots/Heatmap_Predictions/Predictions_SN_included{}.pdf".format(suffix), dpi=300)
            plt.show()
        else:
            patho = patho.drop(15) #CPu
            patho= patho.drop(48) #iSN
            plt.figure(figsize=(10, 25))
            norm_path = patho[['MPI1', 'MPI3', 'MPI6']].values
            norm_path = (norm_path - np.min(norm_path)) / (np.max(norm_path) - np.min(norm_path))
            plt.imshow(norm_path, cmap='BuPu', aspect=.1)
            plt.yticks(np.arange(patho.shape[0]), patho['regions'])
            plt.xticks(np.arange(patho.columns[1:4].shape[0]), patho.columns[1:4])
            plt.title("Vulnerability - CPu excluded")
            plt.colorbar()
            plt.savefig("../plots/Heatmap_Predictions/Predictions_SN_excluded{}.pdf".format(suffix), dpi=300)
            plt.savefig("../plots/Heatmap_Predictions/Predictions_SN_excluded{}.png".format(suffix), dpi=300)
            plt.show()
        return patho

    def compute_stability(self, graph=False, Sliding_Window=None, suffix=''):
        stability = Robustness_Stability.stability(exp_data=exp_data, connectivity_ipsi=self.connectivity_ipsi,
                                              connectivity_contra=self.connectivity_contra,
                                              nb_region=nb_region, timepoints=timepoints, c_rng=self.c_rng)
        if graph == True:
            if Sliding_Window == None:
                for idx, time in enumerate(timepoints):
                    sns.scatterplot(stability["Number Regions"], stability["MPI{}".format(time)])
                    sns.regplot(stability["Number Regions"], stability["MPI{}".format(time)], order=2)
                    plt.hlines(y=r[idx], xmin=0, xmax=nb_region, color="red", label="Best r fit")
                    plt.title("Stability of the model, MPI{}".format(time))
                    plt.ylabel("Fit(r)")
                    plt.xlabel("Number of regions used")
                    plt.legend()
                    plt.savefig('../plots/Stab_Grp/{}Stability_MPI{}.png'.format(suffix, time), dpi=300)
                    plt.savefig('../plots/Stab_Grp/{}Stability_MPI{}.pdf'.format(suffix, time), dpi=300)
                    plt.show()
            else:
                for idx, time in enumerate(timepoints):
                    print("Computing Sliding Window Mean")
                    for i in tqdm(range(2, Sliding_Window+1)):
                        mean_stab = stability["MPI{}".format(time)].rolling(i).mean()
                        #plt.plot(stability["Number Regions"].values, mean_stab.values, 'o-')
                        sns.scatterplot(stability["Number Regions"], mean_stab.values)
                        sns.regplot(stability["Number Regions"], mean_stab.values, order=2)
                        plt.hlines(y=r[idx], xmin=0, xmax=nb_region, color="red", label="Best r fit")
                        plt.title("Stability of the model, MPI{}, Sliding Window Mean = {}".format(time, i))
                        plt.ylabel("Sliding_Mean Fit(r)")
                        plt.xlabel("Number of regions used")
                        plt.xlim(xmin=0, xmax=nb_region)
                        plt.ylim(ymin=0, ymax=1)
                        plt.grid()
                        plt.savefig("../plots/Stab_Grp_Sliding/{}Mean_MPI{}_SW{}.png".format(suffix,time, i), dpi=300)
                        plt.savefig("../plots/Stab_Grp_Sliding/{}Mean_MPI{}_SW{}.pdf".format(suffix, time, i), dpi=300)
                        plt.show()
                    print("Computing Sliding Window SD")
                    for i in tqdm(range(2, Sliding_Window+1)):
                        std_stab = stability["MPI{}".format(time)].rolling(i).std()
                        #plt.plot(stability["Number Regions"].values, std_stab.values, 'o-')
                        sns.scatterplot(stability["Number Regions"].values, std_stab.values)
                        sns.regplot(stability["Number Regions"].values, std_stab.values, order=2)
                        plt.title("Stability of the model, MPI{}, Sliding Window SD = {}".format(time, i))
                        plt.ylabel("Sliding_SD Fit(r)")
                        plt.xlabel("Number of regions used")
                        plt.xlim(xmin=0, xmax=nb_region)
                        plt.ylim(ymin=-0.05, ymax=0.275)
                        plt.grid()
                        plt.savefig("../plots/Stab_Grp_Sliding/{}SD_MPI{}_SW{}.png".format(suffix,time, i), dpi=300)
                        plt.savefig("../plots/Stab_Grp_Sliding/{}SD_MPI{}_SW{}.pdf".format(suffix, time, i), dpi=300)
                        plt.show()

        else:
            print("")
        return stability

    def compute_individual_stability(self, exp_data, nb_region):
        ind_table = Robustness_Stability.stability_individual(exp_data=exp_data,
                                             connectivity_ipsi=dm.connectivity_ipsi,
                                             connectivity_contra=dm.connectivity_contra, nb_region=nb_region,
                                             timepoints=timepoints,
                                             c_rng=dm.c_rng)
        return ind_table

if __name__ == '__main__':

    # DataFrame with header, pS129-alpha-syn quantified in each brain region
    exp_data = pd.read_csv("./Data83018/data.csv", header=0)

    synuclein = pd.read_csv("./Data83018/SncaExpression.csv", index_col=0, header=None) # Gene Expression ==> Generalization

    timepoints = [1, 3, 6]

    # Load data and computation Laplacian matrices
    dm = DataManager(exp_data=exp_data,
                     synuclein=synuclein,
                     timepoints=timepoints,
                     seed='iCPU',
                     use_expression_values=None)
    dm.compute_graph() # Returns different l_out
                       # Need to consider the gene expression

    #### First checking dm
    #### Second checking next

    # Use experimental data to fit the model and find the best c (scaling parameter) and best r

    c, r = dm.find_best_c()
    #c_r_table = dm.find_c_r()

    # Controls to test the robustness of the model - If ALL TRUE then takes 25 min to run the code
    #dm.model_robustness(best_c=c, best_r=r, exp_data=exp_data, timepoints=timepoints,
    #                    RandomSeed=False, RandomAdja=False, RandomPath=False, suffix='')

    # Predict pathology
    #predicted_pathology = dm.predict_pathology(c_Grp=c)
    #predicted_pathology_seeding_sn = dm.predict_pathology(c_Grp=c, seeding_region='iSN', suffix='_seedSN')
    #predicted_pathology_SCNA = dm.predict_pathology(c_Grp=c, suffix='_SNCA') #SCNA

    # Computing the Vulnerability heatmap
    #dm.predicted_heatmap(Drop_Seed=True)
    #patho=dm.predicted_heatmap(Drop_Seed=True, suffix="_seedSN")
    #dm.predicted_heatmap(Drop_Seed=True, suffix="_SNCA")

    # Compare model-based prediction with observed data to extrapolate region vulnerability
    #dm.compute_vulnerability(Xt_Grp=predicted_pathology, c_Grp=c)
    #dm.compute_vulnerability(Xt_Grp=predicted_pathology_seeding_sn, c_Grp=c, suffix='_seedSN')
    #dm.compute_vulnerability(Xt_Grp=predicted_pathology_SNCA, c_Grp=c, suffix='_SNCA')


    # Stability of the model --> All animals included
    nb_region = 57
    #stab = dm.compute_stability(graph=True, Sliding_Window=6)
    #stab = dm.compute_stability(graph=True, Sliding_Window=None, suffix="Seed_SN_")
    #stab = dm.compute_stability(graph=True, Sliding_Window=None, suffix="Snca_")

    # Stability of the model --> Individuals
    #ind_table = dm.compute_individual_stability(exp_data=exp_data, nb_region=58)