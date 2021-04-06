import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import expm
from scipy.stats import pearsonr, linregress
import seaborn as sns
from statsmodels.stats.multitest import multipletests


import fitfunctions

def extract_c_and_r_iter(log_path, L_out, tp, seed, c_rng, roi_names):
    """
    Extract each c and r values corresponding while iterating on c.
    ---
    Inputs:
    log_path --> log10 of grp_mean. Grp_mean is the Dataframe with mean pathology per group, timepoints and regions
    L_out --> Laplacian matrice, array
    tp --> Timepoint, list
    c_rng --> Constant to tune the time scale
    roi_names --> ROInames
    ---
    Outputs:
    extracted --> Panda DataFrame with Multi-indexed columns. Contains for each timepoint c values and corresponding
    r values
    """
    global best_c_per_mpi1, best_c_per_mpi3
    Xo = fitfunctions.make_Xo(seed, roi_names)
    # Exclusion mask; we do not count the regions with 0 path
    mask = log_path != -np.inf
    # Compute fit at each time point for range of time
    multi = []
    for time in tp:
        for output in ["c", "r"]:
            multi.append((str(time), output))

    col = pd.MultiIndex.from_tuples(multi, names=["MPI", "Condition"])
    c_idx = [i for i in range(0,len(c_rng))]
    extracted = pd.DataFrame(0, index=c_idx, columns=col)
    for time in range(0, len(tp)):

        for c_idx, c_value in enumerate(c_rng):
            exp_val = log_path.iloc[:, time][mask.iloc[:, time]].values
            if tp[time] == 1:
                Xt_1 = np.dot(expm(-L_out * c_value * 1), Xo) + 0
                predict_val = np.log10(Xt_1[mask.iloc[:, time]])
                r, _ = pearsonr(exp_val, predict_val)
                extracted[str(tp[time]), "c"].loc[c_idx] = + c_value
                extracted[str(tp[time]), "r"].loc[c_idx] = + r
                idx = np.where(extracted[str(tp[time]), "r"] == np.max(extracted[str(tp[time]), "r"]))[0][0]
                best_c_per_mpi1 = extracted[str(tp[time]), "c"][idx]

            if tp[time] == 3:
                Xt_3 = np.dot(expm(-L_out * c_value * 3), np.dot(expm(-L_out * best_c_per_mpi1 * 1), Xo) + 0) + np.dot(expm(-L_out * best_c_per_mpi1 * 1), Xo) + 0
                predict_val = np.log10(Xt_3[mask.iloc[:, time]])
                r, _ = pearsonr(exp_val, predict_val)
                extracted[str(tp[time]), "c"].loc[c_idx] = + c_value
                extracted[str(tp[time]), "r"].loc[c_idx] = + r
                idx = np.where(extracted[str(tp[time]), "r"] == np.max(extracted[str(tp[time]), "r"]))[0][0]
                best_c_per_mpi3 = extracted[str(tp[time]), "c"][idx]

            if tp[time] == 6:
                Xt_6 = np.dot(expm(-L_out * c_value * 6), (np.dot(expm(-L_out * best_c_per_mpi3 * 3), np.dot(expm(-L_out * best_c_per_mpi1 * 1), Xo) + 0) + np.dot(expm(-L_out * best_c_per_mpi1 * 1), Xo) + 0)) \
                       + np.dot(expm(-L_out * best_c_per_mpi3 * 3), np.dot(expm(-L_out * best_c_per_mpi1 * 1), Xo) + 0) + np.dot(expm(-L_out * best_c_per_mpi1 * 1), Xo) + 0
                predict_val = np.log10(Xt_6[mask.iloc[:, time]])
                r, _ = pearsonr(exp_val, predict_val)
                extracted[str(tp[time]), "c"].loc[c_idx] = + c_value
                extracted[str(tp[time]), "r"].loc[c_idx] = + r
                idx = np.where(extracted[str(tp[time]), "r"] == np.max(extracted[str(tp[time]), "r"]))[0][0]
                best_c_per_mpi6 = extracted[str(tp[time]), "c"][idx]
    return extracted


# def predict_model_iter(L_out, Xo, c, t=1):
#
#     if t == 1:
#         Xt = np.dot(expm(-L_out * c * t), Xo) + 0
#         return Xt
#
#     if t == 3:
#         Xt = np.dot(expm(-L_out * c * t), predict_model_iter(L_out, Xo, c, t - 2)) + predict_model_iter(L_out, Xo, c,
#                                                                                                       t - 2)
#         return Xt
#
#     if t == 6:
#         Xt = np.dot(expm(-L_out * c * t), predict_model_iter(L_out, Xo, c, t - 3)) + predict_model_iter(L_out, Xo, c,
#                                                                                                       t - 3)
#         return Xt

def predict_pathology_iter(self, timepoints):
    # Initialization
    if self.use_expression_values:
        gene_exp = '_SNCA'
        suffix = "_{}{}".format(self.seed, gene_exp)
    else:
        suffix = "_{}".format(self.seed)

    try:
        os.mkdir('../Iterative_Model/')
    except WindowsError:  # For Mac users need to replace by OSError.
        print("")

    Xo = fitfunctions.make_Xo(ROI=self.seed, ROInames=self.ROInames)
    print("suffix is ", suffix)
    c_r = extract_c_and_r_iter(log_path=np.log10(self.grp_mean),
                                                 L_out=self.l_out,
                                                 tp=timepoints,
                                                 seed=self.seed,
                                                 c_rng=self.c_rng,
                                                 roi_names=self.ROInames)
    Xt_Grp = []
    for i in timepoints:
        idx = np.where(c_r[str(i), "r"] == np.max(c_r[str(i), "r"]))[0][0]
        best_c_per_mpi = c_r[str(i), "c"][idx]
        if i == 1:
            Xt = np.dot(expm(-self.l_out * best_c_per_mpi * 1), Xo) + 0
            c_0 = best_c_per_mpi
            Xt_Grp.append(Xt)
        if i == 3:
            Xt = np.dot(expm(-self.l_out * best_c_per_mpi * 3), Xt_Grp[0]) + Xt_Grp[0]
            Xt_Grp.append(Xt)
        if i == 6:
            Xt = np.dot(expm(-self.l_out * best_c_per_mpi * 6), Xt_Grp[1]) + Xt_Grp[1]
            Xt_Grp.append(Xt)

        #print('---------------------\n','timepoint',i,'Xt is',Xt_Grp)

    data_to_export = pd.DataFrame(np.transpose(Xt_Grp), columns=['MPI{}'.format(i) for i in timepoints])
    data_to_export['regions'] = self.ROInames
    data_to_export.to_csv('../Iterative_Model/iter_predicted_pathology{}.csv'.format(suffix))

    stats_df = []
    masks = dict()
    print('---------------------------------------------------')
    print('------------------ITERATIVE MODEL------------------')
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
        Df.to_csv('../Iterative_Model/iter_model_output_MPI{}{}.csv'.format(timepoints[M], suffix))

    # Saving the lollipop plots
    for time in timepoints:
        mpi = pd.read_csv('../Iterative_Model/iter_model_output_MPI{}{}.csv'.format(time, suffix))
        mpi = mpi.rename(columns={'Unnamed: 0': 'region'})
        plt.figure()
        plt.vlines(mpi["ndm_data"], mpi['linreg_data'], mpi['linreg_data'] + mpi['residual'] - 0.04,
                   lw=0.8, color='blue', linestyles="dotted", label="Residual")
        sns.regplot(x=mpi["ndm_data"], y=mpi["experimental_data"], data=mpi,
                    scatter_kws={'s': 40, 'facecolor': 'blue'})
        plt.xlabel("Log(Predicted)")
        plt.ylabel("Log(Path)")
        plt.title("Iterative Model - Month Post Injection {} - Conditions{}".format(time, suffix))
        plt.legend()

        plt.savefig('../Iterative_Model/iter_Predicted_VS_Path_MPI{}{}.png'.format(time, suffix), dpi=300)
        plt.savefig('../Iterative_Model/iter_Predicted_VS_Path_MPI{}{}.pdf'.format(time, suffix), dpi=300)

        plt.show()
    # Saving the density Vs Residual plots
    plt.figure()
    for time in timepoints:
        mpi = pd.read_csv('../Iterative_Model/iter_model_output_MPI{}{}.csv'.format(time, suffix))
        mpi = mpi.rename(columns={'Unnamed: 0': 'region'})
        sns.kdeplot(x='residual', data=mpi, label='{} MPI'.format(time))
        plt.title("Iterative Model - Density(residual) - Conditions{}".format(suffix))
        plt.legend(title='Timepoints')

    plt.savefig('../Iterative_Model/Density_VS_residual{}.png'.format(suffix), dpi=300)
    plt.savefig('../Iterative_Model/Density_VS_residual{}.png'.format(suffix), dpi=300)
    plt.show()

    stats_df = pd.DataFrame(stats_df)
    # Boneferroni method for correction of pvalues
    _, stats_df['adj_p_value'], _, _ = multipletests(stats_df['p_value'], method="bonferroni")

    stats_df.to_csv('../Iterative_Model/stats{}.csv'.format(suffix))

