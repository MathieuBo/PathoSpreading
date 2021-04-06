import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import percentileofscore
from scipy import stats
from tqdm import tqdm


import process_files
import Laplacian
import fitfunctions
from fitfunctions import make_Xo, predict_Lout

# np.random.seed(0o1)

# Computing the robustness of the model
def random_robustness(self, exp_data, timepoints, best_c, best_r, RandomSeed=False, RandomAdja=False, RandomPath=False, suffix=""):
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
        print("Plotting the {r} Fit versus Fits of random seed regions... ".format(r=self.seed))
        RndSeed = pd.DataFrame(r_val, columns=["MPI1", "MPI3", "MPI6"])  # should be same as grp.mean
        sns.swarmplot(data=RndSeed, size=4, zorder=0)
        for time in range(0, len(timepoints)):
            plt.plot(time, best_r[time], "o", color="red", markersize=3)
        plt.title("{r} seed VS random region seed".format(r=self.seed), fontsize=18)
        plt.ylabel("Fit(r)", fontsize=19)

        plt.savefig('../plots/Model_Robustness/Random_Seed.png', dpi=300)
        plt.savefig('../plots/Model_Robustness/Random_Seed.pdf', dpi=300)
        plt.show()
    else:
        print("Robustness- Random seeding ignored")
    # Random Adjacency Matrix
    if RandomAdja == True:
        c_adj = []
        r_adj = []
        print("Loading of Random Shuffling test (Adjacency matrix):")
        for i in tqdm(range(0, 150)):  # Adding an input for the iteration?
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

        percentile = percentileofscore(c_adj, best_c)

        print('{r} seed is the {p}th percentile\n'.format(r='iCPu', p=percentile))
        print("Plotting the adjacency matrix Fit versus Fits of random adjacency matrices... ")
        RndAdj = pd.DataFrame(r_adj, columns=["MPI1", "MPI3", "MPI6"])
        sns.swarmplot(data=RndAdj, size=4, zorder=0)
        for time in range(0, len(timepoints)):
            plt.plot(time, best_r[time], "o", color="red", markersize=3)
        plt.title("Adjacency Fit VS random Adjacency shuffle Fits", fontsize=18)
        plt.ylabel("Fit(r)", fontsize=19)

        plt.savefig('../plots/Model_Robustness/Random_Adja{}.png'.format(suffix), dpi=300)
        plt.savefig('../plots/Model_Robustness/Random_Adja{}.pdf'.format(suffix), dpi=300)
        plt.show()
    else:
        print("Robustness- Random Shuffle of Adjacency matrix ignored")
    # Random Pathology Mean Matrix
    if RandomPath == True:
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
        percentile = percentileofscore(c_path, best_c)
        print('{r} seed is the {p}th percentile\n'.format(r='iCPu', p=percentile))
        print("Plotting the non-shuffled pathology Fit versus shuffled pathology fits ")
        RndPath = pd.DataFrame(r_path, columns=["MPI1", "MPI3", "MPI6"])
        sns.swarmplot(data=RndPath, size=4, zorder=0)
        for time in range(0, len(timepoints)):
            plt.plot(time, best_r[time], "o", color="red", markersize=3)
        plt.title("Pathology Fit VS shuffled pathology Fits", fontsize=18)
        plt.ylabel("Fit(r)", fontsize=19)

        plt.savefig('../plots/Model_Robustness/Random_Patho{}.png'.format(suffix), dpi=300)
        plt.savefig('../plots/Model_Robustness/Random_Patho{}.pdf'.format(suffix), dpi=300)
        plt.show()
    else:
        print("Robustness- Random Shuffle of Pathology Mean matrix ignored")

# Initialization of the stability computation

def initialization_stability(exp_data, connectivity_ipsi,connectivity_contra, nb_region, seed):
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

    conn_names_wX_seed = [i for i in conn_names if i not in [seed[1::]]]
    i_seq_rand = ['i' + seed[1::]] + ["i" + k for k in np.random.choice(conn_names_wX_seed, size=nb_region-1, replace=False)]
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
    connectivity_ipsi = connectivity_ipsi[[seed[1::]] + [c for c in connectivity_ipsi if c not in [seed[1::]]]]
    connectivity_ipsi = connectivity_ipsi.reindex(
        [seed[1::]] + [c for c in connectivity_ipsi.index if c not in [seed[1::]]])

    connectivity_contra = connectivity_contra[[seed[1::]] + [c for c in connectivity_contra if c not in [seed[1::]]]]
    connectivity_contra = connectivity_contra.reindex(
        [seed[1::]] + [c for c in connectivity_contra.index if c not in [seed[1::]]])

        # Slicing the Adjacency using ROInames
    connectivity_names =[i[1::] for i in i_seq_rand]
    connectivity_contra = connectivity_contra.loc[connectivity_names,connectivity_names]
    connectivity_ipsi = connectivity_ipsi.loc[connectivity_names,connectivity_names]
        # Adjacency
    W = pd.concat([pd.concat([connectivity_ipsi, connectivity_contra], axis=1),
                   pd.concat([connectivity_contra, connectivity_ipsi], axis=1)], axis=0)

    return W, path_data, ROInames

# All groups included

def stability(exp_data, connectivity_ipsi, connectivity_contra, nb_region, timepoints,
              c_rng, r, Sliding_Window, suffix, seed):
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
    c_rng: Array from 1 to 10
    graph: Bool- True leads to graphs plots
    Sliding_windows: Int- Plots the Sliding Mean and Sliding SD
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
                                                          connectivity_contra=connectivity_contra, nb_region=regions,
                                                          seed=seed)

        if regions > 3:

            L = Laplacian.get_laplacian(adj_mat=W, expression_data=None, return_in_degree=False)
                    # Need a minimum of 2 values to have a valid Laplacian
            grp_mean = process_files.mean_pathology(timepoints=timepoints, path_data=path_data)

            c, r = fitfunctions.c_fit(log_path=np.log10(grp_mean), L_out=L, tp=timepoints, seed=seed, c_rng=c_rng,
                                        roi_names=ROInames)
            #List of the c and r values
            c_stab.append(c)
            r_stab.append(r)
            rep.append(regions) # List that cumulates the number of regions
        else:
             print("")
    r_stab_df = pd.DataFrame(r_stab, columns=["MPI1", "MPI3", "MPI6"])
    regions = pd.DataFrame(rep, columns=["Number Regions"])
    stability = pd.concat([regions, r_stab_df], axis=1)

    # Graphs
    for idx, time in enumerate(timepoints):
        sns.scatterplot(x=stability["Number Regions"], y=stability["MPI{}".format(time)])
        sns.regplot(x=stability["Number Regions"], y=stability["MPI{}".format(time)], order=2)
        plt.hlines(y=r[idx], xmin=0, xmax=nb_region, color="red", label="Best r fit")
        plt.title("Stability of the model, MPI{} - Conditions{}".format(time, suffix))
        plt.ylabel("Fit(r)")
        plt.xlabel("Number of regions used")
        plt.legend()
        plt.savefig('../plots/Stab_Grp/Stability_MPI{}{}.png'.format(time, suffix), dpi=300)
        plt.savefig('../plots/Stab_Grp/Stability_MPI{}{}.pdf'.format(time, suffix), dpi=300)
        plt.show()

    if Sliding_Window ==None:
        print('')
    else:
        for idx, time in enumerate(timepoints):
            print("Computing Sliding Window Mean")
            for i in tqdm(range(2, Sliding_Window + 1)):
                mean_stab = stability["MPI{}".format(time)].rolling(i).mean()
                sns.scatterplot(stability["Number Regions"], mean_stab.values)
                sns.regplot(stability["Number Regions"], mean_stab.values, order=2)
                plt.hlines(y=r[idx], xmin=0, xmax=nb_region, color="red", label="Best r fit")
                plt.title("Stability of the model, MPI{}, Sliding Window Mean = {} \n"
                          " Conditions{}".format(time, i, suffix))
                plt.ylabel("Sliding_Mean Fit(r)")
                plt.xlabel("Number of regions used")
                plt.xlim(xmin=3, xmax=nb_region)
                plt.ylim(ymin=0, ymax=1)
                plt.grid()
                plt.savefig("../plots/Stab_Grp_Sliding/Mean_MPI{}_SW{}{}.png".format(time, i, suffix), dpi=300)
                plt.savefig("../plots/Stab_Grp_Sliding/Mean_MPI{}_SW{}{}.pdf".format(time, i, suffix), dpi=300)
                plt.show()
            print("Computing Sliding Window SD")
            for i in tqdm(range(2, Sliding_Window + 1)):
                std_stab = stability["MPI{}".format(time)].rolling(i).std()
                # plt.plot(stability["Number Regions"].values, std_stab.values, 'o-')
                sns.scatterplot(stability["Number Regions"].values, std_stab.values)
                sns.regplot(stability["Number Regions"].values, std_stab.values, order=2)
                plt.title("Stability of the model, MPI{}, Sliding Window SD = {} \n"
                          " Conditions{}".format(time, i, suffix))
                plt.ylabel("Sliding_SD Fit(r)")
                plt.xlabel("Number of regions used")
                plt.xlim(xmin=3, xmax=nb_region)
                plt.ylim(ymin=-0.05, ymax=0.275)
                plt.grid()
                plt.savefig("../plots/Stab_Grp_Sliding/SD_MPI{}_SW{}{}.png".format(time, i, suffix), dpi=300)
                plt.savefig("../plots/Stab_Grp_Sliding/SD_MPI{}_SW{}{}.pdf".format(time, i, suffix), dpi=300)
                plt.show()
    return stability


# Computing stability at the scale of an individual

def ind_pathology(timepoints, path_data):
    """
    Process experimental data to return individual pathology mean
    ---
    Inputs:
        timepoints: list of experimental timepoints
        path_data: path_data created in process_path_data. Contains the pathology data quantified.
    ---
    Outputs:
        ind_grp: Multi-index Dataframe: first column index (1,3,6) (MPI), second column index (1,2,3,...)
        (Number of animals used) to call a specific column ==> ind_grp.loc[:, ('1', '1')]
        multi_index: Returns the MultiIndex Dataframe
    """
    mice = []
    ind_grp = pd.DataFrame()
    for idx, time in enumerate(
            timepoints):  # Creation of a list of 3 panda dataframe. These 3 df correspond to the 3 tp
        l = path_data[path_data['Time post-injection (months)'] == time][path_data['Conditions'] == 'NTG'][
            path_data.columns[2::]]
        l = l.reset_index(drop=True)
        multi_array = [["{}".format(time) for k in range(0, len(l))],
                       ["{}".format(k + 1) for k in range(0, len(l))]]
        tuples = list(zip(*multi_array))
        multi_index = pd.MultiIndex.from_tuples(tuples, names=["MPI", "Mouse"])
        l = l.transpose()
        l.columns = multi_index
        mice.append(l)  # list of Dataframe
        ind_grp = pd.concat([ind_grp, mice[idx]], axis=1)
    return ind_grp


def c_fit_individual(ind_patho, L_out, tp, c_rng, seed, roi_names):
    """
    Iterates the c-values to extract subsequent predicted magnitude Xt and compare them to
    the quantified data to return the best tuning constant c et the equivalent R correlation coefficient
    To use with a multiindex DataFrame with first column index (1,3,6) (MPI), second column index (1,2,3,...)
    (Number of animals used) where calling a specific column needs to be processed this ways
    ind_grp.loc[:, ('1', '1')]
    ---
    Inputs:
    log_path --> log10 of grp_mean. Grp_mean is the Dataframe with mean pathology per group, timepoints and regions
    L_out --> Laplacian matrice, array
    tp --> Timepoint, list
    c_rng --> Constant to tune the time scale
    roi_names --> ROInames
    ---
    Outputs:
    c_fit_ani --> Panda Dataframe. Rows = 2 {c_fit, r} and Columns = Number of animals used
    """
    Xo = make_Xo(seed, roi_names)
    # Compute fit at each time point for range of time
    c_fit_animal = pd.DataFrame(np.zeros((2, len(ind_patho.columns))), columns=ind_patho.columns, index=["c_fit", "r"])
    for time in tp:
        for mouse in range(1, len(ind_patho.loc[:, str(time)].columns) + 1):
            log_path = np.log10(ind_patho[str(time)][str(mouse)])
            mask = log_path != -np.inf
            exp_val = log_path[mask]

            local_c = 0
            local_r = 0
            for c_idx, c in enumerate(c_rng):

                predict_val = np.log10(predict_Lout(L_out, Xo, c, t=time))[mask]

                r, _ = stats.pearsonr(exp_val, predict_val)

                if r > local_r:
                    local_r = r
                    local_c = c

            c_fit_animal.loc["c_fit", (str(time), str(mouse))] = local_c
            c_fit_animal.loc["r", (str(time), str(mouse))] = local_r
    return c_fit_animal


def stability_individual(exp_data, connectivity_ipsi, connectivity_contra,
                         nb_region, timepoints, c_rng, suffix, seed, r, Sliding_Window):
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
    # Using previous data to create r_ind_per_reg
    W, path_data, conn_names, ROInames = process_files.process_pathdata(exp_data=exp_data,
                                                                        connectivity_contra=connectivity_contra,
                                                                        connectivity_ipsi=connectivity_ipsi)
    ind_grp = ind_pathology(timepoints=timepoints, path_data=path_data)
    r_ind_per_reg = pd.DataFrame(np.zeros((0, len(ind_grp.columns))), columns=ind_grp.columns)

    for regions in tqdm(range(1, nb_region + 1)):

        W, path_data, ROInames = initialization_stability(exp_data=exp_data, connectivity_ipsi=connectivity_ipsi,
                                                          connectivity_contra=connectivity_contra, nb_region=regions,
                                                          seed=seed)

        if regions > 3:

            ind_grp = ind_pathology(timepoints=timepoints, path_data=path_data)
            L = Laplacian.get_laplacian(adj_mat=W, expression_data=None, return_in_degree=False)
            # Need a minimum of 2 values to have a valid Laplacian
            c_fit_ani = c_fit_individual(ind_patho=ind_grp,
                                         L_out=L,
                                         tp=timepoints,
                                         c_rng=c_rng,
                                         seed=seed,
                                         roi_names=ROInames)
            r_ind_per_reg.loc[str(regions), :] = c_fit_ani.loc['r', :].copy()

    for time in timepoints:
        for mouse in range(1, len(r_ind_per_reg.loc[:, str(time)].columns) + 1):
            plt.figure(figsize=(15, 6))
            sns.scatterplot(x=r_ind_per_reg.index.values, y=r_ind_per_reg[str(time)][str(mouse)].values)
            plt.xlabel("Number of Regions")
            plt.ylabel("Fit (R)")
            plt.title("R(Number of regions used) - MPI {} - Mouse {} \n"
                      "Conditions{}".format(time, mouse, suffix))
            plt.savefig("../plots/Stab_Ind/stab_ind_mpi_{}_mouse{}{}.png".format(time, mouse, suffix), dpi=300)
            plt.savefig("../plots/Stab_Ind/stab_ind_mpi_{}_mouse{}{}.pdf".format(time, mouse, suffix), dpi=300)
            plt.show()
    if Sliding_Window ==None:
        print('')
    else:
        for idx, time in enumerate(timepoints):
            print("Computing Sliding Window Mean and SD")
            for mouse in tqdm(range(1, len(r_ind_per_reg.loc[:, str(time)].columns) + 1)):
                for i in range(2, Sliding_Window + 1):
                    mean_stab = r_ind_per_reg[str(time), str(mouse)].rolling(i).mean()
                    plt.figure(figsize=(15,10))
                    sns.scatterplot(r_ind_per_reg.index.values, mean_stab.values)
                    #sns.regplot(r_ind_per_reg.index.values, mean_stab.values, order=2)
                    plt.hlines(y=r[idx], xmin=0, xmax=nb_region, color="red", label="Best r fit")

                    plt.title("Stability Individual n°{}, MPI{}, Sliding Window Mean = {} \n"
                              " Conditions{}".format(mouse, time, i, suffix))
                    plt.ylabel("Sliding_Mean Fit(r)")
                    plt.xlabel("Number of regions used")
                    plt.xlim(xmin=3, xmax=nb_region)
                    plt.ylim(ymin=0, ymax=1)
                    plt.grid()
                    plt.savefig("../plots/Stab_Ind_Sliding/Ind_{}_Mean_MPI{}_SW{}{}.png".format(mouse, time, i, suffix), dpi=300)
                    plt.savefig("../plots/Stab_Ind_Sliding/Ind_{}_Mean_MPI{}_SW{}{}.pdf".format(mouse, time, i, suffix), dpi=300)
                    plt.show()

                    std_stab = r_ind_per_reg[str(time), str(mouse)].rolling(i).std()
                    plt.figure(figsize=(15, 10))
                    sns.scatterplot(r_ind_per_reg.index.values, std_stab.values)
                    #sns.regplot(r_ind_per_reg.index.values, std_stab.values, order=2)


                    plt.title("Stability Individual n°{}, MPI{}, Sliding Window SD = {} \n"
                              " Conditions{}".format(mouse, time, i, suffix))
                    plt.ylabel("Sliding_SD Fit(r)")
                    plt.xlabel("Number of regions used")
                    plt.xlim(xmin=3, xmax=nb_region)
                    plt.ylim(ymin=-0.05, ymax=0.275)

                    plt.savefig("../plots/Stab_Ind_Sliding/Ind_{}_SD_MPI{}_SW{}{}.png".format(mouse, time, i, suffix),
                                dpi=300)
                    plt.savefig("../plots/Stab_Ind_Sliding/Ind_{}_SD_MPI{}_SW{}{}.pdf".format(mouse, time, i, suffix),
                                dpi=300)
                    plt.show()


    return r_ind_per_reg