import numpy as np
from scipy.linalg import expm
from scipy import stats

# Functions
def make_Xo(ROI, ROInames):
    """From a ROI (str) and by matching with the index of the ROInames,
    it returns a column vector filled with 0 except at the index of the
    ROI where the value will be 1.
    ---
    Inputs:
    ROI --> String of the ROI e.g "iCPu"
    ROInames --> Ordered list of ROI created in process_pathdata
    ---
    Outputs:
    Xo --> DataFrame (vector columns). Size = len(ROInames) and contains zeros except at the ROI index where the value is 1
    """
    n_regions = len(ROInames)
    # Initialisation of the Xo Vector at t = 0
    Xo = np.zeros((n_regions, ))
    idx = [i for i in range(0, len(ROInames)) if ROInames[i] == ROI]
    Xo[idx] += 1
    return Xo


def predict_Lout(L_out, Xo, c, t=1):
    """From the Laplacian Matrix, the seed Xo and a time constant c
    this algorithm predicts the regional alpha-synuclein pathology x(t)
    Returns the column vector Xt
    ---
    Inputs:
    L_out --> Laplacian Matrix as computed in Laplacian.py. Array.
    Xo --> DataFrame. Column of 0 beside at the seed index position.
    c --> Constant to tune the time scale. Integer
    t --> Timepoint. t=1 if not explicit.
    ---
    Outputs:
    Xt --> Column array that contains the magnitude of the alpha-synuclein spreading
    """
    Xt = np.dot(expm(-L_out*c*t), Xo)
    return Xt

def c_fit(log_path, L_out, tp, seed, c_rng, roi_names):
    """
    Iterates the c-values to extract subsequent predicted magnitude Xt and compare them to
    the quantified data to return the best tuning constant c et the equivalent R correlation coefficient
    ---
    Inputs:
    log_path --> log10 of grp_mean. Grp_mean is the Dataframe with mean pathology per group, timepoints and regions
    L_out --> Laplacian matrice, array
    tp --> Timepoint, list
    c_rng --> Constant to tune the time scale
    roi_names --> ROInames
    ---
    Outputs:
    c --> Best c fit.
    r --> Best correlation coefficient r
    """
    Xo = make_Xo(seed, roi_names)
    Xt_sweep = np.zeros((len(c_rng), len(tp)))
    # Exclusion mask; we do not count the regions with 0 path
    mask = log_path != -np.inf
    # Compute fit at each time point for range of time
    reg = []
    for time in range(0, len(tp)):

        for c_idx, c in enumerate(c_rng):
            exp_val = log_path.iloc[:, time][mask.iloc[:, time]].values

            predict_val = np.log10(predict_Lout(L_out, Xo, c, t=tp[time])[mask.iloc[:, time]])

            r, _ = stats.pearsonr(exp_val, predict_val)

            Xt_sweep[c_idx, time] += r
        reg.append(np.max(Xt_sweep[:, time])) #
    normalized_Xt_sweep = np.mean(Xt_sweep, axis=1)
    c_Grp = c_rng[np.where(normalized_Xt_sweep == np.max(normalized_Xt_sweep))][0]  # [0] to access the value

    return c_Grp, reg

def extract_c_and_r(log_path, L_out, tp, seed, c_rng, roi_names):
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
    Xo = make_Xo(seed, roi_names)
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

            predict_val = np.log10(predict_Lout(L_out, Xo, c_value, t=tp[time])[mask.iloc[:, time]])

            r, _ = stats.pearsonr(exp_val, predict_val)
            extracted[str(tp[time]), "c"].loc[c_idx] =+ c_value
            extracted[str(tp[time]), "r"].loc[c_idx] =+ r
    return extracted

def c_fit_individual(ind_patho, L_out, tp, seed, c_rng, roi_names):
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
    # Exclusion mask; we do not count the regions with 0 path
    log_path = np.log10(ind_patho)
    mask = log_path != -np.inf
    # Compute fit at each time point for range of time
    c_fit_animal = pd.DataFrame(np.zeros((2, len(ind_patho.columns))), columns=ind_patho.columns, index=["c_fit", "r"])
    for time in tp:
        for mouse in range(1, len(log_path.loc[:, str(time)].columns) + 1):
            exp_val = log_path.loc[:, (str(time), str(mouse))][
                mask.loc[:, (str(time), str(mouse))]].values

            local_c = 0
            local_r = 0

            for c_idx, c in enumerate(c_rng):
                predict_val = np.log10(predict_Lout(L_out, Xo, c, t=time)[mask.loc[:, (str(time), str(mouse))]])

                r, _ = stats.pearsonr(0.1+exp_val, predict_val)

                if r > local_r:
                    local_r = r
                    local_c = c

                #Xt_sweep.loc[c_idx,("{}".format(time),"{}".format(mouse))] +=r

            c_fit_animal.loc["c_fit", (str(time), str(mouse))] = local_c
            c_fit_animal.loc["r", (str(time),str(mouse))] = local_r
    return c_fit_animal


###################
import pandas as pd
import numpy as np
import process_files
import Laplacian

connectivity_ipsi = pd.read_csv("./Data83018/connectivity_ipsi.csv", index_col=0)
connectivity_contra = pd.read_csv("./Data83018/connectivity_contra.csv", index_col=0)
exp_data = pd.read_csv("./Data83018/data.csv", header=0)
c_rng = np.linspace(start=0.01, stop=10, num=100)
timepoints =[1,3,6]

W, path_data, conn_names, ROInames = process_files.process_pathdata(exp_data=exp_data,
                                                                                        connectivity_contra=connectivity_contra,
                                                                                        connectivity_ipsi=connectivity_ipsi)
ind_grp = process_files.ind_pathology(timepoints=timepoints, path_data=path_data)
l_out = Laplacian.get_laplacian(adj_mat=W)

c_fit_ani = c_fit_individual(ind_patho=ind_grp, L_out=l_out, tp=timepoints, seed="iCPu", c_rng=c_rng, roi_names=ROInames)



