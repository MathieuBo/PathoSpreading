import numpy as np
from scipy.linalg import expm
from scipy import stats
from tqdm import tqdm
import pandas as pd

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
    if ROI not in ROInames:
        print("[ERROR in make_Xo]: ", ROI, "is not in the list of ROI. Please check the seed.")
    else:
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
    c_idx = [i for i in range(0, len(c_rng))]
    extracted = pd.DataFrame(0, index=c_idx, columns=col)

    for time in range(0, len(tp)):

        for c_idx, c_value in enumerate(c_rng):

            exp_val = log_path.iloc[:, time][mask.iloc[:, time]].values
            predict_val = np.log10(predict_Lout(L_out, Xo, c_value, t=tp[time])[mask.iloc[:, time]])

            r, _ = stats.pearsonr(exp_val, predict_val)

            extracted[str(tp[time]), "c"].loc[c_idx] += c_value
            extracted[str(tp[time]), "r"].loc[c_idx] += r

    return extracted
