"""fitfunctions"""
"""04/02/2021"""
# Importing modules
import numpy as np
from scipy.linalg import expm
from scipy import stats

# Functions
def make_Xo(ROI, ROInames):
    """From a ROI (str) and by matching with the index of the ROInames,
    it returns a column vector filled with 0 except at the index of the
    ROI where the value will be 1. """
    n_regions = len(ROInames)
    # Initialisation of the Xo Vector at t = 0
    Xo = np.zeros((n_regions, ))
    idx = [i for i in range(0, len(ROInames)) if ROInames[i] == ROI]
    Xo[idx] += 1
    return Xo

def predict_Lout(L_out, Xo, c, t=1):
    """From the Laplacian Matrix, the seed Xo and a time constant c
    this algorithm predicts the regional alpha-synuclein pathology x(t)
    Returns the column vector Xt"""
    Xt = np.dot(expm(-L_out*c*t), Xo)
    return Xt

def c_fit(log_path,L_out,tp,ROI,c_rng,ROInames):
    "Returns the best c fit"
    Xo = make_Xo(ROI,ROInames)
    Xt_sweep = np.zeros((len(c_rng), len(tp)))
    # Exclusion mask; we do not count the regions with 0 path
    mask = log_path != -np.inf
    # Compute fit at each time point for range of time
    for time in range(0, len(tp)):

        store = []
        for c_idx, c in enumerate(c_rng):
            exp_val = log_path.iloc[:, time][mask.iloc[:, time]].values

            predict_val = np.log10(predict_Lout(L_out, Xo, c, t=tp[time])[mask.iloc[:, time]])

            r, _ = stats.pearsonr(exp_val, predict_val)

            Xt_sweep[c_idx, time] += r
    normalized_Xt_sweep = np.mean(Xt_sweep, axis=1)
    best_c = c_rng[np.where(normalized_Xt_sweep == np.max(normalized_Xt_sweep))][0]# Ajout de [0]
    return best_c