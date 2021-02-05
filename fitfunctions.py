"""fitfunctions"""
"""04/02/2021"""

## For the tests only ########
from process_pathconn import *
from getLout import *
from scipy.linalg import expm
from scipy import stats
from analyzespread import log_path,c_rng
from scipy import stats
##############################

def make_Xo(ROI, ROInames):
    """From a ROI (str) and by matching with the index of the ROInames,
    it returns a column vector filled with 0 except at the index of the
    ROI where the value will be 1. """
    n_regions = len(ROInames)
    # Initialisation of the Xo Vector at t = 0
    Xo = np.zeros((n_regions, 1))
    idx = [i for i in range(0, len(ROInames)) if (ROInames[i] == ROI) == True]
    Xo[idx] = 1
    return Xo

# Test 1 z = make_Xo('iTC',ROInames)


def predict_Lout(L_out, Xo, c, t=1):
    """From the Laplacian Matrix, the seed Xo and a time constant c
    this algorithm predicts the regional alpha-synuclein pathology x(t)
    Returns the column vector Xt"""
    Xt = np.dot(expm(-L_out*c*t), Xo)
    Xt = pd.DataFrame(Xt, index=path_data.columns._index_data[2::], columns=["Predictions"]) # Need to make sure that this is compatible with pathdata
    return Xt

#Test Xt = predict_Lout(L_out, z, 0.15, t=1)

def c_fit(log_path,L_out,tp,ROI,c_rng,ROInames):
    Xo = make_Xo(ROI,ROInames)
    # Exclusion mask; we do not count the regions with 0 path
    mask = log_path != -np.inf
    # Compute fit at each time point for range of time
#a = log_path[mask].dropna()
#log_path["1 MPI"][mask["1 MPI"]].dropna()
#for time in range(0, len(tp)):
#predict_Lout(L_out, z, 0.15, t=3)[mask["1 MPI"].reset_index(drop = True)]

###Test
os.chdir(savedir)
tp = np.array([1, 3, 6])
mask = log_path != -np.inf
z = make_Xo('iCg',ROInames)
store = []
Xt_sweep = []
for time in range(0,len(tp)):
    for c in c_rng:
        store.append(stats.pearsonr(log_path[mask.columns._index_data[time]][mask[mask.columns._index_data[time]]], np.log10(predict_Lout(L_out, z, c, t = tp[time]))[mask[mask.columns._index_data[time]]].values[:, 0])[0])
    Xt_sweep.append([store])
    store = []
