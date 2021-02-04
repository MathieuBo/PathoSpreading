"""fitfunctions"""
"""04/02/2021"""

## For the tests only ########
from process_pathconn import *
from getLout import *
from scipy.linalg import expm
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
    return Xt

#Test Xt = predict_Lout(L_out, z, 0.15, t=1)

def c_fit(log_path,L_out,tp,ROI,c_rng,ROInames):
    Xo = make_Xo(ROI,ROInames)
    #Exclusion mask; we do not count the regions with 0 path
    #####Need to write the code until log.path before