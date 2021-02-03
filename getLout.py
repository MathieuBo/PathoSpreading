"""getLout"""
"""03/02/2021"""
# Need to clear wspace ==> one line missing

#######################
#######LOAD DATA#######
#######################

# Importing modules
import os
import pandas as pd
import numpy as np
from scipy.io import loadmat
import pickle

# Importing the Pipeline
from Pipeline import *

# Setting the environment
######Not necessary, need to check
basedir = params[0]  # Extract the basedir from params
os.chdir(basedir)  # Sets the wd
savedir = os.path.join(basedir, opdir, "processed")
try:
    os.mkdir(savedir)
except WindowsError as error:
    print(error)  # Prevent the algorithm to stop if the folder is already created. For Mac users need to replace by OSError.
#######

# Loading path_data, conn_names, orig_order, n_regions from the pickle file  PICKLE or import algo to check
os.chdir(savedir)
path_data_pickle_in = open("path_data_pickle_out","rb")
path_data_objects = pickle.load(path_data_pickle_in)
path_data = path_data_objects[0]
conn_names = path_data_objects[1]
orig_order = path_data_objects[2]
n_regions = path_data_objects[3]

#####################################
### Tract tracing connectome only ###
#####################################
W = pd.read_csv("W.csv", index_col = 0)

# Conversion to numpy
W_numpy = W.values
np.fill_diagonal(W_numpy, 0)
W_numpy = W_numpy / np.max(np.real(np.linalg.eig(W_numpy)[0])) # Extracting the max eigenvalue from our matrix and scaling the whole matrix

# Computation of in- and out-degree
out_degree = W_numpy.sum(axis=1) # To sum over rows
in_degree = W_numpy.sum(axis=0) # To sum over columns

# Creation of the Laplacian matrix
L_out = np.diag(out_degree) - W_numpy  #DOUBLE CHECK STILL REQUIRED

# Saving the Laplacian matrix using pickle


##################################
### Synuclein weighted matrix ####
##################################