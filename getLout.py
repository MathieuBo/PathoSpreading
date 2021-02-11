"""getLout"""
"""03/02/2021"""

#######################
#######LOAD DATA#######
#######################

#Def getLout

# Importing modules
import os
import pandas as pd
import numpy as np
from scipy.io import loadmat
import pickle

# Importing the Pipeline
from Pipeline import *  # This will be removed after defining it as a function

# Setting the environment
basedir = params[0]  # Extract the basedir from params # Need cuz after path_conn wd = "\\asyndiffusion3\\processed"
os.chdir(basedir)  # Sets the wd to
savedir = os.path.join(basedir, opdir, "processed") # opdir ="asyndiffusion3"
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
W = W.values
np.fill_diagonal(W, 0)
W = W / np.max(np.real(np.linalg.eig(W)[0])) # Extracting the max eigenvalue from our matrix and scaling the whole matrix

# Computation of in- and out-degree
out_degree = W.sum(axis=1) # To sum over rows
in_degree = W.sum(axis=0) # To sum over columns

# Creation of the Laplacian matrix
L_out = np.diag(out_degree) - W  #DOUBLE CHECK STILL REQUIRED

# Saving the Laplacian matrix
np.savetxt('L_out.csv',L_out)

##################################
### Synuclein weighted matrix ####
##################################
os.chdir(basedir)  # Sets the wd
# Loading Snca and diagonalizing it
Snca = pd.read_csv("Data83018/Snca.csv")
Snca = Snca.values[:,0]
Snca = np.diag(Snca)

# Extraction of W and Synuclein weighted matrix
os.chdir(savedir)
W = pd.read_csv("W.csv", index_col = 0)
W = W.values #Converting to numpy
np.fill_diagonal(W, 0)
W_snca = np.dot(Snca, W) # Synuclein weighted matrix
W_snca = W_snca / np.max(np.real(np.linalg.eig(W_snca)[0])) #Scaling with max eigenvalue

# Computation of in- and out-degree
out_degree = W_snca.sum(axis=1)  # To sum over rows
in_degree = W_snca.sum(axis=0)  # To sum over columns

# Creation of the Synuclein Laplacian matrix
L_out = np.diag(out_degree) - W_snca

# Saving the Laplacian matrix
np.savetxt('L_out_scna.csv',L_out) #####NEED TO CHANGE

# To remove: Need to check that W produced in python and R are the same
# os.chdir(savedir)
# W_sn = pd.read_csv("TESTW_snca.csv", header = None)
# W_snR = pd.read_csv("TEST_W_R.csv", index_col = 0)
# a = (W_snR.values == W_sn.values)