"""analyzespread"""
"""04/02/2021"""
# import modules
import os
import pickle
import numpy as np
from process_pathconn import *
from Pipeline import *
# from fitfunctions import * to unlock when all functions will be written

# Setting the environment
basedir = params[0]  # Extract the basedir from params
os.chdir(basedir)  # Sets the wd
# Creating the folder
savedir = os.path.join(basedir, opdir, "diffmodel")
try:
    os.mkdir(savedir)
except WindowsError as error:
    print(error)  # Prevent the algorithm to stop if the folder is already created. For Mac users need to replace by OSError.

savedir = os.path.join(savedir, "roilevel")
try:
    os.mkdir(savedir)
except WindowsError as error:
    print(error)  # Prevent the algorithm to stop if the folder is already created. For Mac users need to replace by OSError.

# Loading path_data, conn_names, orig_order, n_regions from the pickle file  PICKLE or import algo to check
os.chdir(os.path.join(basedir, opdir, "processed")) #Accessing the path of the data
path_data_pickle_in = open("path_data_pickle_out","rb")
path_data_objects = pickle.load(path_data_pickle_in)
path_data = path_data_objects[0]
conn_names = path_data_objects[1]
orig_order = path_data_objects[2]
n_regions = path_data_objects[3]

# Processing them and creation of variables
nROIs =len(conn_names)*2
ROInames = ["R " + i for i in conn_names] + ["L " + i for i in
                                            conn_names]
#Here in R they ordered the list using factor, not done here ==> Ask

# Get mean of each month
tp = np.array([1, 3, 6])
Mice = []
for time in tp:
    l = path_data[path_data['Time post-injection (months)'] == time][path_data['Conditions'] == 'NTG'][path_data.columns[2::]]
    Mice.append(l)
    #PROBLEM INDEX ?

##################################################
### Train model to predict path from iCPu seed ###
##################################################

# Loading L_out
L_out = np.loadtxt('L_out.csv')
c_rng = np.arange(0.01, 10, step = 0.1) # Step =0.1 for a total length of 100

