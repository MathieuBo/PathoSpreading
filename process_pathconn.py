"""Process_pathconn"""
"""01/02/2021"""

#######################
#######LOAD DATA#######
#######################

# Need to clear wspace ==> one line missing
import os
import pandas as pd
import numpy as np
from scipy.io import savemat
import pickle

from Pipeline import *


basedir = params[0]  # Extract the basedir from params
os.chdir(basedir)  # Sets the wd
savedir = os.path.join(basedir, opdir, "processed")
try:
    os.mkdir(savedir)
except WindowsError as error:
    print(error)  # Prevent the algorithm to stop if the folder is already created. For Mac users need to replace by OSError.
data = pd.read_csv("Data83018\\data.csv", header=0)  # Stored as a Dataframe with header
connectivity_ipsi = pd.read_csv("Data83018\\connectivity_ipsi.csv", index_col=0)
connectivity_contra = pd.read_csv("Data83018\\connectivity_contra.csv", index_col=0)

########################
##PROCESS REGION NAMES##
########################

conn_names = [i.split(' (')[0] for i in connectivity_contra.columns]
path_names = data.columns[2::]  # Extraction of the path names, they start from the column 3.

c_path_names_contra = [path_names[i] for i in range(0, len(path_names)) if
                       path_names[i][0] == "c"]  # Extraction of the names
path_names_contra = [i[1:] for i in c_path_names_contra]  # Same without the first letter (to fit with conn.names)

i_path_names_ipsi = [path_names[i] for i in range(0, len(path_names)) if path_names[i][0] == "i"]
path_names_ipsi = [i[1:] for i in i_path_names_ipsi]

# Order path and connectivity by the same name
path_data_ipsi = data.loc[:, i_path_names_ipsi]  # Creates a df with reorganized columns
path_data_contra = data.loc[:, c_path_names_contra]

ordered_matched_ipsi = []
for i in range(0, len(conn_names)):
    for k in range(0, len(path_names_ipsi)):
        if path_names_ipsi[k] == conn_names[i]:
            ordered_matched_ipsi.append(
                k)  # Returns a list containing the ranks of path_names_ipsi to fit in conn.names

ordered_matched_contra = []
for i in range(0, len(conn_names)):
    for k in range(0, len(path_names_contra)):
        if path_names_contra[k] == conn_names[i]:
            ordered_matched_contra.append(k)

# Creation of path_data
path_data = pd.concat([data.loc[:, data.columns[0:2]._index_data],
                       path_data_ipsi.loc[:, path_data_ipsi.columns._index_data[ordered_matched_ipsi]],
                       path_data_contra.loc[:, path_data_contra.columns._index_data[ordered_matched_contra]]], axis=1)
path_data = path_data.rename(columns={"MBSC Region": "Conditions"})  # Renaming a column

# tile matrix such that sources are rows, columns are targets (Oh et al. 2014 Fig 4)
connectivity_ipsi.columns = conn_names  # Sets the names of the columns and the index to be the same
connectivity_ipsi.index = conn_names

connectivity_contra.columns = conn_names
connectivity_contra.index = conn_names

# Creation of the adjacency matrix
W = pd.concat([pd.concat([connectivity_ipsi, connectivity_contra], axis=1),
               pd.concat([connectivity_contra, connectivity_ipsi], axis=1)], axis=0)
n_regions = len(W)

# Checking if the matrix was tiled properly
# W_array = W.to_numpy()
# if (sum(W_array[0:57,0:57] != W_array[58:117,58:117]) > 0):
#    print("Matrix failed to concatenate")

# retain indices to reorder like original data variable for plotting on mouse brains
ROInames = ["i" + i for i in conn_names] + ["c" + i for i in
                                            conn_names]  # List of ROI w/ first the controlateral regions and then the ipsilateral regions.
orig_order = []
for i in range(0, len(ROInames)):
    for k in range(0, len(data.columns._index_data) - 2):
        if ROInames[k] == data.columns._index_data[2::][i]:
            orig_order.append(k)

# Save the adjacency matrix
os.chdir(savedir)
W.to_csv('W.csv')

#Saving path_data, conn_names, orig_order, n_regions in a file located in the folder processed
path_data_objects = [path_data, conn_names, orig_order, n_regions] #Will depend on whether we will import directly this whole code ==> Need to check
path_data_pickle_out = open("path_data_pickle_out","wb")
pickle.dump(path_data_objects, path_data_pickle_out)

os.chdir(basedir)  # Sets the wd back

###########################
##PROCESS ROI COORDINATES##
###########################
coor = pd.read_csv("Data83018\\ROIcoords.csv")
coor.rename(columns = {'Unnamed: 0':'ROI'}, inplace = True)
idx = []
for i in range(0, len(ROInames)):
    for k in range(0, len(coor['ROI'])):
        if ROInames[i] == coor['ROI'][k]:
            idx.append(k)
coor = coor.loc[idx,:] # From the new index created we reorganize the table by index.

# Saving the reorganized coordinates
os.chdir(savedir) #Saving in the folder processed
coor.to_csv('coor.csv')
os.chdir(basedir)  # Sets the wd back

###############################
### Process Snca Expression ###
###############################
synuclein = pd.read_csv("Data83018\\SncaExpression.csv", index_col = 0, header= None)
synuclein_ordered = []
for i in range(0, len(ROInames)):
    for k in range(0, len(synuclein.index._index_data)):
        if ROInames[i] == synuclein.index._index_data[k]:
            synuclein_ordered.append(synuclein.index._index_data[k])
synuclein = synuclein.loc[synuclein_ordered,:] # Reordered

# Saving the reorganized Snca expression
os.chdir(savedir) #Saving in the folder processed
synuclein.to_csv('Snca.csv')
os.chdir(basedir)  # Sets the wd back
