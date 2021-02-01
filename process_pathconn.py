"""Process_pathconn"""
"""01/02/2021"""

#######################
#######LOAD DATA#######
#######################

#Need to clear wspace ==> one line missing
import pandas as pd
import numpy as np
from Pipeline import *
basedir = params[0]#Extract the basedir from params
os.chdir(basedir)#Sets the wd
opdir = "processed"
try:
    os.mkdir(opdir)
except WindowsError as error:
    print(error) # Prevent the algorithm to stop if the folder is already created. For Mac users need to replace by OSError.
data = pd.read_csv("Data83018\\data.csv", header=0)#Stored as a Dataframe with header
connectivity_ipsi = pd.read_csv("Data83018\\connectivity_ipsi.csv", index_col=0)
connectivity_contra = pd.read_csv("Data83018\\connectivity_contra.csv", index_col=0)

########################
##PROCESS REGION NAMES##
########################

conn_names = [i.split(' (')[0] for i in connectivity_contra.columns]
path_names = data.columns[2::] #Extraction of the path names, they start from the column 3.
cInd = [i for i in range(0,len(path_names)) if path_names[i][0] == "c"] # Extraction of the index
iInd = [i for i in range(0,len(path_names)) if path_names[i][0] == "i"]

c_path_names_contra = [path_names[i] for i in range(0,len(path_names)) if path_names[i][0] == "c"] # Extraction of the name
path_names_contra = [i.replace("c","") for i in c_path_names_contra] # Same without the first letter (to fit with conn.names)

i_path_names_ipsi = [path_names[i] for i in range(0,len(path_names)) if path_names[i][0] == "i"]
path_names_ipsi = [i.replace("i","") for i in i_path_names_ipsi]


#Order path and connectivity by the same name
path_data_ipsi = data.loc[:,i_path_names_ipsi]#Returns the columns corresponding to the proper order path
path_data_contra = data.loc[:,c_path_names_contra]
