"""analyzespread"""
"""04/02/2021"""
# Importing modules
import os
import pickle
import numpy as np
# Modules for plotting
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from process_pathconn import *
from Pipeline import *
from fitfunctions import c_fit, make_Xo, predict_Lout

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
tp = np.array([1, 3, 6]) # To generalized if different timepoint?
Mice = []
for time in tp: # Creation of a list of 3 panda dataframe. These 3 df correspond to the 3 tp
        l = path_data[path_data['Time post-injection (months)'] == time][path_data['Conditions'] == 'NTG'][path_data.columns[2::]]
        l = l.reset_index(drop = True) # Reset the index, drop = True is used to remove the old index as it is by default added as a column
        Mice.append(l)# list of panda
Grp_mean = []
for i in range(0,3):
    Grp_mean.append(Mice[i].mean()) # Careful, now the mean are display as columns and not as rows anymore
Grp_mean = pd.concat([Grp_mean[0], Grp_mean[1], Grp_mean[2]], axis = 1)
Grp_mean.columns = ["1 MPI","3 MPI","6 MPI"]

##################################################
### Train model to predict path from iCPu seed ###
##################################################

# Loading L_out
L_out = np.loadtxt('L_out.csv')

# Fit time scaling parameter
# c_rng = np.arange(0.01, 10, step = 0.1) # Step =0.1 for a total length of 100
c_rng = np.linspace(start=0.01, stop=10, num=100)

log_path = np.log10(Grp_mean)
c_Grp = c_fit(log_path, L_out, tp,'R CPu', c_rng, ROInames) # Returns a best fit number. For the 'R Cpu' returns 1.6245

#############################################################
### Test model at observed points for group (NTG or G20)  ###
#############################################################
Xo = make_Xo('R CPu', ROInames) # Where we seed our pathology
vulnerability = mask = pd.DataFrame() # To double check but mask can be removed
Xt_Grp = [predict_Lout(L_out, Xo, c_Grp,i) for i in tp]
p_SC = p_vuln = c_tests = pd.DataFrame()
r_SC = pd.DataFrame(columns=["MPI","Pearson"]) # Result df to store our correlation coefficients

for M in range(0,len(tp)):
    Df = pd.DataFrame({"Path" : np.log10(Grp_mean.iloc[:,M]).values, "Xt" : np.log10(Xt_Grp[M])}, index= Grp_mean.index) # Runtime Warning
    # exclude regions with 0 pathology at each time point for purposes of computing fit
    Df = Df[Df["Path"] != -np.inf][Df["Xt"] != -np.inf][Df["Xt"] != np.nan] # Excluding Nan, and -Inf values for each tp
    cor = {"MPI" : "%s" % (M), "Pearson": stats.pearsonr(Df["Path"], Df["Xt"])[0]}
    r_SC = r_SC.append(cor, ignore_index=True) #Question, How to set index as MPI?
    print("Month Post Injection %s"%tp[M])
    print("Number of Regions used: ", len(Df))
    print("Pearson correlation coefficient", r_SC['Pearson'][M])
    # Plotting the Log(Predicted) vs Log(Path)
    fig = plt.figure()
    plt.scatter(Df["Xt"], Df["Path"], c="r")
    sns.regplot(Df["Xt"], Df["Path"], data=Df, color="blue")
    plt.xlabel("Log(Predicted)")
    plt.ylabel("Log(Path)")
    plt.title("Month Post Injection %s"%tp[M])
    plt.grid(color="grey")
    plt.show()
    #NEED TO EXTRACT EVERY GRAPH ==> in p_SC
    # Plotting the Vulnerability graph
    #Same story with the vulnerability df
    vulnerability
