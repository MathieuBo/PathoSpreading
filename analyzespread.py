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
    print(
        error)  # Prevent the algorithm to stop if the folder is already created. For Mac users need to replace by OSError.

savedir = os.path.join(savedir, "roilevel")
try:
    os.mkdir(savedir)
except WindowsError as error:
    print(
        error)  # Prevent the algorithm to stop if the folder is already created. For Mac users need to replace by OSError.

# Loading path_data, conn_names, orig_order, n_regions from the pickle file  PICKLE or import algo to check
os.chdir(os.path.join(basedir, opdir, "processed"))  # Accessing the path of the data
path_data_pickle_in = open("path_data_pickle_out", "rb")
path_data_objects = pickle.load(path_data_pickle_in)
path_data = path_data_objects[0]
conn_names = path_data_objects[1]
orig_order = path_data_objects[2]
n_regions = path_data_objects[3]

# Processing them and creation of variables
nROIs = len(conn_names) * 2
ROInames = ["R " + i for i in conn_names] + ["L " + i for i in
                                             conn_names]
# Here in R they ordered the list using factor, not done here ==> Ask

# Get mean of each month
tp = np.array([1, 3, 6])  # To generalized if different timepoint?
Mice = []
for time in tp:  # Creation of a list of 3 panda dataframe. These 3 df correspond to the 3 tp
    l = path_data[path_data['Time post-injection (months)'] == time][path_data['Conditions'] == 'NTG'][
        path_data.columns[2::]]
    l = l.reset_index(
        drop=True)  # Reset the index, drop = True is used to remove the old index as it is by default added as a column
    Mice.append(l)  # list of panda
Grp_mean = []
for i in range(0, 3):
    Grp_mean.append(Mice[i].mean())  # Careful, now the mean are display as columns and not as rows anymore
Grp_mean = pd.concat([Grp_mean[0], Grp_mean[1], Grp_mean[2]], axis=1)
Grp_mean.columns = ["1 MPI", "3 MPI", "6 MPI"]

##################################################
### Train model to predict path from iCPu seed ###
##################################################

# Loading L_out
L_out = np.loadtxt('L_out.csv')

# Fit time scaling parameter
# c_rng = np.arange(0.01, 10, step = 0.1) # Step =0.1 for a total length of 100
c_rng = np.linspace(start=0.01, stop=10, num=100)

log_path = np.log10(Grp_mean)
c_Grp = c_fit(log_path, L_out, tp, 'R CPu', c_rng,
              ROInames)  # Returns a best fit number. For the 'R Cpu' returns 1.6245

#############################################################
### Test model at observed points for group (NTG or G20)  ###
#############################################################
Xo = make_Xo('R CPu', ROInames)  # Where we seed our pathology
vulnerability = pd.DataFrame(columns=["MPI 1", "MPI 3", "MPI 6"])  # To double check but mask can be removed
Xt_Grp = [predict_Lout(L_out, Xo, c_Grp, i) for i in tp]
p_SC = p_vuln = c_tests = pd.DataFrame()
r_SCc = pd.DataFrame(columns=["MPI", "Pearson r"])
r_SCp = pd.DataFrame(columns=["MPI", "p_value"])  # Result df to store our correlation coefficients
p_values_cor = list()

os.chdir(os.path.join(basedir, opdir, "diffmodel"))
for M in range(0, len(tp)):  # M iterates according to the number of timepoint
    Df = pd.DataFrame({"Path": np.log10(Grp_mean.iloc[:, M]).values, "Xt": np.log10(Xt_Grp[M])},
                      index=Grp_mean.index)  # Runtime Warning
    # exclude regions with 0 pathology at each time point for purposes of computing fit
    Df = Df[Df["Path"] != -np.inf][Df["Xt"] != -np.inf][
        Df["Xt"] != np.nan]  # Excluding Nan, and -Inf values for each tp
    cor = {"MPI": "%s" % (M), "Pearson r": stats.pearsonr(Df["Path"], Df["Xt"])[0]}
    p_val = {"MPI": "%s" % (M), "p_value": stats.pearsonr(Df["Path"], Df["Xt"])[1]}
    r_SCc = r_SCc.append(cor, ignore_index=True)  # Question, How to set index'name as MPI?
    r_SCp = r_SCp.append(p_val, ignore_index=True)
    print("Month Post Injection %s" % tp[M])
    print("Number of Regions used: ", len(Df))
    print("Pearson correlation coefficient", r_SCc['Pearson r'][M])
    # Plotting the Log(Predicted) vs Log(Path)
    plt.scatter(Df["Xt"], Df["Path"], c="r")
    corr_graph = sns.regplot(Df["Xt"], Df["Path"], data=Df, color="blue")
    plt.xlabel("Log(Predicted)")
    plt.ylabel("Log(Path)")
    plt.title("Month Post Injection %s" % tp[M])
    plt.grid(color="grey")
    # plt.savefig(f'PredictVSPath_Month_{tp[M]}.png', dpi=300)
    # plt.savefig(f'PredictVSPath_Month_{tp[M]}.eps')
    # plt.show()

    # Plotting the Vulnerability graph
    slope, intercept, r_value, p_value, std_err = stats.linregress(x=Df['Xt'], y=Df['Path'])
    pred_values = slope * Df['Xt'] + intercept
    Residual = Df['Path'] - pred_values
    vulnerability["MPI %s" % tp[M]] = Residual ### TO CHECK

    plt.figure(figsize=(20, 5), constrained_layout=True)
    plt.bar(np.arange(len(Residual)), Residual)
    plt.xticks(np.arange(len(Residual)), Df.index, rotation='vertical')
    plt.xlim(xmin=-0.6, xmax=len(Residual) + 0.6)
    # Saving the vulnerability graph
    # plt.savefig(f'Vulnerability_Month_{tp[M]}.png', dpi=300)
    # plt.savefig(f'Vulnerability_Month_{tp[M]}.eps')
    # plt.show()

# Correcting the p_values using Bonferroni method.
# We multiply the p_values to the number of comparison. Here 3 timepoints
r_SCp['p_value'] = r_SCp['p_value'] * len(tp)
for M in range(0, len(tp)):
    print("p_value corrected:", r_SCp['p_value'][M], 'for month %s' % tp[M])

# Saving r_SC
r_SC = pd.concat([r_SCc, r_SCp["p_value"]], axis=1)  # Contains Pearson R and P-values for each timepoint
r_SC_pickle_out = open("r_SC_pickle_out", "wb")
pickle.dump(r_SC, r_SC_pickle_out)

# compare predicted ts to observed for lowest vulnerability region
Xt_Grp = pd.DataFrame({"Xt_MPI 1": Xt_Grp[0], "Xt_MPI 3": Xt_Grp[1], "Xt_MPI 6": Xt_Grp[2]})
Df_mean_Xt = pd.concat([Grp_mean.reset_index(), Xt_Grp], axis=1).set_index("index")
Df_mean_Xt = np.log10(Df_mean_Xt)
Df_mean_Xt = Df_mean_Xt[
    Df_mean_Xt[Df_mean_Xt.columns] != -np.inf].dropna()  # Dropping rows containing Nan and -Inf values #96 regions

Diff = Df_mean_Xt["Xt_MPI 1"] - Df_mean_Xt["1 MPI"]
for t in tp:
    Df_mean_Xt["Xt_MPI %d" % t] = Df_mean_Xt["Xt_MPI %d" % t] - Diff # Normalization of the predicted values (log)

vulnerability = vulnerability.dropna() #Check should the same as df.v

# find regions with low average vulnerability at all time points
