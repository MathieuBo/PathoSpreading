"""analyzespread"""
"""04/02/2021"""
# Importing modules
# Modules for plotting
from scipy import stats
import numpy as np
from Pipeline import *
from fitfunctions import c_fit, make_Xo, predict_Lout
from process_pathconn import *
from matplotlib import pyplot as plt
import seaborn as sns

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
Grp_mean.columns = ["MPI 1", "MPI 3", "MPI 6"]

##################################################
### Train model to predict path from iCPu seed ###
##################################################
# listing the directories so that the User can see the names of the files
print(os.listdir(os.path.join(basedir, opdir, "processed")))

# Loading the Laplacian matrix
Laplacian = input("Please type the Laplacian to use")
if Laplacian == "L_out_scna.csv":
    Lap = "SCNA_"
    L_out = np.loadtxt(Laplacian)
if Laplacian == "L_out.csv":
    Lap = ""
    L_out = np.loadtxt(Laplacian)

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
vulnerability = pd.DataFrame(0,columns=["MPI 1", "MPI 3", "MPI 6"], index=Grp_mean.index)  # To double check but mask can be removed
Xt_Grp = [predict_Lout(L_out, Xo, c_Grp, i) for i in tp]
r_SCc = pd.DataFrame(columns=["MPI", "Pearson r"])
r_SCp = pd.DataFrame(columns=["MPI", "p_value"])  # Result df to store our correlation coefficients
p_values_cor = list()

masks = dict()
os.chdir(os.path.join(basedir, opdir, "diffmodel"))
for M in range(0, len(tp)): # M iterates according to the number of timepoint
    Df = pd.DataFrame({"Path": np.log10(Grp_mean.iloc[:, M]).values, "Xt": np.log10(Xt_Grp[M])},
                       index=Grp_mean.index)  # Runtime Warning
    # exclude regions with 0 pathology at each time point for purposes of computing fit
    mask = (Df["Path"] != -np.inf) & (Df['Xt'] != -np.inf) & (Df['Xt'] != np.nan)
    print(np.sum(mask))
    masks["MPI %s" % tp[M]] = mask
    Df = Df[mask]

    cor = {"MPI": "%s" % (M), "Pearson r": stats.pearsonr(Df["Path"], Df["Xt"])[0]}
    p_val = {"MPI": "%s" % (M), "p_value": stats.pearsonr(Df["Path"], Df["Xt"])[1]}
    r_SCc = r_SCc.append(cor, ignore_index=True)  # Question, How to set index'name as MPI?
    r_SCp = r_SCp.append(p_val, ignore_index=True)
    print("Month Post Injection %s" % tp[M])
    print("Number of Regions used: ", len(Df))
    print("Pearson correlation coefficient", r_SCc['Pearson r'][M])
    # Plotting the Log(Predicted) vs Log(Path)

    fig = plt.figure()
    plt.scatter(Df["Xt"], Df["Path"], c="r")
    sns.regplot(Df["Xt"], Df["Path"], data=Df, color="blue")
    plt.xlabel("Log(Predicted)")
    plt.ylabel("Log(Path)")
    plt.title(f"{Lap}Month Post Injection {tp[M]}")
    plt.grid(color="grey")
    fig.savefig(f'{Lap}PredictVSPath_Month_{tp[M]}.png', dpi=300)
    fig.savefig(f'{Lap}PredictVSPath_Month_{tp[M]}.eps')
    #plt.show()
    #The three months in the same fig + Correlation coefficient?

    # Plotting the Vulnerability graph
    slope, intercept, r_value, p_value, std_err = stats.linregress(x=Df['Xt'], y=Df['Path'])
    pred_values = slope * Df['Xt'] + intercept
    Residual = Df['Path'] - pred_values
    vulnerability["MPI %s" % tp[M]] = Residual

    plt.figure(figsize=(20, 6), constrained_layout=True)
    plt.bar(np.arange(len(Residual)), Residual)
    plt.xticks(np.arange(len(Residual)), Df.index, rotation='vertical')
    plt.xlim(xmin=-0.6, xmax=len(Residual) + 0.6)
    plt.title(f"{Lap}Vulnerability_Month_{tp[M]}")
    # Saving the vulnerability graph
    plt.savefig(f'{Lap}Vulnerability_Month_{tp[M]}.png', dpi=300)
    plt.savefig(f'{Lap}Vulnerability_Month_{tp[M]}.eps')
    #plt.show()
    #Three subplots? + RCG in R while here iCg ==> Ticks

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

Diff = Df_mean_Xt["Xt_MPI 1"] - Df_mean_Xt["MPI 1"] # Fixed values
for t in tp:
    Df_mean_Xt["Xt_MPI %d" % t] = Df_mean_Xt["Xt_MPI %d" % t] - Diff # Normalization of the predicted values (log)

# find regions with low average vulnerability at all time points
Df_v = vulnerability[masks]
Df_v = Df_v.dropna()
low_vuln = np.where(np.abs(Df_v).mean(axis=1) < 0.2)[0] # Returns an array containing the index of the areas of low vulnerability
os.chdir(savedir) # Saving in ROIlevels
for ROI in low_vuln:
    fig = plt.figure()# Ask Mathieu
    plt.plot(tp,Df_mean_Xt.iloc[ROI,0:3], marker='o', color="r", linestyle="none")
    plt.plot(tp,Df_mean_Xt.iloc[ROI,3:6],  color="b")
    plt.legend(["Path","Predicted values"], loc="lower right")
    plt.xlabel("Month Post Injection")
    plt.ylabel("Log(Path)")
    plt.title(f"{Lap}Region of Low Vulnerability: {ROInames[ROI]}")
    plt.grid(color="grey")
    #plt.show()
    fig.savefig(f'{Lap}Low_vulnerability_region_{ROInames[ROI]}.png', dpi=300)
    fig.savefig(f'{Lap}Low_vulnerability_region_{ROInames[ROI]}.eps')

# unit test for vulnerability names matching up:
# regions with vulnerability = 0 should also have group mean pathology = 0
for i in range(0, 3):
    path_0 = np.where(Grp_mean.iloc[:, i] == 0)
    vuln_0 = np.where(vulnerability.iloc[:, i].isna())
    for k in range(0, len(path_0[0])):
        if path_0[0][k] != vuln_0[0][k]:
            print("Vulnerability does not match up with pathology ")

# Saving vulnerability & Predicted_Path

os.chdir(os.path.join(basedir, opdir, "diffmodel")) # Saved in the folder diffmodel
vulnerability.to_csv(f'{Lap}vulnerability.csv')
Xt_Grp.to_csv(f'{Lap}predicted_path.csv')

# Reordering and saving
vulnerability_reordered = (pd.concat([Grp_mean, vulnerability], axis=1, join="outer")).iloc[:,3:6]
vulnerability_reordered = vulnerability_reordered.iloc[orig_order,:]

predicted_path = Xt_Grp.iloc[orig_order,:]
v_names = ["i" + i for i in conn_names] + ["c" + i for i in
                                             conn_names]
v_names_ordered =[]
for i,j in enumerate(predicted_path.index):
    v_names_ordered.append(v_names[j])
predicted_path.index = v_names_ordered

vulnerability_reordered.to_csv(f'{Lap}vulnerabilityReordered.csv')
predicted_path.to_csv(f'{Lap}predicted_pathReordered.csv')