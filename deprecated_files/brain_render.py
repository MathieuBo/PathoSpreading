import brainrender
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

tp = [1,3,6]
#Loading
data = pd.read_csv('../../output/predicted_pathology.csv', index_col=4)
data = data.drop(['Unnamed: 0'], axis=1)
brainR = pd.read_csv('../Data83018/BrainR_ROInames.csv', sep=",", usecols=['BrainR', 'ROInames', 'Side'])
cmap = plt.get_cmap('Reds')


#Normalization of Data values
data = (data - np.min(data.values))/(np.max(data.values)-np.min(data.values))

#Associating BrainR Regions with their values
for month in range(0,len(tp)):
    brainR.insert(3+month,"Val_MPI{}".format(tp[month]),"")
    for idx, regions in enumerate(brainR["ROInames"]):
        brainR['Val_MPI{}'.format(tp[month])][idx] = data.loc[regions, "MPI{}".format(tp[month])]

###Still need ==> to implement this for each month
# Ipsilateral Data
ipsi_pred = brainR[brainR['Side'] == 'ipsi']

# Contralateral Data
contra_pred = brainR[brainR['Side'] == 'contra']

#BrainRenders
ipsi_pred_mpi1 = ipsi_pred[["BrainR", "Val_MPI1"]]
contra_pred_mpi1 = contra_pred[["BrainR", "Val_MPI1"]]

scene = brainrender.Scene(atlas_name="allen_mouse_10um")

#Adding the ipsilateral regions

for reg in contra_pred_mpi1.index:
    region = contra_pred_mpi1.loc[reg, 'BrainR']
    color = cmap(contra_pred_mpi1.loc[reg, 'Val_MPI1'])[:3]
    scene.add_brain_region(reg, color=color, alpha=0.5)
# #scene.axes_indices(None)
scene.render()

#Adding the controlateral regions
