"""Pipeline_Synuclein_spread"""
"""29/01/2021"""

import os

basedir = "C:\\Users\\thoma\\Documents\\M1_Neurasmus\\NeuroBIM_M1\\Internship\\GitRepo\\PathoSpreading"
os.chdir(basedir)  # Sets the wd
opdir = "asyndiffusion3"

try:
    os.mkdir(opdir)
except WindowsError as error: # Prevent the algorithm to stop if the folder is already created. For Mac users need to replace by OSError.
    print(error)

grp = ["NTG"]  # List of the group studied
params = [basedir, opdir, grp]  # List of three str elements

# Here should be written the source for the packages (if used).

############################
####### Process Data #######
############################
