"""Pipeline_Synuclein_spread"""
"""29/01/2021"""
import os
# Need to clear workspace by restarting the interactive interpreter OR need to create a function to do so

basedir = "C:\\Users\\thoma\\Documents\\M1_Neurasmus\\NeuroBIM_M1\\Internship\\GitRepo\\PathoSpreading"
os.chdir(basedir)#Sets the wd
print("Current working directory: ", os.getcwd())
opdir = "asyndiffusion3"
try:
    os.mkdir(opdir)
except WindowsError as error:
    print(error) # Prevent the algorithm to stop if the folder is already created. For Mac users need to replace by OSError.
grp = ["NTG"] # List
params = [basedir, opdir, grp] # List of three str elements
# In params no matlab.path='C:/Program Files/MATLAB/R2020b/bin' in the list as we will try to use something different

# Here should be written the source for the packages (if used).

############################
####### Process Data #######
############################

