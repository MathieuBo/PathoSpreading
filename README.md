# Spreading of pathologenic proteins through the brain connectome
Mathieu Bourdenx - Thomas Paul (@Thomas-Paul-Fr) - 2021

Code to model the spread of pathogenic proteins through the mouse brain connectome. This is a replication from by [Henderson et al. *Nat. Neurosci*, 2019](https://www.nature.com/articles/s41593-019-0457-5). The original code has been developped using RStudio 3.3.3 and can be found [here](https://github.com/ejcorn/connectome_diffusion). The code has been here replicated in Python. 

This ReadMe is subdivided into different sections and aimed at clarifying the main inputs required and the main outputs obtained after running the code.

## Requirements
This code has been written for Python 3.6/3.7 and successfully tested on Microsoft Windows 10 and MacOS 11.2.3 (BigSur). 
It requires the following packages:
- numpy
- matplotlib
- seaborn		
- pandas
- scipy
- tqdm
- statsmodels

## Datasets
All data are contained in the folder Data83018. The files are the following ones:
- data.csv [37rowsx118columns]
- connectivity_ipsi.csv [58rowsx58columns]
- connectivity_contra.csv [58rowsx58columns]
- SncaExpression.csv [116rowsx2columns]

## Scripts

**Pipeline.py**
This is the main file allowing to control and run the model. Several functions have been implemented to allow to run the model it is simple version or combining expression values. It also allows to run some robustness tests. 

Main *inputs*:
- output_path: The path where to save all outputs
- exp_data: Quantification of alpha-synuclein pathology (values from data.csv)
- timepoints: by default [1,3,6]. Can be modified when changing the experimental data set.
- connectivity_ipsi.csv & connectivity_contra.csv: connectivity matrices obtained from [Oh et al. 2014](https://www.nature.com/articles/nature13186)
- synuclein: Energy (i.e. expression) values found in SncaExpression.csv. By defaut ```use_expression_values= False```. 
- seed: seeding region. By default: iCPu. If the seed is not found in seed list returns [ERROR in make_Xo].
- nb_of_region: by default: 58 (maximum value). Used to compute the robustness of the model.

Main *outputs*:
All different outputs are stored in a main folder named PathoSpreading_OUTPUT.
According to the initial conditions (choice of a seed and use -or not- of the expression values) the algorithm produces an specific ouput folder named {seed} or {seed_SNCA} that contains two main folders:

In the folder "Tables":
- predicted_pathology{seed}{timepoint}.csv: CSV table with the predicted pathology values for each timepoint.
- model_output_MPI{timepoint}{seed}.csv: CSV table that contains the results of the linear regression for a specific timepoint.
- stats{seed}.csv: Contains for each timepoint the Pearson's r coefficient, the p-value and the corrected p-value. (Bonferroni's correction)

In the folder "plots"
- Predicted_VS_Path_MPI{timepoint}_{seed}.png/.pdf: Scatterplots showing the predicted vs pathology data.
- In the folder "Density_vs_residuals" --> density_VS_residual{seed}_MPI{timepoint}.png/.pdf: Density plots of the residuals for a specific seed.
				       --> Lollipop_MPI{timepoint}_{seed}.png/.pdf : Lollipop plots showing the predicted vs pathology data.
- In the folder "Fits(c)" --> plot_r_(c)_MPI{timepoint}_{seed}.png/.pdf : Plots of the evolution of r with respect to c for a specific timepoint.

- In the folder "Heatmap_Predictions" --> Predictions_{seed}.png/.pdf Heatmap with normalized values of predicted pathology. By default ```Drop_seed : True```, allowing to drop the seeded region from the heatmap toget a better dynamic range in visualization. 
Predictions_{seed}_excluded{seed}.png/.pdf.

- In the folder "Model_Robustness" --> Random_Seed/Random_patho/Random_Adja.png/.pdf : Plots of the the fits obtained when we used different controls.

*Note: By default ```display_plots : False``` to avoid that the user needs to manually close all plot before proceeding. All plots are saved to the aforementioned folders.*

**process_files.py**
	Processes the data inputs (Quantified pathology matrix, ipsilateral connectivity matrix, 
	and contralateral connectivity matrix). Its main outputs are the adjacency matrix, a 
	pathology matrix, and the list of regions of interest shaped as panda DataFrames. 
	More information on each of the functions defined in process_files.py can be found in the code.

**Laplacian.py**
	Computes from the adjacency matrix the out-degree Laplacian matrix. Computes the Laplacian adjacency
	matrix weighted with the energy value of snca gene expression when ```use_expression_values=True```.

**fitfunctions.py**
	Set of functions with different roles. "make_Xo" has been reproduced as described in Henderson's code and
	produces of a column vector (here shaped as a Pandas DataFrame) that contains 0 for non-seed regions or 1 
	for the seed region.
	"predict_Lout" predicts the regional alpha-synuclein pathology when given the Laplacian, the seed, a timepoint 
	of prediction, and the tuning constant c. 
	"c_fit" returns the value of the best fit c and the best correlation coefficient using "predict_Lout" and 
	Pearson's correlation test.

**Robustness_Stability.py**
	Breaks the matrices into pieces to compute different test of stability. Confere the definition of the functions for more
	information.

