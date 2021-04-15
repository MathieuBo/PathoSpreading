--------------------------------------------------------------------------------------------------------------------------------------------------
---------------------------------------------------------------------README-----------------------------------------------------------------------
--------------------------------------------------------------------------------------------------------------------------------------------------

		This Python Algorithm is a replication from the original code produced by Henderson et Al.
		and presented in Nature vol. 22 in 2019. The original code has been developped using RStudio 3.3.3.

		This ReadMe is subdivided into different sections and aimed at clarifying the main inputs required
		and the main outputs obtained after running the code.
	
		A> Codes
		All codes have been written using Python 3.7 and WindowsOS 10.
			-Pipeline.py
			-process_files.py
			-Laplacian.py
			-fitfunctions.py
			-Robustness_Stability.py
			-summative_model.py

		B> Data sets
			-data.csv
			-connectivity_ipsi.csv
			-connectivity_contra.csv
			-SncaExpression.csv

		Packages used:
			-numpy
			-matplotlib
			-seaborn		
			-pandas
			-scipy
			-tqdm
			-statsmodels.stats.multitest

			
		A> Codes
		--->Pipeline.py
			Made of a DataManager to easily change the different parameters such as the timepoints, 
			the seed, or the quantified pathology matrix. pipeline.py successively calls the scripts
			mentioned below and saves the data as tables and figures. 

		Main inputs:

			--> exp_data = Quantified synucleinopathy compiled in an excel folder.

			--> timepoints = by default [1,3,6]. Can be modified when changing the experimental
			data set.

			--> connectivity_ipsi.csv & connectivity_contra.csv = Contains the connectivity data
			from the Allen Brain Institute.

			--> synuclein = Energy Value excel sheet of the snca gene expression imported from the Allen Brain Institute.
			By defaut use_expression_values= False. When True the code weights the Laplacian matrix
			with the snca gene expression in Laplacian.py.

			--> seed = By default: iCPu. Seed in which the disease in injected. 
			If not in seed list returns [ERROR in make_Xo]:  seed is not in the list of ROI.
 
			--> nb_of_region = by default: 58 (maximum value). Used the compute the robustness of the model.

		Main outputs:
		The main outputs are saved in two main folders.
			-In the folder "output":
				--> predicted_pathology{seed}.csv = Excel table with the predicted pathology values for each timepoint.

				--> model_output_MPI{timepoint}{seed}.csv = Excel table that contains the results of the 
				linear regression for a specific timepoint.

				--> stats{seed}.csv = Contains for each timepoint the pearson r coefficient, the p-value and the corrected
				p-value. (Bonferroni's correction)

			-In the folder "plots"
				--> Predicted_VS_Path_MPI{timepoint}{seed}.png/.pdf = lollipop plot showing the predicted vs pathology data.

				--> density_VS_residual{seed}.png/.pdf = Density plots of the residuals for a specific seed.

				--> predicted_pathology{seed}.csv = Heatmap with normalized values of predicted pathology. 
				By default Drop_seed : False. If turned True then it drops the seed from the Heatmap and plots
				Predictions_{seed}_excluded{seed}.png/.pdf.

				-->plot_r_(c)_MPI{timepoint}{seed}.png/.pdf = Plots of the evolution of r with respect to c
				for a specific timepoint.

				--> Random_Seed/Random_patho/Random_Adja.png/.pdf = Plots of the the fits obtained when we used different controls.

				-->Stability_MPI{timepoint}{seed}.png/.pdf = Graph used to compute the minimal regions needed to
				obtain a stable fit r

				-->Mean_MPI{timepoint}_SW{Sliding_number}{seed}.png/.pdf = Same as before but when computing a mean with a Sliding
				window.

				-->SD_MPI{timepoint}_SW{Sliding_number}{seed}.png/.pdf = Same as before but when computing a Standard Deviation 
				with a Sliding window.

			The same tables and figures are also produced when running the scripts for the Iterative model or considering
			individualised data for each mouse.
	


		--->process_files.py
				Processes the data inputs (Quantified pathology matrix, ipsilateral connectivity matrix, 
				and contralateral connectivity matrix). Its main outputs are the adjacency matrix, a 
				pathology matrix, and the list of regions of interest shaped as panda DataFrames. 
				More information on each of the functions defined in process_files.py can be found in the script code.

		--->Laplacian.py
				Computes from the adjacency matrix the out-degree Laplacian matrix. Computes the Laplacian adjacency
				matrix weighted with the energy value of snca gene expression when use_expression_values= True.

		--->fitfunctions.py
				Set of functions with different roles. "make_Xo" has been reproduced as described in Henderson's code and
				produces of a column vector (here shaped as a panda DataFrame) that contains 0 for non-seed regions or 1 
				for the seed region.
				"predict_Lout" predicts the regional alpha-synuclein pathology when given the Laplacian, the seed, a timepoint 
				of prediction, and the tuning constant c. 
				"c_fit" returns the value of the best fit c and the best correlation coefficient using "predict_Lout" and 
				Pearson's correlation test.

		--->Robustness_Stability.py
				Breaks the matrices into pieces to compute different test of stability. Confere the definition of the functions for more
				information.

		--->summative_model.py
				Iterative model. The main modifications from the original code resides in the prediction function that uses a different
				equation.

		B> Data Sets
		The Data Sets used being the main inputs we will expose here the shape they should have to be processed properly by the algorithmn.
		More generally, Henderson et al. measured the pathology in 58 regions/ side and created two 58x58 connectivity matrices. Finally, 
		using the API from Allen Brain Institute they extracted the gene expression from the 58 regions of interest.

			--> data.csv is shaped as presented below:

		Time post-injection (months)	MBSC Region	iM2		iM1		iAI		iPrL		iVO		...
		1				NTG		0.014842145	0		0.0036289	0.0066497	0.011567358	...
		1				NTG		0.227463	0.01637225	0.139844055	0.044689435	0.01536177      ...
		1				NTG		0.26633468	0.00869148	0.028367347	0.092924027	0.00972719      ...
		.														    .
		.													 	    .
		.													            .
		37rowsx118columns
		The ROI are not ordered and will be using the order from the connectivity data.
		MBSC Region correspond to the type of animal here NTG stand for Non-Transgenic. (Our code process the NTG data only)
		The Time post-injection is 1, 3 or 6 months here. If modified, need to modify the timepoint input as well.

			--> connectivity_ipsi.csv & connectivity_contra.csv are shaped as presented below:

					Cg (ACAd + ACAv)	Acb (ACB)	TC (AHN, LHA, TU, VMH)	AI (AId)	AI-b (Alv, Ald, GU)	...
		Cg (ACAd + ACAv)	0.70376136		0.076145675	0.027158017		0.011908716	0.011292776		...
		Acb (ACB)		0			0.074182913	0.065485206		0		0.003016284		...
		TC (AHN, LHA, TU, VMH)	0			0.01076834	0.108167074		0		0			...
		AI (AId)		0			0		0			0.311219424	0.211814808		...
		AI-b (Alv, Ald, GU)	0.089091663		1.650240251	0.072525433		0.58024648	0.436141508		...
		AIP (AIP)		0.039485547		0.486595031	0.041829649		0.630972509	0.355623613		...
		.													     .
		.													     .
		.													     .

		58rowsx58columns
		The algorithmn process the connectivity data and rename the column and index without the content from the parenthesis. e.g

			Cg		Acb		TC		AI		AI-b			...
		Cg	0.70376136	0.076145675	0.027158017	0.011908716	0.011292776		...
		Acb	0		0.074182913	0.065485206	0		0.003016284		...
		TC	0		0.01076834	0.108167074	0		0			...
		AI	0		0		0		0.311219424	0.211814808		...
		AI-b	0.089091663	1.650240251	0.072525433	0.58024648	0.436141508		...
		AIP	0.039485547	0.486595031	0.041829649	0.630972509	0.355623613		...
		.									     .
		.									     .
		.									     .
		This allows to easily label the index and columns but also extract the list of ROI
		Changing data.csv require then to use a new connectivity matrix.

			--> SncaExpression.csv
		The gene expression can be extracted from the API of the Allen Brain Institute. Here is the shape of the SncaExpression used:

		iM2	24.36416667
		iM1	17.44552632
		iAI	17.9625
		iPrL	21.48090909
		iMO	15.80214286
		iVO	7.081
		iDP	12.77176923
		cM2	24.36416667
		 .	    .
		 .	    .							    
 		 .          .
		116rowsx2columns

															Last modification : 13/04/2021
