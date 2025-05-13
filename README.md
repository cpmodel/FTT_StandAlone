# FTT StandAlone

## Future Technology Transformation
This repository contains a family of Future Technology Transformation (FTT) models. Models that are included are:

* FTT:Power (Mercure, 2012) - data up to 2022
* FTT:Heat (Knobloch et al, 2017) data up to 2020
* FTT:Transport (Mercure et al, 2018) - data up to 2022
* FTT:Freight (Nijsse et al, under review) - data up to 2023

## Theoretical background
The FTT family of models are based on [evolutionary economics](https://en.wikipedia.org/wiki/Evolutionary_economics). The uptake of new technologies typically follows an S-curve, which can be represented well with evolutionary dynamics (Mercure et al, 2012). The core equations for all of the models in the model family are coupled logistic equations of the [Lotka-Volterra family](https://en.wikipedia.org/wiki/Lotka%E2%80%93Volterra_equations), also known as the predator-prey equations. These equations are used to determine the evolution of the shares of various technologies in the models. Each model contains between ~10 to 25 technologies competing for market share. 

## FTT and E3ME
This repository contains the public standalone version of FTT, written in Python. A FORTRAN version of the model family is often used together with a macro-economic model as: [E3ME-FTT](https://www.e3me.com/). This model is managed by Cambridge Econometrics, and informs some of the inputs for the standalone model. In specific, energy demand is an output from the coupled model. 

## Installation
1. Run the install_ce_conda_3.9_external_users.cmd script in _Python_installation to install the prerequisite packages. On top of Anaconda's standard packages, bottle and paste are required. You can install these two packages with pip. Paste is being deprecated; If you cannot install paste, you can remove calls to paste in the Backend_FTT.py (this might prevent you from opening two instances at the same time). 

The model has been tested on Windows, using Python 3.10 and 3.11. 

## Running the model
1. You can run the front-end of the model in your browser by double clicking FTT_Stand_Alone_Launcher.cmd. Select the models to run and scenarios and explore the output. S0 is the baseline scenario. Output is available via the frontend, but also in the Output folder as a pickle file. Running the code should take a few minutes. 
   1. The first time you run the model, csv input files will be created. This takes a few additional minutes. 
2. Alternatively, you can run the model from the run_file.py script. Select the models and scenarios from the settings.ini file. 
3. Create new scenarios by adding a new folder in the Inputs folder. Data is read in first from this folder, and missing data is read from the S0 baseline folder.

## Generating relevant scenario files
The Analysis folder contains a set of csv files that are used to generate scenario files for the paper, using the Generate_scenario_files.py script. At the bottom of the script, you can change the scenario set to be used (for instance Policies_sector_by_policy.csv). Then run the models from the frontend, selecting all four models and all the scenarios that are analysed within a single graph.

## Generating graphs
The same Analysis folder contains scripts to generate the graphs for the paper. Save the data from model output by renaming (i.e. Results_sxp.pickle, for generation_graph_4x4.py). 

## References
* Knobloch, F., Pollitt H., Chewpreecha U., Daioglou V. and Mercure J-F. (2018) ‘[Simulating the deep decarbonisation of residential heating for limiting global warming to 1.5°C](https://link.springer.com/article/10.1007/s12053-018-9710-0)’, Energy Efficiency **12**, Issue 2, pp 521–550.
* Mercure (2012) FTT:Power : [A global model of the power sector with induced technological change and natural resource depletion](https://www.sciencedirect.com/science/article/pii/S0301421512005356 ). Energy Policy **48**.
* Mercure, J-F., Lam, A., Billington, S. and Pollitt, H. (2018) ‘[Integrated assessment modelling as a positive science: private passenger road transport policies to meet a climate target well below 2°C](https://pubmed.ncbi.nlm.nih.gov/30930506/)’, Climatic Change, November 2018, Volume 151, Issue 2, pp 109–129.

