# FTT StandAlone

## Future Technology Transformation
This repository contains a family of Future Technology Transformation (FTT) models. Models that are included are:

* FTT:Power (Mercure, 2012) - data up to 2018, update to 2021 expected in June
* FTT:Heat (Knobloch et al, 2017) data up to 2020
* FTT:Industrial heat *under construction*
* FTT:Transport (Mercure et al, 2018) - data up to 2022
* FTT:Freight (*under review*) - data up to 2023
* FTT:Hydrogen (*under construction*)

## Theoretical background
The FTT family of models are based on [evolutionary economics](https://en.wikipedia.org/wiki/Evolutionary_economics). The uptake of new technologies typically follows an S-curve, which can be represented well with evolutionary dynamics (Mercure et al, 2012). The core equations for all of the models in the model family are coupled logistic equations of the [Lotka-Volterra family](https://en.wikipedia.org/wiki/Lotka%E2%80%93Volterra_equations), also known as the predator-prey equations. These equations are used to determine the evolution of the shares of various technologies in the models. Each model contains between ~10 to 25 technologies competing for market share. 

## FTT and E3ME
This repository contains the public standalone version of FTT, written in Python. A FORTRAN version of the model family is often used together with a macro-economic model as: [E3ME-FTT](https://www.e3me.com/). This model is managed by Cambridge Econometrics, and informs some of the inputs for the standalone model. In specific, energy demand is an output from the coupled model. 

## Installation
1. Run the install_ce_conda_3.9_external_users.cmd script in _Python_installation to install the prerequisite packages. On top of Anaconda's standard packages, bottle and paste are required. You can install these two packages with pip. Paste is being deprecated. If you cannot install paste, you can remove calls to paste in the Backend_FTT.py.

## Running the model
1. You can run the front-end of the model in your browser by double clicking FTT_Stand_Alone_Launcher.cmd. Select the models to run and scenarios and explore the output.
   1. The first time you run the model, csv input files will be created. This takes a few additional minutes. 
2. Alternatively, you can run the model from the run_file.py script. Output is saved to a pickle file in the Output folder. Select the models and scenarios from the settings.ini file.
3. Create new scenarios by adding a new folder in the Inputs folder. Data is read in first from this folder, and missing data is read from the S0 baseline folder. 

## References
* Heat: Knobloch, F., Pollitt H., Chewpreecha U., Daioglou V. and Mercure J-F. (2018) ‘[Simulating the deep decarbonisation of residential heating for limiting global warming to 1.5°C](https://link.springer.com/article/10.1007/s12053-018-9710-0)’, Energy Efficiency **12**, Issue 2, pp 521–550.
* Power: Mercure (2012): [A global model of the power sector with induced technological change and natural resource depletion](https://www.sciencedirect.com/science/article/pii/S0301421512005356 ). Energy Policy **48**.
* Power: Nijsse et al. (2023): [The momentum of the solar energy transition ](https://doi.org/10.1038/s41467-023-41971-7). Nature Communications **14**
* Passenger transport: Mercure, J-F., Lam, A., Billington, S. and Pollitt, H. (2018) ‘[Integrated assessment modelling as a positive science: private passenger road transport policies to meet a climate target well below 2°C](https://pubmed.ncbi.nlm.nih.gov/30930506/)’, Climatic Change, November 2018, Volume 151, Issue 2, pp 109–129.
* Steel: Vercoulen, P.; Lee, S.; Han, X.; Zhang, W.; Cho, Y.; Pang, J. (2023) '[Carbon-Neutral Steel Production and Its Impact on the Economies of China, Japan, and Korea: A Simulation with E3ME-FTT:Steel](https://www.mdpi.com/1996-1073/16/11/4498). Energies **16**, 4498. https://doi.org/10.3390/en16114498 
