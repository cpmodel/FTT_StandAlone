# FTT StandAlone

## Future Technology Transformation
This repository contains a family of Future Technology Transformation (FTT) models. Models that are included are:

* FTT:Power (Mercure, 2012) - updated to 2022 (generation) and 2023 (prices)
* FTT:Heat (Knobloch et al, 2017) data up to 2020
* FTT:Industrial heat
* FTT:Transport (Mercure et al, 2018) - data up to 2022
* FTT:Freight (*under review*) - data up to 2023
* FTT:Hydrogen (*under review*)

## Theoretical background
The FTT family of models are based on [evolutionary economics](https://en.wikipedia.org/wiki/Evolutionary_economics). The uptake of new technologies typically follows an S-curve, which can be represented well with evolutionary dynamics (Mercure et al, 2012). The core equations for all of the models in the model family are coupled logistic equations of the [Lotka-Volterra family](https://en.wikipedia.org/wiki/Lotka%E2%80%93Volterra_equations), also known as the predator-prey equations. These equations are used to determine the evolution of the shares of various technologies in the models. Each model contains between ~10 to 25 technologies competing for market share. 

## FTT and macro-economic models
This repository contains the main version of FTT, written in Python. A FORTRAN version of the model family is often used together with a macro-economic model as: [E3ME-FTT](https://www.e3me.com/). This model is managed by Cambridge Econometrics, and informs some of the inputs for the standalone model. In specific, energy demand is an output from the coupled model. 

## Installation

Before you start, make sure that git is installed on your system, for instance by [installing GitHub Desktop](https://docs.github.com/en/desktop/installing-and-authenticating-to-github-desktop/installing-github-desktop)

1. Open your terminal at a location where you want to install ftt. Type the following in your terminal to download the package from GitHub:

   ```bash
   git clone https://github.com/cpmodel/FTT_StandAlone.git
   ```
2. The python package requirements are curated in the `environment.yml` file.
   Change directory to the repo, and then install the environment using:

   ```bash
   conda env create -f environment.yml
   ```
3. On Windows, you can start the frontend with FTT_Stand_Alone_Launcher.cmd. If Python is not yet added to your path, [ensure you add this first](https://realpython.com/add-python-to-path/).

Alternatively, you can download ftt by clicking the green `Code` button in the top right, and selecting `Open with Github Desktop` if you have this installed. You can import the environment in Anaconda Navigator.

## Running the model
1. You can run the front-end of the model in your browser by double clicking FTT_Stand_Alone_Launcher.cmd. Select the models to run and scenarios and explore the output.
   1. The first time you run the model, csv input files will be created. This takes a few additional minutes. 
2. Alternatively, you can run the model from the run_file.py script. Output is saved to a pickle file in the Output folder. Select the models and scenarios from the settings.ini file.
3. Create new scenarios by adding a new folder in the Inputs folder. Data is read in first from this folder, and missing data is read from the S0 baseline folder.

## How to contribute
We welcome contributions from everyone. You can report issues, fix bugs, improve the documentation, or write and propose model changes and provide updated data. 
1. New contributors can open pull requests with suggested code improvements by first forking the repository
2. Join our [Open community meetings](https://teams.microsoft.com/dl/launcher/launcher.html?url=%2F_%23%2Fl%2Fmeetup-join%2F19%3Ameeting_NTA1YmM0MGUtN2JmMS00ZjQ3LWFiM2UtNDkzNTM3OTFhMjNh%40thread.v2%2F0%3Fcontext%3D%257b%2522Tid%2522%253a%2522912a5d77-fb98-4eee-af32-1334d8f04a53%2522%252c%2522Oid%2522%253a%25222273eeaa-a79f-4eff-a90d-3083812f1175%2522%257d%26anon%3Dtrue&type=meetup-join&deeplinkId=68922fd8-cc1e-4549-83cd-af2c34badff2&directDl=true&msLaunch=true&enableMobilePage=true&suppressPrompt=true), typically on the last Friday of the month.
3. When you have questions, ask them on Github, so other people can benefit from the answers. Bugs and feature requests should be raised in [GitHub Issues](https://github.com/cpmodel/FTT_StandAlone/issues). Questions should be posted at the GitHub Discussions tab.
4. Whether you open a PR or ask questions, ensure that you're using the latest version of the code. Rebase your branch before you open a PR.

### Collaborations and publications
If you plan to publish work using this codebase, please let us know. Where capacity allows, we are happy to review results or confirm that analyses are consistent with the implementation.

We encourage a community-driven approach. If you need more detailed support, we welcome contributions back to the project through improvements to code, training material or data, to help strengthen the work for everyone.

## References
* Heat: Knobloch, F., Pollitt H., Chewpreecha U., Daioglou V. and Mercure J-F. (2018) ‘[Simulating the deep decarbonisation of residential heating for limiting global warming to 1.5°C](https://link.springer.com/article/10.1007/s12053-018-9710-0)’, Energy Efficiency **12**, Issue 2, pp 521–550.
* Power: Mercure (2012): [A global model of the power sector with induced technological change and natural resource depletion](https://www.sciencedirect.com/science/article/pii/S0301421512005356 ). Energy Policy **48**.
* Power: Nijsse et al. (2023): [The momentum of the solar energy transition ](https://doi.org/10.1038/s41467-023-41971-7). Nature Communications **14**
* Passenger transport: Mercure, J-F., Lam, A., Billington, S. and Pollitt, H. (2018) ‘[Integrated assessment modelling as a positive science: private passenger road transport policies to meet a climate target well below 2°C](https://pubmed.ncbi.nlm.nih.gov/30930506/)’, Climatic Change, November 2018, Volume 151, Issue 2, pp 109–129.
* Steel: Vercoulen, P.; Lee, S.; Han, X.; Zhang, W.; Cho, Y.; Pang, J. (2023) '[Carbon-Neutral Steel Production and Its Impact on the Economies of China, Japan, and Korea: A Simulation with E3ME-FTT:Steel](https://www.mdpi.com/1996-1073/16/11/4498). Energies **16**, 4498. https://doi.org/10.3390/en16114498 
