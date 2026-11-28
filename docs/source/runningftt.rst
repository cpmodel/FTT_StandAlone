Running FTT
==========================

Running FTT from the command line
#############################################

Run ``python run_file.py`` from the project folder. Configure the run in
``settings.ini`` in the same folder:

.. image:: settings.jpg
    :width: 400px
    :align: center
    :alt: settings.ini


The different modules in FTT can be enabled in settings.ini, and it is possible to enable 
multiple different modules at the same time by using commas to separate, e.g. 
  
  enable_modules = FTT-P, FTT-Fr

S0 is FTT's baseline scenario, which incorporates current policies and projections 
into the model's simulation. Scenarios which differ from the baseline due to changes in policy inputs
can be run alongside the baseline. See :doc:`howto` for creating scenarios.

It is recommended that **model_start** and **simulation_start** are left unchanged, as the model will be 
unable to run if there is a gap between the start year of the simulation and the last year of historical data. 
The simulation can be shortened by changing **simulation_end**. If extending
it, check that the required input data covers the chosen years.

Errors and warnings are shown in the command window.

.. image:: command.jpg
    :width: 600px
    :align: center
    :alt: frontend


.. image:: command2.jpg
    :width: 600px
    :align: center
    :alt: frontend

The results are saved to ``Output/Results.pickle``. They can be loaded in Python
or viewed in the frontend.

Running FTT from the front end
##########################################

The frontend can be launched with ``launch_frontend.bat`` on Windows, or by
running ``python run_frontend.py`` from the command line. It can visualise the
simulation results.

.. image:: frontend.jpg
    :width: 600px
    :align: center
    :alt: frontend

To run the model, go to the **RUN** page and select the desired settings:

To run a standard baseline scenario for any module, select scenario S0 and the 
end year to 2050. Click **RUN**. The status of the run will display in the panel on the right. Click the
**RESULTS** page to view the model results. This page is split in two, with the upper part showing the 
results plot and the lower part containing filters for the data:

.. image:: frontend_results.jpg
    :width: 600px
    :align: center
    :alt: frontend

To view results, first select which results file to read from the **Available files** select box.
Each run of FTT will create a new pickle file which can contain results for multiple scenarios and models. If your results
file contains multiple scenarios, you can select which scenario(s) you wish to view on the right hand side, along with which
scenario is your baseline. Next, navigate to the **ANALYSIS** tab within the **RESULTS** page:

.. image:: frontend_analysis.jpg
    :width: 600px
    :align: center
    :alt: frontend
 
Select which variable to display in the plot using the **Select variable** select box. Variables can be searched for by using
either their code or their description. Next, pick the values of the variable's dimensions. This is frequently **Region** and **Technology**.
If you have selected more than one scenario and a specified a baseline, you can also view the absolute or relative difference of the variable
from the baseline by using the **Compare with baseline** select box. The **DOWNLOAD DATA** button will export the data currently displayed
in the plot to a CSV file.

The start and end year of the plot as well as whether dark mode is enabled can be changed in the **SETTINGS** page. 
