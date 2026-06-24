How to
=================

Add a new variable
------------------

Adding a variable is not too difficult. The steps depend on whether the variable uses a new classification and whether it has historical data.

1. **Add the variable to `Utilities/VariableListing.csv`.**  
   Choose the right dimension, as described in `classification_titles`:  

   - `Dim1` is only used for 3D variables, for the regions.  
   - `Dim2` refers to the rows.  
   - `Dim3` refers to the columns.  
   - The `summary` and `detailed` columns indicate how the variable is displayed in the frontend.

2. **Add new dimensions (if needed).**  
   If your variable uses a new dimension, add it to `classification_titles`.

3. **Add historical data (if any).**  
   You can add historical data in two ways:  

   1. Add the data in CSV format in the `Inputs` folder. Ensure that the dimensions match those in `VariableListing.csv`.  
   2. Alternatively, add your variable as a sheet to the Masterfile spreadsheet, and include it in `Inputs/_MasterFiles/FTT_Variables.xlsx`.  
    
      - Place it in the tab related to the model where the variable belongs.  
      - If the variable has historical data, also add it in the `"Time_Horizons"` tab.  
    
        - The format is `"tl_" + year` when the series starts.  
        - Ensure dimensions match `VariableListing.csv`.  
        - Ensure there is enough information in the Masterfile for all years of the horizon (usually until 2100) to prevent errors.

4. **Update the manager metadata.**  
   Run::

      python manager_new/update_manager_metadata.py

   This ensures your variable is added to the JSON file required by the frontend manager.
