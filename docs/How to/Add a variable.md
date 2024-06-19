Adding a variable is simple. There are three or four steps, depending on the type of variable:
1. Add the variable to Utilities/VariableListing.csv. Choose the right dimension, as described in classification_titles. Dim1 is only used for 3D variables, for the regions. Dim2 refers to the rows, and Dim3 refers to the columns. The summary and detailed columns are to indicate how the variable is displayed in the frontend.
2. If your variable uses a new dimension, you can add this to classification_titles. 
3. If your variable has historical data, you can add this data in csv format in the Inputs folder. Ensure that the dimensions match with those in the VariableListing file. 
4. When you've added your variable, run manager_new/update_manager_metadata.py. This ensures your variable is added to the json file which is needed for the front end manager. 