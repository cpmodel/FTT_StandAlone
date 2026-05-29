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


Review pull requests
--------------------

Pull requests (PRs) are used to review changes before they are merged into the main codebase. The goal is to ensure changes are scientifically correct, understandable, and do not break model coupling.

Before starting
^^^^^^^^^^^^^^^

- Read the PR description
- Check which files were changed
- Confirm the purpose of the change
- Ask questions if anything is unclear

Reviewing changes
^^^^^^^^^^^^^^^

Review model output:

- Run the code locally
- Check that it executes without errors
- Compare outputs with the previous version
- Check that scientific behaviour is sensible
- Look for unintended side effects

Review code quality:

- Clear and consistent variable names
- Readable structure
- Appropriate documentation
- Avoiding unnecessary complexity (in particular, if code is partially LLM-written)

Coupled model variables
^^^^^^^^^^^^^^^

Some variables are coupled with MSET, which (will) import this repo as a package. Changes to these can break compatibility, including for investment and fuel use.

Check carefully for changes to:

- Variable names
- Dimensions or indexing
- Units and definitions

Review comments
^^^^^^^^^^^^^^^

- Leave comments directly on relevant lines
- Be specific with feedback, ensure that issues can be reproduced
- PR authors should resolve comments

Approving and merging
^^^^^^^^^^^^^^^

Approve the PR when:

- The code runs correctly
- The scientific changes are justified
- Review comments are resolved
- The changes are safe to merge

Merging:

- Small PRs may be merged directly
- Merge conflicts should be resolved by the author
- Delete the branch unless it is still needed
