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

2. **Add new dimensions (if needed).**  
   If your variable uses a new dimension, add it to `classification_titles`.

3. **Add historical data (if any).**  
   Historical data can be added by creating a CSV file in the correct model folder in `Inputs/S0`. For example, if the 'domain' field in `VariableListing.csv` is 'FTT-Tr', the file should be placed in `Inputs/S0/FTT-Tr`. 
   The file name should match the variable name and the dimensions should match those in `VariableListing.csv`.
   If your new variable is critical to the model, you may want to specify 'Y' under the 'Is input variable' column in `VariableListing.csv`. This will ensure that the model will not run if the variable is missing.

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
- Be specific with feedback and ensure that issues can be reproduced
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
