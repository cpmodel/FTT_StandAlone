How to
=================

Create a new scenario
---------------------

Scenarios are made from alternative input data. A scenario only needs to contain
the files that differ from the baseline.

Scenario names are normally written as ``S1``, ``S2``, and so on. Use a name
that is unique within ``Inputs``.

Create a module subfolder for the inputs that will change. For example, a
Power scenario called ``S3`` should have:

.. code-block:: text

   Inputs/
   └── S3/
       └── FTT-P/

The available module folder names include ``FTT-P``, ``FTT-H``, ``FTT-Tr``
and ``FTT-Fr``. A scenario can also contain a ``General`` subfolder for
general inputs. Add the relevant subfolders when the scenario changes their
inputs. **Do not add input files directly inside ``Inputs/S3``.** The model
reads CSV files from the scenario subfolders.

Copy the CSVs that need to change from the corresponding ``S0`` subfolder,
then edit their values. Keep the file names, row labels, column labels,
dimensions, units and year columns consistent with the ``S0`` files.
Unchanged files do not need to be copied.

To run the scenario in the GUI:

1. Start the frontend using ``launch_frontend.bat``
2. Open the **RUN** page, select the scenario from **Scenario(s) to run**,
   and click **RUN**.

When running from the command line instead, add the scenario name to the
comma-separated ``scenarios`` setting in ``settings.ini``.

Publish the next model version
------------------------------

The package version is defined in ``pyproject.toml``:

.. code-block:: toml

   [project]
   version = "1.3.0"

Use a three-part version number, ``MAJOR.MINOR.PATCH``. The three components
have different meanings:

* **Major** (the first number) is for incompatible changes, such as breaking
  the public Python API, changing required input formats, or changing output
  structures in a way that requires users to update their code.
* **Minor** (the middle number, sometimes described as the medium version) is
  for new, backwards-compatible functionality, such as adding a new model
  capability or an optional input. Existing users should be able to continue
  using the previous interface.
* **Patch** (the final number) is for backwards-compatible bug fixes,
  documentation updates, and small corrections that do not add a new
  capability or change the public interface.

To publish a release:

1. Update the ``version`` under ``[project]`` in ``pyproject.toml``. For
   example, a backwards-compatible feature changes ``1.3.0`` to ``1.4.0``.
2. Commit and merge that change.
3. Create and **publish** a GitHub Release with the matching tag, e.g.
   ``v1.4.0``. In the repository's **Releases** page, click **Draft a new
   release**, enter the matching tag and release title, and click **Publish
   release**.

Publishing the GitHub Release automatically starts
``.github/workflows/pypi-publish.yml``. The workflow checks that the release
tag matches the version in ``pyproject.toml``, builds the package, and
publishes it to PyPI. Updating ``pyproject.toml`` by itself does not start the
Action or publish a new version.

Add a new variable
------------------

1. Add the variable to ``Utilities/titles/VariableListing.csv``. Specify its
   unit, domain and dimensions in order in ``Dim1`` through ``Dim4``. For
   example, a regional transport variable over time can use ``RTI``, ``VTTI``
   and ``TIME``. Leave unused dimension columns blank.

2. If a new classification is needed, add its labels to
   ``Utilities/titles/classification_titles.csv``. ``TIME`` is supplied by the
   simulation timeline rather than this file.

3. If the variable has input data, add a CSV named after the variable in the
   corresponding ``Inputs/S0`` domain folder. For example, an ``FTT-Tr``
   variable belongs in ``Inputs/S0/FTT-Tr``. Match its coordinates and labels
   to the declared dimensions; time-dependent inputs have coordinate columns
   followed by year columns, while scalar inputs use a ``value`` column.
   Set ``Is input variable`` to ``Y`` for a required input so the loader
   checks that it exists.

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
