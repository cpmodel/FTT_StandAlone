# Contents of celib/io/tests/data

## Inputs

| File             | Description                           | Source                                                                                                                                                                                        |
|------------------+---------------------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| lms.csv          | Example ONS CSV-format data file      | Subset of 12 July 2017 ONS labour market statistics release: https://www.ons.gov.uk/employmentandlabourmarket/peopleinwork/employmentandemployeetypes/datasets/labourmarketstatistics/current |
| tabls_e3me.dat   | E3ME tabls file to test Tabls class   | svn://svnserver/var/subversion/E3ME/Master/Model/E3ME/Dats/Tabls.dat, r1406                                                                                                                   |
| tabls_mdm-e3.dat | MDM-E3 tabls file to test Tabls class | svn://svnserver/var/subversion/MDM/Branch/C162/REG/Model/MDM/Dats/tabls.dat, r12888                                                                                                           |

## Expected outputs

| File                 | Description                                                                                 | Source                                      |
|----------------------+---------------------------------------------------------------------------------------------+---------------------------------------------|
| lms_out_a.csv        | Annual data from lms.csv (see above) as would be returned by `load_ons_db()`                | Extracted from lms.csv (from above)         |
| lms_out_q.csv        | Quarterly data from lms.csv (see above) as would be returned by `load_ons_db()`             | Extracted from lms.csv (from above)         |
| lms_out_m.csv        | Monthly data from lms.csv (see above) as would be returned by `load_ons_db()`               | Extracted from lms.csv (from above)         |
| e3me_titles.csv      | E3ME titles entries as structured CSV file (excludes 'titles' identified as 'dummy scalar') | Exported from tabls_e3me.dat (from above)   |
| e3me_variables.csv   | Subset of E3ME variables entries as structured CSV file (first 5 entries)                   | Exported from tabls_e3me.dat (from above)   |
| mdm-e3_titles.csv    | Full set of MDM-E3 titles entries as structured CSV file                                    | Exported from tabls_mdm-e3.dat (from above) |
| mdm-e3_variables.csv | Subset of MDM-E3 variables entries as structured CSV file (first 5 entries)                 | Exported from tabls_mdm-e3.dat (from above) |
