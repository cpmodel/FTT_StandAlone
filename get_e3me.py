import os
from copy import deepcopy
import pandas as pd
import numpy as np
from celib import DB1, MRE, Tabls

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

TITLES_ALT = {var: f'{var[0]}RTI' for var in ['QTI', 'YTI', 'KTI', 'CTI']}
TITLES_CONV = {var: 'CRC' if var == 'CTI' else 'QQ69' for var in TITLES_ALT}
DB_NAMES = {'QQC':  'QQ46',
            'QQ69': 'QQ64',

            # DB-vs-model variable name mapping below is accurate, but ultimately
            # useless for our purposes because Tabls doesn't store dimensions
            # of 3D variables effectively (max 2 dimensions listed in Tabls)
            # 'QCC':  'BQRC',
            # 'QKC':  'BQRK',
            # 'QGC':  'BQRG',
            # 'QY00': 'BQRY',
            # 'QY05': 'BQRY',
            # 'QY10': 'BQRY',
            # 'QY15': 'BQRY',
            }

#%%
# --------------------------------------------------------------------------- #
#  E3ME variable metadata
# --------------------------------------------------------------------------- #

def get_dim_titles(e3me_dir, dim_var):
    with DB1(os.path.join(e3me_dir, 'databank', 'U.db1')) as dbu:
        try:
            return dbu[dim_var]
        except KeyError:
            return None


def get_var_titles(e3me_dir, variables, regs_short=False, alt=True):
    if type(variables)==str:
        variables = [variables]

    dimensions = get_var_dimensions(e3me_dir, variables, regs_short, reduce=False)

    titles = {}
    for var in variables:
        for dim in dimensions[var]:
            if dim not in titles:
                titles[dim] = get_dim_titles(e3me_dir, dim)
                if dim == 'RTI':
                    titles['RSHORTTI'] = get_dim_titles(e3me_dir, 'RSHORTTI')
    if alt:
        # Get alternate titles
        for var in TITLES_ALT:
            if var in titles:
                var_alt = TITLES_ALT[var]
                titles[var_alt] = get_dim_titles(e3me_dir, var_alt)

    return titles


def get_var_dimensions(e3me_dir, variables, regs_short=False, reduce=True):
    tabls_fp = os.path.join(e3me_dir, 'Model', 'E3ME', 'Dats', 'Tabls.dat')

    if type(variables)==str:
        variables = [variables]

    is_var_ts = is_var_time_series(e3me_dir, variables, reduce=False)

    # Hardcoded exceptions - mostly 3D converter variables
    # TODO: Ugly!! Clean up if possible
    tabls_exceptions = {
                  'QCC':  ['RTI', 'YTI', 'CTI'],
                  'QKC':  ['RTI', 'YTI', 'YTI'],
                  'QGC':  ['RTI', 'YTI', 'GTI'],
                  'QY00':  ['RTI', 'YTI', 'YTI'],
                  'QY05':  ['RTI', 'YTI', 'YTI'],
                  'QY10':  ['RTI', 'YTI', 'YTI'],
                  'QY15':  ['RTI', 'YTI', 'YTI'],
                  'CRC':  ['CTI', 'CRTI'],
                  'TWIY': ['RTI', 'VTTI', 'Year'],
                  'TEWW': ['VTTI', 'Year'],
                  'HEWW': ['HTTI', 'Year'],
                  # 'RHSM': ['RTI', 'Year'],
                  # 'TETH': ['RTI', 'VYTI', 'Year'],
                  'BCET': ['RTI', 'T2TI', 'C2TI'],
                  'BTTC': ['RTI', 'VTTI', 'C3TI'],
                  'BHTC': ['RTI', 'HTTI', 'C4TI'],
                  'BSTC': ['RTI', 'STTI', 'C5TI'],
                  'G': ['RTI', "GTI", 'Year'],
                  'F1CC10': ['FUTI', 'RTI'],
                  'F1KC10': ['FUTI', 'RTI'],
                  'F1UC10': ['FUTI', 'RTI'],
                  'F1OC10': ['FUTI', 'RTI'],
                  'F1MC10': ['FUTI', 'RTI'],
                  'F1DC10': ['FUTI', 'RTI'],
                  'F1GC10': ['FUTI', 'RTI'],
                  'F1WC10': ['FUTI', 'RTI']
                  }
    dims = {}
    for var in variables:
        # Exceptions first - hardcode dimensions
        if var in tabls_exceptions:
            dims[var] = tabls_exceptions[var]

        else:
            var_ = DB_NAMES[var] if var in DB_NAMES else var
            dims[var] = list(Tabls(tabls_fp).get_var(var_)['dimnames'])

            # Add year dimension if exists
            if is_var_ts[var]:
                # Reverse order from Tabls module output
                # -> regions then sectors
                dims[var] = list(reversed(dims[var]))
                dims[var] = dims[var]+['Year']

            # 3D variables, non-time series: add region dimension
            elif dims[var][0] != 'RTI':
                with open(tabls_fp) as tabls:
                    for i, line in enumerate(tabls.readlines()):
                        # Location in Tabls roughly indicates whether it is a 3D variable
                        if 2634<=i<=3072 and line[:4].strip() == var:
                            dims[var] = ['RTI'] + list(reversed(dims[var]))
                            break

        dims[var] = tuple(dims[var])

    if regs_short:
        for var in variables:
            if 'RTI' in dims[var]:
                dims[var] = tuple('RSHORTTI' if dim == 'RTI' else dim for dim in dims[var])

    if reduce:
        dims = reduce_data_dict(dims)

    return dims


def is_var_time_series(e3me_dir, variables, reduce=True):
    if type(variables)==str:
        variables = [variables]

    ts = {var: False for var in variables}
    with open(os.path.join(e3me_dir, 'In', 'History.idiom')) as file:
        lines = file.readlines()
        for var in variables:
            var_alt = var[:3]+'A'
            get_cmd = [f'GET {var}',
                       f'GETA {var}',
                       f'GET {var_alt}',
                       f'GETA {var_alt}']
            for cmd in get_cmd:
                if not ts[var]:
                    for line in lines:
                        line = line.strip()
                        if line.startswith(cmd+' ') or line.startswith(cmd+'('):
                            ts[var] = line[-10:-6] == 'YEAR'
                            break


    with open(os.path.join(e3me_dir, 'In', 'EnForecast.idiom')) as file:
        lines = file.readlines()
        for var in variables:
            var_alt = var[:3]+'A'
            get_cmd = [f'GET {var}',
                       f'GETA {var}',
                       f'GET {var_alt}',
                       f'GETA {var_alt}']
            for cmd in get_cmd:
                if not ts[var]:
                    for line in lines:
                        line = line.strip()
                        if line.startswith(cmd+' ') or line.startswith(cmd+'('):
                            ts[var] = line[-10:-6] == 'YEAR'
                            break

    with open(os.path.join(e3me_dir, 'In', 'DAN1.idiom')) as file:
        lines = file.readlines()
        for var in variables:
            if not ts[var]:
                for line in lines:
                    if var in line:
                        ts[var] = True
                        break

    if reduce:
        ts = reduce_data_dict(ts)

    return ts


def get_var_description(e3me_dir, variables, reduce=True):
    if type(variables) == str:
        variables = [variables]
    tabls_fp = os.path.join(e3me_dir, 'Model', 'E3ME', 'Dats', 'Tabls.dat')
    desc = {var: Tabls(tabls_fp).get_var(var)['desc'] for var in variables}
    if reduce:
        desc = reduce_data_dict(desc)
    return desc

#%%
# --------------------------------------------------------------------------- #
#  Databank metadata
# --------------------------------------------------------------------------- #

def get_db_index(e3me_dir, db):
    # Allow 'db' reference to be single letter only
    if len(db) == 1:
        db += '.db1'
    with DB1(os.path.join(e3me_dir, 'databank', db)) as db1:
        return db1.index


def get_db_var_list(e3me_dir, db):
    db_index = get_db_index(e3me_dir, db)
    db_vars = list(sorted(set([var.split('_')[0] for var in db_index.name])))
    return db_vars


def is_var_on_db(e3me_dir, variables, db, reduce=True):
    if type(variables)==str:
        variables = [variables]
    db_var_list = get_db_var_list(e3me_dir, db)
    var_on_db = {var: var in db_var_list for var in variables}
    if reduce:
        var_on_db = reduce_data_dict(var_on_db)
    return var_on_db


def get_dbs_with_var(e3me_dir, variables, reduce=True):
    if type(variables)==str:
        variables = [variables]
    db_list = [fp for fp in os.listdir(os.path.join(e3me_dir, 'databank')) if fp[-4:]=='.db1']
    var_on_db = {var: [db for db in db_list if is_var_on_db(e3me_dir, var, db)] for var in variables}
    if reduce:
        var_on_db = reduce_data_dict(var_on_db)
    return var_on_db


def get_db_var_code(e3me_dir, db, variables, all_codes=False, region=None, reduce=True):
    if type(variables)==str:
        variables = [variables]

    db_index = get_db_index(e3me_dir, db)

    var_code = {}
    for var in variables:
        if region:
            codes = db_index[db_index.name == f'{var}_{region}'].index.tolist()
        else:
            codes = db_index[db_index.name.str.split('_').str[0] == var].index.tolist()

        if all_codes:
            if len(codes) > 0:
                var_code[var] = codes
            else:
                var_code[var] = None
        else:
            try:
                var_code[var] = min(codes)
            except ValueError:
                var_code[var] = None

    if reduce:
        var_code = reduce_data_dict(var_code)

    return var_code


def get_db_var_start_year(e3me_dir, db, variables, dimensions=None, reduce=True):
    if type(variables) in [int, str]:
        variables = [variables]

    if len(db) == 1:
        db += '.db1'
    index = get_db_index(e3me_dir, db)

    regions = get_dim_titles(e3me_dir, 'RSHORTTI')


    is_on_db = is_var_on_db(e3me_dir, variables, db, reduce=False)
    for var in variables:
        if not is_on_db[var]:
            raise ValueError(f'{var} not found on {db}!')

    if not dimensions:
        dimensions = get_var_dimensions(e3me_dir, variables, reduce=False)


    start_years = {}
    for var in variables:
        if dimensions[var][-1] == 'Year':
            if type(var)==str:
                var_3d = len(dimensions[var]) == 3 and dimensions[var][0] == 'RTI'
                var_db = f'{var}_{regions[0]}' if var_3d else var
                year = index.loc[index.name==var_db, 'start_year'].unique().tolist()[0]

                i = min(index.loc[index.name==var_db, 'start_year'].index.tolist())
                if var_3d:
                    i -= 100

            elif type(var)==int:
                year = index.loc[var, 'start_year']
                i = var

            else:
                raise TypeError('Variable must be a str or int.')

            # If no start year info logged in DB index, look in History/EnForecast file
            if year == 0:
                fn = 'EnForecast.idiom' if db in ['F.db1', 'C.db1'] else 'History.idiom'
                with open(os.path.join(e3me_dir, 'In', fn)) as file:
                    for line in file.readlines():
                        # Check for GET/GETA line for this variable, and extract year
                        # e.g. GET DSO3 91330023,C(YEAR-2004)
                        # Check for
                        check_line = var[:-1], str(i), 'YEAR'
                        if all(x in line for x in check_line):
                            line = line.split('|')[0]   # Remove inline comments
                            year = line.strip()[-5:-1]  # Get year from IDIOM
                            year = int(year) + 1        # Start year is following year


            # Exception for single-year time-series variables which are read into
            # EnForecast using different syntax
            ## TODO: Ugly!! Clean up if possible
            if db=='C.db1':
                if var in ['TEWW', 'TESH']:
                    year = 2012
                if var in ['HEWW']:
                    year = 2014

            start_years[var] = year if year > 0 else None

        else:
            start_years[var] = None

    if reduce:
        start_years = reduce_data_dict(start_years)

    return start_years


#%%
# --------------------------------------------------------------------------- #
#  Databank data
# --------------------------------------------------------------------------- #

def get_db_data(e3me_dir, db, variables,
                regions=None, sectors=None, years=None,
                is_e3me=True, regs_short=False, align_sectors=False,
                long_format=False, reduce=True, log=True):
    if len(db) == 1:
        db += '.db1'
    if type(variables) in [str, int]:
        variables = [variables]
    if type(regions) == str:
        regions = [regions]

    # Check if variable is available in databank
    is_on_db = is_var_on_db(e3me_dir, variables, db, reduce=False)
    for var in variables:
        if not is_on_db[var]:
            raise ValueError(f'{var} not found on {db}!')

    # Get metadata
    dimensions = get_var_dimensions(e3me_dir, variables, regs_short, reduce=False)
    titles = get_var_titles(e3me_dir, variables, regs_short)
    start_years = get_db_var_start_year(e3me_dir, db, variables, reduce=False)
    if align_sectors:
        sec_conv = {ti: get_db_data(e3me_dir, 'X', conv, log=False) for ti, conv in TITLES_CONV.items() if ti in titles}

    # Get data
    db_fp = os.path.join(e3me_dir, 'databank', db)
    if log:
        print(f'\nReading DB1 data from {db_fp}')

    with DB1(db_fp) as db1:
        data = {}
        for var in variables:
            if log:
                print(' ', var)

            tt = [titles[t] for t in dimensions[var]]
            start_year = start_years[var]

            # 1D variable -> add 2nd dim of length 1
            if len(dimensions[var]) == 1:
                dimensions[var] = (dimensions[var][0], '0')
                tt.append(['0'])

            # 2D variable
            if len(dimensions[var]) == 2:
                df = pd.DataFrame(db1[var], index=tt[0])

                # 2D variable with no year dimension (usually a converter)
                if tt[1]:
                    if var == 'CRC':
                        # CRC converter has missing category
                        # -> Replicate methodology from frontend "alt class" button
                        df.insert(19, 'buses', 0)
                        df.iloc[24, 19] = 0.5
                        df.iloc[24, 21] = 0.5
                        df.iloc[-1, -1] = 1

                    df.columns = tt[1]

                    if var == 'QQ69':
                        tt_alt = titles[TITLES_ALT[dimensions[var][1]]]
                        df = df.iloc[:, :len(tt_alt)]
                        df.columns = tt_alt

                # 2D variable with year dimension
                else:
                    df.columns = range(start_year, start_year+df.shape[1])
                    if years:
                        df = df[years]

                if regions:
                    if dimensions[var][0] == 'RTI':
                        df = df.loc[regions]
                    elif dimensions[var][1] == 'RTI':   # It does happen! e.g. HJFC
                        df = df[regions]

                data[var] = df.copy()

            elif len(dimensions[var]) == 3:
                data[var] = {}
                get_regs = regions if regions else tt[0]
                for r, reg in enumerate(tt[0]):
                    if reg in get_regs:     # Region filter
                        reg_short = titles['RSHORTTI'][r]
                        df = pd.DataFrame(db1['{}_{}'.format(var, reg_short)], index=tt[1])

                        # 3D variables without/with year dimension
                        if tt[2]:
                            df.columns = tt[2]
                        else:
                            df.columns = range(start_year, start_year+df.shape[1])
                            if years:
                                df = df[years]


                        # Sector adjustments only required in E3ME, not E3-US etc, hence need for "is_e3me" parameter
                        if is_e3me:
                            d1, d2 = dimensions[var][1], dimensions[var][2]

                            # Remove empty rows in non-European regions
                            if r>=33:
                                if d1 in TITLES_ALT:
                                    tt_alt = titles[TITLES_ALT[d1]]
                                    df = df.iloc[:len(tt_alt)]
                                    df.index = tt_alt
                                if d2 in TITLES_ALT:
                                    tt_alt = titles[TITLES_ALT[d2]]
                                    df = df.iloc[:, :len(tt_alt)]
                                    df.columns = tt_alt

                            # Align sectors (a.k.a. "alt class" button)
                            ## Only required in E3ME, not E3-US etc, hence need for "is_e3me" parameter
                            if align_sectors and r<33:
                                # Align index
                                if d1 in sec_conv:
                                    df = sec_conv[d1].T.dot(df)
                                # Align columns
                                if d2 in sec_conv:
                                    df = df.dot(sec_conv[d2])

                        # Filter sectors
                        if sectors:
                            df = df.loc[sectors]

                        data[var][reg] = df.copy()

    # Convert to long format
    # TODO: Complicated approach, relies on recursive function -> clean up?
    if long_format:
        data = data_dict_to_long(data, dimensions)
        data = data.rename(columns={'index':'Variable'})

    # Remove redundant dimensions in nested dictionary
    elif reduce:
        data = reduce_data_dict(data)

    return data


#%%
# --------------------------------------------------------------------------- #
#  MRE results
# --------------------------------------------------------------------------- #

def get_mre_data(e3me_dir, scenarios, variables, start_year=2010,
                 regions=None, sectors=None, years=None,
                 is_e3me=True, align_sectors=False,
                 regs_short=False, long_format=False, reduce=True,
                 diffs=None, ba=None):
    if type(scenarios) == str:
        if scenarios.endswith('.mre'):
            scenarios = {'scen': scenarios}
    if type(variables) == tuple:
        variables = list(variables)
    if type(variables) == str:
        variables = [variables]
    if type(regions) == str:
        regions = [regions]
    if type(sectors) == str:
        sectors = [sectors]


    # Get metadata
    dimensions = get_var_dimensions(e3me_dir, variables, regs_short, reduce=False)
    titles = get_var_titles(e3me_dir, variables, regs_short)
    if align_sectors:
        sec_conv = {ti: get_db_data(e3me_dir, 'X', conv, log=False) for ti, conv in TITLES_CONV.items() if ti in titles}
    rti = 'RSHORTTI' if regs_short else 'RTI'

    vars_2d = [var for var in variables if len(dimensions[var])==2]
    vars_3d = [var for var in variables if len(dimensions[var])==3]

    # Get data
    output_dir = os.path.join(e3me_dir, 'Output')
    print(f'\nReading MRE data from {output_dir}')
    data = {}
    for scen, scenfile in scenarios.items():
        print(' ', scen)
        with MRE(os.path.join(output_dir, scenfile)) as mre:
            df = mre.get_full_vars(variables) # !!!: Super fast! ~10x faster than previous best practice
            df.index = range(start_year, start_year+df.shape[0])
            df = df.T.stack().reset_index()
            data[scen] = df.copy()
    data = pd.concat(data).droplevel(1).reset_index()

    # Sort out region vs. sector dimension
    ## Different dimension alignment in MRE output for 2D vs 3D variables
    ## There are one or two exceptions (e.g. ERRY), but this pattern is true in 99% of cases
    data.columns = 'Scenario', 'Variable', 'D1', 'D2', 'Year', 'Value'
    data.loc[data.Variable.isin(vars_2d), 'reg_index'] = data.loc[data.Variable.isin(vars_2d), 'D2']
    data.loc[data.Variable.isin(vars_2d), 'sec_index'] = '-'
    data.loc[data.Variable.isin(vars_3d), 'reg_index'] = data.loc[data.Variable.isin(vars_3d), 'D1']
    data.loc[data.Variable.isin(vars_3d), 'sec_index'] = data.loc[data.Variable.isin(vars_3d), 'D2']
    data = data[['Scenario', 'Variable', 'reg_index', 'sec_index', 'Year', 'Value']]

    # Fill region names
    data['Region'] = data['reg_index'].replace(dict(enumerate(titles[rti])))

    # Fill sector names
    ## Get sector dimension
    sec_dims = list(set([dimensions[var][1] for var in vars_3d]))
    for var in vars_3d:
        dim = dimensions[var][1]
        data.loc[data.Variable==var, 'sec_dim'] = dim
    ## E3ME alt titles: use for non-European regions
    for dim in sec_dims:
        if is_e3me and dim in TITLES_ALT:
            alt_dim = TITLES_ALT[dim]
            sec_dims.append(alt_dim)
            data.loc[(data.sec_dim==dim) & (data.Region.isin(titles[rti][33:])), 'sec_dim'] = alt_dim

    ## Get sector titles
    sec_titles = pd.concat({sec: pd.Series(titles[sec]) for sec in sec_dims}).reset_index()
    sec_titles.columns = 'sec_dim', 'sec_index', 'Sector'

    data = data.merge(sec_titles, on=['sec_dim', 'sec_index'], how='left')
    data.loc[data.sec_index=='-', 'Sector'] = '-'
    ## Drop empty sectors (e.g. 45-70 for YRTI)
    data = data.dropna(subset='Sector')

    ## Optional: align sectors for all regions to alt class
    if is_e3me and align_sectors and dim in TITLES_ALT:
        alt_conv = pd.concat(sec_conv).stack().reset_index()
        alt_conv.columns = 'sec_dim', 'Sector', 'alt', 'conv_val'
        alt_conv = alt_conv[alt_conv.conv_val>0]

        data = data.merge(alt_conv, on=['sec_dim', 'Sector'], how='left')
        data['Value'] *= data['conv_val'].fillna(1)     # Account for converters with values between 0 and 1
        data['Sector'] = data['alt'].fillna(data['Sector'])
        data = data.drop(columns=['alt', 'conv_val'])

    # Tidy columns
    cols = ['Scenario', 'Variable', 'Region', 'Sector', 'Year', 'Value']
    data = data[cols].groupby(cols[:-1], sort=False).sum().reset_index()

    # Apply filters
    if regions:
        data = data[data.Region.isin(regions)]
    if years:
        data = data[data.Year.isin(years)]
    if sectors:
        data = data[data.Sector.isin(sectors)]


    # Calculate differences from baseline
    if diffs in ['abs', 'pct']:
        if not ba:
            raise ValueError('If requesting absolute or percentage differences, '
                             'please indicate name of baseline scenario')
        data = mre_diffs(data, diffs, ba)

    # Convert to dict format
    if not long_format:
        print('  Converting to dictionary format')
        df = data.copy()
        df = df.pivot_table(index=['Scenario', 'Variable', 'Region', 'Sector'], columns='Year', values='Value', aggfunc='sum', sort=False)

        data = {}
        for scen in scenarios:
            data[scen] = {}
            for var in variables:
                if var in vars_2d:
                    data[scen][var] = df.xs([scen, var]).droplevel('Sector').copy()

                elif var in vars_3d:
                    data[scen][var] = {}
                    get_regs = regions if regions else titles[rti]
                    for reg in get_regs:
                        data[scen][var][reg] = df.xs([scen, var, reg]).copy()

        # Remove redundant dimensions in nested dictionary
        if reduce:
            data = reduce_data_dict(data)

    return data


def mre_diffs(raw_data, transform, ba, drop_ba=False):
    if isinstance(raw_data, pd.DataFrame):
        if 'Scenario' not in raw_data.columns:
            raise ValueError('Baseline scenario not found in DataFrame columns')

        diffs = raw_data.copy()
        cols = diffs.columns
        other_dims = [col for col in cols if col not in ['Scenario', 'Value']]
        diffs = diffs.pivot_table(index=other_dims, columns='Scenario', values='Value', aggfunc='sum')
        if transform == 'abs':
            diffs = diffs.sub(diffs[ba], axis=0)
        elif transform == 'pct':
            diffs = diffs.div(diffs[ba], axis=0) - 1
        else:
            raise ValueError("Please enter 'abs' or 'pct' format")

        diffs = diffs.stack().reset_index().rename(columns={0:'Value'})
        diffs = diffs[cols]

        if drop_ba:
            diffs = diffs[diffs.Scenario!=ba]


    elif type(raw_data) == dict:
        if ba not in raw_data:
            raise ValueError('Baseline scenario not found in data dictionary')

        diffs = {}
        for scen in raw_data:
            diffs[scen] = {}
            for var in raw_data[scen]:
                var_data_ba = raw_data[ba][var].copy()
                var_data_scen = raw_data[scen][var].copy()

                if isinstance(var_data_scen, pd.DataFrame) or isinstance(var_data_scen, pd.Series):
                    if transform == 'abs':
                        diffs[scen][var] = var_data_scen.copy() - var_data_ba.copy()
                    elif transform == 'pct':
                        diffs[scen][var] = var_data_scen.copy() / var_data_ba.copy() - 1
                        diffs[scen][var] = diffs[scen][var].fillna(0)
                    else:
                        raise ValueError("Please enter 'abs' or 'pct' format")

                else:
                    diffs[scen][var] = {}
                    for reg in var_data_scen:
                        if transform == 'abs':
                            diffs[scen][var][reg] = var_data_scen[reg].copy() - var_data_ba[reg].copy()
                        elif transform == 'pct':
                            diffs[scen][var][reg] = var_data_scen[reg].copy() / var_data_ba[reg].copy() - 1
                            diffs[scen][var][reg] = diffs[scen][var][reg].fillna(0)
                        else:
                            raise ValueError("Please enter 'abs' or 'pct' format")

        if drop_ba:
            diffs = {scen: diffs[scen] for scen in diffs if scen != ba}

    return diffs


#%%
# --------------------------------------------------------------------------- #
#  Helper functions
# --------------------------------------------------------------------------- #

def reduce_data_dict(data):
    """
    reduce_data_dict
    ================
    Recursive function to collapse nested dictionary levels with only one key.

    For example: {'Key1': {'Key2': {'Key3A': 'ValA',
                                    'Key3B': 'ValB'}}}
    Becomes: {'Key3A': 'ValA',
              'Key3B': 'ValB'}

    Inputs:
        - Nested dictionary of DataFrames
    Outputs:
        - Series/DataFrame or nested dictionary of Series/DataFrame

    """
    for key, val in data.items():
        if type(val) == dict:
            data[key] = reduce_data_dict(val)

        if len(data)==1:
            data = list(data.values())[0]
    return data


def data_dict_to_long(data, dims):
    data = deepcopy(data)
    for key, val in data.items():
        if type(val) == dict:
            val = data_dict_to_long(val, dims[key])

            if type(dims[key]) == tuple:
                val = val.rename(columns={'index': dims[key][0]})

            data[key] = val.copy()

        elif type(val) == pd.DataFrame:
            val = val.stack().reset_index()
            dims_ = dims[key] if type(dims) == dict else dims
            val.columns = list(dims_[-2:]) + ['Value']
            data[key] = val.copy()

        # elif type(val) == pd.Series:
        #     val.name = 'Value'
        #     data[key] = val.copy().reset_index()

    data = pd.concat(data)
    data.index = data.index.droplevel(1)
    data = data.reset_index()

    # Reorder columns
    if 'Year' in data.columns:
        data_ = data.drop('Year', axis=1)
        data = pd.concat([data_, data['Year']], axis=1)

    data_ = data.drop('Value', axis=1)
    data = pd.concat([data_, data['Value']], axis=1)


    return data


