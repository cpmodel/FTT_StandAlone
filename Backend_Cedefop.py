# -*- coding: utf-8 -*-
"""
backend.py
CEDEFOP Simulation tool Backend
Code to handle backend requests for data. Using bottle for server foundations
"""

# Libraries

from bottle import (route, run, request, response, static_file)
import socket
import glob
import os
import json
import time
import datetime
import csv
import sys
import pandas as pd
import numpy as np
import shutil

import configparser
from threading import Timer, Thread
import tkinter as tk
from tkinter import messagebox
import psutil
import pickle
from collections import OrderedDict
import Codes.Simulation_tool as SimulationTool

# Switch for build
PRODUCTION = True if len(sys.argv) == 1 else False
terminate_run = False

# File paths
try:
    rundir = sys._MEIPASS
    rootdir = os.path.abspath(sys._MEIPASS)
except AttributeError:
    rootdir = os.path.abspath(os.getcwd())
    rundir = rootdir

KILL_RUN = False

# cache for runs
run_entries_cache = {}

def console_message(error, message, elapsed):
    return str({'error': error, 'message': message,\
         'elapsed_time': elapsed, 'timestamp': str(datetime.datetime.now())})

# the decorator to allow calling the API endpoint from the outside
def enable_cors(fn):
    """Allows for data to be passed between the backend and frontend in a local application"""

    if PRODUCTION:
        return fn
    else:
        def _enable_cors(*args, **kwargs):
            # set CORS headers
            print("CORS set")
            response.headers['Access-Control-Allow-Origin'] = '*'
            response.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, OPTIONS'
            response.headers['Access-Control-Allow-Headers'] = 'Origin, Accept, Content-Type, X-Requested-With, X-CSRF-Token'

            if request.method != 'OPTIONS':
                # actual request; reply with the actual response
                return fn(*args, **kwargs)

        return _enable_cors

p = dict()

@route('/api/run/initialize/', method=['OPTIONS','POST'])
@enable_cors
def init_model():
    """Loads inputs for running the model"""
    body = request.body.read()
    p=json.loads(body.decode("utf-8"))

    global run_entries_cache
    run_entries_cache = p

    return {'status':'success'}

# API endpoint for running the model
#
#   WARNING: this call return an EVENT-STREAM and therefore needs to be handled as such
#   Calling with this function with the proper paramters results in running the model
#   and giving real-time feedback on its progress through server-sent events
#
@route('/api/run/start/', method=['GET'])
@enable_cors
def run_model():
    """Call batch script for generating the baseline. Run each step call
    induividually to allow for piping progress to frontend"""
    response.content_type = 'text/event-stream; charset=UTF-8'



    yield("event: status_change\n")
    yield("data: initialising\n\n")
    error = False
    yield("event: processing\n")
    # WARNING: hardcoded!
    yield("data: message:Processing started...;\n\n")
    global KILL_RUN
    KILL_RUN = False

    # Load initalised settings
    global run_entries_cache
    scenarios = run_entries_cache["data"]
    start_year = run_entries_cache["start_year"]
    yield("event: status_change\n")
    yield("data: running\n\n")
    yield("data: items;{};\n".format(len(scenarios)))
    start_time = time.time()
    print(scenarios)
    for sc in scenarios:
        if KILL_RUN is True:
            yield("event: status_change\n")
            yield("data: terminate\n\n")
            return {'done':'false'}
        yield("event: processing\n")
        yield("data: ;message:Running {};\n\n".format(sc["scenario"]))
        try:

            print("running scenario")
            tool_response = SimulationTool.run_backend(f'Assumptions_{sc["scenario"]}', int(start_year))
            print(tool_response)
            print(isinstance(tool_response, str))
            print("scenario_ran")
            elapsed_time = time.time() - start_time
            yield("event: processing\n")
            yield("data: progress;{}; \n".format(sc["scenario"]))
            yield("data: elapsed;{} \n\n".format(elapsed_time))
            message = 'done'
        except (KeyError, FileNotFoundError) as e:
            print("error occured")
            error = True
            message = e
            print(e)
    if(isinstance(tool_response, str)):
        yield("event: processing\n")
        yield("data: message;message:{}; \n\n".format(tool_response))
        yield("event: status_change\n")
        yield("data: finished_w_errors\n\n")
    else:
        yield("event: processing\n")
        yield("data: message;message:Finished generating baseline in {}; \n\n".format(elapsed_time))
        yield("event: status_change\n")
        yield("data: finished\n\n")
    return {'done':'true'}

@route('/api/kill_run', method=['GET'])
@enable_cors
def kill_run():
    """
    Set global property to allow termination of model run
    """
    global KILL_RUN
    KILL_RUN = True

    return {'status':'true'}
def restore_baseline():
    dst = "{}\\Scenarios\\Assumptions_Baseline.xlsx".format(rootdir)
    src = "{}\\utilities\\Backup\\Assumptions_Baseline.xlsx".format(rootdir)
    # copy directory if it does not exist
    if os.path.isdir(dst) == False:
        shutil.copy(src,dst)


# API endpoint for getting scenarios
#
#   Returns scenarios that can be run from the inputs folder
#
@route('/api/available_scenarios', method=['GET'])
@enable_cors
def available_scenarios():

    list_files_from_results = []


    scenarios  = glob.glob("{}\\Scenarios\\Assumptions_*".format(rootdir))

    scenarios = [s.split("Assumptions_")[1].split(".")[0] for s in scenarios]

    if "Baseline" not in scenarios:
        restore_baseline()
        scenarios = ["Baseline"] + scenarios
    list_files_from_results.extend(scenarios)

    list_files_from_results = list(set(list_files_from_results))

    print("sort_list")
    print(list_files_from_results)
    #Sort to Baseline followed by alpahbetical
    if "Baseline" in list_files_from_results:
        list_files_from_results.remove("Baseline")
        list_files_from_results.sort()
        sort_list = ["Baseline"] + list_files_from_results
    else:
        list_files_from_results.sort()
        sort_list = list_files_from_results

    scenids = []
    for scen in sort_list:
        if scen == "Baseline":
            scenid =  {"id":scen,"label":scen,"locked":True}
        else:
            scenid =  {"id":scen,"label":scen,"locked":False}
        scenids.append(scenid)

    return {'scenarios': scenids}

#
#   Returns scenarios that that have been run in latest model run
#
@route('/api/default_run_id', method=['GET'])
@enable_cors
def get_run_id():
    config = configparser.ConfigParser()
    config.read('settings.ini')
    default_run_id = config.get('settings', 'run_name')

    return {"default_run_id":default_run_id}

@route('/api/scenarios_ran', method=['GET'])
@enable_cors
def scenarios_ran():

    exist = []
    #Get all runs log_files
    runs = glob.glob("{}\\Output\\Output_*.xlsx".format(rootdir))
    print(runs)
    # Get model run metadata for scenarios
    for r in runs:
        info = {}
        info["scenario"] = r.split("Output_")[-1].split(".")[0]
        exist.append(info)
    print("exist")
    print(exist)
    # Return scenario metadata
    return{'exist':exist}

#
# Get the metadata for all model variables
#
@route('/api/results/variables', method=['GET'])
@enable_cors
def retrieve_variables():

    # Load preprocessed metadata file (generated by seprate python script /manager_new/Update manager metadata.py)
    with open('{}//measures_meta.json'.format(rootdir)) as f:
        variable_meta = json.load(f)
    convs = pd.read_excel('{}\\Utilities\\Unit_conversions.xlsx'.format(rootdir),
                           index_col=0,sheet_name=None)
    energy_conv = convs["energy"]
    vol_conv = convs["volume"]
    gas_conv = convs["gas"]
    energy_unit_options = ["model unit"] + list(energy_conv.columns)
    vol_unit_options = ["model unit"] + list(vol_conv.columns)
    gas_unit_options = ["model unit"] + list(gas_conv.columns)

    config = configparser.ConfigParser()
    config.read('settings.ini')
    start_year_default = config.get('settings', 'simulation_start')
    start_year = int(config.get('settings', 'simulation_start'))
    end_year = int(config.get('settings', 'simulation_end'))
    simulation_years = list(range(start_year,end_year+1))
    simulation_years = [str(x) for x in simulation_years]
    start_year = int(config.get('settings', 'model_start'))
    end_year = int(config.get('settings', 'model_end'))
    years = list(range(start_year,end_year+1))
    years = [str(x) for x in years]

    return {'vars': variable_meta,"energy_unit_options":energy_unit_options,
            "vol_unit_options":vol_unit_options,
            "gas_unit_options":gas_unit_options,
            "years":years,"default_start_yr":start_year_default,"simulation_years":simulation_years}
# API endpoint for getting dimensions for a var for a given county
#
#   returns labels for specific title dimension
#   handles both grouped and ungrouped variables
#
@route('/api/info/<title>', method=['GET'])
@enable_cors
def retrieve_titles(title):
    agg_all = pd.read_excel("{}\\Utilities\\titles\\Grouping.xlsx".format(rootdir),sheet_name=None)

    #Check if a dimension has a hierachical structure or not
    if title in agg_all.keys():
        with open('{}//var_groupings.json'.format(rootdir)) as f:
            data = json.load(f)

            data = json.dumps({"Sectors": data[title]})
    elif title != "None":
        df = pd.read_excel('{}\\Utilities\\titles\\classification_titles.xlsx'.format(rootdir),sheet_name=title)
        df = df.reset_index()
        title_data = list(df['Full name'].unique())

        #Handle numerical titles (like age of technology) and force to string
        if isinstance(title_data[0],np.int64):
            title_data = [str(x) for x in title_data]
        data = json.dumps(title_data)
    else:
        data = json.dumps(["None"])
    return data

# API endpoint for getting all titles used for classification page
#
#   returns a dataframe in JSON format for all model variables
#
@route('/api/info/titles', method=['GET'])
@enable_cors
def retrieve_all_titles():

    title_dict = pd.read_excel('{}\\Utilities\\Titles\\classification_titles.xlsx'.format(rootdir),sheet_name=None)
    titles = []
    for t,title in title_dict.items():
        if t=="Cover":
            continue
        title = title.reset_index()

        temp = list(title['Full name'].unique())
        titles.append( {"name":t,"title":[str(x) for x in temp]})

    data = json.dumps(titles)

    return data

# API endpoint for getting all titles used for classification page
#
#   returns a dataframe in JSON format for all model variables
@route('/api/info/vars', method=['GET'])
@enable_cors
def retrieve_var_data():

    vars_meta = pd.read_excel('{}\\Utilities\\Titles\\VariableListing.xlsx'.format(rootdir),sheet_name="Sheet1")
    vars_meta = vars_meta.fillna("None")
    vars_meta_dict = vars_meta.to_dict("records")
    # Get FAQ data
    faq = pd.read_csv('{}\\Utilities\\Faqs.csv'.format(rootdir),encoding='cp1252')
    faq_dict = faq.to_dict("records")

    return {"items":vars_meta_dict,"faqs":faq_dict}
    # return pd.DataFrame(list(df['variable'].unique())).to_json()
#
# Function for getting the position of each requested element in a dimension
#

def get_dim_pos(title_code,agg_all,dims,title):

    #Check if dimension has a hierachy structure with aggregates
    if title_code in agg_all.keys():
        group = agg_all[title_code]
        group_options = group.columns
        if "Hierachy" in group.index:
            hierachy = set(group.loc["Hierachy",:])
        else:
            hierachy = []
        dims_pos = []
        for x in dims:
            if "|-" in x:
                dim = x.split("|-")[0]
            else:
                dim = x
            #Check if requested element is an aggregation
            if dim in group_options:
                items = list(group[group[dim] == 1].index)
                dims_pos.append([title.index(y) for y in items])
            #else find position of specifc child element
            elif dim in hierachy:
                childs = [y for y in group.columns if group.loc["Hierachy",y]==dim]

                items = list(group.loc[(group[childs] == 1).any(axis=1),:].index)

                dims_pos.append([title.index(y) for y in items])

            else:
                dims_pos.append([title.index(dim)])

            #Allow custom removal of region from grouping
            for r in x.split("|-")[1:]:
                dims_pos[-1].remove(title.index(r))
    else:
        dims_pos = [[title.index(x)] for x in dims]
    return dims_pos

def create_aggregate(data, dims_pos, dims,title, title_code,years):
    "Generate aggregate variables and add to data frame"
    data_plus_agg = data.copy()
    # Check for aggregate catergorie with multiple items
    for d,dim in enumerate(dims_pos):
        if len(dim)> 1:
            #Filter to aggregate commponents
            filter_labels = [title[x] for x in dim]
            data_agg = data_plus_agg[data_plus_agg[title_code.lower()].isin(filter_labels)]

            data_agg[title_code.lower()] = dims[d]

            #Set group by for all non year columns
            temp = list(data_agg.columns)

            temp = temp[:len(temp)-len(years)]
            data_agg = data_agg.groupby(temp).sum().reset_index()

            data_plus_agg = pd.concat([data_plus_agg,data_agg])


    return data_plus_agg

# API endpoint for retrieving model results data
#
#   returns a dataframe in JSON format for the given query parameters
#   paramters should be supported through the parameters object of the request
#
@route('/api/results/data/<type_>', method=['GET'])
@enable_cors
def retrieve_chart_data(type_):

    #Load requests
    p = request.query;

    #extract parameters passed
    variables = p.getlist("variable[]")
    dims = p.getlist("dimensions[]")
    dims2_master = p.getlist("dimensions2[]")
    dims3_master = p.getlist("dimensions3[]")

    title_codes = p.getlist("title[]")
    title2_codes = p.getlist("title2[]")
    title3_codes = p.getlist("title3[]")
    var_labels = p.getlist("variable_label[]")

    scenarios_ = p.getlist("scenarios[]")
    baseline = p.get("baseline")
    agg = p.get("aggregate")
    agg2 = p.get("aggregate2")
    agg3 = p.get("aggregate3")
    calc_type = p.get("calculation")
    time = p.get("time")
    unit = p.get("unit")
    start_year = p.get("start_year")
    end_year = p.get("end_year")
    report_extract = p.get("report_extract")

    print([variables,dims,dims2_master,dims3_master,title_codes,title2_codes,title3_codes,var_labels])

    # Add baseline to scenarios to extract for calculating difference from baseline
    if baseline not in scenarios_ and calc_type != 'Levels':
        scenarios = scenarios_ + [baseline]
    else:
        scenarios = scenarios_
    full_df = None

    # Load latest model run results
    runs_dict = {}
    print("Loading data")
    # for r in scenarios:
    #     runs_dict[r] = pd.read_excel(f'Output\Outputs_{r}.xlsx',sheet_name=None)
    print("Data loaded")
    #Get titles
    title_list = pd.read_excel('{}\\Utilities\\Titles\\classification_titles.xlsx'.format(rootdir),sheet_name=None)
    agg_all = pd.read_excel("{}\\Utilities\\Titles\\Grouping.xlsx".format(rootdir),sheet_name=None,index_col=0)


    # Loop through variables
    for v,var in enumerate(variables):

        #Get variable data
        var_info = json.loads(var_labels[v])
        var_label = var_info["label"]

        #retrieve title codes
        title_code = title_codes[v]
        title2_code = title2_codes[v]
        title3_code = title3_codes[v]

        #Handle null case for spare dimensions otherwise retrieve full
        #titles for each dimension
        if title_code == "None":
            title = ["None"]
        else:
            title = list(title_list[title_code]['Full name'].unique())
        if title2_code == "None":
            title2 = ["None"]
        else:
            title2 = list(title_list[title2_code]['Full name'].unique())
        if title3_code == "None":
            title3 = ["None"]
        else:
            title3 = list(title_list[title3_code]['Full name'].unique())

        #Ensure all titles are strings
        title = [str(x) for x in title]
        title2 = [str(x) for x in title2]
        title3 = [str(x) for x in title3]
        # If all passed then set selection to full list of labels
        if dims[0] == "All":
            dims = title
        if dims2_master[0] == "All":
            dims2 = title2
        else:
            dims2 = dims2_master
        if dims3_master[0] == "All":
            dims3 = title3
        else:
            dims3 = dims3_master
        # Get position of all selected elements in each dimension
        dims_pos = get_dim_pos(title_code,agg_all,dims,title)
        dims2_pos = get_dim_pos(title2_code,agg_all,dims2,title2)
        dims3_pos = get_dim_pos(title3_code,agg_all,dims3,title3)


        config = configparser.ConfigParser()
        config.read('settings.ini')
        start_year_max = config.get('settings', 'model_start')
        end_year_max = config.get('settings', 'model_end')
        years_max = list(range(int(start_year_max),int(end_year_max) +1))
        years = list(range(int(start_year),int(end_year) +1))
        if time == "Yes":
           years_max = [str(x) for x in years_max]
           years = [str(x) for x in years]
        else:
            years = ["None"]

        #Iterate through scenario to extract all request data for the variable
        for scenario in scenarios:
            scenario_df = None


            data = pd.read_csv(f'Output\\{scenario}\\{var}.csv')
            #drop code columns
            print(data.columns)
            col_drop = [x for x in data.columns if "_code" in x]
            print(col_drop)
            col_drop += ["units"]
            print(col_drop)
            #data = data.drop(col_drop, axis=1)

            # Prepare required aggregates
            data = create_aggregate(data , dims_pos, dims,title, title_code,years_max)
            data = create_aggregate(data , dims2_pos, dims2,title2, title2_code,years_max)
            data = create_aggregate(data, dims3_pos, dims3,title3, title3_code,years_max)


            #data = data[var]
            #filter through each active dimension
            if title_code != "None":
                data = data[data[title_code.lower()].isin(dims)]
            if title2_code != "None":
                data = data[data[title2_code.lower()].isin(dims2)]
            if title3_code != "None":
                data = data[data[title3_code.lower()].isin(dims3)]


            print(data)

            #Rename title columns to make easier for frontend manipulation
            data = data.rename(columns={title_code.lower():"dimension",
                                        title2_code.lower():"dimension2",
                                        title3_code.lower():"dimension3"})
            data.columns = [str(x) for x in data.columns]

            if title2_code == "None":
                data["dimension2"] = "None"
            if title3_code == "None":
                data["dimension3"] = "None"

            # Collapse year dimension
            scenario_df = pd.melt(data, id_vars=["dimension","dimension2","dimension3"]+ col_drop, value_name="variables")

            #Add additional metadat
            scenario_df['scenario'] = scenario
            scenario_df = scenario_df.rename(columns={"variable":"year"})
            if len(years)>0:
                scenario_df = scenario_df[scenario_df["year"].isin(years)]
            scenario_df["variable"] = var
            scenario_df["Variable Name"] = var_label

            #Collate into single data frame for all scenarios and variables
            full_df = scenario_df if full_df is None else full_df.append(scenario_df)

    print(f'Unit: {var_info["unit"]}')
    # Sum across each dimensions if aggregate is set
    if agg == "true":
        full_df = full_df.groupby(['year','scenario',"variable","Variable Name","dimension2","dimension3"]+col_drop).sum().reset_index().copy()
        full_df["dimension"] = ", ".join(dims)
    if agg2 == "true":
        full_df = full_df.groupby(['year','scenario',"variable","Variable Name","dimension","dimension3"]+col_drop).sum().reset_index().copy()
        full_df["dimension2"] = ", ".join(dims2)
    if agg3 == "true":
        full_df = full_df.groupby(['year','scenario',"variable","Variable Name","dimension","dimension2"]+col_drop).sum().reset_index().copy()
        full_df["dimension3"] = ", ".join(dims3)
    # Transform data for difference in baseline
    if calc_type in ['absolute_diff','perc_diff']:
        baseline_df = full_df[full_df['scenario'] == baseline].copy().drop(['scenario'], axis=1)
        full_df = full_df.merge(baseline_df, how="left", left_on=["year","variable","Variable Name","dimension","dimension2","dimension3"]+col_drop, right_on=["year","variable","Variable Name","dimension","dimension2","dimension3"])
        if calc_type == 'absolute_diff':
            full_df["variables"] = full_df.apply(lambda row: row["{}_x".format("variables")] - row["{}_y".format("variables")], axis=1)
        else:
            full_df["variables"] = full_df.apply(lambda row: ((row["{}_x".format("variables")] / row["{}_y".format("variables")]) - 1) *100 if row["{}_y".format("variables")] != 0 else 0 , axis=1)
        # import pdb; pdb.set_trace()
        full_df = full_df.drop(["{}_x".format("variables"),"{}_y".format("variables")], axis=1)

    # Remove baseline data if difference from baseline but baseline is not selected
    if baseline not in scenarios_ and calc_type != 'Levels':
        full_df = full_df[full_df['scenario'] != baseline].copy()

    if calc_type in ['Annual growth rate','Incremental']:
        if calc_type == 'Annual growth rate':

            full_df['lagged'] = full_df.groupby(['scenario',"variable","Variable Name","dimension","dimension2","dimension3"]+col_drop)["variables"].shift(1)
            full_df["variables"] = (full_df["variables"] / full_df['lagged'] - 1)*100

            full_df = full_df.drop(columns=['lagged'])
        if calc_type == 'Incremental':
            full_df['lagged'] = full_df.groupby(['scenario',"variable","Variable Name","dimension","dimension2","dimension3"]+col_drop)["variables"].shift(1)
            full_df["variables"] = (full_df["variables"] - full_df['lagged'] - 1)*100

            full_df = full_df.drop(columns=['lagged'])
    if isinstance(unit,str):
        if unit != "model unit":
            common_unit = var_info["unit"]
            convs = pd.read_excel('{}\\Utilities\\Unit_conversions.xlsx'.format(rootdir),
                           index_col=0,sheet_name=None)
            energy_conv = convs["energy"]
            vol_conv = convs["volume"]
            gas_conv = convs["gas"]
            if unit in energy_conv.columns:
                conv = energy_conv
            elif unit in vol_conv.columns:
                conv = vol_conv
            elif unit in gas_conv.columns:
                conv = gas_conv

            full_df["variables"] = full_df["variables"] * conv.loc[common_unit,unit]
            var_info["unit"] = unit
    if calc_type in ['Annual growth rate','perc_diff']:
        var_info["unit"] = "%"
    dims_all =  ['scenario','year',"variable","variables","Variable Name",'dimension','dimension2','dimension3']+col_drop
    # Handle div zero errors set to 0
    full_df.fillna(0)
    full_df = full_df.loc[:,dims_all]
    full_df = full_df.reset_index().drop("index",axis=1)
    json_ = full_df.copy()
    if type_ == "json":
        # Apply rounding for presentation
        settings = pd.read_excel('{}\\Utilities\\Titles\\VariableListing.xlsx'.format(rootdir),
                                 sheet_name="Sheet1",index_col=0)
        settings = settings.loc[var,:]
        print(f'Calc_type:{calc_type}')

        if calc_type in ['Annual growth rate','perc_diff']:
            rounding = 3
        else:
            rounding = settings.loc["decimal_round"]
        json_["variables"] = round(json_["variables"] ,int(rounding))

    piv = json_.copy()
    json_ = json_.rename(columns={"dimension":title_code,
                              "dimension2":title2_code,
                              "dimension3":title3_code})
    if (title2_code == "None" and title3_code == "None"):
       json_ =  json_.iloc[:,:-1]
    json_ = json_.to_json(orient='records')
    full_df = full_df.rename(columns={"dimension":title_code,
                              "dimension2":title2_code,
                              "dimension3":title3_code})


    dims = ['scenario',"variable","Variable Name",'dimension','dimension2','dimension3']+col_drop


    if type_ == 'csv':
        if isinstance(report_extract,str):
            with open('{}//manager_new//reverse_lookup.json'.format(rootdir)) as f:
                reverse = json.load(f)
            if title_code in reverse.keys():
                region_lookup = reverse[title_code]

                piv["hierachy"] = piv.apply(lambda row: region_lookup[row["dimension"]]["hierachy"],axis=1)
                piv["group"] = piv.apply(lambda row: region_lookup[row["dimension"]]["group"],axis=1)
                dims = ["hierachy", "group"] + dims

        # Generate csv file for frontend to offer as download
        if time == "Yes":
            piv = piv.pivot_table(index=dims, columns=['year'])
        else:
            piv = piv.pivot_table(index=dims, columns=['dimension3'])
        piv.columns = piv.columns.droplevel(0)
        #If single variable extraction, add a header with metadata
        if len(variables)==1:
            meta_dict = {}
            meta_dict["Variable Name"] = variables[0]
            meta_dict["Desc"] = var_label
            if calc_type == "Annual growth rate" or calc_type == "per_diff" :
                meta_dict["Unit"] = "%"
            else:
                meta_dict["Unit"] = var_info["unit"]
            meta_dict["Presented as"] = calc_type
            meta_dict["Info"] = var_info["info"]


            metadata = pd.Series(meta_dict).to_csv(quoting=csv.QUOTE_NONNUMERIC, header=False)

            piv = piv.reset_index().drop(columns=["variable","Variable Name"])
            piv = piv.rename(columns={"variables": variables[0],
                                      "dimension":title_code,
                                      "dimension2":title2_code,
                                      "dimension3":title3_code})
            data = piv.to_csv(quoting=csv.QUOTE_NONNUMERIC,index=False)
            return metadata + data
        else:
            data = piv.to_csv(quoting=csv.QUOTE_NONNUMERIC)
            return data


        return piv.to_csv(quoting=csv.QUOTE_NONNUMERIC)
    else:
        #Generate data for use in front end tables and charts

        piv = piv.drop(columns=list(set(piv.columns)-set(dims_all)))
        if len(variables)==1:
            piv = piv.rename(columns={"variables": variables[0]} )

        if time == "Yes":
            piv = piv.pivot_table(index=dims, columns=['year'])
        else:
            piv = piv.pivot_table(index=dims, columns=['dimension3'])
        piv.columns = piv.columns.droplevel(0)


        piv = piv.reset_index()
        piv = piv.rename(columns={"dimension":title_code,
                                  "dimension2":title2_code,
                                  "dimension3":title3_code})
        if (title2_code == "None" and title3_code == "None"):
            piv =  piv.drop(piv.loc[:,'None'].columns,axis = 1)
        if type_ == "json":

            piv_ = piv.to_json(orient='records', double_precision=4)
            return {'info': list(full_df.columns),'results': json_, 'pivot': piv_,'round':str(rounding)}
        #Experimental version for jexcel table
        else:
            piv_ = piv.to_json(orient='values', double_precision=4)
            return {'info': list(full_df.columns),'results': json_, 'pivot': piv_,"pivot_columns": list(piv.columns),'round':str(rounding)}


###Report page queries###

#
# Retrieve list of available report graphics
#
@route('/api/Report/Options', method=['GET'])
@enable_cors
def retrieve_report_graphics():
    graphics = pd.read_excel('{}\\Utilities\\Titles\\ReportGraphics.xlsx'.format(rootdir),
                             sheet_name="Graphic_Definitions",index_col="ref")
    categories = list(set(graphics["Category"]))

    graphics_dict = {}

    for cat in categories:
        cat_df = graphics[graphics["Category"]==cat]
        graphics_dict[cat] = {}

        chart_df = cat_df[cat_df["Type"] == "Chart"]
        #graphic_dict[cat] = cat_df.to_dict(orient="index")
        graphics_dict[cat]["charts"] = list(chart_df["Figure label"])
        table_df = cat_df[cat_df["Type"] == "Table"]
        graphics_dict[cat]["tables"] = list(table_df["Figure label"])
    return{"category": categories, "graphics": graphics_dict}

@route('/api/Report/Values/<graphic>/<type_>/<scenario_run>', method=['GET'])
@enable_cors
def construct_graphic_data(graphic,type_,scenario_run):


    graphics = pd.read_excel('{}\\Utilities\\Titles\\ReportGraphics.xlsx'.format(rootdir),
                         sheet_name="Graphic_Definitions",index_col="Figure label")
    settings = graphics.loc[graphic.replace("-"," ")]
    print(settings)
    command = settings.loc["Vars"].split("|")

    vars = command[1].split(",")

    dims = settings.loc["Dim1"].split(",")
    dims2 = settings.loc["Dim2"].split(",")
    dims3 = settings.loc["Dim3"].split(",")

    #Get titles from var listing
    vars_meta = pd.read_excel('{}\\Utilities\\Titles\\VariableListing.xlsx'.format(rootdir),
                              sheet_name="Sheet1",index_col=0)
    #Assume al variables needed have same dimension as first for processing
    vars_meta = vars_meta.fillna("None")

    title_code = vars_meta.loc[vars[0],"Dim1"]
    title2_code = vars_meta.loc[vars[0],"Dim2"]
    title3_code = vars_meta.loc[vars[0],"Dim3"]


    scenario,run_id = scenario_run.split("|")
    scenarios = [scenario]
    time = "Yes"

    full_df = None

    with open('Output\{}.pickle'.format(run_id), 'rb') as f:
        output = pickle.load(f)
    title_list = pd.read_excel('{}\\Utilities\\Titles\\classification_titles.xlsx'.format(rootdir),sheet_name=None)
    agg_all = pd.read_excel("{}\\Utilities\\Titles\\Grouping.xlsx".format(rootdir),sheet_name=None,index_col=0)
    if title_code == "None":
        title = ["None"]
    else:
        title = list(title_list[title_code]['Full name'].unique())
    if title2_code == "None":
        title2 = ["None"]
    else:
        title2 = list(title_list[title2_code]['Full name'].unique())
    if title3_code == "None":
        title3 = ["None"]
    else:
        title3 = list(title_list[title3_code]['Full name'].unique())
    if dims[0] =="All":
       dims = title
    if dims2[0] =="All":
       dims2 = title2
    if dims3[0] =="All":
       dims3 = title3

    # Get position of all selected elements in each dimension
    dims_pos = get_dim_pos(title_code,agg_all,dims,title)
    dims2_pos = get_dim_pos(title2_code,agg_all,dims2,title2)
    dims3_pos = get_dim_pos(title3_code,agg_all,dims3,title3)

    for d,dim in enumerate(dims):
        if "|-" in dim:
            dims[d] = "Other " + dim.split("|-")[0]
    scen_meta = "{}\\Output\\Scenarios_{}.json".format(rootdir,run_id)
    with open(scen_meta, 'r+') as f:
        meta = json.load(f)
        for scen,value in meta.items():
            years = value["years"]

    if time == "Yes":
       years = [str(x) for x in years]
    else:
        years = ["None"]

    for scenario in scenarios:
        scenario_df = None

        data = output[scenario]
        data_filter = []
        dims_list = []
        dims2_list = []
        dims3_list = []
        var_list = []
        for v,var in enumerate(vars):

            for d1,dim1 in enumerate(dims):
                for d2,dim2 in enumerate(dims2):
                    for d3,dim3 in enumerate(dims3):
                        dims_list.append(dim1)
                        dims2_list.append(dim2)
                        dims3_list.append(dim3)
                        var_list.append(var)
                        if isinstance(dims_pos[d1],list) or isinstance(dims2_pos[d2],list) or isinstance(dims3_pos[d3],list):
                            #Need to use advanced indexing to cut across multiple dimensions
                            index = np.ix_(dims_pos[d1],dims2_pos[d2],dims3_pos[d3],range(data[var].shape[3]))
                            temp = data[var][index]
                            no_dims = len(temp.shape)
                            sum_cuts = tuple(range(no_dims-1))
                            data_filter.append(np.sum(temp,axis=sum_cuts))
                        else:
                            data_filter.append(data[var][dims_pos[d1],dims2_pos[d2],dims3_pos[d3],:])
        data_filter = np.vstack(data_filter)

        df = pd.DataFrame(data_filter,columns=years)
        df["dimension"] = pd.Categorical(dims_list,dims)
        if len(dims2) >1:
            df["dimension2"] = pd.Categorical(dims2_list,dims2)
        else:
            df["dimension2"] = dims2_list
        if len(dims3) >1:
            df["dimension3"] = pd.Categorical(dims3_list,dims3)
        else:
            df["dimension3"] = dims3_list

        df["indic"] = var_list
        scenario_df = pd.melt(df, id_vars=["indic","dimension","dimension2","dimension3"])

        scenario_df['scenario'] = scenario

        full_df = scenario_df if full_df is None else full_df.append(scenario_df)

    full_df = full_df.rename(columns={"variable":"year"})

    #Based on command and time specified transform data
    full_df = full_df.set_index(["indic","dimension","dimension2","dimension3","scenario","year"])
    if command[0] =="DIVIDE":
        #Divde two variables of the same size
        #unstack variable dimension
        full_df = full_df.unstack(level=0).droplevel(0, axis=1)
        #divide variable value columns
        full_df["value"] = full_df.loc[:,vars[0]]/full_df.loc[:,vars[1]]
        #drop divisors
        full_df = full_df.drop(vars,axis=1)
        #Second optional arguement for multiplier
        if len(command) >2:
            full_df = full_df * int(command[2])
    if command[0] == "SHARE":
        #Calculate share of total in each year

        full_df = full_df.unstack(level=-1).droplevel(0, axis=1)
        full_df = full_df/full_df.sum(axis=0)
        #Second optional arguement for multiplier
        if len(command) >2:
            full_df = full_df * int(command[2])

        full_df = pd.melt(full_df.reset_index(), id_vars=["indic","dimension","dimension2","dimension3","scenario"], value_name="value")
    fields = ['scenario','dimension','dimension2','dimension3']

    if command[0] == "SUM":
        if "1" in command[2]:
            group = ["dimension2","dimension3","scenario","year"]

            full_df = full_df.groupby(by=group).sum()
            full_df['dimension'] = "All"

        if "2" in command[2]:
            group = ["dimension","dimension3","scenario","year"]

            full_df = full_df.groupby(by=group).sum()
            full_df['dimension2'] = "All"

        if "3" in command[2]:
            group = ["dimension","dimension2","scenario","year"]

            full_df = full_df.groupby(by=group).sum()
            full_df['dimension3'] = "All"

        if "4" in command[2]:
            group = ["dimension","scenario","year"]

            full_df = full_df.groupby(by=group).sum()
            full_df['dimension2'] = "All"
            full_df['dimension3'] = "All"

        if "5" in command[2]:
            group = ["dimension2","scenario","year"]

            full_df = full_df.groupby(by=group).sum()
            full_df['dimension'] = "All"
            full_df['dimension3'] = "All"
            main_dim = "dimension2"
            main_dims = dims2
            drop_dim = "dimension3"

        if "6" in command[2]:
            group = ["dimension3","scenario","year"]

            full_df = full_df.groupby(by=group).sum()
            full_df['dimension'] = "All"
            full_df['dimension2'] = "All"
            main_dim = "dimension3"
            main_dims = dims3
            drop_dim = "dimension2"

        #Disaggregated sum function
        elif "D" in command[2]:
            group = ["dimension",'dimension2', 'dimension3',"scenario","year"]

            full_df = full_df.groupby(by=group).sum()

    piv = full_df.pivot_table(index=fields, columns=['year']).droplevel(0, axis=1)
    time_select = settings.loc["Dim4"].split("|")
    if settings.loc["layer transform"] != "None":
        time_select.append(settings.loc["layer transform"])
    #Check unit against standard listing
    report_unit = settings.loc["unit"]

    if (command[0] != "DIVIDE") or (report_unit != "%"):
        common_unit = vars_meta.loc[vars[0],"Unit"]
        if common_unit != report_unit:
            conv = pd.read_excel('{}\\Utilities\\Unit_conversions.xlsx'.format(rootdir),
                               index_col=0, sheet_name = None)
            unit_cat = settings['unit_category']
            piv = piv * conv[unit_cat].loc[common_unit,report_unit]
    for t in time_select:
        if "Growth" in t:
            #Denotes absolute change
            com = t.split(" ")[1]
            coms = com.split("-")
            piv[t] = piv[coms[1]]-piv[coms[0]]
        elif "-" in t:
            #Average annual Percentage growth rate
            coms = t.split("-")
            diff = int(coms[1]) - int(coms[0])
            piv[t] = ((piv[coms[1]]/piv[coms[0]])**(1/diff)-1)*100


    piv = piv.loc[:,time_select]
    for t in piv.columns:
        piv[t] = round(piv[t],settings.loc["decimal_round"])
    full_df = piv.stack().reset_index()

    # Handle Waterfall chart special case

    if settings.loc["Chart type"] == "waterfall":
        # set waterfall_groups
        groups = settings.loc["waterfall_groups"].split("|")
        groups = {x.split(",")[0]:x.split(",")[1:] for x in groups}
        full_df = full_df.drop(drop_dim,axis=1)
        base_sum = full_df.groupby(["scenario","dimension","year"]).sum()
        base_sum = base_sum.loc[slice(None, None, int(time_select[0]))]

        waterfall = []
        for g,group in groups.items():
            g_mask = full_df[main_dim].isin(group)
            #Set group in year field
            temp = full_df.loc[g_mask].set_index(["scenario","dimension",main_dim,"year"]).unstack(-1)
            temp.columns = time_select
            temp = temp[(time_select[1])]-temp[(time_select[0])]

            temp.columns = [g]

            if len(waterfall) ==0:
                temp_base = base_sum.copy().reset_index()
                temp_base[main_dim] = "Base"
                temp_base = temp_base.drop(["year"],axis=1)
                temp_base = temp_base.set_index(["scenario","dimension",main_dim])

            else:
                temp_base =  waterfall[-1].reset_index().groupby(["scenario","dimension"]).sum()
                temp_base[main_dim] = "Base"
                temp_base = temp_base.reset_index().set_index(["scenario","dimension",main_dim])

            temp = temp.reset_index()
            temp  = temp.set_index(["scenario","dimension",main_dim])
            temp.columns = [g]
            temp_base.columns = [g]
            #Adjust waterfall base for reductions

            adjustment = temp[temp[g]<0].sum(axis=0)

            temp_base = temp_base + adjustment

            #Sort difference largest to smallest and then shift to positive
            temp = temp.sort_values(by=g,ascending=False)
            #Change to absolute values (will include orginal values for labels)


            #Update column names
            temp_base = temp_base.astype(float)

            temp = pd.concat([temp_base,temp],axis=0)

            waterfall.append(temp)

        waterfall_df = pd.concat(waterfall)
        waterfall_df = waterfall_df.stack()
        waterfall_df.columns = ["year",0]
        waterfall_df = waterfall_df.reset_index()
        waterfall_df.columns = ["scenario","dimension",main_dim,"year",0]

        full_df = full_df.groupby(["scenario","dimension","year"]).sum().reset_index()
        full_df[main_dim] = "Total"
        full_df = pd.concat([full_df,waterfall_df])
        full_df[drop_dim] = "None"
        full_df["value2"] = full_df[0].abs()


        #reorder data
        year_order = [time_select[0]] + list(groups.keys()) +[time_select[1]]
        print(year_order)
        full_df = full_df.set_index(["year"])
        full_df = full_df.loc[year_order]
        full_df = full_df.reset_index()
        full_df = pd.concat([full_df[full_df[main_dim]!="Base"],full_df[full_df[main_dim]=="Base"]])
        cat_order = ['Total'] + main_dims + ['Base']
        full_df[main_dim] = pd.Categorical(full_df[main_dim], cat_order, ordered = True)

    # Handle div zero errors set to 0
    full_df = full_df.rename(columns={0:"value"})
    full_df["value"].fillna(0)


    axis_max = full_df["value"].max()
    axis_min = full_df["value"].min()

    axis_set = "merge"
    layer_label = settings.loc["unit"] + " "
    #Map layer dims to seperate value2
    if (settings.loc["layer transform"] != "None") & ('Growth' not in settings.loc["layer transform"]):
        layer = full_df.copy()[full_df["year"]==settings.loc["layer transform"]]
        layer["%"] = layer["value"]
        layer = layer.drop("value",axis=1)
        stitch = pd.concat([full_df[full_df["year"]!=settings.loc["layer transform"]],layer])
        axis_set = "dock"
        layer_label = "%"
    elif (settings.loc["layer transform"] != "None") & ('Growth' in settings.loc["layer transform"]):
        layer = full_df.copy()[full_df["year"]==settings.loc["layer transform"]]
        layer[settings.loc["unit"] + " "] = layer["value"]
        layer = layer.drop("value",axis=1)
        stitch = pd.concat([full_df[full_df["year"]!=settings.loc["layer transform"]],layer])
        if settings.loc["Chart type"] =="stacked-bar":
            bar_check = full_df[full_df["dimension"]!=settings.loc["layer_dim1"]].pivot_table(index=fields, columns=['year'],values="value")

            bar_check_sum = bar_check.sum(axis=0)

            axis_max = bar_check_sum.max()
            axis_min = bar_check.min().min()
            axis_set = "merge"
    elif (settings.loc["layer_dim1"] != "None"):
        layer = full_df.copy()[full_df["dimension"]==settings.loc["layer_dim1"]]
        layer[settings.loc["unit"] + " "] = layer["value"]
        layer = layer.drop("value",axis=1)
        stitch = pd.concat([full_df[full_df["dimension"]!=settings.loc["layer_dim1"]],layer])
        if settings.loc["Chart type"] =="stacked-bar":
            bar_check = full_df[full_df["dimension"]!=settings.loc["layer_dim1"]].pivot_table(index=fields, columns=['year'],values="value")

            bar_check_sum = bar_check.sum(axis=0)

            axis_max = bar_check_sum.max()
            axis_min = bar_check.min().min()
            axis_set = "merge"

    else:
        stitch = full_df.copy()

    #full_df.to_csv("Test.csv")
    print(stitch)
    json_ = stitch.to_json(orient='records')

    piv = full_df.copy()
    if settings.loc["Chart type"] == "waterfall":
        piv = piv.drop("value2",axis=1)
    #piv['year'] = piv['year'].apply(lambda d_: d_[:4])

    brewer_dict = {}
    #Retrieve all colour codes for brewer
    if settings.loc["Type"] == "Chart":
        label_set = set(list(full_df["dimension"]) + list(full_df["dimension2"]) + list(full_df["dimension3"]))
        if settings.loc["color"] == "year":
            label_set = set(list(full_df["dimension"]) + list(full_df["dimension2"]) + list(full_df["dimension3"]) + list(time_select))

        try:
            label_set.remove("None")
        except:
            pass
        try:
            label_set.remove("All")
        except:
            pass
        colours = pd.read_excel('{}\\Utilities\\Titles\\ReportGraphics.xlsx'.format(rootdir),
                             sheet_name="ColoursMap",index_col=0)
        colours.index = [str(x) for x in colours.index]

        rgb_values = pd.read_excel('{}\\Utilities\\Titles\\ReportGraphics.xlsx'.format(rootdir),
                             sheet_name="RGB_values",index_col=0)
        brewer_dict = {}
        colour_codes = [colours.loc[x,"WOO_code"] for x in label_set if x in label_set]
        colour_codes_dict = dict(zip( label_set,colour_codes))
        #Handle dual colour classes
        for k,v in colour_codes_dict.items():
            val = str(v)
            if len(val)>2:
                values = val.split("/")
                #Check if first value is already in use
                if values[0] in colour_codes:
                    colour_codes_dict[k] = values[1]
                else:
                    colour_codes_dict[k] = values[0]

        for lab in label_set:
            colour_code = int(colour_codes_dict[lab])
            # Modify colour to red if variable is in layer
            if lab == settings.loc["layer transform"]:
                colour_code = 3
            rgb = "rgb({},{},{})".format(rgb_values.loc[colour_code,"R"],rgb_values.loc[colour_code,"G"],rgb_values.loc[colour_code,"B"])
            brewer_dict[lab] = rgb
    if type_ == 'csv':
        if time == "Yes":
            piv = piv.pivot_table(index=fields, columns=['year'])

        else:
            piv = piv.pivot_table(index=fields, columns=['dimension3'])
        piv.columns = piv.columns.droplevel(0)
        if settings.loc["Chart type"] == "waterfall":
            piv = piv.loc[:,year_order]
        return piv.to_csv(quoting=csv.QUOTE_NONNUMERIC)
    else:
        if settings.loc["x"] == "dimension3":
            print(piv)
            fields = ['scenario','dimension','dimension2','year']
            piv = piv.pivot_table(index=fields, columns=['dimension3'])
            time = 'No'
        else:
            piv = piv.pivot_table(index=fields, columns=['year'])
        print(piv)
        piv.columns = piv.columns.droplevel(0)
        print(piv)
        piv = piv.rename(columns=str).reset_index()
        if type_ == "json":
            print(brewer_dict)
            if settings.loc["Chart type"] == "waterfall":
                settings.loc["Chart type"] = "stacked-bar"
            print(piv)
            piv_ = piv.to_json(orient='records', double_precision=4)
            print(piv_)
            return {'info': list(full_df.columns),
                    'results': json_, 'pivot': piv_,
                    "x":settings.loc["x"],"y":settings.loc["y"],
                    "type":settings.loc["Chart type"],
                    "color":settings.loc["color"],
                    "label":settings.loc["label"],
                    "unit":settings.loc["unit"],"brewer":brewer_dict,
                    "layer_type":settings.loc["layer_type"],
                    "axis_max":axis_max,"axis_min":axis_min,
                    "axis_set":axis_set,"layer_label":layer_label,
                    'time': time, 'information': settings.loc['Information']}


#Deliver the main page of the frontend
#
@route('/main', method=['GET'])
def frontend():
    config = configparser.ConfigParser()
    config.read('settings.ini')
    gui_mode = config.get('settings', 'GUI')
    if gui_mode == "Full":
        return static_file("{}\\frontend\\index.html".format(rootdir), "frontend")
    elif gui_mode == "Simple":
        return static_file("{}\\frontend_simple\\index.html".format(rootdir), "frontend_simple")
    else:
        print("Incorrect GUI mode set. Choose either Simple or Full in setting.ini")

@route('/frontend/:filename#.*#', method=['GET'])
def statics(filename):

    return static_file(filename, "{}\\frontend\\".format(rootdir))

@route('/documentation/:filename#.*#', method=['GET'])
def statics_docs(filename):

    return static_file(filename, "{}\\documentation\\".format(rootdir))
#
# Terminate the application
#
@route('/api/exit', method=['GET'])
def exit_():
    q = Timer(2.0, os._exit(1), ())
    return None


@route('/api/info/region_titles', method=['GET'])
@enable_cors
def retrieve_region_titles():
       df = pd.read_excel('{}\\Utilities\\Titles\\classification_titles.xlsx'.format(rootdir),sheet_name="demand_region")
       df = df.reset_index()
       data = json.dumps(list(df['Full name'].unique()))
       return data


@route('/api/assumption/save/', method=['GET'])
@enable_cors
def create_new_assumption():
    """
    Create assumption file for new scenario by copying the assumption folder
    from base scenario chosen
    Inputs need from frontend
    scen_copy - existing scenario to use as base file


    Returns
    -------
    None.

    """
    p = request.query
    print("saving scenario")
    scen_copy = p.get("scen_copy")
    scen_name = p.get("scen_name")
    src = "{}/Scenarios/Assumptions_{}.xlsx".format(rootdir,scen_copy)
    dst =  "{}/Scenarios/Assumptions_{}.xlsx".format(rootdir,scen_name)

    # copy directory if it does not exist
    if os.path.isdir(dst) == False:
        shutil.copy(src,dst)

@route('/api/assumption/open/<scen_name>', method=['GET'])
@enable_cors
def open_assumption(scen_name):
    """
    Opens assumption file requested by the user in excel
    Parameters
    ----------
    scen_name : string
        Scenario name of assumption file

    Returns
    -------
    None.

    """
    print(scen_name)


    dst =  "{}/Scenarios/Assumptions_{}.xlsx".format(rootdir,scen_name)

    os.system(f"start EXCEL.exe \"{dst}\"")

@route('/api/assumption/delete/<scen_name>', method=['GET'])
@enable_cors
def delete_assumption(scen_name):

    # Delete the assumption file
    dst = "{}/Scenarios/Assumptions_{}.xlsx".format(rootdir, scen_name)
    os.remove(dst)

    # Delete results file if present
    dst = "{}/Output/Output_{}.xlsx".format(rootdir, scen_name)
    print(dst)
    print(os.path.isfile(dst))
    if os.path.isfile(dst):
        os.remove(dst)
    dst = "{}/Output/{}".format(rootdir, scen_name)
    print(dst)
    print(os.path.isdir(dst))
    if os.path.isdir(dst):
        shutil.rmtree(dst)


# MAIN APP
# ==========

message_cache = []

if __name__ == '__main__':

    def splash():
        root = tk.Tk()
        root.overrideredirect(True)
        width = root.winfo_screenwidth()
        height = root.winfo_screenheight()

        img_width = 533
        img_height = 300

        x0 = width / 2 - img_width / 2
        x1 = width / 2 + img_width / 2
        y0 = height / 2 - img_height / 2
        y1 = height / 2 + img_height / 2

        root.geometry('%dx%d+%d+%d' % (img_width, img_height, x0, y0))
        root.attributes("-alpha", 0.9)

        if os.path.exists("{}//splash.png".format(rootdir)):
            image_file = "{}//splash.png".format(rootdir)
        else:
            image_file = "{}//splash.png".format(rundir)

        image = tk.PhotoImage(file=image_file)
        canvas = tk.Canvas(root, height=img_height, width=img_width)
        canvas.create_image(0, 0, image=image, anchor=tk.NW)
        canvas.pack()
        # show the splash screen for 5000 milliseconds then destroy
        root.after(5000, root.destroy)
        root.call('wm', 'attributes', '.', '-topmost', '1')
        root.mainloop()
    # splash()

    # testing if the given port is available, if not, test other ones
    bport = 4000  # base port number, to be added on to
    success = False
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    for jj in range(100):
        port = bport + jj
        try:
            s.bind(('127.0.0.1', port))
            success = True
            s.close()
            break
        except socket.error as e:
            # print(e)
            continue

    if not success:
        raise IOError('Failed to open a port, the front end will not start.')

    port = 4000
    addr = 'http://localhost:%s' % port
    if PRODUCTION:

        already_running = False
        count = 0
        for p in psutil.process_iter(attrs=['pid', 'name']):
            if "CE_LEM_Launcher.exe" in p.info['name']:
                count += 1
                if count > 1:
                    root = tk.Tk()
                    root.withdraw()
                    if messagebox.askokcancel("Error","Application already running. Do you want to go to the running instance?"):
                        os.system("start {}/main".format(addr))
                    os._exit(1)

        start = lambda: run(host='localhost', port=int(port), server='paste', reloader=False)
        thr = Thread(target=start)
        thr.setDaemon(True)
        thr.start()

        os.system("start {}/main".format(addr))

    else:
        print("Manager running in development mode...")
        run(host='localhost', port=int(port), server='paste', reloader=True)
