# -*- coding: utf-8 -*-
"""
Update manager metadata
Collate various bits of metadata needed for the model manager into single json
Created on Wed Jul 24 11:08:31 2019

@author: jp
"""

from collections import OrderedDict
import json
import pandas as pd

# Load variables metadata
var_meta = pd.read_csv("../utilities/titles/VariableListing.csv", index_col=0)
var_meta = var_meta.reset_index()
var_meta = var_meta.fillna("None")
var_meta = var_meta.rename(columns={"Variable name":"Variable",
                         "Variable description":"label",
                         "Unit":"unit",
                         "Domain":"Group"})
var_meta["label_short"] = var_meta["label"]
fields = ["Variable","label","unit","Group","Dim1","Dim2","Dim3","Dim4","Summary","Detailed"]
var_meta = var_meta.loc[:,fields]

var_meta = var_meta.set_index("Variable")

var_meta = var_meta[var_meta["Dim4"].isin(["BSM_TIME", "TIME"])]
labels = var_meta.to_dict("index")

var_indicies = var_meta.copy().reset_index()

var_indicies = var_indicies.loc[:,["Variable","label"]]
var_indicies.columns = ["id","label"]
indicies = var_indicies.to_dict("records")

groups = [[x] for x in var_meta.index]


## Load variables metadata
#assum_meta = pd.read_csv("Assumption_description.csv",index_col=0)
#
#assum_indicies = assum_meta.copy().reset_index()
#assum_indicies = assum_indicies.loc[:,["code","desc"]]
#assum_indicies.columns = ["value","text"]
#assum = assum_indicies.to_dict("records")
#
#
##Read in master list of specification definitions
#spec_desc = pd.read_csv("Specification_Descriptions.csv",
#                        index_col=[0,1],
#                        keep_default_na=False)

# Get valid specifications
#specs_dict = {}
#config = configparser.ConfigParser()
#config.read('../settings.ini')
#spec_no = int(config.get('settings', 'specno'))
#specs_indicies = []
#for i in range(spec_no):
#    var = config.get('settings', 'spec{}'.format(i+1)).split(' - ')[0]
#    var = var.lower()
#    if var == "bqxx":
#        continue
#    specs = config.get('settings', 'spec{}'.format(i+1)).split(' - ')[1].split(', ')
#
#    specs_indicies.append({"id": var, "label":labels[var]["label_short"] })
#    specs_dict[var] = []
#    # Compile a dictionary of permitted specifications for relevant vars
#    for key in specs:
#        dict_temp ={}
#        dict_temp["Name"] = key
#        if spec_desc.index.isin([(var, key)]).any():
#
#            dict_temp["Desc"] = spec_desc.loc[(var, key),"Desc"]
#        else:
#            dict_temp["Desc"] = spec_desc.loc[("Generic", key),"Desc"]
#        specs_dict[var].append(dict_temp)

out_dict = {}
#out_dict["indicies"] = indicies
out_dict["groups"] = groups
#out_dict["specs"] = specs_dict
out_dict["labels"] = labels
#out_dict["specs_indicies"] = specs_indicies
#out_dict["assumptions_desc"] = assum

# title dimensions
class_map = var_meta.copy().reset_index()
class_map = class_map.rename(columns={"Dim1":"title",
                                      "Dim2":"title2",
                                      "Dim3":"title3",
                                      "Dim4":"title4",
                                      "Variable":"Var"
                                      })
class_map = class_map.loc[:,["Var","title","title2","title3","title4"]]
class_map = class_map.set_index("Var")
class_map_dict = class_map.to_dict("index")

out_dict["title_map"] = class_map_dict

# chart colours
colour_list = pd.read_csv("ChartColours.csv")
colour_list = colour_list.astype("str")
colour_list['rgb'] = colour_list[["R", "G", "B"]].apply(lambda x: ','.join(x),
                                                        axis=1)
colour_list['rgb'] = colour_list['rgb'].apply(lambda x: f"rgb({x})")
colour_list = colour_list['rgb'].values
out_dict["chart_colours"] = colour_list.tolist()

#Update sector groupings
var_grouping = var_meta.copy().reset_index()
var_grouping_summary = var_grouping[var_grouping["Summary"]=="Y"]
var_grouping_detailed = var_grouping[var_grouping["Detailed"]=="Y"]
var_grouping = var_grouping.loc[:,["Group","Variable","label"]]
var_grouping_summary = var_grouping_summary.loc[:,["Group","Variable","label"]]
var_grouping_detailed = var_grouping_detailed.loc[:,["Group","Variable","label"]]
grouping_dict = {}

#grouping_dict = pd.read_excel("../utilities/titles/Grouping.xlsx",sheet_name=None)
grouping_dict["Variables"] = var_grouping
grouping_dict["Variables_Summary"] = var_grouping_summary
grouping_dict["Variables_Detailed"] = var_grouping_detailed
final_json = {}
for key, agg in grouping_dict.items():
    agg_dict = OrderedDict()
    agg_dict_labels = OrderedDict()
    agg_json = []
    if key in ["Variables", "Variables_Summary", "Variables_Detailed"]:
        items = agg["Group"].values
        groups = list(OrderedDict.fromkeys(items))
    else:
        agg = agg.set_index("Items")
        groups = list(agg.columns)

    for g in groups:
        if key in ["Variables", "Variables_Summary", "Variables_Detailed"]:
            agg_dict[g] = list(agg.loc[agg["Group"]==g,"Variable"].values)
            agg_dict_labels[g] = list(agg.loc[agg["Group"]==g,"label"].values)
        else:
            agg_mask = agg[g] == 1
            agg_dict[g] = list(agg.loc[agg_mask,g].index)
            agg_dict_labels[g] = agg_dict[g]


        #Covert to json format
        sub_dict = OrderedDict()
        sub_dict["id"] = g
        sub_dict["label"] = g
        if len(agg_dict[g]) > 1:
            sub_dict["children"] = []
            for v,var in enumerate(agg_dict[g]):
                child_dict = OrderedDict()
                child_dict["id"] = var
                child_dict["label"] = agg_dict_labels[g][v]
                sub_dict["children"].append(child_dict)
        agg_json.append(sub_dict)
    if "Hierachy" in agg.index:
        hierachy = agg.loc["Hierachy"]
        items = set(hierachy.values)
        agg_json_children = agg_json.copy()
        agg_json = []
        for i in items:
            hier_dict = OrderedDict()
            hier_dict["id"] = i
            hier_dict["label"] = i
            hier_dict["children"] = []
            children_mask = hierachy == i
            children = list(hierachy[children_mask].index)
            if len(children) >1:
                for c in children:
                    hier_dict["children"].append(agg_json_children[groups.index(c)])
            agg_json.append(hier_dict)

    final_json[key] = agg_json

#
with open('var_groupings.json', 'w') as fp:
    json.dump(final_json, fp)

out_dict["indicies"] = final_json["Variables"]
out_dict["indicies_summary"] = final_json["Variables_Summary"]
out_dict["indicies_detailed"] = final_json["Variables_Detailed"]


with open('..\\measures_meta.json', 'w') as fp:
    json.dump(out_dict, fp)
