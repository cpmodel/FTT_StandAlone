<template>
  <div class="container">
    <b-row>
      <div class="col-12 draw_area">
        <b-row>
         <b-col cols="8">
          <div class="selected_variable">{{selected.variable_label.label}}
           <b-badge class="info" :style="{display: isSet()? '' : 'none'}" v-b-tooltip.hover variant="info" size="sm" :title="selected.variable_label.desc">i</b-badge>
         </div>
        </b-col>
        <b-col cols="3" class="download">
          <b-col>
            <b-form-group>
              <b-form-radio-group
              id="btn-radios-1"
              v-model="chart_not_table"
              :options="[{text: 'Chart', value: true}, {text: 'Table', value: false}]"
              buttons
              name="radios-btn-default"
              button-variant="outline-primary"
              ></b-form-radio-group>
            </b-form-group>
          </b-col>
          <b-col cols="3">
          <b-button size="sm"  v-on:click="download_data">Download data</b-button>

          </b-col>
        </b-col>
      </b-row>
      <div class="overlay" :style="{display: isSet()? 'none' : 'block'}">
        <span>{{selection_warning}}</span>
      </div>

      <div id='line' :style="{display: chart_not_table? 'block' : 'none'}"></div>
      <div id="data-table-main" :style="{display: chart_not_table? 'none' : 'block'}">
      <b-table
        selectable small sticky-header
        head-variant="light"
        :items="JSON.parse(pivot)['items']"
        :fields="JSON.parse(table_fields)"
        responsive="sm"
        class='table table-hover'
        style="min-height:400px">
      </b-table>
    </div>

  </div>

  </b-row>
  <div class="row">
    <b-card class="box" id="controlpanel">
      <b-tabs card>
        <b-tab title="Scenario" class="tab-pane active " id="Scenario">
           <b-row>

            <b-table
              ref="selectableTable"
              selectable small sticky-header
              head-variant="light"
              :items="scenarios"
              :fields="fields"
              @row-selected="onRowSelected"
              :no-border-collapse="true"
              responsive="sm"
              class='table table-hover'>
            </b-table>
          </b-row>
          <b-row>
            <b-col cols=3>
              <label class="selection_label">Baseline to compare to:</label>
            </b-col>
            <b-col >
              <treeselect :multiple="false" v-model="selected.baseline" :options="scenarios_drop"/>
                </b-col>
            </b-row>

          </b-tab>

          <b-tab title="Indicator" class="tab-pane container smallselecttext" id="VariableSet">
 
            <b-row>
            
              <b-col cols="5">
            <b-row>
                
                           
            
              <label class="selection_label">Indicator</label>
              <b-form-radio-group
              id="btn-radios-2"
              size=sm
              inline=true
              v-model="indic_detail"
              :options="[{text: 'Simple', value: 0}, {text: 'Detail', value: 1},{text: 'All', value: 2}]"
           
              name="radios-btn-default"
              
              ></b-form-radio-group>
              <label class="labels"> Label </label> 
              <b-form-checkbox class="inline" v-model="code_label" name="check-button" switch>
              Code
              </b-form-checkbox>
            
            </b-row>
                <treeselect v-model="selected.variable" :multiple="false" :options="variables" :alwaysOpen="true" :clearable="false"
                :appendToBody="false" openDirection="below" :maxHeight="150" :sortValueBy="'INDEX'" >
                </treeselect>
  
              </b-col>
              <b-col >
                <b-row>
                  <b-col>
                    <label class="selection_label">Dim 1</label>
                  </b-col>
                </b-row>
                <b-row>
                    <b-col>
                    <b-form-checkbox class="inline" v-model="selected.aggregate" name="check-button" :disabled="no_aggregate" >
                      Agg?
                    </b-form-checkbox>
                  </b-col>
                  <b-col>
                    <b-form-checkbox class="inline" v-model="select_all" name="check-button" switch>
                       All?
                    </b-form-checkbox>
                  </b-col>
                </b-row>
                <div class="dimension_box">
                  <treeselect v-model="selected.dimensions" :multiple="true" :options="dimensions" :alwaysOpen="true" :clearable="true"
                  :showCount="false" :appendToBody="false" :defaultExpandLevel="Levels" :maxHeight="150" :sortValueBy="'LEVEL'" :limit="Levels"
                  openDirection="below" :flat="true" :limitText="tree_limit_text" />
                </div>
              </b-col>
              <b-col>
                <b-row>
                  <b-col>
                    <label class="selection_label">Dim 2</label>
                  </b-col>
                </b-row>
                <b-row>
                    <b-col>
                    <b-form-checkbox class="inline" v-model="selected.aggregate2" name="check-button" :disabled="no_aggregate">
                      Agg?
                    </b-form-checkbox>
                  </b-col>
                  <b-col>
                    <b-form-checkbox class="inline" v-model="select_all2" name="check-button" switch>
                       All?
                    </b-form-checkbox>
                  </b-col>
                </b-row>
                <div class="dimension_box">
                  <treeselect v-model="selected.dimensions2" :multiple="true" :options="dimensions2" :alwaysOpen="true" :clearable="true"
                  :showCount="false" :appendToBody="false" :defaultExpandLevel="Levels" :maxHeight="150" :sortValueBy="'INDEX'" :limit="Levels"
                  openDirection="below" :flat="true" search-nested :limitText="tree_limit_text" />
                </div>
              </b-col>
              <b-col>
                <b-row>
                  <b-col>
                    <label class="selection_label">Dim 3</label>
                  </b-col>
                </b-row>
                <b-row>
                    <b-col>
                    <b-form-checkbox class="inline" v-model="selected.aggregate3" name="check-button" :disabled="no_aggregate">
                      Agg?
                    </b-form-checkbox>
                  </b-col>
                  <b-col>
                    <b-form-checkbox class="inline" v-model="select_all3" name="check-button" switch>
                       All?
                    </b-form-checkbox>
                  </b-col>
                </b-row>
                <div class="dimension_box">
                  <treeselect v-model="selected.dimensions3" :multiple="true" :options="dimensions3" :alwaysOpen="true" :clearable="true"
                  :showCount="false" :appendToBody="false" :defaultExpandLevel="Levels" :maxHeight="150" :sortValueBy="'INDEX'" :limit="Levels"
                  openDirection="below" :flat="true" :limitText="tree_limit_text" />
                </div>
              </b-col>
            </b-row>
          </b-tab>

          <b-tab title="Settings" class="tab-pane container " id="ChartSet">
            <b-row>
              <b-col>
              <b-form-group  label="Color" class="selection_label">
                <b-form-radio-group v-b-tooltip.hover title="Dimension to seperate by colour" buttons button-variant="outline-primary"  id="color" v-model="color" name="radioColor" size="sm">
                  <b-form-radio  value="dimension" :disabled="isColourActive(1)">Dim1</b-form-radio>
                  <b-form-radio value="dimension2" :disabled="isColourActive(2)">Dim2</b-form-radio>
                  <b-form-radio value="scenario" :disabled="isColourActive(3)">scenario</b-form-radio>
                </b-form-radio-group>
              </b-form-group>
                <b-form-group label="Facet - y" class="selection_label">
                  <b-form-radio-group v-b-tooltip.hover title="Dimension to plot as seperate facet plots along y axis" buttons button-variant="outline-primary" id="facet" :options="[]" v-model="facet" name="radioFacet" size="sm">
                    <b-form-radio :disabled="isFacetActive(4)">None</b-form-radio>
                     <b-form-radio value="dimension" :disabled="isFacetActive(1)">Dim1</b-form-radio>
                    <b-form-radio value="dimension2" :disabled="isFacetActive(2)">Dim2</b-form-radio>
                    <b-form-radio value="scenario" :disabled="isFacetActive(3)">scenario</b-form-radio>
                  </b-form-radio-group>
                </b-form-group>
                <b-form-group label="Facet - x" class="selection_label">
                  <b-form-radio-group v-b-tooltip.hover title="Dimension to plot as seperate facet plots along x axis" buttons button-variant="outline-primary" id="facet" :options="[]" v-model="facet_x" name="radioFacet" size="sm">
                    <b-form-radio :disabled="isFacetActive_x(4)">None</b-form-radio>
                    <b-form-radio value="dimension" :disabled="isFacetActive_x(1)">Dim1</b-form-radio>
                    <b-form-radio value="dimension2" :disabled="isFacetActive_x(2)">Dim2</b-form-radio>
                    <b-form-radio value="scenario" :disabled="isFacetActive_x(3)">scenario</b-form-radio>
                  </b-form-radio-group>
                </b-form-group>

            </b-col>
            <b-col>
                <b-form-group label="Show"  class="selection_label top_margin">
                  <b-form-radio-group id="calculation" v-model="selected.calculation"
                  :options="calculation_methods" name="radioCalc">
                  </b-form-radio-group>
                </b-form-group>
                <b-form-group label="Start Year"  class="selection_label top_margin">
                  <b-form-select id="years" v-model="start_year"
                  :options="years">
                  </b-form-select>
                </b-form-group>
                <b-form-group label="End Year"  class="selection_label top_margin">
                  <b-form-select id="years" v-model="end_year"
                  :options="years">
                  </b-form-select>
                </b-form-group>
            </b-col>
         
            </b-row>
          </b-tab>

        </b-tabs>

      </b-card>
    </div>
  </div>
</template>

<script>
import lodash from "lodash";
import axios from "axios";
import * as taucharts from "taucharts";
import tau_tooltip from "taucharts/dist/plugins/tooltip";
import tau_quickfilter from "taucharts/dist/plugins/quick-filter";
import tau_legend from "taucharts/dist/plugins/legend";

import Treeselect from "@riophae/vue-treeselect";
import "@riophae/vue-treeselect/dist/vue-treeselect.css";
import { globalStore } from "../main.js";
var taucharts_tooltip = taucharts.api.plugins.get("tooltip")({
  formatters: {
    year: {
      label: "year",
      format: "%Y"
    }
  }
});
var config = {
  data: [
    { dimension: "us", y_real: 20, q_real: 23, year: "2013", dimension2: "a" }
  ],
  dimensions: {
    year: { type: "measure", scale: "time" },
    y_real: { type: "measure" },
    q_real: { type: "measure" },
    dimension: { type: "category" },
    dimension2: { type: "category" },
    dimension3: { type: "category" }
  },
  guide: [
    {
      y: { label: { text: "test" } },
      color: {
        brewer: [
          "rgb(197, 68, 110)",
          "rgb(73, 201, 197)",
          "rgb(170, 183, 29)",
          "rgb(0, 99, 152)",
          "rgb(0, 122, 178)",
          "rgb(100, 100, 100)"
        ]
      }
    },
    {
      y: { label: { text: "test" } },
      color: {
        brewer: [
          "rgb(197, 68, 110)",
          "rgb(73, 201, 197)",
          "rgb(170, 183, 29)",
          "rgb(0, 99, 152)",
          "rgb(0, 122, 178)",
          "rgb(100, 100, 100)"
        ]
      }
    }
  ],

  plugins: [taucharts_tooltip, tau_legend()],
  type: "line",
  x: "year",
  y: "y_real",
  order: "year",
  color: ["scenario", "dimension"] // there will be two lines with different colors on the chart
};
var chart = new taucharts.Chart(config);

export default {
  name: "Results",
  data: () => {
    return {
      pivot: "[]",
      pivot_columns: [],
      table_fields: "[]",
      facet: null,
      facet_x: null,
      color: null,
      json_: [],
      display_data: [],
      error: "",
      chart: {},
      chart_colours: [],
      variables: [],
      variables_summary: [],
      variables_detailed: [],
      variables_all: [],
      var_labels: [],
      dimensions: [],
      dimensions2: [],
      dimensions3: [],
      json_vars: [],
      var_groups: [],
      scenarios: [],
      years: [],
      start_year:2000,
      end_year:2050,
      fields: [
        { key: "scenario", sortable: true },
        { key: "description", sortable: true },
        { key: "Last Run", sortable: true }
      ],
      scenarios: [],
      selected_table: [],
      titles: [],
      quickfilter_on: false,
      settings_loaded: true,
      select_all: false,
      select_all2: false,
      select_all3: false,
      loading: false,
      first_run: true,
      chart_not_table: true,
      indic_detail: 0,
      code_label: false,
      no_aggregate: false,
      dimensions_drawn: [],
      dimensions2_drawn: [],
      dimensions3_drawn: [],
      scenarios_drawn: ["S0"],
      title_old: "null",
      title2_old: "null",
      title3_old: "null",
      selection_warning:
        "Please set the scenario, variable and dimensions to display results.",
      selected: {
        variable: "",
        variable_label: {},
        dimensions: [],
        dimensions2: [],
        dimensions3: [],
        title: "",
        title2: "",
        title3: "",
        time: "Yes",
        scenarios: ["S0"],
        aggregate: false,
        aggregate2: false,
        aggregate3: false,
        baseline: "S0",
        calculation: "Levels",
        scen_index: [0]
      },
      calculation_methods: [
        { text: "Absolute value", value: "Levels" },
        { text: "Absolute difference from baseline", value: "absolute_diff" },
        { text: "Percentage difference from baseline", value: "perc_diff" },
        { text: "Year-on-Year change", value: "Annual growth rate" }
      ],
      aggregation_methods: [
        { text: "None", value: "0" },
        { text: "Sum", value: "sum" }
      ]
    };
  },
  components: { Treeselect },
  methods: {
    initialise: function() {
      //Get available variables
      this.get_vars();

      axios.get("http://localhost:5000/api/scenarios_ran").then(res => {
        let scenarios = res.data.exist;
        this.scenarios = scenarios;
        let years = res.data.years
        this.years = years
        this.scenarios_drop = scenarios.map(s => {
                return { label: s["scenario"], id: s["scenario"] }
        })
      });
    },
    onRowSelected(items) {
      this.selected_table = items;
      this.selected.scenarios = [];
      this.selected.scen_index = [];
      for (var i = 0; i < this.selected_table.length; i++) {
        if (
          this.selected.scenarios.includes(this.selected_table[i]["scenario"]) == false) {
          this.selected.scenarios.push(this.selected_table[i]["scenario"]);
        }
        for (var j = 0; j < this.scenarios.length; j++) {
          if ( this.selected_table[i]["scenario"] == this.scenarios[j]["scenario"])
            this.selected.scen_index.push(j);
        }
      }
    },
    selectFirstRow() {
      // Rows are indexed from 0, so the third row is index 2
      for (var j = 0; j < this.selected.scen_index.length; j++) {
        this.$refs.selectableTable.selectRow(this.selected.scen_index[j]);
      }
    },
    get_vars: function() {
      axios.get("http://localhost:5000/api/results/variables").then(
        res => {
          let variables = res.data.vars.indicies;
          this.var_groups = res.data.vars.groups;
          this.var_labels = res.data.vars.labels;
          this.variables_all = variables;
          this.variables_detailed = res.data.vars.indicies_detailed;
          this.variables_summary = res.data.vars.indicies_summary;
          this.variables = this.variables_summary;
          this.titles = res.data.vars.title_map;
          this.chart_colours = res.data.vars.chart_colours;
        },
        err => {
          this.error_message(err, "variables");
        }
      );
    },
    get_dimensions: function() {
      if (this.selected.scenarios.length == 0) return;

      this.selected.titles = this.titles[this.selected.variable];
      if (this.selected.titles["title4"] == "TIME") {
        this.selected.time = "Yes";
      } else {
        this.selected.time = "No";
      }

      axios
        .get(`http://localhost:5000/api/info/${this.selected.titles["title"]}`)
        .then(
          res => {
            if ("Sectors" in res.data) {
              this.dimensions = res.data.Sectors;
            } else {
              let dimensions = res.data;
              this.dimensions = dimensions.map(s => {
                return { label: s, id: s };
              });
            }
          },
          err => {
            this.error_message(err, "dimensions");
          }
        );
      axios
        .get(`http://localhost:5000/api/info/${this.selected.titles["title2"]}`)
        .then(
          res => {
            if ("Sectors" in res.data) {
              this.dimensions2 = res.data.Sectors;
            } else {
              let dimensions2 = res.data;
              this.dimensions2 = dimensions2.map(s => {
                return { label: s, id: s };
              });
            }
          },
          err => {
            this.error_message(err, "dimensions2");
          }
        );

      axios
        .get(`http://localhost:5000/api/info/${this.selected.titles["title3"]}`)
        .then(
          res => {
            if ("Sectors" in res.data) {
              this.dimensions3 = res.data.Sectors;
            } else {
              let dimensions3 = res.data;
              this.dimensions3 = dimensions3.map(s => {
                return { label: s, id: s };
              });
            }
          },
          err => {
            this.error_message(err, "dimensions");
          }
        );

      this.selected.title = this.titles[this.selected.variable]["title"];
      this.selected.title2 = this.titles[this.selected.variable]["title2"];
      this.selected.title3 = this.titles[this.selected.variable]["title3"];
    },
    update_chart: function() {
      // Prevents update chart calling prematurely

      if (this.selected.time == "Yes") {
        (config.x = "year"), (config.type = "line");
      } else {
        (config.x = "dimension3"), (config.type = "line");
      }
      // Prevents update chart calling prematurely
      if (this.isSet() == false) return;

      if (this.isdrawn()) return;
      if (this.var_groups.length == 0) return;
      if (this.loading == true) return;
      if (this.facet == this.color && this.facet != null) return;
      if (this.facet_x == this.color && this.color != null) return;
      if (this.facet == this.facet_x && this.facet_x != null) return;
      let vars = this.var_groups.filter(
        g => g.indexOf(this.selected.variable) != -1
      );
      let params = lodash.cloneDeep(this.selected);

      params["variable"] = [this.selected.variable];
      params["title"] = [this.selected.title];
      params["title2"] = [this.selected.title2];
      params["title3"] = [this.selected.title3];
      params["variable_label"] = [this.selected.variable_label];
      params["Start_Year"] = this.start_year
      params["End_Year"] = this.end_year

      axios
        .get(`http://localhost:5000/api/results/data/json`, { params: params })
        .then(
          res => {
            this.json_ = res.data;
          },
          err => {
            this.error_message(err, "requested chart data");
          }
        );

      this.title_update();

      chart.updateConfig(config);
      this.title_fix();
      if (this.first_run == true) {
        this.selectFirstRow();
      }
      this.first_run = false;
      this.dimensions_drawn = this.selected.dimensions;
      this.dimensions2_drawn = this.selected.dimensions2;
      this.dimensions3_drawn = this.selected.dimensions3;
      this.scenarios_drawn = this.selected.scenarios;
    },
    download_data: function() {
      let params = lodash.cloneDeep(this.selected);

      params["variable"] = [this.selected.variable];
      params["title"] = [this.selected.title];
      params["title2"] = [this.selected.title2];
      params["title3"] = [this.selected.title3];
      params["variable_label"] = [this.selected.variable_label];

      axios
        .get(`http://localhost:5000/api/results/data/csv`, { params: params })
        .then(
          res => {
            let blob = new Blob([res.data], { type: "text/csv" }),
              url = window.URL.createObjectURL(blob);
            var fileLink = document.createElement("a");
            fileLink.href = url;
            fileLink.download = `FTT_data_download_${new Date().toISOString()}.csv`;
            fileLink.click();
          },
          err => {
            this.error_message(err, "download data");
          }
        );
    },
    update_table: function() {
      let table_fields = [
        {
          key: "scenario",
          sortable: false,
          stickyColumn: true,
          isRowHeader: true
        },
        {
          key: "dimension",
          sortable: false,
          stickyColumn: true,
          isRowHeader: true
        },
        {
          key: "dimension2",
          sortable: false,
          stickyColumn: true,
          isRowHeader: true
        },
        {
          key: this.selected.time == "Yes" ? "dimension3" : "year",
          sortable: false,
          stickyColumn: true,
          isRowHeader: true
        }
      ];
      if (this.pivot.length > 0) {
        Object.keys(this.pivot[0])
          .slice(0, Object.keys(this.pivot[0]).length - 3)
          .forEach(y => {
            table_fields.push({ key: y });
            // }
          });
      }
      this.table_fields = JSON.stringify(table_fields);
      this.pivot = JSON.stringify({ items: this.pivot });

      // localStorage.variable = this.selected.variable_label.label
    },
    isSet: function() {
      if (this) {
        return (
          (this.selected.variable != "") &
          (this.selected.dimensions.length > 0) &
          (this.selected.dimensions2.length > 0) &
          (this.selected.dimensions3.length > 0) &
          (this.selected.scenarios.length > 0)
        );
      } else {
        return false;
      }
    },
    isdrawn: function() {
      if (
        this.arraysEqual(this.selected.dimensions, this.dimensions_drawn) ==
        false
      )
        return false;
      if (
        this.arraysEqual(this.selected.dimensions2, this.dimensions_drawn2) ==
        false
      )
        return false;
      if (
        this.arraysEqual(this.selected.dimensions3, this.dimensions_drawn3) ==
        false
      )
        return false;
      if (
        this.arraysEqual(this.selected.scenarios, this.scenarios_drawn) == false
      )
        return false;

      return true;
    },
    arraysEqual: function(a, b) {
      if (a === b) return true;
      if (a == null || b == null) return false;
      if (a.length !== b.length) return false;

      // If you don't care about the order of the elements inside
      // the array, you should sort both arrays here.
      // Please note that calling sort on an array will modify that array.
      // you might want to clone your array first.

      for (var i = 0; i < a.length; ++i) {
        if (a[i] !== b[i]) return false;
      }
      return true;
    },
    isFacetActive: function(facet) {
      var multiples = [];
      if (this.selected.scenarios.length > 1) multiples.push("scenario");
      if (this.selected.dimensions.length > 1) multiples.push("dimension");
      if (this.selected.dimensions2.length > 1) multiples.push("dimension2");
      if (facet == 4) {
        return multiples.length >= 2;
      } else if (multiples.length == 1) return true;
      else if (facet == 1) return this.selected.dimensions.length <= 1;
      else if (facet == 2) return this.selected.dimensions2.length <= 1;
      else if (facet == 3) return this.selected.scenarios.length <= 1;
      else if (facet == 4) return multiples.length >= 2;
      else return false;
    },
    isFacetActive_x: function(facet_x) {
      var multiples = [];
      if (this.selected.scenarios.length > 1) multiples.push("scenario");
      if (this.selected.dimensions.length > 1) multiples.push("dimension");
      if (this.selected.dimensions2.length > 1) multiples.push("dimension2");
      if (facet_x == 1) {
        if (multiples.length != 3) return true;
        else return false;
      } else if (facet_x == 2) {
        if (multiples.length != 3) return true;
        else return false;
      } else if (facet_x == 3) {
        if (multiples.length != 3) return true;
        else return false;
      } else if (facet_x == 4) {
        if (multiples.length == 3) return true;
        else return false;
      } else return false;
    },
    isColourActive: function(colour) {
      if (colour == 1) return this.selected.dimensions.length <= 1;
      else if (colour == 2) return this.selected.dimensions2.length <= 1;
      else if (colour == 3) return this.selected.scenarios.length <= 1;
      else return false;
    },

    title_update: function() {
      let full_axis_label = "";

      let unit = "";
      if (
        this.selected.calculation == "absolute_diff" ||
        this.selected.calculation == "perc_diff"
      )
        full_axis_label = "Difference from baseline";
      else if (this.selected.calculation == "Annual growth rate")
        full_axis_label = "Year on Year growth";
      if (
        this.selected.calculation == "Levels" ||
        this.selected.calculation == "absolute_diff"
      )
        unit = this.selected.variable_label.unit;
      else unit = "%";

      full_axis_label = full_axis_label + " (" + unit + ")";
      var dimension_text = "Scenarios: " + this.selected.scenarios;
      if (this.selected.dimensions.length > 4) {
        dimension_text =
          dimension_text +
          " / Elements: " +
          "Multiple selected (" +
          this.selected.dimensions.length +
          ")";
      } else {
        dimension_text =
          dimension_text + " / Elements: " + this.selected.dimensions;
      }

      if (this.facet != null) {
        config.guide = [
          {
            y: { label: { text: full_axis_label } },
            color: {
              brewer: this.chart_colours
            }
          },
          {
            y: { label: { text: this.facet } },
            x: { label: { text: dimension_text } },
            color: {
              brewer: this.chart_colours
            }
          }
        ];
      } else {
        config.guide = {
          y: { label: { text: full_axis_label } },
          x: { label: { text: dimension_text } },
          color: {
            brewer: this.chart_colours
          }
        };
      }
    },
    tree_limit_text: function() {
      return "";
    },
    title_fix: function() {
      // Warning crude fix to get regular refresh of axis labels by searching through class properties
      let full_axis_label = "";

      let unit = "";
      if (
        this.selected.calculation == "absolute_diff" ||
        this.selected.calculation == "perc_diff"
      )
        full_axis_label = "Difference from baseline";
      else if (this.selected.calculation == "Annual growth rate")
        full_axis_label = "Year on Year growth";
      if (
        this.selected.calculation == "Levels" ||
        this.selected.calculation == "absolute_diff"
      )
        unit = this.selected.variable_label.unit;
      else unit = "%";

      full_axis_label = full_axis_label + " (" + unit + ")";
      var dimension_text = "Scenarios: " + this.selected.scenarios;
      if (this.selected.dimensions.length > 4) {
        dimension_text =
          dimension_text +
          " / Elements: " +
          "Multiple selected (" +
          this.selected.dimensions.length +
          ")";
      } else {
        dimension_text =
          dimension_text + " / Elements: " + this.selected.dimensions;
      }
      if (document.getElementsByClassName("label-token").length > 0) {
        document.getElementsByClassName(
          "label-token"
        )[0].innerHTML = dimension_text;
      }

      if (document.getElementsByClassName("label-token").length == 2) {
        document.getElementsByClassName(
          "label-token"
        )[1].innerHTML = full_axis_label;
      } else if (
        this.facet != null &&
        this.facet_x == null &&
        document.getElementsByClassName("label-token").length > 2
      ) {
        document.getElementsByClassName(
          "label-token"
        )[2].innerHTML = full_axis_label;
      } else if (
        this.facet_x != null &&
        document.getElementsByClassName("label-token").length > 3
      ) {
        document.getElementsByClassName(
          "label-token"
        )[3].innerHTML = full_axis_label;
      }
    },
    check_facet_color: function() {
      var multiples = [];
      if (this.selected.scenarios.length > 1) multiples.push("scenario");
      if (this.selected.dimensions.length > 1) multiples.push("dimension");
      if (this.selected.dimensions2.length > 1) multiples.push("dimension2");
      var nulls = [];
      if (this.color == null) nulls.push("color");
      if (this.facet == null) nulls.push("facet");
      if (this.facet_x == null) nulls.push("facet_x");

      // Only colour is needed
      if (multiples.length == 1) {
        this.color = multiples[0];
        this.facet = null;
        this.facet_x = null;
      }
      // Color and facet y is needed
      else if (multiples.length == 2) {
        this.color = multiples[0];
        this.facet = multiples[1];
        this.facet_x = null;
      }
      // All facets needed
      else if ((multiples.length == 3) & (nulls.length > 0)) {
        this.color = multiples[0];
        this.facet = multiples[1];
        this.facet_x = multiples[2];
      }
    },
    check_facet_color_swap: function(dim_check, value) {
      var multiples = [];
      if (this.selected.scenarios.length > 1) multiples.push("scenario");
      if (this.selected.dimensions.length > 1) multiples.push("dimension");
      if (this.selected.dimensions.length > 1) multiples.push("dimension2");
      if (multiples.length == 1) return;
      if (dim_check == "facet") {
        if (value == this.color) {
          this.color = multiples.filter(
            e => (e !== value) & (e !== this.facet_x)
          )[0];
        } else if (value == this.facet_x) {
          this.facet_x = multiples.filter(
            e => (e !== value) & (e !== this.color)
          )[0];
        }
      } else if (dim_check == "color") {
        if (value == this.facet) {
          this.facet = multiples.filter(
            e => (e !== value) & (e !== this.facet_x)
          )[0];
        } else if (value == this.facet_x) {
          this.facet_x = multiples.filter(
            e => (e !== value) & (e !== this.facet)
          )[0];
        }
      } else if (dim_check == "facet_x") {
        if (value == this.color) {
          this.color = multiples.filter(
            e => (e !== value) & (e !== this.facet)
          )[0];
        } else if (value == this.facet) {
          this.facet = multiples.filter(
            e => (e !== value) & (e !== this.color)
          )[0];
        }
      }
    },
    selection_warning_update: function() {
      this.selection_warning = "Please set the ";
      var iCount = 0;
      iCount += this.selected.dimensions.length == 0 ? 1 : 0;
      iCount += this.selected.dimensions2.length == 0 ? 1 : 0;
      iCount += this.selected.scenarios.length == 0 ? 1 : 0;
      if (this.selected.dimensions.length == 0) {
        this.selection_warning = this.selection_warning + "Dim 1(s)";
        if ((iCount == 2) & (this.selected.dimensions.length == 0))
          this.selection_warning = this.selection_warning + " and ";
        else if (iCount == 3)
          this.selection_warning = this.selection_warning + ", ";
      }
      if (this.selected.dimensions2.length == 0)
        this.selection_warning = this.selection_warning + "Dim 2(s)";
      if (this.selected.scenarios.length == 0) {
        if (iCount > 1)
          this.selection_warning = this.selection_warning + " and ";
        this.selection_warning = this.selection_warning + "scenario(s)";
      }
    },
    saveselection: function() {
      if (globalStore.saved == false) {
        globalStore.results_selected = this.selected;
        globalStore.saved = true;
      }
    },

    error_message: function(err, item) {
      var msg = "";
      if (!err.response) {
        var msg =
          "Failed to get " +
          item +
          ", the manager is either not running or encountered an error (Check backend is running).";
      } else {
        msg =
          "Failed to get " +
          item +
          ", the manager is either not running or encountered an error." +
          "(" +
          err.response.statusText +
          ")";
      }
      alert(msg);
    }
  },
  beforeMount() {
    this.initialise();
  },
  mounted() {
    localStorage.pivot = "";
    localStorage.table_fields = "";
    localStorage.Variable = "";
    chart.renderTo(document.getElementById("line"));
  },
  beforeUpdate() {},
  beforeDestroy() {
    //this.saveselection()
  },
  watch: {
    variables: function() {
      if (
        this.settings_loaded == true &&
        this.selected.variable != this.variables[0].children[0].id
      ) {
        this.selected.variable = this.variables[0].children[0].id;

        this.get_dimensions();
      }
    },
    years: function(){
      this.start_year = this.years[0]
      this.end_year = this.years[this.years.length - 1]
    },
    start_year: function(){
      this.update_chart();
    },
    end_year: function(){
      this.update_chart();
    },
    scenarios: function() {},
    dimensions: function() {
      if (this.dimensions.length == 1)
        this.selected.dimensions = [this.dimensions[0].id];
      else if (
        this.selected.title == this.title_old &&
        this.dimensions_drawn.length > 0
      )
        this.selected.dimensions = this.dimensions_drawn;
      else if (this.settings_loaded == true)
        this.selected.dimensions = [this.dimensions[0].id];
      if (this.selected.title != "") this.title_old = this.selected.title;
    },
    dimensions2: function() {
      if (this.dimensions2.length == 1)
        this.selected.dimensions2 = [this.dimensions2[0].id];
      else if (
        this.selected.title2 == this.title2_old &&
        this.dimensions2_drawn.length > 0
      )
        this.selected.dimensions2 = this.dimensions2_drawn;
      else if (this.settings_loaded == true)
        this.selected.dimensions2 = [this.dimensions2[0].id];
      if (this.selected.title2 != "") this.title2_old = this.selected.title2;
    },
    dimensions3: function() {
      if (this.dimensions3.length == 1)
        this.selected.dimensions3 = [this.dimensions3[0].id];
      else if (
        this.selected.title3 == this.title3_old &&
        this.dimensions3_drawn.length > 0
      )
        this.selected.dimensions3 = this.dimensions3_drawn;
      else if (this.settings_loaded == true)
        this.selected.dimensions3 = [this.dimensions3[0].id];
      if (this.selected.title3 != "") this.title3_old = this.selected.title3;
    },
    chart_not_table: function() {
      if (this.chart_not_table) this.update_chart();
    },
    "selected.scenarios": function() {
      this.selection_warning_update();
      this.update_chart();
    },

    "selected.variable": function() {
      this.loading = true;
      if (this.selected.variable != "") {
        this.selected.dimensions = [];
        this.selected.dimensions2 = [];
        this.selected.dimensions3 = [];
        this.get_dimensions();
      }
      this.selected.variable_label = this.var_labels[this.selected.variable];
      if (this.selected.variable_label.unit.includes("/"))
        this.no_aggregate = true;
      else this.no_aggregate = false;
      this.loading = false;
      if (this.isSet) this.update_chart();
    },
    "selected.dimensions": function() {
      if (this.isSet) {
        this.update_chart();
        this.selection_warning_update();
      }
    },
    "selected.dimensions2": function() {
      if (this.isSet) {
        this.update_chart();
        this.selection_warning_update();
      }
    },
    "selected.dimensions3": function() {
      if (this.isSet) {
        this.update_chart();
        this.selection_warning_update();
      }
    },
    "selected.aggregation": function() {
      this.update_chart();
    },
    "selected.calculation": function() {
      this.update_chart();
    },
    "selected.baseline": function() {
      this.update_chart();
    },
    "selected.aggregate": function() {
      this.update_chart();
    },
    "selected.aggregate2": function() {
      this.update_chart();
    },
    "selected.aggregate3": function() {
      this.update_chart();
    },
    select_all: function() {
      this.selected.dimensions = [];
      if (this.select_all) {
        for (var i = 0; i < this.dimensions.length; i++)
          this.selected.dimensions.push(this.dimensions[i].id);
      } else this.selected.dimensions.push(this.dimensions[0].id);
    },
    select_all2: function() {
      this.selected.dimensions2 = [];
      if (this.select_all2) {
        for (var i = 0; i < this.dimensions2.length; i++)
          this.selected.dimensions2.push(this.dimensions2[i].id);
      } else this.selected.dimensions2.push(this.dimensions2[0].id);
    },
    select_all3: function() {
      this.selected.dimensions3 = [];
      if (this.select_all3) {
        for (var i = 0; i < this.dimensions3.length; i++)
          this.selected.dimensions3.push(this.dimensions3[i].id);
      } else this.selected.dimensions3.push(this.dimensions3[0].id);
    },
    indic_detail: function() {
      if (this.indic_detail == 0) this.variables = this.variables_summary;
      else if (this.indic_detail == 1) this.variables = this.variables_detailed;
      else if (this.indic_detail == 2) this.variables = this.variables_all;
      if (this.code_label)
        this.variables = this.variables.map(s => {
          return {
            label: s.id,
            id: s.label,
            children: s.children.map(v => {
              return { label: v.id, id: v.id };
            })
          };
        });
    },
    code_label: function() {
      if (this.indic_detail == 0) this.variables = this.variables_summary;
      else if (this.indic_detail == 1) this.variables = this.variables_detailed;
      else if (this.indic_detail == 2) this.variables = this.variables_all;
      if (this.code_label)
        this.variables = this.variables.map(s => {
          return {
            label: s.id,
            id: s.label,
            children: s.children.map(v => {
              return { label: v.id, id: v.id };
            })
          };
        });
    },
    json_: function() {
      this.pivot = JSON.parse(this.json_.pivot);
      this.display_data = JSON.parse(this.json_.results);
      // this.pivot_columns = this.json_.pivot_columns.filter(v => ['county','dimension','scenario'].indexOf(v) == -1)
      if (this.selected.time == "Yes")
        this.json_vars = this.json_.info.filter(
          v =>
            [
              "year",
              "dimension",
              "dimension2",
              "dimension3",
              "scenario"
            ].indexOf(v) == -1
        );
      else
        this.json_vars = this.json_.info.filter(
          v =>
            [
              "dimension3",
              "dimension",
              "dimension2",
              "dimension2",
              "scenario"
            ].indexOf(v) == -1
        );

      if (this.selected.dimensions.length > 0) {
        this.update_table();
      }
    },
    display_data: function() {
      if (globalStore.saved == true) return;
      if (this.loading == true) return;
      config.data = this.display_data;
      if (this.selected.time == "Yes") {
        config.dimensions = {
          year: { type: "measure", scale: "time" },
          dimension: { type: "category" },
          dimension2: { type: "category" },
          scenario: { type: "category" }
        };
      } else {
        config.dimensions = {
          dimension3: { type: "catergory" },
          dimension: { type: "category" },
          dimension2: { type: "category" },
          scenario: { type: "category" }
        };
      }
      for (var i = 0; i < this.json_vars.length; i++)
        config.dimensions[this.json_vars[i]] = { type: "measure" };

      this.check_facet_color();

      config.color = this.color;
      config.y = this.facet ? ["variables", this.facet] : ["variables"];
      config.x = this.facet_x ? ["year", this.facet_x] : ["year"];

      if (this.quickfilter_on)
        config.plugins = [
          taucharts_tooltip,
          tau_legend(),
          tau_quickfilter(this.json_vars)
        ];
      else config.plugins = [taucharts_tooltip, tau_legend()];

      this.title_update();
      chart.updateConfig(config);
      this.title_fix();
    },
    facet: function() {
      if ((this.facet == this.color) | (this.facet == this.facet_x))
        this.check_facet_color_swap("facet", this.facet);
      config.y = this.facet ? ["variables", this.facet] : ["variables"];
      if (this.quickfilter_on)
        config.plugins = [
          taucharts_tooltip,
          tau_legend(),
          tau_quickfilter(this.json_vars)
        ];
      else config.plugins = [taucharts_tooltip, tau_legend()];
      this.title_update();
      chart.updateConfig(config);
      this.title_fix();
    },
    facet_x: function() {
      if ((this.facet_x == this.facet) | (this.facet_x == this.color))
        this.check_facet_color_swap("facet_x", this.facet_x);
      config.x = this.facet_x ? ["year", this.facet_x] : ["year"];

      chart.updateConfig(config);
      this.title_fix();
    },
    color: function() {
      if ((this.color == this.facet) | (this.color == this.facet_x))
        this.check_facet_color_swap("color", this.color);
      config.color = this.color;
      if (this.quickfilter_on)
        config.plugins = [
          taucharts_tooltip,
          tau_legend(),
          tau_quickfilter(this.json_vars)
        ];
      else config.plugins = [taucharts_tooltip, tau_legend()];
      this.title_update();
      chart.updateConfig(config);
      this.title_fix();
    },
    quickfilter_on: function() {
      if (this.quickfilter_on)
        config.plugins = [
          taucharts_tooltip,
          tau_legend(),
          tau_quickfilter(this.json_vars)
        ];
      else config.plugins = [taucharts_tooltip, tau_legend()];
      chart.updateConfig(config);
      this.title_fix();
    }
  }
};
</script>

<!-- Add "scoped" attribute to limit CSS to this component only -->
<style scoped lang="scss">
.box {
  background-color: #e5e5e5;
  width: 100%;
  height: 315px;
}
// #controlpanel{
//   z-index: 1100;
// }
.overlay {
  background-color: rgba(255, 255, 255, 1);
  position: absolute;
  left: 0;
  width: 100%;
  height: 90%;
  z-index: 1000;
  transition: 1.5s;
  span {
    margin-top: 25%;
    display: block;
    font-size: 24px;
    font-weight: 700;
  }
}
.table {
  background-color: whitesmoke;
  height: 175px;
  overflow-y: scroll;
}
.content {
  height: 100%;
  // overflow-y: auto;
}
.dimension_box {
  height: 150px;
}
.draw_area {
  height: 100%;
}
.card-body {
  padding-top: 0.1rem;
  padding-bottom: 0.1rem;
}
body,
#line {
  height: 50vh;
}
h1 {
  font-size: 48px;
  text-align: left;
  font-weight: 900;
}
.inline {
  display: inline;
}
.selection_label {
  font-size: 14px;
  font-weight: 700;
  text-align: left;
  width: 100%;
  margin-top: 10px;
  * {
    font-weight: 400;
  }
}
.download {
  margin-top: 10px;
  padding: 0.2em;
  display: flex;
  align-items: center;
}
.selected_variable {
  font-size: 30px;
  font-weight: 700;
  text-align: left;
  padding: 0.25em;
}
.info {
  border-radius: 30% !important;
  font-size: 14px;
}

.button_ow {
  color: white;
  &:hover {
    text-decoration: none;
  }
}
.smallselecttext {
  font-size: 0.85rem;
}
.tau-chart__filter__wrap .resize.w text {
  text-anchor: start;
  font-size: 12px;
}

#data-table-main {
  font-size: 0.75em;
}
.tau-chart__tooltip {
  z-index: 1100;
}
</style>

<style lang="css" scoped>
   @import 'http://cdn.jsdelivr.net/npm/taucharts@2/dist/taucharts.min.css'
</style>
