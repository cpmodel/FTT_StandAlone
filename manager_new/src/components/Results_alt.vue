<template>
  <div class="container">
    <b-row>
      <b-col cols="4">
        <b-card class="box" id="controlpanel">
          <b-tabs card >
            <b-tab title="Scenario" class="tab-pane active " id="Scenario">
              <b-col>
                <b-table
                  ref="selectableTable"
                  selectable
                  small
                  sticky-header
                  head-variant="light"
                  :items="scenarios"
                  :fields="fields"
                  @row-selected="onRowSelected"
                  :no-border-collapse="true"
                  responsive="sm"
                  class="table table-hover"
                >
                </b-table>
              </b-col>
              <b-col>
                <b-row rows="3">
                  <label class="selection_label">Baseline to compare to:</label>
                </b-row>
                <b-row>
                  <treeselect
                    :multiple="false"
                    v-model="selected.baseline"
                    :options="scenarios_drop"
                  />
                </b-row>
              </b-col>
            </b-tab>

            <b-tab
              title="Indicator"
              class="tab-pane container smallselecttext"
              active
              id="VariableSet"
            >
              <b-row>
                <label class="selection_label">Indicator</label>
              <label class="labels"> Label </label> 
              <b-form-checkbox class="inline" v-model="code_label" name="check-button" switch>
              Code
              </b-form-checkbox>
                <div class="indic_box">
                  <treeselect
                    v-model="selected.variable"
                    :disable-branch-nodes="true"
                    :options="variables"
                    :alwaysOpen="true"
                    :clearable="false"
                    :appendToBody="false"
                    openDirection="below"
                    :maxHeight="95"
                    :sortValueBy="'INDEX'"
                    :clearOnSelect="true"
                    :levels=2
                  >
                  </treeselect>
                </div>
              </b-row>
              <b-row>
                <label class="selection_label">{{ this.selected.title }}</label>
                <b-form-checkbox
                  class="inline"
                  v-model="selected.aggregate"
                  name="check-button"
                  :disabled="no_aggregate"
                >
                  Aggregate
                </b-form-checkbox>
                <b-col>
                  <b-form-checkbox
                    class="inline"
                    v-model="select_all"
                    name="check-button"
                    switch
                  >
                    All
                  </b-form-checkbox>
                </b-col>

                <div class="dimension_box">
                  <treeselect
                    v-model="selected.dimensions"
                    
                    :multiple="true"
                    :options="dimensions"
                    :disable-branch-nodes="no_aggregate"
                    :alwaysOpen="true"
                    :clearable="true"
                    :showCount="false"
                    :appendToBody="false"
                    :maxHeight="100"
                    :sortValueBy="'LEVEL'"
                    :limit=Levels
                    openDirection="below"
                    :flat="true"
                    :limitText="tree_limit_text"
                    :clearOnSelect="true"
                  />
                </div>
              </b-row>

              <b-row>
                <label class="selection_label">{{
                  this.selected.title2
                }}</label>

                <b-form-checkbox
                  class="inline"
                  v-model="selected.aggregate2"
                  name="check-button"
                  :disabled="no_aggregate2"
                >
                  Aggregate
                </b-form-checkbox>

                <b-col>
                  <b-form-checkbox
                    class="inline"
                    v-model="select_all2"
                    name="check-button"
                    switch
                  >
                    All
                  </b-form-checkbox>
                </b-col>

                <div class="dimension_box">
                  <treeselect
                    v-model="selected.dimensions2"
                    :multiple="true"
                    :options="dimensions2"
                    :alwaysOpen="true"
                    :clearable="true"
                    :showCount="false"
                    :appendToBody="false"
                    :maxHeight="100"
                    :sortValueBy="'LEVEL'"
                    :limit=Levels
                    openDirection="below"
                    :flat="true"
                    :limitText="tree_limit_text"
                    :clearOnSelect="true"
                  />
                </div>
              </b-row>
              <b-row>
                <label class="selection_label">{{
                  this.selected.title3
                }}</label>

                <b-form-checkbox
                  class="inline"
                  v-model="selected.aggregate3"
                  name="check-button"
                  :disabled="no_aggregate3"
                >
                  Aggregate
                </b-form-checkbox>

                <b-col>
                  <b-form-checkbox
                    class="inline"
                    v-model="select_all3"
                    name="check-button"
                    switch
                  >
                    All
                  </b-form-checkbox>
                </b-col>

                <div class="dimension_box">
                  <treeselect
                    v-model="selected.dimensions3"
                    :multiple="true"
                    :options="dimensions3"
                    :alwaysOpen="true"
                    :clearable="true"
                    :showCount="false"
                    :appendToBody="false"
                    :maxHeight="100"
                    :sortValueBy="'LEVEL'"
                    :limit=Levels
                    openDirection="below"
                    :flat="true"
                    :limitText="tree_limit_text"
                    :clearOnSelect="true"
                  />
                </div>
              </b-row>
            </b-tab>

            <b-tab title="Settings" class="tab-pane container" id="ChartSet">
              <b-col>
                <b-row>
                  <b-form-group label="Color" class="selection_label ">
                    <b-form-radio-group 
                      v-b-tooltip.hover
                      title="Dimension to seperate by colour"
                      buttons
                      button-variant="light"
                      id="color"
                      v-model="color"
                      name="radioColor"
                      size="sm"
                      class="cedefop_button"
                    >
                      <b-form-radio 
                        :value=this.selected.title
                        :disabled="isColourActive(1)"
                        >{{this.selected.title}}</b-form-radio
                      >
                      <b-form-radio
                        :value=this.selected.title2
                        :disabled="isColourActive(2)"
                        >{{this.selected.title2}}</b-form-radio
                      >
                      <b-form-radio
                        :value=this.selected.title3
                        :disabled="isColourActive(3)"
                        >{{this.selected.title3}}</b-form-radio
                      >
                      <b-form-radio
                        value="scenario"
                        :disabled="isColourActive(4)"
                        >Scenario</b-form-radio
                      >
                    </b-form-radio-group>
                  </b-form-group>
                  <b-form-group label="Axis - y" class="selection_label">
                    <b-form-radio-group
                      v-b-tooltip.hover
                      title="Dimension to plot as seperate plots along y axis"
                      buttons
                      button-variant="light"
                      id="facet"
                      :options="[]"
                      v-model="facet"
                      name="radioFacet"
                      size="sm"
                      class="cedefop_button"
                    >
                      <b-form-radio :disabled="isFacetActive(5)"
                        >None</b-form-radio
                      >
                      <b-form-radio
                        :value=this.selected.title
                        :disabled="isFacetActive(1)"
                        >{{this.selected.title}}</b-form-radio
                      >
                      <b-form-radio
                      :value=this.selected.title2
                        :disabled="isFacetActive(2)"
                        >{{this.selected.title2}}</b-form-radio
                      >
                      <b-form-radio
                        :value=this.selected.title3
                        :disabled="isFacetActive(3)"
                        >{{this.selected.title3}}</b-form-radio
                      >
                      <b-form-radio
                        value="scenario"
                        :disabled="isFacetActive(4)"
                        >Scenario</b-form-radio
                      >
                    </b-form-radio-group>
                  </b-form-group>
                  <b-form-group label="Axis - x" class="selection_label">
                    <b-form-radio-group
                      v-b-tooltip.hover
                      title="Dimension to plot as seperate plots along x axis"
                      buttons
                      button-variant="light"
                      id="facet_x"
                      :options="[]"
                      v-model="facet_x"
                      name="radioFacet"
                      size="sm"
                      class="cedefop_button"
                    >
                      <b-form-radio :disabled="isFacetActive_x(5)"
                        >None</b-form-radio
                      >
                      <b-form-radio
                      :value=this.selected.title
                        :disabled="isFacetActive_x(1)"
                        >{{this.selected.title}}</b-form-radio
                      >
                      <b-form-radio
                      :value=this.selected.title2
                        :disabled="isFacetActive_x(2)"
                        >{{this.selected.title2}}</b-form-radio
                      >
                      <b-form-radio
                      :value=this.selected.title3
                        :disabled="isFacetActive_x(3)"
                        >{{this.selected.title3}}</b-form-radio
                      >
                      <b-form-radio
                        value="Scenario"
                        :disabled="isFacetActive_x(4)"
                        >Scenario</b-form-radio
                      >
                    </b-form-radio-group>
                  </b-form-group>
                </b-row>
                <b-row>
                <b-form-group label="Chart type" class="selection_label top_margin">
                    <b-form-radio-group
                      id="chart_type"
                      v-model="selected.chart_type"
                      :options="chart_type"
                      name="radiocharttype"
                    >
                    </b-form-radio-group>
                </b-form-group>
                  <b-form-group label="Show" class="selection_label top_margin">
                    <b-form-radio-group
                      id="calculation"
                      v-model="selected.calculation"
                      :options="calculation_methods"
                      name="radioCalc"
                    >
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
                  <!-- <b-form-group
                    label="Unit"
                    class="selection_label top_margin"
                    :style="{ display: isUnit() ? '' : 'none' }"
                  >
                    <b-form-radio-group
                      id="energyunitselect"
                      v-model="selected.unit"
                      :options="unit_options"
                      name="radioCalc2"
                    >
                    </b-form-radio-group> -->
                  <!-- </b-form-group> -->
                </b-row>
              </b-col>
            </b-tab>
          </b-tabs>
        </b-card>
      </b-col>

      <b-col cols="8">
        <div class="row-12 draw_area">
          <b-row>
            <b-col cols="12">
              <div class="selected_variable">
                {{ selected.variable_label.label }}
                <b-badge
                  class="info"
                  :style="{ display: isSet() ? '' : 'none' }"
                  v-b-tooltip.hover
                  variant="info"
                  size="sm"
                  :title="selected.variable_label.info"
                  >i</b-badge
                >
              </div>
            </b-col>
            <b-col cols="3" class="download">
              <b-col>
                <b-form-group>
                  <b-form-radio-group
                    id="btn-radios-1"
                    v-model="chart_not_table"
                    :options="[
                      { text: 'Chart', value: true },
                      { text: 'Table', value: false },
                    ]"
                    buttons
                    name="radios-btn-default"
                    button-variant="outline-primary"
                  ></b-form-radio-group>
                </b-form-group>
              </b-col>
              <b-col cols="3">
                <b-button size="sm" v-on:click="download_data"
                  >Download data</b-button
                >
              </b-col>
            </b-col>
          </b-row>
          <div class="overlay" :style="{ display: isSet() ? 'none' : 'block' }">
            <span>{{ selection_warning }}</span>
          </div>

          <div
            id="line"
            :style="{ display: chart_not_table ? 'block' : 'none' }"
          ></div>
          <div
            id="data-table-main"
            :style="{ display: chart_not_table ? 'none' : 'block' }"
          >
            <b-table
              selectable
              small
              sticky-header
              head-variant="light"
              :items="JSON.parse(pivot)['items']"
              :fields="JSON.parse(table_fields)"
              responsive="sm"
              class="table table-hover"
              style="min-height: 400px"
            >
            </b-table>
          </div>
        </div>
      </b-col>
    </b-row>
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
  // formatters: {
  //   year: {
  //     label: "variables",
  //     format: "%Y",
  //   },
  // },
});

taucharts.api.tickFormat.add('facet', function(originalValue) {
    if( typeof originalValue == "string"){

      return originalValue
    }
    else{
      return originalValue
    } 
})
var config = {
  data: [
    {
      dimension: "us",
      y_real: 20,
      q_real: 23,
      year: "2013",
      dimension2: "a",
      dimension3: "b",
    },
  ],
  dimensions: {
    year: { type: "category",},
    y_real: { type: "measure" },
    q_real: { type: "measure" },
    dimension: { type: "category" },
    dimension2: { type: "category" },
    dimension3: { type: "category" },
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
          "rgb(100, 100, 100)",
        ],
      },
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
          "rgb(100, 100, 100)",
        ],
      },
    },
  ],

  plugins: [taucharts_tooltip, tau_legend()],
  type: "line",
  x: "year",
  y: "y_real",
  order: "year",
  color: ["scenario", "dimension"], // there will be two lines with different colors on the chart
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
      round:"3",
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
      start_year:2010,
      end_year:2050,
      fields: [
        { key: "scenario", sortable: true },
        { key: "description", sortable: true },
        { key: "Last Run", sortable: true }
      ],
      scenarios: [],
      scenarios_drop: [],
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
      no_aggregate2: false,
      no_aggregate3: false,
      dimension_text:"",
      energy_unit_options: [
        "model unit",
        "ktoe",
        "GWh",
        "TJ",
        "mboe",
        "million BTU",
      ],
      vol_unit_options: ["model unit", "mb"],
      gas_unit_options: ["model unit", "bcm"],
      unit_options: ["model unit"],
      years:[],
      dimensions_drawn: [],
      dimensions2_drawn: [],
      dimensions3_drawn: [],
      scenarios_drawn: ["S0"],
      chart_type: ["bar", "line", "stacked-bar"],
      title_old: "null",
      title2_old: "null",
      title3_old: "null",
      Levels: 0,
      change_detail:true,
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
        scen_index: [0],
        unit: "model unit",
        chart_type:"bar",
        start_year:0,
        end_year:0
      },
      calculation_methods: [
        { text: "Absolute value", value: "Levels" },
        { text: "Absolute difference from baseline", value: "absolute_diff" },
        { text: "Percentage difference from baseline", value: "perc_diff" },
        { text: "% Year-on-Year change", value: "Annual growth rate" },
        { text: "Incremental change (absolute)", value: "Incremental" },
      ],
      aggregation_methods: [
        { text: "None", value: "0" },
        { text: "Sum", value: "sum" },
      ],
    };
  },
  components: { Treeselect },
  methods: {
    initialise: function () {
      //Get available variables
      this.get_vars();

      axios.get("http://localhost:5000/api/scenarios_ran").then((res) => {
        let scenarios = res.data.exist;
        this.scenarios = scenarios;
        this.selected.baseline = this.scenarios[0]["scenario"];
        this.scenarios_drop = scenarios.map((s) => {
          return { label: s["scenario"], id: s["scenario"] };
        });
      });
    },
    onRowSelected(items) {
      this.selected_table = items;
      this.selected.scenarios = [];
      this.selected.scen_index = [];

      for (var i = 0; i < this.selected_table.length; i++) {
        if (
          this.selected.scenarios.includes(
            this.selected_table[i]["scenario"]
          ) == false
        ) {
          this.selected.scenarios.push(this.selected_table[i]["scenario"]);
        }
        for (var j = 0; j < this.scenarios.length; j++) {
          if (
            this.selected_table[i]["scenario"] == this.scenarios[j]["scenario"]
          )
            this.selected.scen_index.push(j);
        }
      }
    },
    selectFirstRow() {
      // Rows are indexed from 0, so the third row is index 2
      this.$refs.selectableTable.selectRow(0);
    },
    get_vars: function () {
      axios.get("http://localhost:5000/api/results/variables").then(
        (res) => {
          console.log(res)
          let variables = res.data.vars.indicies;
          this.var_groups = res.data.vars.groups;
          this.var_labels = res.data.vars.labels;
          this.variables_all = variables;
          this.variables_detailed = res.data.vars.indicies_detailed;
          this.variables_summary = res.data.vars.indicies_summary;
          this.variables = variables;
          this.titles = res.data.vars.title_map;
          this.chart_colours = res.data.vars.chart_colours;
          // this.energy_unit_options = res.data.energy_unit_options;
          // this.vol_unit_options = res.data.vol_unit_options;
          // this.gas_unit_options = res.data.gas_unit_options;
          this.years = res.data.years
          this.selected.start_year = this.years[0]
          this.selected.end_year = this.years[this.years.length -1]
        },
        (err) => {
          this.error_message(err, "variables");
        }
      );
    },
    get_dimensions: function () {
      if (this.selected.scenarios.length == 0) return;

      this.selected.titles = this.titles[this.selected.variable];
      console.log(this.titles)
      console.log(this.selected.variable)
      if (this.selected.titles["title4"] == "TIME") {
        this.selected.time = "Yes";
      } else {
        this.selected.time = "No";
      }

      axios
        .get(`http://localhost:5000/api/info/${this.selected.titles["title"]}`)
        .then(
          (res) => {
            if ("Sectors" in res.data) {
              this.dimensions = res.data.Sectors;
              if (this.selected.variable_label.aggregatable == 0) {
                this.no_aggregate = true;
                this.selected.aggregate = false
                if (this.selected.title == "demand_region") {
                  this.dimensions = this.dimensions[0]["children"];
                }
              }
            } else {
              let dimensions = res.data;
              this.dimensions = dimensions.map((s) => {
                return { label: s, id: s };
              });
            }
          },
          (err) => {
            this.error_message(err, "dimensions");
          }
        );
      axios
        .get(`http://localhost:5000/api/info/${this.selected.titles["title2"]}`)
        .then(
          (res) => {
            if ("Sectors" in res.data) {
              this.dimensions2 = res.data.Sectors;
            } else {
              let dimensions2 = res.data;
              this.dimensions2 = dimensions2.map((s) => {
                return { label: s, id: s };
              });
            }
          },
          (err) => {
            this.error_message(err, "dimensions2");
          }
        );

      axios
        .get(`http://localhost:5000/api/info/${this.selected.titles["title3"]}`)
        .then(
          (res) => {
            if ("Sectors" in res.data) {
              this.dimensions3 = res.data.Sectors;
            } else {
              let dimensions3 = res.data;
              this.dimensions3 = dimensions3.map((s) => {
                return { label: s, id: s };
              });
            }
          },
          (err) => {
            this.error_message(err, "dimensions");
          }
        );

      this.selected.title = this.titles[this.selected.variable]["title"];
      this.selected.title2 = this.titles[this.selected.variable]["title2"];
      this.selected.title3 = this.titles[this.selected.variable]["title3"];

    },
    update_chart: function () {
      // Prevents update chart calling prematurely

      if (this.selected.time == "Yes") {
        (config.x = "year"), (config.type = this.selected.chart_type);
      } else {
        (config.x = this.selected.title3), (config.type = this.selected.chart_type);
      }

      // Prevents update chart calling prematurely
      if (this.isSet() == false) return;

      //if (this.selected.baseline == "Baseline") return;
      // if (this.selected.scenarios[0] == "Baseline") {
      //   this.selectFirstRow();
      //   return;
      // }
      if (this.isdrawn()) return;
      if (this.var_groups.length == 0) return;
      if (this.loading == true) return;
      if (this.facet == this.color && this.facet != null) return;
      if (this.facet_x == this.color && this.color != null) return;
      if (this.facet == this.facet_x && this.facet_x != null) return;
      if (!this.isEnergyUnit() && !this.isVolUnit() && !this.isGasUnit()) {
        this.selected.unit = "model unit";
      }
      if (!this.unit_options.includes(this.selected.variable_label.unit)) {
        this.selected.unit = "model unit";
      }
      if (!this.unit_options.includes(this.selected.unit)) {
        this.selected.unit = "model unit";
      }
      let vars = this.var_groups.filter(
        (g) => g.indexOf(this.selected.variable) != -1
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
          (res) => {
            this.json_ = res.data;
            
          },
          (err) => {
            this.error_message(err, "requested chart data");
          }
        );
      console.log("Title update")
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
    download_data: function () {
      let params = lodash.cloneDeep(this.selected);

      params["variable"] = [this.selected.variable];
      params["title"] = [this.selected.title];
      params["title2"] = [this.selected.title2];
      params["title3"] = [this.selected.title3];
      params["variable_label"] = [this.selected.variable_label];

      axios
        .get(`http://localhost:5000/api/results/data/csv`, { params: params })
        .then(
          (res) => {
            let blob = new Blob([res.data], { type: "text/csv" }),
              url = window.URL.createObjectURL(blob);
            var fileLink = document.createElement("a");
            fileLink.href = url;
            fileLink.download = `FTT_data_download_${new Date().toISOString()}.csv`;
            fileLink.click();
          },
          (err) => {
            this.error_message(err, "download data");
          }
        );
    },
    update_table: function () {
      let table_fields = [
        {
          key: "scenario",
          sortable: false,
          stickyColumn: true,
          isRowHeader: true,
        },
        {
          key: this.selected.title,
          sortable: false,
          stickyColumn: true,
          isRowHeader: true,
        },
        {
          key: this.selected.title2,
          sortable: false,
          stickyColumn: true,
          isRowHeader: true,
        },
        {
          key: this.selected.title3,
          sortable: false,
          stickyColumn: true,
          isRowHeader: true,
        },
        {
          key: "year",
          sortable: false,
          stickyColumn: true,
          isRowHeader: true,
        },
      ];
      if (this.pivot.length > 0) {
        Object.keys(this.pivot[0])
          .slice(0, Object.keys(this.pivot[0]).length - 3)
          .forEach((y) => {
            table_fields.push({ key: y });
            // }
          });
      }
      this.table_fields = JSON.stringify(table_fields);
      this.pivot = JSON.stringify({ items: this.pivot });

      // localStorage.variable = this.selected.variable_label.label
    },
    isSet: function () {
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
    isUnit: function () {
      if (this.isEnergyUnit()) return true;
      else if (this.isVolUnit()) return true;
      else if (this.isGasUnit()) return true;
      else return false;
    },
    isEnergyUnit: function () {
      if (this) {
        if (
          this.energy_unit_options.includes(this.selected.variable_label.unit)
        ) {
          this.unit_options = this.energy_unit_options;
          return true;
        } else {
          this.selected_unit = "model unit";
          return false;
        }
      } else {
        return false;
      }
    },
    isVolUnit: function () {
      if (this) {
        if (this.vol_unit_options.includes(this.selected.variable_label.unit)) {
          this.unit_options = this.vol_unit_options;
          return true;
        } else {
          this.selected_unit = "model unit";
          return false;
        }
      } else {
        return false;
      }
    },
    isGasUnit: function () {
      if (this) {
        if (this.gas_unit_options.includes(this.selected.variable_label.unit)) {
          this.unit_options = this.gas_unit_options;
          return true;
        } else {
          this.selected_unit = "model unit";
          return false;
        }
      } else {
        return false;
      }
    },
    isdrawn: function () {
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
    arraysEqual: function (a, b) {
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
    isFacetActive: function (facet) {
      var multiples = [];
      if (this.selected.scenarios.length > 1) multiples.push("scenario");
      if (this.selected.dimensions.length > 1) multiples.push(this.selected.title);
      if (this.selected.dimensions2.length > 1) multiples.push(this.selected.title2);
      if (this.selected.dimensions3.length > 1) multiples.push(this.selected.title3);
      if (facet == 5) {
        return multiples.length >= 2;
      } else if (multiples.length == 1) return true;
      else if (facet == 1) return this.selected.dimensions.length <= 1;
      else if (facet == 2) return this.selected.dimensions2.length <= 1;
      else if (facet == 3) return this.selected.dimensions3.length <= 1;
      else if (facet == 4) return this.selected.scenarios.length <= 1;
      else if (facet == 5) return multiples.length >= 3;
      else return false;
    },
    isFacetActive_x: function (facet_x) {
      var multiples = [];
      if (this.selected.scenarios.length > 1) multiples.push("scenario");
      if (this.selected.dimensions.length > 1) multiples.push(this.selected.title);
      if (this.selected.dimensions2.length > 1) multiples.push(this.selected.title2);
      if (this.selected.dimensions3.length > 1) multiples.push(this.selected.title3);
      if (multiples.length < 3) return true;
      if (facet_x == 1) return this.selected.dimensions.length <= 1;
      else if (facet_x == 2) return this.selected.dimensions2.length <= 1;
      else if (facet_x == 3) return this.selected.dimensions3.length <= 1;
      else if (facet_x == 4) return this.selected.scenarios.length <= 1;
      else if (facet_x == 5) return multiples.length >= 3;
      else return false;
    },
    isColourActive: function (colour) {
      if (colour == 1) return this.selected.dimensions.length <= 1;
      else if (colour == 2) return this.selected.dimensions2.length <= 1;
      else if (colour == 3) return this.selected.dimensions3.length <= 1;
      else if (colour == 4) return this.selected.scenarios.length <= 1;
      else return false;
    },

    title_update: function () {
      let full_axis_label = "";

      let unit = "";
      if (
        this.selected.calculation == "absolute_diff" ||
        this.selected.calculation == "perc_diff"
      )
        full_axis_label = "Difference from baseline";
      else if (this.selected.calculation == "Annual growth rate")
        full_axis_label = "Year on Year growth";
      else if (this.selected.calculation == "Incremental")
        full_axis_label = "Incremental change";
      if (
        this.selected.calculation == "Levels" ||
        this.selected.calculation == "absolute_diff" ||
        this.selected.calculation == "Incremental"
      ) {
        if (this.selected.unit == "model unit") {
          unit = this.selected.variable_label.unit;
        } else unit = this.selected.unit;
      } 
      else unit = "%";

      full_axis_label = full_axis_label + " (" + unit + ")";

      this.dimension_text = "Scenarios: " + this.selected.scenarios;
      if (this.selected.dimensions.length > 4) {
        this.dimension_text =
        this.dimension_text +
          " / "+this.selected.title +": " +
          "Multiple selected (" +
          this.selected.dimensions.length +
          ")";
      } 
      else {
        this.dimension_text =
          this.dimension_text + " / "+this.selected.title +": " + this.selected.dimensions;
      }
      var dimension_text = this.dimension_text 
      //If there is a facet, you need to pass array of format options for each layer (inside out) 
      //i.e number tick formating for y axis is in first element of guide array
      //Otherwise if no facet only one layer of formatting 
      if (this.facet != null) {
        if (this.selected.chart_type !="line"){
          config.guide = [
            {
              y: { label: { text: full_axis_label },tickFormat: ",."+ Math.max(parseInt(this.round)-2,0)+ "f"},

              color: {
                brewer: this.chart_colours,
              },
            },
            {
              y: { label: { text: this.facet },  },
              x: { label: { text: this.dimension_text }  },
    
              color: {
                brewer: this.chart_colours,
              },
            },
          ];
        
        }
        else{
          config.guide = [
            {
              y: { label: { text: full_axis_label },tickFormat: ",."+ Math.max(parseInt(this.round)-2,0)+ "f"},


              color: {
                brewer: this.chart_colours,
              },
            },
            {
              y: { label: { text: this.facet },  },
              x: { label: { text: this.dimension_text }  },
              
      
              color: {
                brewer: this.chart_colours,
              },
            },
          ];
        
        }
      }
      
      else {
        if (this.selected.chart_type !="line"){
          config.guide = {
            y: { label: { text: full_axis_label },tickFormat: ",."+ Math.max(parseInt(this.round)-2,0)+ "f"},
            x: { label: { text: this.dimension_text } },

            color: {
              brewer: this.chart_colours,
            },
          };
        }
      
        else{
            config.guide = {
              y: { label: { text: full_axis_label },tickFormat: ",."+ Math.max(parseInt(this.round)-2,0)+ "f"},
              x: { label: { text: this.dimension_text } },

              color: {
                brewer: this.chart_colours,
              },
            };
        }      
      }
      
      var formatters_dict = {}
      formatters_dict["variables"] = {
            label: full_axis_label,
            format: ",."+ this.round+ "f",
      }
      formatters_dict["year"] = {
            label: "Year",
      }
      taucharts_tooltip = taucharts.api.plugins.get("tooltip")({
        formatters: formatters_dict
      });
      config.plugins = [
          taucharts_tooltip,
          tau_legend()
        ]
      console.log("dimesnion_text:" + this.dimension_text)
    },
    tree_limit_text: function () {
      return "";
    },
    title_fix: function () {
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
        this.selected.calculation == "absolute_diff" ||
        this.selected.calculation == "Incremental"
      ) {
        if (this.selected.unit == "model unit") {
          unit = this.selected.variable_label.unit;
        } 
        else unit = this.selected.unit;
      } 
      else unit = "%";
      
      if (
        this.selected.calculation == "Levels" ||
        this.selected.calculation == "Incremental"
      ) {
        full_axis_label = unit;
      } else {
        full_axis_label = full_axis_label + " (" + unit + ")";
      }
      if (this.facet != null && this.facet_x != null) {
        console.log("facet active facet x active")
        document.getElementsByClassName("label-token-2")[1].innerHTML = full_axis_label 
        document.getElementsByClassName("label-token-2")[0].innerHTML = this.dimension_text 
      }
      else if (this.facet != null && this.facet_x == null) {
        console.log("facet active facet x inactive")
        document.getElementsByClassName("label-token-2")[0].innerHTML = full_axis_label 
        document.getElementsByClassName("label-token-0")[0].innerHTML = this.dimension_text 
      }
      else{
        console.log("facet inactive facet x inactive")
        var temp = document.getElementsByClassName("label-token-0")[document.getElementsByClassName("label-token-0").length-1].innerHTML
        for (let i=0; i<document.getElementsByClassName("label-token-0").length; i++){
          if (document.getElementsByClassName("label-token-0")[i].innerHTML == temp)
          document.getElementsByClassName("label-token-0")[i].innerHTML = full_axis_label 
          else{
            document.getElementsByClassName("label-token-0")[i].innerHTML = this.dimension_text   
          }
        }
      }
      console.log(this.facet)
      console.log(this.facet_x)
      console.log("label:")
      console.log(document.getElementsByClassName("label-token-2"))
      console.log(document.getElementsByClassName("label-token-0"))
    },
    check_facet_color: function () {
      var multiples = [];
      if (this.selected.scenarios.length > 1) multiples.push("scenario");
      if (this.selected.dimensions.length > 1) multiples.push(this.selected.title);
      if (this.selected.dimensions2.length > 1) multiples.push(this.selected.title2);
      if (this.selected.dimensions3.length > 1) multiples.push(this.selected.title3);
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
      else if (multiples.length == 3) {
        this.color = multiples[0];
        this.facet = multiples[1];
        this.facet_x = multiples[2];
      }
    },
    check_facet_color_swap: function (dim_check, value) {
      var multiples = [];
      if (this.selected.scenarios.length > 1) multiples.push("scenario");
      if (this.selected.dimensions.length > 1) multiples.push(this.selected.title);
      if (this.selected.dimensions2.length > 1) multiples.push(this.selected.title2);
      if (this.selected.dimensions3.length > 1) multiples.push(this.selected.title3);
      if (multiples.length == 1) return;
      if (dim_check == "facet") {
        if (value == this.color) {
          this.color = multiples.filter(
            (e) => (e !== value) & (e !== this.facet_x)
          )[0];
        } else if (value == this.facet_x) {
          this.facet_x = multiples.filter(
            (e) => (e !== value) & (e !== this.color)
          )[0];
        }
      } else if (dim_check == "color") {
        if (value == this.facet) {
          this.facet = multiples.filter(
            (e) => (e !== value) & (e !== this.facet_x)
          )[0];
        } else if (value == this.facet_x) {
          this.facet_x = multiples.filter(
            (e) => (e !== value) & (e !== this.facet)
          )[0];
        }
      } else if (dim_check == "facet_x") {
        if (value == this.color) {
          this.color = multiples.filter(
            (e) => (e !== value) & (e !== this.facet)
          )[0];
        } else if (value == this.facet) {
          this.facet = multiples.filter(
            (e) => (e !== value) & (e !== this.color)
          )[0];
        }
      }
    },
    selection_warning_update: function () {
      this.selection_warning = "Please set the ";
      var iCount = 0;
      
      iCount += this.selected.dimensions.length == 0 ? 1 : 0;
      iCount += this.selected.dimensions2.length == 0 ? 1 : 0;
      iCount += this.selected.dimensions3.length == 0 ? 1 : 0;
      iCount += this.selected.scenarios.length == 0 ? 1 : 0;
      var iFilled = 0
      if (this.selected.dimensions.length == 0) {
        this.selection_warning = this.selection_warning + this.selected.title;
        iFilled += 1
        if (iFilled == iCount-1) this.selection_warning = this.selection_warning + " and ";
        else if (iFilled < iCount-1) this.selection_warning = this.selection_warning + ", ";
      }
      if (this.selected.dimensions2.length == 0){ this.selection_warning = this.selection_warning + this.selected.title2;
        iFilled += 1
        if (iFilled == iCount-1) 
          this.selection_warning = this.selection_warning + " and ";
        else if (iFilled < iCount-1) 
          this.selection_warning = this.selection_warning + ", ";
    }
        if (this.selected.dimensions3.length == 0){
        this.selection_warning = this.selection_warning + this.selected.title3;
        iFilled += 1
        if (iFilled == iCount-1) this.selection_warning = this.selection_warning + " and ";
        else if (iFilled < iCount-1)
          this.selection_warning = this.selection_warning + ", ";
      }
      if (this.selected.scenarios.length == 0) {
        this.selection_warning = this.selection_warning + "scenario(s)";
      }
    },
    saveselection: function () {
      if (globalStore.saved == false) {
        globalStore.results_selected = this.selected;
        globalStore.saved = true;
      }
    },

    error_message: function (err, item) {
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
    },
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
    variables: function () {
      if (
        this.settings_loaded == true &&
        this.selected.variable != this.variables[0].id
      ) {
        this.selected.variable = this.variables[0].id;
        this.get_dimensions();
      }
    },
    start_year: function(){
      this.update_chart();
    },
    end_year: function(){
      this.update_chart();
    },
    scenarios: function () {
      this.selectFirstRow();
    },
    dimensions: function () {
      if (this.dimensions.length == 1)
        this.selected.dimensions = [this.dimensions[0].id];
      else if (
        this.selected.title == this.title_old && this.dimensions_drawn.length > 0){
          this.selected.dimensions = this.dimensions_drawn;
          if (this.selected.title == "demand_region" && this.no_aggregate ) {
              var temp = Array.from(this.dimensions.map(({ id }) => id));
              this.selected.dimensions = this.selected.dimensions.filter( function( el ) {
                return temp.includes(el);
              })
          }
      }
      else if (this.settings_loaded == true)
        this.selected.dimensions = [this.dimensions[0].id];
      if (this.selected.title != "") this.title_old = this.selected.title;
    },
    dimensions2: function () {
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
    dimensions3: function () {
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
    chart_not_table: function () {
      if (this.chart_not_table) this.update_chart();
    },
    "selected.scenarios": function () {
      this.selection_warning_update();
      this.update_chart();
    },

    "selected.variable": function () {
      this.loading = true;
      if (this.selected.variable != "") {
        this.selected.dimensions = [];
        this.selected.dimensions2 = [];
        this.selected.dimensions3 = [];
        this.get_dimensions();
      }
      this.selected.variable_label = this.var_labels[this.selected.variable];
      if (this.selected.variable_label.aggregatable == 0) {
        this.no_aggregate = true;
        this.selected.aggregate = false        
      } else {
        this.no_aggregate = false;
      }
      if (this.selected.variable_label.aggregatable2 == 0){
        this.no_aggregate2 = true;
        this.selected.aggregate2 = false
      }
      else {
        this.no_aggregate2 = false;
        
      }
      if (this.selected.variable_label.aggregatable3 == 0){
        this.no_aggregate3 = true;
        this.selected.aggregate3 = false
      }
      else {
        this.no_aggregate3 = false;
        
      }
      this.loading = false;

      if (this.isSet) this.update_chart();
    },
    "selected.dimensions": function () {
      //Check if selection is valid
      if (this.isSet) {
        this.update_chart();
        this.selection_warning_update();
      }
    },
    "selected.dimensions2": function () {
      if (this.isSet) {
        this.update_chart();
        this.selection_warning_update();
      }
    },
    "selected.dimensions3": function () {
      if (this.isSet) {
        this.update_chart();
        this.selection_warning_update();
      }
    },
    "selected.aggregation": function () {
      this.update_chart();
    },
    "selected.calculation": function () {
      this.update_chart();
    },
    "selected.baseline": function () {
      this.update_chart();
    },
    "selected.aggregate": function () {
      this.update_chart();
    },
    "selected.aggregate2": function () {
      this.update_chart();
    },
    "selected.aggregate3": function () {
      this.update_chart();
    },
    "selected.unit": function () {
      this.update_chart();
    },
    "selected.chart_type": function () {
      this.update_chart();
    },
    "selected.start_year": function () {
      this.update_chart();
    },
    "selected.end_year": function () {
      this.update_chart();
    },  
    "selected.unit": function () {
      this.update_chart();
    },    
    select_all: function () {
      this.selected.dimensions = [];
      if (this.select_all) {
        for (var i = 0; i < this.dimensions.length; i++)
          this.selected.dimensions.push(this.dimensions[i].id);
      } else this.selected.dimensions.push(this.dimensions[0].id);
    },
    select_all2: function () {
      this.selected.dimensions2 = [];
      if (this.select_all2) {
        for (var i = 0; i < this.dimensions2.length; i++)
          this.selected.dimensions2.push(this.dimensions2[i].id);
      } else this.selected.dimensions2.push(this.dimensions2[0].id);
    },
    select_all3: function () {
      this.selected.dimensions3 = [];
      if (this.select_all3) {
        for (var i = 0; i < this.dimensions3.length; i++)
          this.selected.dimensions3.push(this.dimensions3[i].id);
      } else this.selected.dimensions3.push(this.dimensions3[0].id);
    },
    indic_detail: function () {
      if (this.indic_detail == 0) this.variables = this.variables_all;
      else if (this.indic_detail == 1) this.variables = this.variables_all;
      else if (this.indic_detail == 2) this.variables = this.variables_all;
      if (this.code_label)
        this.variables = this.variables.map((s) => {
          if ("children" in s)
            return {
              label: s.label,
              id: s.id,

              children: s.children.map((v) => {
                return { label: v.id, id: v.id };
              }),
            };
          else
            return {
              label: s.label,
              id: s.id,
            };
        });
    },
    code_label: function () {
      if (this.indic_detail == 0) this.variables = this.variables_all;
      else if (this.indic_detail == 1) this.variables = this.variables_all;
      else if (this.indic_detail == 2) this.variables = this.variables_all;
      if (this.code_label)
        this.variables = this.variables.map((s) => {
          if ("children" in s)
            return {
              label: s.label,
              id: s.id,

              children: s.children.map((v) => {
                return { label: v.id, id: v.id };
              }),
            };
          else
            return {
              label: s.label,
              id: s.id,
            };
        });
    },
    json_: function () {
      this.pivot = JSON.parse(this.json_.pivot);
      this.display_data = JSON.parse(this.json_.results);
      // this.pivot_columns = this.json_.pivot_columns.filter(v => ['county','dimension','scenario'].indexOf(v) == -1)
      if (this.selected.time == "Yes")
        this.json_vars = this.json_.info.filter(
          (v) =>
            [
              "year",
              this.selected.title,
              this.selected.title2,
              this.selected.title3,
              "scenario",
            ].indexOf(v) == -1
        );
      else
        this.json_vars = this.json_.info.filter(
          (v) =>
            [this.selected.title, this.selected.title2, this.selected.title3, "scenario"].indexOf(v) ==
            -1
        );

      if (this.selected.dimensions.length > 0) {
        this.update_table();
      }
    },
    display_data: function () {
      if (globalStore.saved == true) return;
      if (this.loading == true) return;
      config.data = this.display_data;

      if (this.selected.time == "Yes") {
        config.type = this.selected.chart_type
        if (this.selected.chart_type == "line"){
          config.dimensions = {
          year: { type: "order", order: this.years},
          [this.selected.title]: { type: "category" },
          [this.selected.title2]: { type: "category" },
          [this.selected.title3]: { type: "category" },
          scenario: { type: "category" },
        };
        }
        else{
        config.dimensions = {
          year: { type: "order", order: this.years},
          [this.selected.title]: { type: "category" },
          [this.selected.title2]: { type: "category" },
          [this.selected.title3]: { type: "category" },
          scenario: { type: "category" },
        };
      }
      } else {
        config.dimensions = {
          [this.selected.title]: { type: "category" },
          [this.selected.title2]: { type: "category" },
          [this.selected.title3]: { type: "category" },
          scenario: { type: "category" },
        };
      }
      for (var i = 0; i < this.json_vars.length; i++)
        if (this.json_vars[i] != "year"){ 
          config.dimensions[this.json_vars[i]] = { type: "measure" };
        }
      this.check_facet_color();

      config.color = this.color;
      config.y = this.facet ? ["variables", this.facet] : ["variables"];
      if (this.selected.chart_type == "line"){ 
        config.x = this.facet_x ? [this.facet_x,"year"] : ["year"];
      }
      else{
        config.x = this.facet_x ? [this.facet_x,"year"] : ["year"];
      }
      if (this.quickfilter_on)
        config.plugins = [
          taucharts_tooltip,
          tau_legend(),
          tau_quickfilter(this.json_vars),
        ];
      else config.plugins = [taucharts_tooltip, tau_legend()];

      this.title_update();
      chart.updateConfig(config);
      
      this.title_fix();

    },
    facet: function () {
      if ((this.facet == this.color) | (this.facet == this.facet_x)) {
        this.check_facet_color_swap("facet", this.facet);
      }
      config.y = this.facet ? ["variables", this.facet] : ["variables"];
      if (this.quickfilter_on)
        config.plugins = [
          taucharts_tooltip,
          tau_legend(),
          tau_quickfilter(this.json_vars),
        ];
      else config.plugins = [taucharts_tooltip, tau_legend()];
      this.title_update();
      chart.updateConfig(config);
      this.title_fix();
    },
    facet_x: function () {
      if ((this.facet_x == this.facet) | (this.facet_x == this.color)) {
        this.check_facet_color_swap("facet_x", this.facet_x);
      }
      config.x = this.facet_x ? [this.facet_x,"year"] : ["year"];
      this.title_update();
      chart.updateConfig(config);
      this.title_fix();
    },
    color: function () {
      if ((this.color == this.facet) | (this.color == this.facet_x)) {
        this.check_facet_color_swap("color", this.color);
      }
      config.color = this.color;
      if (this.quickfilter_on)
        config.plugins = [
          taucharts_tooltip,
          tau_legend(),
          tau_quickfilter(this.json_vars),
        ];
      else config.plugins = [taucharts_tooltip, tau_legend()];
      this.title_update();
      chart.updateConfig(config);
      this.title_fix();
    },
    quickfilter_on: function () {
      if (this.quickfilter_on)
        config.plugins = [
          taucharts_tooltip,
          tau_legend(),
          tau_quickfilter(this.json_vars),
        ];
      else config.plugins = [taucharts_tooltip, tau_legend()];
      chart.updateConfig(config);
      this.title_fix();
    },
  },
};
</script>

<!-- Add "scoped" attribute to limit CSS to this component only -->
<style scoped lang="scss">
.box {
  background-color: #e5e5e5;
  width: 100%;
  height: fit-content;
  border-radius: 0.2rem;
  z-index: 1100;
}
.control_header{
  background-color: #4B62BE;
}
#tab-content {
  overflow: scroll;
}
.indic_box {
  height: 10em;
}
 #controlpanel{
   z-index: 1000;
 }
.overlay {
  background-color: rgba(255, 255, 255, 1);
  position: absolute;
  left: 0;
  width: 100%;
  height: 100%;
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

.dimension_box {
  height: 11em;
}
.draw_area {
  height: 50em;
  //overflow-y: scroll;


}
#line{
  max-height: 80%;
}
.card-body {
  padding-top: 0.1rem;
  padding-bottom: 0.1rem;
}
body,
#line {
  height: 80%;
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
.selection_label.btn
.cedefop_button{

  color: #4B62BE;
  background-color: white;
  font-weight: bold;
}
.btn-light.disabled{
    background-color: grey;
  }

.btn-light:not(:disabled):not(.disabled):active, .btn-light:not(:disabled):not(.disabled).active, .show > .btn-light.dropdown-toggle{
  color: white;
  background-color: #4B62BE;
  font-weight: bold;
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
  font-size: 0.75rem;
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
.tau-chart__legend {
  margin-right: 0px;
  width: 100%;
}
@media (min-width: 1920px) {
  .container {
    max-width: 1800px;
  }
}
.vue-treeselect__menu {
  line-height: 110%;
}
.vue-treeselect__control {
  margin-top: 5px;
}

.btn-sm{
  font-size:0.8em !important
}
</style>

<style lang="css" scoped>
@import "http://cdn.jsdelivr.net/npm/taucharts@2/dist/taucharts.min.css";
</style>
