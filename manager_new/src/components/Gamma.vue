<template>
 <div class=container>
 <div class=container :style="{display: initial? 'block' : 'none'}">

         <b-col cols="2">
        <label class="selection_label">Model to calibrate:</label>
        <treeselect
          :multiple="false"
          v-model="selected.model"
          :options="ftt_models"
        />
      </b-col>
      <b-col cols="2">
      <label class="selection_label">End year</label>
      <b-form-input type="number" v-model="end_year"/>

      </b-col>
      <b-col cols="2">
        <br>
        <b-button  v-on:click="initialise_model">Initialise model</b-button>
      </b-col>
  
  </div> 

 <div class=container :style="{display: loading? 'block' : 'none'}">



      <h2>Initialising model... </h2>
      <b-spinner style="width: 3rem; height: 3rem;" label="Spinning"  variant="info"></b-spinner>
  </div>
  <div class=container :style="{display: loaded? 'block' : 'none'}">  
    <h3>"{{selected.model}}"</h3>
    <b-row>
      <b-col cols="2">
        <label class="selection_label">Region to calibrate:</label>
        <treeselect
          :multiple="false"
          v-model="selected.region"
          :options="regions"
        />
      </b-col>
   

      <b-col cols="2">
      <label class="selection_label">Slider limit</label>
      <b-form-input type="number" v-model="slider_limit"/>

      </b-col>
      <b-col cols="2">
      <label class="selection_label">Start_year</label>
      <b-form-select :options="years" type="number" v-model="start_year"/>

      </b-col>

      <b-col cols="2">   
      <label class="selection_label">Slider step size</label>
      <b-form-input type="number" v-model="slider_step"/>

      </b-col>

      <b-col cols="2">
        <br>
        <b-button  v-on:click="call_run_ftt">Run FTT</b-button>
      </b-col>
      <b-col cols="2">
        <br>
        <b-button  v-on:click="save_gamma_values">
          Save Gamma values
        </b-button>
      </b-col>
      <b-col cols="2">
        <br>
        <b-button  v-on:click="select_other_FTT">
          Select other FTT
      </b-button>
      </b-col>
    </b-row>
    <b-row>
      <!--Gamma value editor  -->
      <div class="col-4 draw_area">
        <div class="col-md-12 form-group">
          <!--Inputs generator for each technology  -->
          <div
            class="form-row"
            v-for="(value, key) in form"
            v-bind:key="(value, key)"
          >
            <label class="col-md-5 col-form-label dim_label" :for="key">
              {{ key }}
            </label>
            <!--Spin button for each technology  -->
            <b-form-input
              class="col-md-5 spec_select"
              type="range"
              :id="key"
              v-model="form[key]"
              :step="slider_step"
              :min="slider_limit_min"
              :max="slider_limit"
              :number="true"
            >
                      </b-form-input>
                        <label class="col-md-2 col-form-label dim_label" :for="key" > {{value.toFixed(6)}} </label>

          </div>
        </div>
      </div>
      <!--Chart/Table area  -->
      <div class="col-8 draw_area">
        <b-row>
          <b-col cols="8">
            <div class="selected_variable">
              {{ selected.variable_label.label }}
              <b-badge
                class="info"
                :style="{ display: isSet() ? '' : 'none' }"
                v-b-tooltip.hover
                variant="info"
                size="sm"
                :title="selected.variable_label.desc"
                >i</b-badge
              >
            </div>
          </b-col>
          <b-col cols="3" class="download">
              <b-button size="sm" v-on:click="download_data"
                >Download data</b-button
              >



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
          <!-- Array.from(Object.keys(pivot[0])).map(e => /[0-9]{4}/g.exec(e) == null ? {key: `hello ${e}`, stickyColumn: true} : e) -->
        </div>
      
        <b-row>
          <table>
            <tr>
              <td>
                <div class="status_text status_title">Status:</div>
              </td>
              <td>
                <div class="status_text status_state">
                  <span class="dot" :style="{ backgroundColor: status_state }"></span>{{ status_text }}
                </div>
              </td>
            </tr>
          </table>
        </b-row>
        <b-row class="current_progress">
          <div class="progress_text">
            <span>
              Progress of the current run
            </span>
            <span class="right" v-if="processing_remaining != 0">
              Est. time remaining {{processing_remaining | format_time}}
            </span>
          </div>
          <b-progress class='w-100'
           :value="processing_processed" :max="processing_length > 0? processing_length : 100"
           show-progress height="2rem"></b-progress>
        </b-row>
        </div>
      </b-row>  
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
let msgServer;
var taucharts_tooltip = taucharts.api.plugins.get("tooltip")({
  // formatters: {
  //   year: { label: "year", format: "%Y" },
  // },
});
var config = {
  data: [
    { dimension: "us", y_real: 20, q_real: 23, year: "2013", dimension2: "a" },
  ],
  dimensions: {
    year: { type: "measure", scale: "time" },
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
      slider_limit: 100,
      slider_limit_min: -100,
      slider_step: 1,
      start_year: 2000,
      years:[],
      end_year: 2025,
      loaded: false,
      loading: false,
      initial: true,
      form: {},
      ftt_models: [],
      regions: [],
      regions_list: [],
      console_log: [],
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
      var_labels: [],
      dimensions: [],
      dimensions2: [],
      dimensions3: [],
      json_vars: [],
      var_groups: [],
      scenarios: [],
      fields: [
        { key: "scenario", sortable: true },
        { key: "description", sortable: true },
        { key: "Last Run", sortable: true },
      ],
      scenarios_county: [],
      sort_by: "scenario",
      selectMode: "multi",
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
      indic_detail: false,
      no_aggregate: false,
      selection_warning:
        "Please set the scenario, variable and dimensions to display results.",
      selected: {
        variable: "",
        variable_label: {},
        dimensions: [],
        dimensions2: [],
        dimensions3: [],
        title: [],
        title2: [],
        title3: [],
        time: "Yes",
        scenarios: ["Baseline"],
        aggregate: false,
        aggregate2: false,
        aggregate3: false,
        baseline: "Baseline",
        calculation: "0",
        scen_index: [0],
        region: "",
        model: "power_gen",
        type: "Chart",
      },
      error: '',
      status_text: 'Standing by',
      status_state: '#006398',
      isRunning: false,
      processing_length: 0,
      processing_processed: 0,
      processing_avgtime: 0,
      processing_remaining: 0,
      console_log: [],
      calculation_methods: [
        { text: "Absolute value", value: "0" },
        { text: "Absolute difference from baseline", value: "absolute_diff" },
        { text: "Percentage difference from baseline", value: "perc_diff" },
        { text: "Year-on-Year change", value: "yoy" },
      ],
      aggregation_methods: [
        { text: "None", value: "0" },
        { text: "Sum", value: "sum" },
      ],
      minmax_year: [2006, 2016],
    };
  },
  components: { Treeselect },
  methods: {
    initialise: function () {

      axios.get("http://localhost:5000/api/info/ftt_options").then((res) => {
        let ftt_models = res.data;
        this.ftt_models = ftt_models.map((s) => {
          return { label: s, id: s };
        });
      });
    
      
    },
    isSet: function () {
      if (this) {
        return this.selected.latest != "";
      } else {
        return false;
      }
    },
    update_chart: function () {
      chart.updateConfig(config);
      if (this.first_run == true) {
        this.selectFirstRow();
      }
      this.first_run = false;
    },
    update_data: function () {
      axios
        .get(
          `http://localhost:5000/api/gamma/chart/${this.selected.model}/${this.selected.region}/${this.start_year}/json`
        )
        .then(
          (res) => {
            this.json_ = res.data;
          },
          (err) => {
            this.error_message(err, "requested chart data");
          }
        );
    },
    set_table_fields: function () {
      let table_fields = [
        {
          key: "scenario",
          sortable: false,
          stickyColumn: true,
          isRowHeader: true,
        },
        {
          key: "dimension",
          sortable: false,
          stickyColumn: true,
          isRowHeader: true,
        },
        {
          key: "dimension2",
          sortable: false,
          stickyColumn: true,
          isRowHeader: true,
        },
        {
          key: this.selected.time == "Yes" ? "dimension3" : "year",
          sortable: false,
          stickyColumn: true,
          isRowHeader: true,
        },
      ];
      this.update_data();
      this.pivot = JSON.parse(this.json_.pivot);

      if (this.pivot[0].length > 0) {
        Object.keys(this.pivot[0])
          .slice(0, Object.keys(this.pivot[0]).length - 3)
          .forEach((y) => {
            // if(y >= this.minmax_year[0] && y <= this.minmax_year[1]){
            table_fields.push({ key: y });
            // }
          });
      }
      //this.table_fields = JSON.stringify(table_fields);

      this.table_fields = table_fields;
      //this.pivot = JSON.stringify({'items':this.pivot});

      // localStorage.variable = this.selected.variable_label.label
    },
    download_data: function () {
      axios
        .get(
          `http://localhost:5000/api/gamma/chart/${this.selected.model}/${this.selected.region}/${this.year_from_hist}/csv`
        )
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
    load_current_gamma: function () {
      axios
        .get(
          `http://localhost:5000/api/Gamma/values/${this.selected.model}/${this.selected.region}`
        )
        .then(
          (res) => {
            this.form = {}
            this.form = res.data.gamma;
          },
          (err) => {
            this.error =
              "Failed to get specifications, the manager is either not running or encountered an error. (" +
              err.response.statusText +
              ")";
          }
        );
    },
    call_run_ftt: function(){
      this.run_ftt()
    },
    initialise_model: function(){
      this.initial = false
      this.loading = true
      axios
          .get(`http://localhost:5000/api/run_gamma/initialize/${this.end_year}`).then((res) => {
        
              let years = res.data.years
              this.years = years
              this.selected.region = this.regions_list[0] 
              this.loading= false
              this.loaded = true
              
          }); 
        // first load the available Regions and models
        axios.get("http://localhost:5000/api/info/region_titles").then(
          (res) => {
            let regions = res.data;
            this.regions_list = regions
            this.regions = regions.map((s) => {
              return { label: s, id: s };
            });
          },
          (err) => {
            this.error_message(err, "region titles");
          }
        );
    },
    run_ftt: function () {
      //Call model run
      this.processing_length = 0;
      this.processing_processed= 0;
      this.processing_avgtime= 0;
      this.processing_remaining= 0;
      
      let json_body = {
        data: this.form,
        model: this.selected.model,
        region: this.selected.region,
        region_pos: this.regions_list.indexOf(this.selected.region),
      };
      this.log("message;message:Started process...;");
      this.log("message;message:Processing data...;");

      axios
        .post(`http://localhost:5000/api/run_gamma/update_gamma/`, json_body, {
          headers: { "Content-Type": "application/json" },
        })
        .then(
          (res) => {
            this.log("message;message:Data processed. Running modelling.;");
            this.$sse(`http://localhost:5000/api/run_gamma/start/`, {
              format: "plain",
            }).then((sse) => {
              msgServer = sse;
              sse.onError((e) => {
                sse.close();
                this.update_status("system_error");
              });
              sse.subscribe("status_change", (data) => {
                if (data != "running") {
                  this.update_data();
                  sse.close();
                }
                this.update_status(data);
                this.log(data);
              });
              sse.subscribe("processing", (data) => {
                this.update_progress(data);
                if (data.split(";")[0] != "progress") {
                  this.log(data);
                }
              });
            });
          },
          (err) => {
            this.$bvToast.toast("Unexpected error", {
              title: "Running failed!",
              toaster: "b-toaster-bottom-right",
              appendToast: true,
              autoHideDelay: 7000,
              variant: "danger",
            });
          }
        );

      //Send updated gamma values
      //Update chart
      
    },
    log: function (message) {
      let date = new Date();
      date = date.toISOString();
      var regexp = /(?:;message:)([a-zA-Z0-9 \.\-\,]*)(?:;)/g;
      var match = [];
      match = regexp.exec(message);

      if (match.length != 0) {
        let object = { timestamp: date, message: match[1] };
        this.console_log.push(object);
      }
    },
    update_status: function (state) {
      switch (state) {
        case "running":
          this.isRunning = true;
          this.status_text = "Model is running";
          this.status_state = "#F1C40F";
          break;
        case "finished":
          this.isRunning = false;
          this.status_text = "Run finished";
          this.status_state = "#99FFCC";
          break;
        default:
          this.isRunning = false;
          this.status_text = "Run finished with errors";
          if (state == "system_error") this.status_text = "System error!";
          this.status_state = "#F06449";
          this.log(
            "message;message:System error - if all runs have finished running and this error is displayed it is likely that at least one of the scenarios have failed to converge after 100 iterations;"
          );
      }
    },
    update_progress: function (data) {
      let arr = data.split(";");
      let id = arr[0];
      if (id == "items") {
        this.processing_length = parseInt(arr[1]);
      } else if (id == "progress") {
        this.processing_processed += 1;
        let n = this.processing_processed;
        let elapsed = parseFloat(arr[3]);
        this.processing_avgtime =
          this.processing_avgtime == 0
            ? elapsed
            : (this.processing_avgtime * n + elapsed) / (n + 1);
        this.processing_remaining =
          (this.processing_length - n) * this.processing_avgtime;
      } else if (id == "processed" || id == "error") {
        // this.processing_remaining = "Finished"
      }
    },
    save_gamma_values: function(){
        let json_body = {
        data: this.form,
        model: this.selected.model,
        region: this.selected.region,
        region_pos: this.regions_list.indexOf(this.selected.region),
      };
        axios
        .post(`http://localhost:5000/api/run_gamma/save_gamma/`, json_body, {
          headers: { "Content-Type": "application/json" },
        })
    },
    select_other_FTT: function(){
      this.initial = true
      this.loaded = false
    }

  },
  beforeMount() {
    this.initialise();
    this.log("message;message:Standing by...;");
    //this.loadselection();
  },
  beforeDestroy(){
        msgServer.close();
  },
  mounted() {
    localStorage.pivot = "";
    localStorage.table_fields = "";
    localStorage.Variable = "";
    chart.renderTo(document.getElementById("line"));
  },
  watch: {
    category: function () {
      this.charts = this.graphics[this.category].charts;
      this.tables = this.graphics[this.category].tables;
    },

    "selected.region": function () {
      this.load_current_gamma();
      this.update_data();
    },
    "selected.model": function () {
  
    },
    json_: function () {

      this.pivot = JSON.parse(this.json_.pivot);

      //if (this.selected.type == "chart"){
      this.display_data = JSON.parse(this.json_.results);

      // this.pivot_columns = this.json_.pivot_columns.filter(v => ['county','dimension','scenario'].indexOf(v) == -1)
      this.json_vars = this.json_.info.filter(
        (v) =>
          ["year", "dimension", "dimension2", "dimension3", "scenario"].indexOf(
            v
          ) == -1
      );
      //}
    },
    slider_limit:function(){
      this.slider_limit_min =  this.slider_limit * -1
    },
    display_data: function () {
      // if (globalStore.saved == true) return
      if (this.selected.type == "table") return;

      config.data = this.display_data;

      config.dimensions = {
        year: { type: "category" },
        dimension: { type: "category" },
        dimension2: { type: "category" },
        scenario: { type: "category" },
      };

      config.x = this.json_.x;
      config.y = this.json_.y;
      config.color = this.json_.color;

      config.type = this.json_.type;

      if (this.json_.label != "None") {
        config.label = this.json_.label;
      } else {
        config.label = "";
      }


      config.guide[0].y.label.text = this.json_.unit;
      for (var i = 0; i < this.json_vars.length; i++)
        config.dimensions[this.json_vars[i]] = { type: "measure" };

      if (this.quickfilter_on)
        config.plugins = [
          taucharts_tooltip,
          tau_legend(),
          tau_quickfilter(this.json_vars),
        ];
      else config.plugins = [taucharts_tooltip, tau_legend()];

      //this.title_update();
      chart.updateConfig(config);
      //this.title_fix()
    },

  },
  filters: {
    format_time: function (value) {
      if (!value) return '';
      let minutes = Math.floor(value / 60);
      let seconds = parseInt(value % 60);
      return `${minutes} m ${seconds} s`;
    }
  }
};
</script>


<!-- Add "scoped" attribute to limit CSS to this component only -->
<style scoped lang="scss">
.box {
  background-color: #e5e5e5;
  width: 100%;
  height: 295px;
}
#controlpanel {
  z-index: 1100;
}
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
.current_progress{
  .progress_text{
    width: 100%;
    margin-bottom: 5px;
  }
  span{
    font-size: 14px;
    float: left;
    &.right{
      float: right;
      font-size: 12px;
      text-align:right;
      color: #8C8C8C;
    }
  }
    margin: 20px -15px 20px -15px;
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
.county_box {
  height: 150px;
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
.color-us {
  stroke: blue;
}
.color-bug {
  stroke: red;
}
h1 {
  font-size: 48px;
  text-align: left;
  font-weight: 900;
}
.inline {
  display: inline;
}
.spinner {
  border-radius: 100%;
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
@import "http://cdn.jsdelivr.net/npm/taucharts@2/dist/taucharts.min.css";
</style>


