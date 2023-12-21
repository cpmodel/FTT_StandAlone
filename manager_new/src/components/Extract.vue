<template>
  <div class="container">

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
              <label class="selection_label">Indicators</label>
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
            <b-row class="form-row" v-for="(value,key) in variables" v-bind:key=(value,key)>
              <b-card no-body class="mb-1">
                <b-card-header header-tag="header" class="p-1" role="tab">
                  <b-button block v-b-toggle=value.id variant="info">{{value.id}}

                  </b-button>
                </b-card-header>
                <b-collapse :id="value.id" visible accordion="my-accordion" role="tabpanel">
                  <b-form-checkbox-group class="checkboxgroup" v-model="selected.variable" :options=value.children value-field="id" text-field="label"  stacked></b-form-checkbox-group>
                </b-collapse>
              </b-card>
            </b-row>
       
              </b-col>
              <b-col >
                <b-row>
                  <b-col>
                    <label class="selection_label">Regions</label>
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
                  :showCount="false" :appendToBody="false" defaultExpandLevel=2 :maxHeight="325" :sortValueBy="'LEVEL'" :limit="Levels"
                  openDirection="below" :flat="true" :limitText="tree_limit_text" />
                </div>
              </b-col>
            </b-row>
          </b-tab>

          <b-tab title="Settings" class="tab-pane container " id="ChartSet">
            <b-row>
            <b-col>
                <b-form-group label="Show"  class="selection_label top_margin">
                  <b-form-radio-group id="calculation" v-model="selected.calculation"
                  :options="calculation_methods" name="radioCalc">
                  </b-form-radio-group>
                </b-form-group>
            </b-col>
         
            </b-row>
          </b-tab>
        </b-tabs>

      </b-card>
      
    </div>
    <br>
        <b-button size="lg"  v-on:click="download_data">Download data</b-button>
  </div>
</template>

<script>
import lodash from 'lodash'
import axios from 'axios'
import * as taucharts from 'taucharts'
import tau_tooltip from 'taucharts/dist/plugins/tooltip'
import tau_quickfilter from 'taucharts/dist/plugins/quick-filter'
import tau_legend from 'taucharts/dist/plugins/legend'

import Treeselect from '@riophae/vue-treeselect'
import '@riophae/vue-treeselect/dist/vue-treeselect.css'
import {globalStore} from '../main.js'
var taucharts_tooltip = taucharts.api.plugins.get('tooltip')({
      formatters:{
        year:{label:"year",
              format:"%Y"
            }
      }
});
var config = {
    data: [
      {dimension:'us', y_real:20, q_real:23, year:'2013', dimension2: 'a'}
    ],
    dimensions: {
      year: {type: 'measure', scale: 'time'},
      y_real: {type: 'measure'},
      q_real: {type: 'measure'},
      dimension: {type: 'category'},
      dimension2: {type: 'category'},
      dimension3: {type: 'category'}
    },
    guide:[
      {
        y: {label:{text:"test"}},
        color:{
        brewer:['rgb(197, 68, 110)', 'rgb(73, 201, 197)', 'rgb(170, 183, 29)', 'rgb(0, 99, 152)', 'rgb(0, 122, 178)', 'rgb(100, 100, 100)']
        }
      },
      {
        y: {label:{text:"test"}},
        color:{
          brewer:['rgb(197, 68, 110)', 'rgb(73, 201, 197)', 'rgb(170, 183, 29)', 'rgb(0, 99, 152)', 'rgb(0, 122, 178)', 'rgb(100, 100, 100)']
        }
      }

    ],

    plugins: [
    taucharts_tooltip,
    tau_legend()],
    type: 'line',
    x: 'year',
    y: 'y_real',
    order: 'year',
    color: ['scenario',"dimension"] // there will be two lines with different colors on the chart
};
var chart = new taucharts.Chart(config);

export default {
  name: 'Results',
  data: () => {
    return {
      pivot: "[]",
      pivot_columns:[],
      table_fields:"[]",
      facet: null,
      facet_x: null,
      color: null,
      json_: [],
      display_data: [],
      error: '',
      chart: {},
      chart_colours: [],
      variables: [],
      variables_summary: [],
      variables_detailed: [],
      variables_all:[],
      var_labels: [],
      dimensions: [],
      json_vars: [],
      var_groups: [],
      scenarios: [],
      scenarios_drop: [],
      fields:[{key:"scenario",sortable: true },{key:"description",sortable: true },{key:"Last Run",sortable: true }],
      scenarios:[],
      sort_by: "scenario",
      selectMode: "multi",
      selected_table: [],
      titles:[],
      titles2:[],
      titles3:[],
      quickfilter_on: false,
      settings_loaded: true,
      select_all: false,
      select_all2: false,
      select_all3: false,
      loading: false,
      first_run:true,
      chart_not_table:true,
      indic_detail:0,
      code_label:false,
      no_aggregate:false,
      dimensions_drawn: [],

      scenarios_drawn: ["baseline"],
      title_old: "null",
      selection_warning: "Please set the scenario, variable and dimensions to display results.",
      selected: {
        variable: [],
        variable_label: {},
        dimensions: [],
        dimensions2: ["All"],
        dimensions3: ["All"],
        title: [],
        title2: [],
        title3: [],
        time:"Yes",
        scenarios: ["Baseline"],
        aggregate: false,
        baseline: 'Baseline',
        calculation: 'Levels',
        scen_index:[0],

      },
      calculation_methods: [
        {text:'Absolute value', value:'Levels'},
        {text:'Absolute difference from baseline', value:'absolute_diff'},
        {text:'Percentage difference from baseline', value:'perc_diff'},
        {text:'Year-on-Year change', value:'Annual growth rate'}
      ],
      aggregation_methods: [
        {text:'None', value:'0'},
        {text:'Sum', value:'sum'}
      ],
      minmax_year: [2006,2016]
    }
  },
  components: { Treeselect },
  methods: {
    initialise() {

      //Get available variables
      this.get_vars()

      axios.get("http://localhost:5000/api/scenarios_ran")
        .then((res) => {
          let scenarios=res.data.exist
          this.scenarios=scenarios
          this.scenarios_drop = scenarios.map(s => {
            return { label: s["scenario"], id: s["scenario"] }
        })
        })
    },
    onRowSelected(items) {
        this.selected_table = items
        this.selected.scenarios = []
        this.selected.scen_index = []
        for (var i = 0; i < this.selected_table.length; i++){
          // if(this.selected.counties.includes(this.selected_table[i]["county"])==false){
          //   this.selected.counties.push(this.selected_table[i]["county"])
          // }
          if(this.selected.scenarios.includes(this.selected_table[i]["scenario"])==false){
            this.selected.scenarios.push(this.selected_table[i]["scenario"])
          }
          for (var j = 0; j < this.scenarios.length; j++){
            if ( this.selected_table[i]["scenario"] == this.scenarios[j]["scenario"])
              this.selected.scen_index.push(j)
          }
        }
    },
    selectFirstRow() {
        // Rows are indexed from 0, so the third row is index 2
          for (var j = 0; j < this.selected.scen_index.length; j++){
            this.$refs.selectableTable.selectRow(this.selected.scen_index[j])
          }



      },
    get_vars(){
      axios.get("http://localhost:5000/api/results/variables")
      .then((res) => {
        
        let variables = res.data.vars.indicies;
        this.var_groups = res.data.vars.groups;
        this.var_labels = res.data.vars.labels;
        this.variables_all = variables
        this.variables_detailed = res.data.vars.indicies_detailed;
        this.variables_summary = res.data.vars.indicies_summary
        this.variables = this.variables_summary;
        this.titles = res.data.vars.title_map;
        this.chart_colours = res.data.vars.chart_colours;
        }, (err) => {
        this.error_message(err, "variables")
      })
    },
    get_dimensions() {
      if (this.selected.scenarios.length==0) return

      this.selected.titles = this.titles[this.selected.variable]
      if (this.selected.titles["title4"] == "TIME"){
        this.selected.time = "Yes"
      }
      else{
        this.selected.time = "No"
      }
  
      axios.get(`http://localhost:5000/api/info/${this.selected.titles["title"]}`)
      .then((res) => {
         if ("Sectors" in res.data){
           
           this.dimensions = res.data.Sectors
         }
         else{
          let dimensions = res.data
          this.dimensions = dimensions.map((s)=>{return {"label":s, "id":s}})
         }
        }, (err) => {
        this.error_message(err, "dimensions")
      })
 
     
      this.selected.title[0] = this.titles[this.selected.variable[0]]["title"]




    },

   
    download_data: function() {
      this.selected.title = []
      this.selected.title2 = []
      this.selected.title3 = []
      this.selected.variable_label = []
      var temp = new Array(0)
      for (var v = 0; v < this.selected.variable.length; ++v) {
          temp.push(this.selected.variable[v])
          this.selected.title.push(this.titles[this.selected.variable[v]]["title"])
          this.selected.title2.push(this.titles[this.selected.variable[v]]["title2"])
          this.selected.title3.push(this.titles[this.selected.variable[v]]["title3"])
          this.selected.variable_label.push(this.var_labels[this.selected.variable[v]])
      }
      this.selected.variable = temp
      
      console.log("Variable:")
      console.log(this.selected.title)
      console.log(this.selected.variable)
      console.log(this.selected.dimensions)

      let params = lodash.cloneDeep(this.selected);

      axios.get(`http://localhost:5000/api/results/data/csv`, {params: params})
      .then((res) => {
        let blob = new Blob([res.data], {type: 'text/csv'}),
        url = window.URL.createObjectURL(blob);
        var fileLink = document.createElement('a');
        fileLink.href = url;
        fileLink.download = `FTT_standalone_download_${new Date().toISOString()}.csv`;
        fileLink.click();
      }, (err) => {
        this.error_message(err, "download data")
      })
    },
    isSet: function() {
      if(this){
        return this.selected.variable != '' & this.selected.dimensions.length > 0 & this.selected.dimensions2.length > 0 & this.selected.dimensions3.length > 0 & this.selected.scenarios.length > 0;
      } else {
        return false;
      }
    },
    isdrawn: function() {

      if (this.arraysEqual(this.selected.dimensions,this.dimensions_drawn)==false) return false
      if (this.arraysEqual(this.selected.dimensions2,this.dimensions_drawn2)==false) return false
      if (this.arraysEqual(this.selected.dimensions3,this.dimensions_drawn3)==false) return false    
      if (this.arraysEqual(this.selected.scenarios,this.scenarios_drawn)==false) return false    

      
       
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
 
     tree_limit_text: function(){
      return ""
    },
      error_message: function(err, item){
      var msg = ""
      if (!err.response){
        var msg = 'Failed to get '+ item +', the manager is either not running or encountered an error (Check backend is running).';
      }
      else{
        msg = 'Failed to get '+ item +', the manager is either not running or encountered an error.' + "(" + err.response.statusText + ")";
      }
      alert(msg)
    }
  },
  beforeMount(){

    this.initialise();
    //this.loadselection();

  },
  mounted(){

    localStorage.pivot = '';
    localStorage.table_fields = '';
    localStorage.Variable = '';
    chart.renderTo(document.getElementById('line'));



  },
  beforeUpdate(){

  },
  beforeDestroy(){

    //this.saveselection()

  },
  watch: {
    'variables': function(){
      
      if (this.settings_loaded == true && this.selected.variable != this.variables[0].children[0].id){
        this.selected.variable = [this.variables[0].children[0].id]
        
        this.get_dimensions()
      }
    },
    'scenarios': function(){
    },
    'dimensions': function(){

        if (this.dimensions.length == 1)
          this.selected.dimensions = [this.dimensions[0].id]
        else if(this.selected.title == this.title_old && this.dimensions_drawn.length>0)
          this.selected.dimensions = this.dimensions_drawn 
        else if(this.settings_loaded == true)
          this.selected.dimensions = [this.dimensions[0].id];
        if (this.selected.title != "")
        this.title_old = this.selected.title
    },
  


    'selected.variable': function(){

    },

    'select_all' :function(){
      this.selected.dimensions = []
      if (this.select_all){
        for (var i = 0; i < this.dimensions.length; i++)
         this.selected.dimensions.push(this.dimensions[i].id)
      }
      else
         this.selected.dimensions.push(this.dimensions[0].id)
    },
    'indic_detail': function(){
      if (this.indic_detail == 0)
        this.variables = this.variables_summary
      else if (this.indic_detail == 1)
        this.variables = this.variables_detailed
      else if (this.indic_detail == 2)
        this.variables = this.variables_all
      if (this.code_label)
        this.variables = this.variables.map((s) => { return { "label": s.id, "id": s.label, "children": s.children.map((v)=>{return{"label":v.id,"id":v.id}}) } })
      
    },
    'code_label': function(){
      if (this.indic_detail == 0)
        this.variables = this.variables_summary
      else if (this.indic_detail == 1)
        this.variables = this.variables_detailed
      else if (this.indic_detail == 2)
        this.variables = this.variables_all
      if (this.code_label)
        this.variables = this.variables.map((s) => { return { "label": s.id, "id": s.label, "children": s.children.map((v)=>{return{"label":v.id,"id":v.id}}) } })

    },    
  }
}
</script>

<!-- Add "scoped" attribute to limit CSS to this component only -->
<style scoped lang="scss">
.box{
  background-color: #E5E5E5;
  width:100%;
  height:700px;
}
// #controlpanel{
//   z-index: 1100;
// }
.overlay{
  background-color: rgba(255,255,255,1);
  position: absolute;
  left: 0;
  width: 100%;
  height: 90%;
  z-index: 1000;
  transition: 1.5s;
  span{
    margin-top: 25%;
    display: block;
    font-size: 24px;
    font-weight: 700;
  }
}
.table{
  background-color:whitesmoke;
  height: 175px;
  overflow-y: scroll
}
.content{
  height: 100%;
  // overflow-y: auto;
}
.county_box{
  height: 150px;
}
.dimension_box{
  height: 150px;
}
.draw_area{
  height: 100%;
}
.card-body{
    padding-top: 0.1rem;
    padding-bottom: 0.1rem;

}
.mb-1{
  width:100%;
}
.checkboxgroup{
  height:200px;
  overflow:scroll; 
  text-align: left;
  margin-left: 10px;
}
body,#line {
    height:50vh;
}
.color-us {
  stroke:blue
}
.color-bug {
  stroke:red
}
h1{
  font-size: 48px;
  text-align: left;
  font-weight: 900;
}
.inline{
  display: inline
}
.selection_label{
  font-size: 14px;
  font-weight: 700;
  text-align: left;
  width: 100%;
  margin-top: 10px;
  *{
    font-weight: 400;
  }
}
.download{
  margin-top: 10px;
  padding: 0.2em;
  display: flex;
  align-items: center;
}
.selected_variable{
  font-size: 30px;
  font-weight: 700;
  text-align: left;
  padding: 0.25em
}
.info{
  border-radius: 30% !important;
  font-size: 14px;
}

.button_ow{
  color: white;
  &:hover{
    text-decoration: none;
  }
}
.smallselecttext{
  font-size: 0.85rem
}
.tau-chart__filter__wrap .resize.w text {
  text-anchor: start;
  font-size: 12px;
}

#data-table-main{
  font-size: 0.75em;
}
.tau-chart__tooltip{
  z-index: 1100;
}
</style>

<style lang="css" scoped>
   @import 'http://cdn.jsdelivr.net/npm/taucharts@2/dist/taucharts.min.css'
</style>


