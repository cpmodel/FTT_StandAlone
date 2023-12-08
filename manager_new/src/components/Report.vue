<template>
  <div class="container">
    <b-row>
      <div class="col-12 draw_area">
        <b-row>
        <b-col cols="9">
        <div class="selected_variable">{{selected.latest}}
          <b-badge class="info" :style="{display: isSet()? '' : 'none'}" v-b-tooltip.hover variant="info" size="sm" :title="selected">i</b-badge>
        </div>
        </b-col>
        <b-col cols="3" class="download">

          
          <b-button size="sm"  v-on:click="download_data">Download data</b-button>
          <!-- <b-form-checkbox  id="TableDisplay" class="inline" v-model="chart_not_table" name="TableDisplay" switch>
            Chart or Table
          </b-form-checkbox> -->
          
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
            :items="pivot"
            :fields="table_fields"
            responsive="sm"
            class='table table-hover'
            style="min-height:400px">
          </b-table>
          <!-- Array.from(Object.keys(pivot[0])).map(e => /[0-9]{4}/g.exec(e) == null ? {key: `hello ${e}`, stickyColumn: true} : e) -->
        </div>

      </div>

    </b-row>
    <div class="row">
      <b-card class="box" id="controlpanel">
        <b-tabs card>

            <b-row>
                <b-col cols=8>
                  <label class="selection_label">Chapter to view:</label>
                  <treeselect :multiple="false" v-model="category" :options="categories"/>
                </b-col>
                <b-col cols=4>
                  <label class="selection_label">Scenario to view:</label>
                  <treeselect :multiple="false" v-model="selected.scenario" :options="scenarios"/>
                </b-col>            
            </b-row>
            <b-row>
            <b-col >
                  <b-form-group  label="Charts" class="selection_label">
                <b-form-radio-group v-b-tooltip.hover title="Charts" buttons button-variant="outline-primary"  id="color" v-model="selected.chart" name="radioChart" :options="charts" stacked>
                </b-form-radio-group>
              </b-form-group>
                </b-col>
                        <b-col >
                  <b-form-group  label="Tables" class="selection_label">
                <b-form-radio-group v-b-tooltip.hover title="Tables" buttons button-variant="outline-primary"  id="color" v-model="selected.table" name="radioChart" :options="tables" stacked>
                </b-form-radio-group>
              </b-form-group>
                </b-col>
            </b-row>

        </b-tabs>

      </b-card>
    </div>
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
      // formatters:{
      //   year:{label:"year",
      //         format:"%Y"
      //       }
      // }
     
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
    label:'',
    guide:[
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
  name: 'Report',
  data: () => {
    return {
      category:'',
      selected: {
        chart:'',
        table:'',
        latest:'',
        type:'',
        time:"Yes",
        scenario:"baseline",
        scenarios:[]

      },
      categories: [],
      charts:[],
      tables:[],
      pivot: [],
      pivot_columns:[],
      table_fields:[],
      chart_not_table:true,
      fields:[{key:"scenario",sortable: true },{key:"description",sortable: true },{key:"Last Run",sortable: true }],
      json_: [],
      json_vars: [],
      display_data: [],
      error: '',
      selection_warning: "Please select a chart or table to view.",
      
      }
  },
  components: { Treeselect },
  methods: {
    initialise: function() {
      // first load the possible counties and variables
      axios.get("http://localhost:5000/api/Report/Options")
      .then((res) => {
        
        let category = res.data.category;

        this.categories = category.map((s)=>{return {"label":s, "id":s}})
        this.category = this.categories[0].label
        this.graphics = res.data.graphics;
        //Get available variables
                }, (err) => {
        this.error_message(err, "scenarios")
      })
      // first load the possible counties and variables
      axios.get("http://localhost:5000/api/results/scenarios")
      .then((res) => {
        let scenarios = res.data.scenarios;
        //Initialse dropdown with first scenario
        this.selected.scenario = scenarios[0] 
        this.scenarios = scenarios.map((s)=>{return {"label":s, "id":s}})

        }, (err) => {
        this.error_message(err, "scenarios")
      })
    },
    isSet: function() {
      if(this){
        return this.selected.latest != '' 
      } else {
        return false;
      }
    },
    update_chart: function() {
    
      chart.updateConfig(config);
      if (this.first_run == true){
        this.selectFirstRow()
      }
      this.first_run = false
    },
    update_data: function(){
      let graphic_label = this.selected.latest
      graphic_label.replace(" ","-")
      axios.get(`http://localhost:5000/api/Report/Values/${graphic_label}/json`)
      .then((res) => {
        this.json_ = res.data;
      }, (err) => {
        this.error_message(err, "requested chart data")
      })
    },
    set_table_fields: function (){
      let table_fields = [{
                            key: 'scenario',
                            sortable: false,
                            stickyColumn: true,
                            isRowHeader: true
                          },
                          {
                            key: 'dimension',
                            sortable: false,
                            stickyColumn: true,
                            isRowHeader: true
                          },
                          {
                            key: 'dimension2',
                            sortable: false,
                            stickyColumn: true,
                            isRowHeader: true
                          },
                          {
                            key: this.selected.time=="Yes"? 'dimension3' : 'year',
                            sortable: false,
                            stickyColumn: true,
                            isRowHeader: true
                          }
                          ]
      this.update_data();
      this.pivot = JSON.parse(this.json_.pivot)

      if(this.pivot[0].length > 0){
          Object.keys(this.pivot[0]).slice(0,Object.keys(this.pivot[0]).length-3).forEach((y)=>{
            // if(y >= this.minmax_year[0] && y <= this.minmax_year[1]){
            table_fields.push({key: y})
            // }
          })
      }
      //this.table_fields = JSON.stringify(table_fields);

      this.table_fields =table_fields
      //this.pivot = JSON.stringify({'items':this.pivot});

      // localStorage.variable = this.selected.variable_label.label
    },
    download_data: function() {
      let graphic_label = this.selected.latest
      graphic_label.replace(" ","-")
      axios.get(`http://localhost:5000/api/Report/Values/${graphic_label}/csv`)
      .then((res) => {
        let blob = new Blob([res.data], {type: 'text/csv'}),
        url = window.URL.createObjectURL(blob);
        var fileLink = document.createElement('a');
        fileLink.href = url;
        fileLink.download = `FTT_data_download_${new Date().toISOString()}.csv`;
        fileLink.click();
      }, (err) => {
        this.error_message(err, "download data")
      })
    },
    error_message: function(err, item){
      var msg = ""
      if (!err.response){
        msg = 'Failed to get '+ item +', the manager is either not running or encountered an error (Check backend is running).';
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
  watch: {


    'category':function(){

        this.charts = this.graphics[this.category].charts
        this.tables = this.graphics[this.category].tables 
    },
    "selected.chart":function(){

      if (this.selected.chart != ''){
        this.selected.table = ''
      this.selected.latest = this.selected.chart
      this.selected.type = "chart"
      this.update_data()
      this.chart_not_table = true
      } 
    },
      "selected.table":function(){
      if (this.selected.table != ''){
        this.selected.chart = ''
      this.selected.latest = this.selected.table
      this.selected.type = "table"
      this.update_data();
  
      this.chart_not_table = false
      }
    },
    json_: function(){
      
      this.pivot = JSON.parse(this.json_.pivot);
      if (this.selected.type == "chart"){
      this.display_data = JSON.parse(this.json_.results);
      // this.pivot_columns = this.json_.pivot_columns.filter(v => ['county','dimension','scenario'].indexOf(v) == -1)
        this.json_vars = this.json_.info.filter(v => ['year','dimension','dimension2','dimension3','scenario'].indexOf(v) == -1);
      }
    },
    display_data: function(){

      // if (globalStore.saved == true) return
      if (this.selected.type == "table") return 

      config.data = this.display_data;
      
      config.dimensions = {
        year: {type: 'category'},
        dimension: {type: 'category'},
        dimension2: {type: 'category'},
        scenario: {type: 'category'}
      };
      
        config.x = this.json_.x
        config.y = this.json_.y
        config.color = this.json_.color
        
        config.type = this.json_.type

        if (this.json_.label != "None"){ 
          config.label = this.json_.label
        }
        else{
          config.label = "" 
        }
        console.log(config)
        console.log(this.json_.brewer)
        config.guide[0].color.brewer = this.json_.brewer
        config.guide[0].y.label.text = this.json_.unit
      for (var i = 0; i < this.json_vars.length; i++)
        config.dimensions[this.json_vars[i]] = {type: 'measure'}

      if (this.quickfilter_on)
        config.plugins = [taucharts_tooltip,tau_legend(),tau_quickfilter(this.json_vars)];
      else
        config.plugins = [taucharts_tooltip,tau_legend()]

      //this.title_update();
      chart.updateConfig(config);
      //this.title_fix()
    },

  }
} 
</script>

<!-- Add "scoped" attribute to limit CSS to this component only -->
<style scoped lang="scss">
.box{
  background-color: #E5E5E5;
  width:100%;
  height:295px
}
#controlpanel{
  z-index: 1100;
}
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
