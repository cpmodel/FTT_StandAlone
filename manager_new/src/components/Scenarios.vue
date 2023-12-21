<template>
  <div class="container">
   <h1>Scenarios</h1>
    <div class="row">
      <div class="box row col-12">

            <div class="col-3">
              <h2>County</h2>
              <p class='subtitle'>Choose county to work on</p>
            </div>
            <div class="col-9">
              <treeselect id="regselect" v-model="county" :multiple="false" :options="counties"
              :alwaysOpen="false" :clearable="false" :defaultExpandLevel="1" :flattenSearchResults="true"
              :appendToBody="true" :maxHeight="200"/>
            </div>
          </div>
    </div>

    <div class="row"> &nbsp;</div>

    <div class="row">
      <div class="box row col-12 no-bottom-padding">
          <div class="col-12">
            <div class='row'>
              <div class="col-8">
                <h2>Scenarios for <span id="county-name" class="highlight">{{ county ? county : "No county selected" }}</span></h2>
                <p class='subtitle'>Create, edit, and delete impact scenarios</p>
              </div>
              <div class="col-4" align="right">
                <!--<b-button size="sm" v-b-modal.modal-new-scenario>Create new scenario</b-button>-->
              </div>
            </div>
          </div>
        </div>
        <div class="row box col-12 no-top-padding">
          <div class="col-12">
          <table v-if="scenarios.length > 0" class='table table-hover'>
            <thead>
              <tr>
                <th>Name</th>
                <th>Description</th>
                <th>Last saved</th>
                <th colspan="4">Actions</th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="s in scenarios" v-bind:key=s>
                <td>{{s.scenario}}</td>
                <td>{{s.description}}</td>
                <td>{{s["Last Saved"] == 0 ? 'never' : parseInt(s["Last Saved"]) | moment("LLLL")}}</td>
                <td>
                  <b-button :disabled="s.locked" title="Delete Scenario"
                  v-on:click="s.locked ? null : confirm_del_scenario(s.scenario)">
                    <font-awesome-icon class="faicon" icon="trash-alt"/>
                  </b-button>
                </td>
                <td>
                  <b-button :disabled="s.locked" title="Edit Scenario"
                  v-on:click="s.locked ? null : load_scenario(s.scenario)" v-ripple>
                    <font-awesome-icon class="faicon"  icon="edit"/>
                  </b-button>
                </td>
                <td>
                  <b-button title="Copy Scenario" v-on:click="copy_scenario(s.scenario)" v-ripple>
                    <font-awesome-icon class="faicon" icon="copy"/>
                  </b-button>
                </td>
              </tr>

            </tbody>
          </table>
          <span v-if="scenarios.length == 0">No scenarios are present / no country is selected</span>
        </div>
        </div>
    </div>

    <div class="row"> &nbsp;</div>

    <div class="row">
      <div class="box row col-12 no-bottom-padding" v-if="scen_name">
          <div class="col-7">
            <h2>Editing <span id="county-name" class="highlight">{{ scen_name }}</span></h2>
            <p class='subtitle'>Last saved {{ last_saved == 'never' ? last_saved : parseInt(last_saved) | moment("LLLL") }}</p>
          </div>
          <div class="col-5">
            <label class="selection_label" for="description_edit">Description of the scenario</label>
            <b-form-textarea id="description_edit" v-model="scen_desc" placeholder="A short description of the scenario"></b-form-textarea>
          </div>
        </div>
        <div class="row box col-12 no-top-padding no-bottom-padding" v-show="scen_name">
          <div class="box form-group col-md-12 no-bottom-padding">
            <hr style="border-width: 1px">
            <b-tabs content-class="mt-3" class="spread_tabs" v-model="tabIndex" @activate-tab="check_tabs">
                <hr style="border-width: 1px">
                <!--<b-tab v-for="s in scen_inputs" v-bind:key=s :title="s">-->

                <b-tab title=Assumptions>
                  <b-form-select v-model="assumption" :options="scen_inputs"></b-form-select>
                  <div class=table_new ref="assumption"></div>
                </b-tab>

                <b-tab title=Impacts>
                  <b-row>
                    <b-col>
                      <table v-if="impacts.length > 0" class='table table-hover'>
                        <thead>
                          <tr>
                            <th>Name</th>
                            <th>Industry</th>
                            <th>Type</th>
                            <th>Last saved</th>
                            <th colspan="4">Actions</th>
                          </tr>
                        </thead>
                        <tbody>
                          <tr v-for="i in impacts" v-bind:key=i>
                            <td>{{i.name}}</td>
                            <td>{{sectors_raw[i.industry]}}</td>
                            <td>{{i.type == 'open' ? 'Opening' : 'Closure'}}</td>
                            <td>{{i.modified}}</td>
                            <td>
                              <b-button title="Delete Impact"
                              v-on:click="confirm_del_impact(`${i.name}_${i.industry}`)" v-ripple>
                              <font-awesome-icon class="faicon" icon="trash-alt" />
                              </b-button>
                            </td>
                            <td>
                              <b-button title="Edit Impact"
                              v-on:click="load_impact(`${i.name}_${i.industry}`, i.type)" v-ripple>
                              <font-awesome-icon class="faicon" icon="edit" />
                              </b-button>
                            </td>
                          </tr>
                        </tbody>
                      </table>
                      <b-button v-ripple block variant='primary' v-on:click="create_new_impact_wizard()" v-if='impacts.length == 0'>Add new impact</b-button>
                    </b-col>
                  </b-row>
                  <b-row>&nbsp;</b-row>
                  <b-row v-if="selected_impact_file">
                    <b-col>
                      <h2>Editing impact <span class="highlight"> {{ selected_impact_file.split("_")[0] }}, </span>
                        <small>industry: {{ sectors_raw[selected_impact_file.split("_")[1]] }} </small></h2>
                      <b-col cols=9>
                        <b-form-group
                          class="mb-0"
                          label=""
                          label-for="impact_tab_select"
                          label-align="left">
                            <b-form-select
                            id="impact_tab_select"
                            v-model="impact_tab"
                            :options="selected_impact_type == 'open' ? [
                            {'text':'Construction phase assumptions','value':'construction'},
                            {'text':'Operation - Additional direct jobs and output','value':'operation'},
                            {'text':'Supply chain - Bought-in goods and services for production','value':'supply_chain'}
                            ] : [
                            {'text':'Operation - Additional direct jobs and output','value':'operation'},
                            {'text':'Supply chain - Bought-in goods and services for production','value':'supply_chain'}
                            ]">
                            </b-form-select>
                        </b-form-group>
                      </b-col>
                    </b-col>
                  </b-row>
                  <b-row>&nbsp;</b-row>
                  <b-row v-if="selected_impact_file">
                    <b-col>
                      <div class=table_new ref="impacts_table"></div>
                    </b-col>
                  </b-row>
                  <!-- <b-form-select v-model="assumption" :options="scen_inputs"></b-form-select> -->
                  <!-- <div class=table_new ref="assumption"></div> -->
                </b-tab>
                <!-- <b-tab title="Investment" class="text-dark"><div class=table_new ref="spreadsheet_1"></div></b-tab>
                <b-tab title="Additional Employment/Output"><div class=table_new ref="spreadsheet_2"></div></b-tab>-->
            </b-tabs>
          </div>
        </div>

        <div class="row box col-12" v-if="scen_name">
          <div class="col-12">
            <hr>
            <b-button v-on:click="save_scenario" v-if="tabIndex == 0">Save</b-button>
            <b-button v-on:click="save_impact" v-if="tabIndex == 1 && selected_impact_file ">Save</b-button>
          </div>
        </div>
    </div>

    <b-modal id="modal-new-scenario" ref="modal_new_scen" title="Create new scenario"
    @show="reset_modal"
    @hidden="cancel_modal"
    @ok="new_scenario"
    >
    <form ref="form" @submit.stop.prevent="check_scenario_name_valid">
      <b-form-group
        label="Scenario name"
        label-for="name-input"
        :state="name_state"
        invalid-feedback="Scenario name must be unique"
      >
        <b-form-input
          id="name-input"
          v-model="new_name"
          required
        ></b-form-input>
        <div class="form-row">
            <b-col>
              <label class="selection_label" for="s_year_input">Start year</label>
              <b-form-input id="s_year_input" v-model="sYear" :disabled="copy"></b-form-input>
            </b-col>
            <b-col>
              <label class="selection_label" for="e_year_input">End year</label>
              <b-form-input id="e_year_input" v-model="eYear" :disabled="copy"></b-form-input>
            </b-col>
        </div>
        <div class="form-row">
            <b-col>
              <label class="selection_label" for="description_input">Description of the scenario (can be edited later):</label>
              <b-form-textarea id="description_input" v-model="scen_desc" placeholder="A short description of the scenario"></b-form-textarea>
            </b-col>
        </div>
      </b-form-group>
    </form>
    </b-modal>

    <b-modal id="modal-delete-scenario" ref="modal_delete_scen" title="Confirm scenario deletion" ok-title="Confirm" @ok="delete_scenario">
      Are you sure to delete scenario "{{ this.scen_name_delete }}" for county {{ this.county }}? <br>
      This action cannot be undone.
    </b-modal>

    <b-modal id="modal-new-impact" ref="modal_new_impact" title="Create new impact" ok-title="Create"
    @ok="create_impact"
    @show="impact_wizard_reset">
      <b-col>
        <b-row>
          <label for="modal_impact_name">Impact name</label><br>
          <b-form-input id="modal_impact_name" v-model="impact_name" placeholder="New impact" :state="validate_impact_name">
          </b-form-input>
        </b-row>
        <b-row>
          <label for="modal_impact_industry">Impacted industry</label><br>
          <treeselect id="modal_impact_industry" v-model="impact_industry" :multiple="false" :options="sectors" :alwaysOpen="false" :clearable="false"
          :appendToBody="true" :defaultExpandLevel="1" :maxHeight="390" :limit="5"
          :value-consists-of="'LEAF_PRIORITY'" zIndex=1060 />
        </b-row>
        <b-row>
          <label for="modal_impact_pricebase">Price base</label><br>
          <b-form-input type='number' id="modal_impact_pricebase" v-model="impact_pricebase"
          min=2015 max=2050 :state="validate_impact_pricebase"/>
        </b-row>
        <b-row>
          <b-form-group label='Impact type'>
            <b-form-radio-group v-model='impact_type' id="radio-group-2" :options="[{'text':'Opening','value':'open'},{'text':'Closure','value':'close'}]" />
          </b-form-group>
        </b-row>
      </b-col>
    </b-modal>

    <b-modal id="modal-delete-impact" ref="modal_delete_impact" title="Confirm impact deletion" ok-title="Confirm" @ok="delete_impact">
      Are you sure to delete this impact? <br>
      This action cannot be undone.
    </b-modal>

  </div>

</template>

<script>
import axios from 'axios'
import Treeselect from '@riophae/vue-treeselect'
import '@riophae/vue-treeselect/dist/vue-treeselect.css'
import jexcel from 'jexcel'
import 'jexcel/dist/jexcel.css'
import VueMoment from 'vue-moment'
// var jExcelObj
import {globalStore} from '../main.js'

function getExcelCol(num){
  // num from 1!

  // 65 - A char, 1st COL
  // 91 - Z char, last 1 char COL
  if (num < 27){
    return String.fromCharCode(64+num)
  } else {
    let first = String.fromCharCode(64 + Math.floor(num / 26.5))
    let second = String.fromCharCode(64 + num - 26 * Math.floor(num / 26.5))
    return `${first}${second}`
  }
}

export default {
  name: 'Scenarios',
  data: () => {
    return {
      tabIndex: 0,
      form: {},
      scenarios: [],
      impacts: [],
      scenario_data: {},
      new_name:'',
      name_state:'valid',
      scen_name: null,
      scen_desc: '',
      scen_input_selected: '',
      scen_inputs:["Investment","Change in Employment and output"],
      impact_name: '',
      impact_industry: "0",
      impact_pricebase: 2015,
      impact_type: 'open',
      impact_name_delete: '',
      impact_data: {},
      selected_impact_file: null,
      assumption:'',
      error: '',
      counties: [],
      sectors: [],
      county: null,
      save_message: '',
      options:{
        data:  [],
        columns:  [],
      },
      sYear: 2020,
      eYear: 2050,
      copy: null,
      copy_name:"",
      tables:{},
      table_drawn: false,
      existing_scenarios:{},
      scen_name_delete: null,
      last_saved: "never",
      last_saved_states:{},
      jExcelObj: null,
      impact_tab: 'construction',
      jExcelObjImpact: null,
      impacts_table: {
        table_drawn: false,
        table: {
          data: [],
          columns: []
        },
        tables: {}
      }
    }
  },
  computed: {
    jExcelOptions() {
      return {
        data: this.options.data,
        columns: this.options.columns,
        rowDrag:false,
        tableOverflow:false,
        allowInsertColumn:false,
        allowInsertRow:false,
        allowManualInsertRow:false,
        allowManualInsertColumn:false,
        allowDeleteRow:false,
        allowDeleteColumn:false,
        allowRenameColumn:false,
        wordWrap:true,
        updateTable:function(instance, cell, col, row, val, label, cellName) {
          // Format first column with indicator values to overwrite default readonly formatting
          if (col == 0) {
            cell.style.color = 'black';
            cell.style.backgroundColor = '#f3f3f3';
            cell.style.textAlign = 'left';
            //cell.style.position = 'sticky';
            //cell.style.left = 0;
          }
          /*if (row == 2){
            cell.className = 'readonly'
            cell.style.color = 'black';
            cell.style.backgroundColor = '#f3f3f3';
          }*/
        },
        onchange: function(instance, cell, col, row, val, label, cellName){
          // Prevent non numerical values being entered and reset to zero
          if (col != 0){
            if (isNaN(parseInt(cell.innerHTML))){
              cell.innerHTML =0
            }
          }
        },
      }
    },
    jExcelOptions_impacts() {
      return {
        data: this.impacts_table.table.data,
        columns: this.impacts_table.table.columns,
        rowDrag:false,
        tableOverflow:false,
        allowInsertColumn:false,
        allowInsertRow:false,
        allowManualInsertRow:false,
        allowManualInsertColumn:false,
        allowDeleteRow:false,
        allowDeleteColumn:false,
        allowRenameColumn:false,
        wordWrap:true,
        updateTable: (instance, cell, col, row, val, label, cellName) => {
          // Format first column with indicator values to overwrite default readonly formatting
          if (col == 0) {
            cell.style.color = 'black';
            cell.style.backgroundColor = '#f3f3f3';
            cell.style.textAlign = 'left';
          }
          switch(this.impact_tab){
            case 'construction':
              if (row == 1 | row == 10) {
                cell.classList.add('readonly');
                cell.style.backgroundColor = '#f3f3f3';
                cell.style.color = 'black';
                if (row == 1 & col != 0){
                  cell.innerHTML = ""
                }
              }
              break;
            case 'operation':
              if (row == 2) {
                cell.classList.add('readonly');
              }
              break;
            case 'supply_chain':

              break;
          }
        },
        onchange: function(instance, cell, col, row, val, label, cellName){
          // Prevent non numerical values being entered and reset to zero
          if (col != 0){
            if (isNaN(parseInt(cell.innerHTML))){
              cell.innerHTML =0
            }
          }
        },
      }
    },
    validate_impact_name() {
      if(this.impact_name.length > 0){
        return /^[a-z0-9_. ()-]+$/i.test(this.impact_name);
      }
      return false
    },
    validate_impact_pricebase() {
      return (this.impact_pricebase >= 2015) & (this.impact_pricebase <= 2050) ? true : false
    }
  },
  beforeMount(){
    this.initialise();
    this.loadselection();
  },
  watch: {
    "options.data":function(){
      //console.log(this.options.data)
    },
    "county": function(){
      this.tabIndex = 0
      this.scen_name = null
      this.selected_impact_file = null
      this.load_scenarios()
    },
    "scen_name": function(){
      this.tabIndex = 0
      this.selected_impact_file = null
      this.listImpacts()
    },
    "counties": function(){
      this.county = this.counties[0].id
    },
    "assumption": function(){
      this.draw_tables()
    },
    "scen_inputs": function(){
      this.assumption = this.scen_inputs[0].value
    },
    "impact_tab": function(){
      this.draw_impact_tables()
    },
},
  components: {Treeselect},
  methods: {
    check_tabs: function(newTabIndex, prevTabIndex, bvEvt) {
      if(newTabIndex == 1 & this.last_saved == 'never'){
        bvEvt.preventDefault()
        this.$bvToast.toast('Please save the scenario first (button at the bottom of the screen), before specifing impacts.', {
          title: 'Missing data',
          toaster: 'b-toaster-bottom-right',
          appendToast: true,
          autoHideDelay: 7000,
          variant: 'danger'
        })
      }
    },
    initialise: function() {
      this.impact_pricebase = this.$moment().year();
      // first load the possible variables
      axios.get("http://localhost:5000/api/available")
      .then((res) => {
        // eslint-disable-next-line
        let reg = res.data.areas;
        this.counties = reg;
      }, (err) => {
        this.error = 'Failed to get area names, the manager is either not running or encountered an error. (' + err.response.statusText + ')';
      })
      axios.get("http://localhost:5000/api/classifications/q")
      .then((res) => {
        let q = res.data['data']
        this.sectors_raw = []
        for(const k in q){
          this.sectors_raw.push(q[k])
        }
        this.sectors = []
        for(const k in Object.keys(q)){
          this.sectors.push({"id": k, "label": q[k]})
        }
      })
    },
    copy_scenario: function(name){
      this.copy = true
      this.copy_name = name

      let pos = this.scenarios.map(x => x['scenario']).indexOf(name);
      let copy_meta = this.scenarios[pos]

      this.sYear = copy_meta['min']
      this.eYear = copy_meta['max']
      this.new_name = `Copy of ${name}`
      this.$refs['modal_new_scen'].show()
    },
    confirm_del_scenario: function(name){
      this.scen_name_delete = name;
      this.$refs['modal_delete_scen'].show()
    },
    new_scenario:function(bvModalEvt){
      if(!this.check_scenario_name_valid()){
        bvModalEvt.preventDefault()
        return
      }
      this.set_new_scenario(this.new_name)
      this.new_name = ""
    },
    check_scenario_name_valid: function(){
      let notunique = this.scenarios.some((e)=>{console.log(e['scenario']); return this.new_name == e['scenario']})
      if (notunique) this.name_state = 'invalid'
      return !notunique
    },
    reset_modal: function(){
      this.name_state = 'valid'
    },
    cancel_modal: function(){
      this.name_state = 'valid'
      this.new_name = ''
      this.copy = null
    },
    draw_tables: function(){
        this.table_drawn = false

        this.options.data = this.scenario_data[this.assumption]["Data"]
        this.options.columns = this.scenario_data[this.assumption]["Columns"]
        jexcel.destroy(this.$refs["assumption"], false);
        this.jExcelObj = jexcel(this.$refs["assumption"], this.jExcelOptions);
        this.tables[this.assumption] = this.jExcelObj
        this.table_drawn = true
    },
    set_new_scenario: function(name){
      this.scen_name = name
      if(!this.copy){
        // (a) NEW scenario - Add blank template for new scenario to available data
        var params = {sYear:this.sYear,eYear:this.eYear}
        axios.get(`http://localhost:5000/api/scenario_template`,{params: params})
        .then((res) => {
          // eslint-disable-next-line
          let scenario_template = res.data.scenario_data;
          for (var i=0; i<this.scen_inputs.length; i++)
            scenario_template[this.scen_inputs[i]]["Data"] = JSON.parse(scenario_template[this.scen_inputs[i]]["Data"])
          this.scenario_data = scenario_template
          this.last_saved = "never"
          //this.draw_tables()
        }, (err) => {
          this.error = 'Failed to get scenario template, the manager is either not running or encountered an error. (' + err.response.statusText + ')';
        })
      } else {
        // (b) EXISTING copy
        axios.get(`http://localhost:5000/api/scenarios/get/${this.county}/${this.copy_name}`)
        .then((res)=>{
          let data = res.data["data"]
          let labels = res.data["labels"]
          this.scen_inputs = labels
          for (var ind in data){
            data[ind]["Data"] = JSON.parse(data[ind]["Data"])
          }
          this.scenario_data = data

          this.last_saved = "never"
          //this.draw_tables()
          this.copy = null
        })
      }
    },
    is_scen_not_set: function(){
        return this.scen_name == null
    },
    is_reg_not_set: function(){
        return this.county == null
    },
    load_scenarios: function(){
      this.selected_impact_file = null
      //Retrieve valid scenarios from backend
      axios.get(`http://localhost:5000/api/scenarios/existing/${this.county}`)
        .then((res) => {
          this.scenarios = res.data.exist;
        }, (err) => {
          this.error = 'Failed to get load scenario data, the manager is either not running or encountered an error. (' + err.response.statusText + ')';
        })
    },
    load_scenario: function(name){
      axios.get(`http://localhost:5000/api/scenarios/get/${this.county}/${name}`)
        .then((res)=>{
          this.scen_name = name

          let data = res.data["data"]
          this.scen_inputs = res.data["labels"]
          for (var ind in data){
           data[ind]["Data"] = JSON.parse(data[ind]["Data"])
          }
          this.scenario_data = data

          // hack -> needs to wait for data processing --> JSON.parse
          setTimeout(()=>{this.draw_tables()}, 300)
          this.getDesc()
          this.getLastSave()
          this.listImpacts()

        }, (err) => {
          this.error = 'Failed to get load scenario data, the manager is either not running or encountered an error. (' + err.response.statusText + ')';
        })
    },
    save_scenario: function(){
      var params = {}
      params["eYear"] = this.eYear
      params["sYear"] = this.sYear
      params["county"] = this.county
      params["scenario_data"] = this.scenario_data
      params["scenario"] = this.scen_name
      params["description"] = this.scen_desc
      params["copy_from"] = this.copy_name
      console.log(params)
      axios.post("http://localhost:5000/api/scenario/save", params, {headers: {'Content-Type': 'application/json'}})
        .then((res) => {
          this.load_scenarios()
          this.$bvToast.toast('Scenario was saved successfully.', {
            title: 'Operation successful',
            toaster: 'b-toaster-bottom-right',
            autoHideDelay: 2000,
            appendToast: true,
            variant: 'success'
          })
        }, (err) => {
          this.error = 'Failed to save scenario file. (' + err.response.statusText + ')';
        })
      var today = new Date();
      var day = today.getDate()<10 ? "0"+today.getDate(): today.getDate()
      var month = (today.getMonth()+1)<10 ? "0"+(today.getMonth()+1):(today.getMonth()+1)
      var date = day +'/'+ month+'/'+ today.getFullYear().toString().substr(-2);
      var minutes = today.getMinutes()<10 ? "0"+today.getMinutes(): today.getMinutes()
      var time = today.getHours() + ":" + minutes
      var dateTime = date+' '+time;
      this.last_saved = dateTime
      this.last_saved_states[this.scen_name] = this.last_saved
      if (this.copy_name != ""){
        this.copy = false
        this.copy_name = ""
      }

    },

    delete_scenario: function(){
      if (this.scen_name_delete == null) return
      axios.get("http://localhost:5000/api/scenarios/delete/" + this.county + "/" + this.scen_name_delete)
      .then((res) => {
        this.load_scenarios();
        this.$bvToast.toast('Scenario deleted.', {
          title: 'Operation successful',
          toaster: 'b-toaster-bottom-right',
          autoHideDelay: 2000,
          appendToast: true,
          variant: 'success'
        })
      }, (err) => {
        this.error = 'Failed to delete scenario, the manager is either not running or encountered an error. (' + err.response.statusText + ')';
      })
      this.scen_name_delete = null
      this.scen_name = null
    },

    // IMPACTS module

    listImpacts: function(){
      axios.get(`http://localhost:5000/api/scenarios/impacts/list/${this.county}/${this.scen_name}`)
      .then((res)=>{
          this.impacts = res.data["impacts"]
          console.log(res.data['partial_error'])
          if(res.data['partial_error'].length > 0){
            this.$bvToast.toast(res.data['partial_error'], {
              title: 'Listing not completed',
              toaster: 'b-toaster-bottom-right',
              appendToast: true,
              noAutoHide: true,
              variant: 'warning'
            })
          }
      })
    },
    create_new_impact_wizard: function(){
      this.$refs['modal_new_impact'].show()
    },
    impact_wizard_reset: function(){
      this.impact_name = ''
    },
    create_impact: function(bvModalEvt){
      // prevent modal from closing
      bvModalEvt.preventDefault()
      // check validation
      if (this.validate_impact_name & this.validate_impact_pricebase){

        // build JSON body
        let json_body = {
          'name': this.impact_name,
          'desc': '',
          'price_base': this.impact_pricebase,
          'industry': this.impact_industry,
          'type': this.impact_type
        }

        // post new impact
        axios.post(`http://localhost:5000/api/scenarios/impacts/create/${this.county}/${this.scen_name}`, json_body, {headers: {'Content-Type': 'application/json'}})
          .then((res) => {
            this.$bvToast.toast("Impact successfully saved.", {
              title: 'Success',
              toaster: 'b-toaster-bottom-right',
              autoHideDelay: 2000,
              appendToast: true,
              variant: 'success'
            })

            // reset impact_name input and hide the modal
            this.impact_name = ''
            this.impact_industry = '0'
            this.impact_pricebase = this.$moment().year()

            this.$refs['modal_new_impact'].hide()
            this.listImpacts()

          }, (err) => {
            if(err.response['data']['error']['code'] == "already_existing"){
              this.$bvToast.toast("Saving failed. There is an impact file already existing with this name for this industry." +
              " If you are setting up multiple impacts for the same industry please make sure that scenario names are unique!", {
                title: 'Already existing impact name',
                toaster: 'b-toaster-bottom-right',
                appendToast: true,
                autoHideDelay: 10000,
                variant: 'danger'
              })
            } else if(err.response['data']['error']['code'] == 'predicted_missing'){
              this.$bvToast.toast("Saving failed. Baseline results cannot be found. Please run the baseline scenario for the given county before setting up impacts!", {
                title: 'Baseline missing',
                toaster: 'b-toaster-bottom-right',
                appendToast: true,
                autoHideDelay: 10000,
                variant: 'danger'
              })
            }
          })
      } else {
        this.$bvToast.toast(`Validation failed. Please check your inputs!`, {
          title: 'Validation failed',
          toaster: 'b-toaster-bottom-right',
          autoHideDelay: 2000,
          appendToast: true,
          variant: 'danger'
        })
      }
    },
    confirm_del_impact: function(file_name){
      this.impact_name_delete = file_name;
      this.$refs['modal_delete_impact'].show()
    },
    delete_impact: function(bvModalEvt){
      bvModalEvt.preventDefault()
      if (this.impact_name_delete == null) return
      let json_body = {
        'county': this.county,
        'scenario': this.scen_name,
        'file_name': this.impact_name_delete
      }
      axios.post("http://localhost:5000/api/scenarios/impacts/delete", json_body, {headers: {'Content-Type': 'application/json'}})
      .then((res) => {
        this.$bvToast.toast("Impact deleted.", {
          title: 'Success',
          toaster: 'b-toaster-bottom-right',
          autoHideDelay: 2000,
          appendToast: true,
          variant: 'success'
        })
        this.listImpacts();
        this.$refs['modal_delete_impact'].hide()
        this.impact_name_delete = null
        this.selected_impact_file = null
      }, (err) => {
        this.$bvToast.toast(err.response.statusText, {
          title: 'Operation cannot be completed',
          toaster: 'b-toaster-bottom-right',
          autoHideDelay: 2000,
          appendToast: true,
          variant: 'danger'
        })
      })
    },
    load_impact: function(file_name, type){
      axios.get(`http://localhost:5000/api/scenarios/impacts/read/${this.county}/${this.scen_name}/${file_name}`)
        .then((res)=>{
          this.impact_data = {}
          this.selected_impact_file = file_name
          this.selected_impact_type = type
          this.impact_tab = type == 'open' ? 'construction' : 'operation'

          let data = JSON.parse(res.data["data"])

          let standard_headers = [
            ...[{'title':'Indicator', 'type':'text', 'width': 200, 'readOnly': true}],
            ...Array(53).fill().map((v,i)=>i+1998).reduce((p, x)=> [...p,...[{'title': x, 'type': 'numeric', 'width': 50}]], [])
          ]

          // construction phase
          let total_row = [...['Total (%)'],
          ...Array(53).fill().map((v,i)=>i+2).reduce((p, x)=> [...p,...[`=SUM(${getExcelCol(x)}3:${getExcelCol(x)}10)`]], [])]

          this.impact_data['construction'] = {
            "Data": [...data['construction'],...[total_row]],
            "Columns": standard_headers
          }

          // operation phase
          let product_row = [...['Gross output'],
          ...Array(53).fill().map((v,i)=>i+2).reduce((p, x)=> [...p,...[`=PRODUCT(${getExcelCol(x)}1:${getExcelCol(x)}2)`]], [])]

          this.impact_data['operation'] = {
            "Data": [...data['operation'],...[product_row]],
            "Columns": standard_headers
          }

          // supply_chain phase
          this.impact_data['supply_chain'] = {
            "Data": data['supply_chain'].map((arr, i) => [...[this.sectors_raw[i]],...arr]),
            "Columns": [
              {'title': 'Industry', 'type': 'text', 'width': 200, 'readOnly': true},
              {'title': 'Production inputs as share of gross output (%)', 'type': 'numeric', 'mask': '0.0000','width': 400},
              {'title': 'Share of each product that is sourced from within the county (%)', 'type': 'numeric', 'mask': '0.0000', 'width': 400},
            ]
          }

          // hack -> needs to wait for data processing --> JSON.parse
          setTimeout(()=>{this.draw_impact_tables()}, 300)
          // this.getDesc()
          // this.getLastSave()
        }, (err) => {
          this.error = 'Failed to get load scenario data, the manager is either not running or encountered an error. (' + err.response.statusText + ')';
        })
    },
    draw_impact_tables: function(){
        this.impacts.table_drawn = false

        this.impacts_table['table']['data'] = this.impact_data[this.impact_tab]["Data"]
        this.impacts_table['table']['columns'] = this.impact_data[this.impact_tab]["Columns"]
        jexcel.destroy(this.$refs["impacts_table"], false);
        const jExcelObjImpact = jexcel(this.$refs["impacts_table"], this.jExcelOptions_impacts);
        Object.assign(this, { jExcelObjImpact });
        this.impacts_table['tables'][this.impact_tab] = this.jExcelObjImpact

        this.impacts.table_drawn = true
    },
    save_impact: function(){

        let json_body = {
          'county': this.county,
          'scenario': this.scen_name,
          'desc': '',
          'file_name': this.selected_impact_file,
          'data': this.impact_data,
          'processed': false
        }

        axios.post(`http://localhost:5000/api/scenarios/impacts/write`, json_body, {headers: {'Content-Type': 'application/json'}})
          .then((res) => {
            this.$bvToast.toast('Impact changes were saved successfully.', {
              title: 'Operation successful',
              toaster: 'b-toaster-bottom-right',
              autoHideDelay: 2000,
              appendToast: true,
              variant: 'success'
            })
          }, (err) => {
            this.$bvToast.toast('Failed to save impact file. (' + err.response.statusText + ')', {
              title: 'Operation cannot be completed',
              toaster: 'b-toaster-bottom-right',
              autoHideDelay: 7000,
              appendToast: true,
              variant: 'danger'
            })
          })

    },

    // UTILITY functions

    saveselection: function(){
      if (globalStore.scen_saved == false){
        globalStore.scen_county = this.county
        globalStore.scen_saved = true
      }
    },
    loadselection: function(){
      if (globalStore.scen_saved == true){
        this.county = globalStore.scen_county
        globalStore.scen_saved = false
      }
    },
    getLastSave: function(){
      let pos = this.scenarios.map(x => x['scenario']).indexOf(this.scen_name);
      if (pos == -1){
        this.last_saved = 'never';
      } else {
        this.last_saved = this.scenarios[pos]['Last Saved']
      }
    },
    getDesc: function(){
      let pos = this.scenarios.map(x => x['scenario']).indexOf(this.scen_name);
      if (pos == -1){
        this.scen_desc = '';
      } else {
        this.scen_desc = this.scenarios[pos]['description']
      }
    },
    currentYear: function(){
      return new Date().getFullYear()
    }

  },
  mounted: function () {
  },
  beforeDestroy(){
    this.saveselection()

  },
}
</script>
<style>
.jexcel > thead > tr:first-child > td{
  left: 0;
  position: sticky;
  z-index: 1000 !important;
}
.jexcel > tbody > tr > td:nth-child(2){
  left: 0;
  position: sticky;
  z-index: 1000
}
</style>
<!-- Add "scoped" attribute to limit CSS to this component only -->
<style scoped lang="scss">
h1{
  font-size: 48px;
  text-align: left;
  font-weight: 900;
}
h2{
  font-size: 24px;
  text-align: left;
  font-weight: 600;
  width: 100%;
}
ul {
  list-style-type: none;
  padding: 0;
}
li {
  display: inline-block;
  margin: 0 10px;
}
a {
  color: #42b983;
}
.selection_label, .spec_desc,.dim_label{
    text-align: left;
}
.bold{
  font-weight: bold;
  text-decoration: underline

}
.box {
  background-color: #E5E5E5;
  padding-top: 2vh;
  padding-bottom: 2vh;
  &.no-top-padding {
    padding-top: 0;
  }
  &.no-bottom-padding {
    padding-bottom: 0;
  }
}
.spec_select{
  margin: auto
}
.selection_label{
  font-size: 14px;
  font-weight: 700;
  text-align: left;
  width: 100%;
  margin-top: 10px;
}
.table_new{
  overflow:scroll;
  width: 100%;
  height: 500px
}
.spread_tabs{
  width: 100%;
}
.nav-item{
  background-color: rgb(0, 99, 152);
}
.meta{
  text-align: left;
}
.subtitle{
  font-size: 12px;
  color: #8C8C8C;
  line-height: 1em;
  margin-top: -5px;
  text-align: left;
}
table{
  text-align: left;
  background-color: white;
}
.jexcel > thead > tr:first-child > td:nth-child(2){
  left: 0;
  position: sticky;
  z-index: 2002
}
.jexcel > tbody > tr > td:nth-child(2){
  left: 0;
  position: sticky;
  z-index: 2001
}
.dotdotdot{
  cursor: pointer;
}
.faicon{
  cursor: pointer;
  &.disabled{
    color:#E5E5E5;
  }
}
h2{
  .highlight{
    color: #3399FF;
  }
}
</style>
