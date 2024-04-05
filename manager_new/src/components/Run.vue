<template>
  <div class="container">
    <h1>Running the model</h1>
    <div class="row">
      <div class="col-7 box">
        <div class="overlay" :style="{display: isRunning? 'block' : 'none'}">
          <span>Model is running, selections are not available.</span>
        </div>
        <div class="row">
          <div class="col-7 no-float">
            <div class="row text_scenarios">
              <h2>{{scenarios.length > 1 ? 'Select scenarios' : scenarios[0].length > 1 ? 'Select scenarios' : 'Scenario'}}</h2>
              <!-- <p>Subtitle / description can come here</p> -->
            </div>
            <div class="row">
              <div class="county_box">
                <treeselect v-model="selected_scenarios" :multiple="true" :options="scenarios" :alwaysOpen="false" :clearable="true"
                :showCount="true" :appendToBody="false" :defaultExpandLevel="1" :maxHeight="390" :limit="5"
                :value-consists-of="'LEAF_PRIORITY'"/>
              </div>
            <div class="row text_scenarios">
              <h2>{{'Select models'}}</h2>
              <!-- <p>Subtitle / description can come here</p> -->
            </div>
            </div>
            <div class="row">
                <div class="model_box">
                <treeselect v-model="selected_models" :multiple="true" :options="models" :alwaysOpen="false" :clearable="true"
                :showCount="true" :appendToBody="false" :defaultExpandLevel="1" :maxHeight="390" :limit="5"
                :value-consists-of="'LEAF_PRIORITY'"/>
              </div>
            </div>
          </div>
          <div class="col-5 no-float">
            <div class="row text_timeframe">
              <h2>Time-frame</h2>
              <!-- <p>Select the start and end year of the simulation</p> -->
            </div>
            <div class="row yr_label">
              <label for='run_s_yr'>FIRST YEAR</label>
              <b-form-input id='run_s_yr' type="number" v-model='run_s_yr' />
            </div>
            <div class="row yr_label">
              <label for='run_e_yr'>LAST YEAR</label>
              <b-form-input id='run_e_yr' type='number' v-model='run_e_yr' />
            </div>
            <div class="row text_run">
              <h2>Run the model</h2>
              <!-- <p>Begin running the model with the current specification</p> -->
            </div>
            <div class="row button_run">
                <button class= "btn btn-primary" id='run' v-on:click="run_and_listen()">RUN</button>
            </div>
            <div class="row text_notice">
              <p>This could take several minutes!</p>
            </div>
          </div>
        </div>
      </div>
      <div class="col-1"></div>
      <div class="col-4 box right">
        <div class="row">
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
        </div>
        <div class="row current_progress">
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
        </div>
        <div class="row">
          <div class="console">
            <p v-for="item in console_log" v-bind:key=item><span class="timestamp">{{item.timestamp}}</span>{{item.message}}</p>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import axios from 'axios'
import Treeselect from '@riophae/vue-treeselect'
import '@riophae/vue-treeselect/dist/vue-treeselect.css'
import {globalStore} from '../main.js'

let msgServer;

export default {
  name: 'Run',
  data: () => {
    return {
      error: '',
      scenarios: '',
      selected: [],
      models:[],
      run_s_yr: 2015,
      run_e_yr: 2050,
      selected_scenarios: null,
      selected_models: null,
      status_text: 'Standing by',
      status_state: '#006398',
      isRunning: false,
      processing_length: 0,
      processing_processed: 0,
      processing_avgtime: 0,
      processing_remaining: 0,
      console_log: []
    }
  },
  components: { Treeselect },
  methods: {
    load_scenarios: function() {
      axios.get("http://localhost:5000/api/available_scenarios")
      .then((res) => {
        // eslint-disable-next-line
        let reg = res.data.scenarios;
        this.scenarios = reg;
        let mod = res.data.models;
        this.models = mod
        console.log(reg)
      }, (err) => {
        this.error = 'Failed to get filenames, the manager is either not running or encountered an error. (' + err.response.statusText + ')';
      })
    },
    save_selection: function() {
      globalStore.county_selection = this.selected_scenarios;
      globalStore.model_start_year = this.run_s_yr;
      globalStore.model_end_year = this.run_e_yr;
    },
    load_values: function() {
      this.selected_scenarios = globalStore.county_selection;
      this.run_s_yr = globalStore.model_start_year;
      this.run_e_yr = globalStore.model_end_year;
    },
    run_and_listen: function() {
      this.processing_length = 0;
      this.processing_processed= 0;
      this.processing_avgtime= 0;
      this.processing_remaining= 0;

      let scenario_arr = []
      this.selected_scenarios.forEach((e)=>{
        scenario_arr.push({
            "scenario": e
        })
        console.log(scenario_arr)
      })
      let model_arr = []
      this.selected_models.forEach((e)=>{
        model_arr.push({
            "model": e
        })
        console.log(model_arr)
      })

      let json_body = {
        "data": scenario_arr,
        "model": model_arr,
        "endyear": this.run_e_yr
      }

      this.log("message;message:Started process...;");
      this.log("message;message:Processing data...;");

      axios.post(`http://localhost:5000/api/run/initialize/`, json_body, {headers: {'Content-Type': 'application/json'}})
        .then((res) => {
          this.log("message;message:Data processed. Running modelling.;");
          this.$sse(`http://localhost:5000/api/run/start/`, {format: 'plain'})
            .then(sse => {
              msgServer = sse;
              sse.onError(e => {
                sse.close();
                this.update_status('system_error');
              });
              sse.subscribe('status_change', data => {
                if(data != 'running'){
                  sse.close()
                }
                this.update_status(data)
                this.log(data);
              });
              sse.subscribe('processing', data => {
                this.update_progress(data);
                if(data.split(";")[0] != 'progress'){
                  this.log(data);
                }
              });
            });
        }, (err) => {
          this.$bvToast.toast("Unexpected error", {
            title: 'Running failed!',
            toaster: 'b-toaster-bottom-right',
            appendToast: true,
            autoHideDelay: 7000,
            variant: 'danger'
          })
        })
    },
    log: function(message){
      console.log(message)
      let date = new Date();
      date = date.toISOString();
      var regexp = /(?:;message:)([a-zA-Z0-9 \.\-\,]*)(?:;)/g;
      var match = []
      match = regexp.exec(message);

      if(match.length != 0){
        let object = {"timestamp": date, "message": match[1]};
        this.console_log.push(object);
      }
    },
    update_status: function(state) {
      switch(state) {
        case 'running':
          this.isRunning = true;
          this.status_text = "Model is running";
          this.status_state = "#F1C40F";
          break;
        case 'finished':
          this.isRunning = false;
          this.status_text = "Run finished";
          this.status_state = "#99FFCC";
          break;
        default:
          this.isRunning = false;
          this.status_text = "Run finished with errors";
          if(state == 'system_error') this.status_text = "System error!";
          this.status_state = "#F06449";
          this.log("message;message:System error - if all runs have finished running and this error is displayed it is likely that at least one of the scenarios have failed to converge after 100 iterations;")
      }
    },
    update_progress: function(data) {
      let arr = data.split(";");
      let id = arr[0];
      if(id == 'items'){
        this.processing_length = parseInt(arr[1]);
      } else if(id == 'progress') {
        this.processing_processed += 1;
        let n = this.processing_processed;
        let elapsed = parseFloat(arr[3]);
        this.processing_avgtime = this.processing_avgtime == 0? elapsed : (this.processing_avgtime*n + elapsed) / (n+1);
        this.processing_remaining = (this.processing_length - n) * this.processing_avgtime
      }
      else if(id == 'processed' || id == 'error') {
        // this.processing_remaining = "Finished"
      }
    }
  },
  beforeMount(){
    this.load_scenarios();
    this.load_values();
    this.log("message;message:Standing by...;");
  },
  beforeDestroy(){
    this.save_selection();
    msgServer.close();
  },
  filters: {
    format_time: function (value) {
      if (!value) return '';
      let minutes = Math.floor(value / 60);
      let seconds = parseInt(value % 60);
      return `${minutes} m ${seconds} s`;
    }
  }
}
</script>

<!-- Add "scoped" attribute to limit CSS to this component only -->
<style scoped lang="scss">
.overlay{
  background-color: rgba(255,255,255,0.8);
  position: absolute;
  left: 0;
  width: 100%;
  height: 100%;
  z-index: 1000;
  transition: 1.5s;
  span{
    margin-top: 50%;
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

.dot {
  height: 25px;
  width: 25px;
  background-color: #bbb;
  border-radius: 50% !important;
  display: inline-flex;
  vertical-align: sub;
  margin-right: 10px;
}

.box.right{
  .status_text{
    text-align: right;
    width: 100%;
    font-size: 24px;
    font-weight: 700;
  }
  .status_title{
    text-align: left;
  }
  .status_state{
    text-align: right;
  }
  table{
    width: 100%;
  }

  .console{
    width: 100%;
    height: 55vh;
    overflow-y: scroll;
    background-color: #0B1F2C;
    padding: 10px 10px;
    p{
      .timestamp{
        color: #8C8C8C;
        margin-right: 10px;
      }
      color: white;
      font-size: 11px;
      text-align: left;
      margin-bottom: 0px;
    }
  }
}

.yr_label{
  font-size: 14px;
  font-weight: 700;
  text-align: right;
  width: 100%;
  label{
    width: 100%;
    margin-bottom: 0px;
    margin-top: 10px;
  }
  input{
    font-size: 24px;
    font-weight: 700;
    text-align: right;
    width: 70%;
    margin-left: auto;
    padding: 30px 10px;
  }
  &:nth-of-type(3){
    margin-top: 15px;
  }
}
h1{
  font-size: 48px;
  text-align: left;
  font-weight: 900;
}
h2{
  font-size: 24px;
  font-weight: 600;
  width: 100%;
}
.text_scenarios{
  text-align: left;
}
.text_timeframe, .text_run, .text_notice{
  text-align: right;
  p{
    margin-left: auto;
    max-width: 170px;
  }
}
.text_scenarios, .text_timeframe, .text_run, .text_notice{
  margin: 20px 10px;
  p{
    font-size: 12px;
    color: #8C8C8C;
    line-height: 1em;
    margin-top: -5px;
  }
}
.button_run{
  width: 100%;
  margin-left: auto;
  text-align: right;
  display: block;
  padding-right: 2vw;
  button{
    height: 60px;
    width: 170px;
    font-weight: 900;
    font-size: 24px;
  }
}
.box{
  &:nth-of-type(1){
    background-color: #E5E5E5;
  }
  height: 70vh;
  margin-top: 5vh;
  margin-bottom: 5vh;
}
.county_box{
  padding: 0px 20px;
  height: 100%;
  *{
    height: 100% !important;
  }
  .vue-treeselect__menu{
    height: 10vh !important;
  }
}
.vue-treeselect__control{
    height: 0px !important;
    visibility: hidden !important;
}
.model_box{
  padding: 0px 20px;
  height: 100%;
  *{
    height: 100% !important;
  }
  .vue-treeselect__menu{
    height: 10vh !important;
  }
}
.vue-treeselect__control{
    height: 0px !important;
    visibility: hidden !important;
}
</style>
