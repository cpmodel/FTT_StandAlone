<template>
  <div class="container">
   <h1>Specification Switches</h1>
    <div class="row content">

       <div class="col-md-8 box form-group">
          <div class="form-row">
          <label class="selection_label col-md-2" for="varselect">Select variable:</label>
          </div>
          <div class="form-row">
            <b-col cols="8">

              <treeselect id="varselect col-md-8" v-model="variable" :multiple="false" :options="variables"
              :alwaysOpen="false" :clearable="false" placeholder="Select a variable..."
              :appendToBody="true" :maxHeight="200"/>
            </b-col>
            <b-col>
              <button class= "btn btn-primary" v-b-modal.modal-apply-specs>Apply Spec to all</button>
            </b-col>
          </div>

        <br>
        <div class="form-row" v-for="(value,key) in form" v-bind:key=(value,key)>
          <label class="col-md-6 col-form-label dim_label" :for="key" > {{key}} </label>
          <b-form-select class="col-md-4 spec_select" :id="key" v-model="form[key]" >
            <option v-for="option in spec_options" v-bind:key=option>{{option.Name}}</option>
          </b-form-select>
        </div>
      </div>
      <div class="col-md-4 form-group">

          <ul class="spec_desc">
            <h2>Available Specifications:</h2>
            <li v-for="item in spec_options" v-bind:key=item>
              <div class="bold">{{ item.Name }}</div> {{ item.Desc }}
            </li>
          </ul>
          <button class= "btn btn-primary" v-on:click="save_current_specs">Save Changes</button>
          <p>{{save_message}}</p>

      </div>


      </div>
      <b-modal id="modal-apply-specs" ref="modal-apply-specs" title="Apply Specification to all elements" ok-title="Confirm" @ok="apply_specs">
          <b-form-select class="col-md-4 spec_select" :id="key" v-model="applied_spec" >
            <option v-for="option in spec_options" v-bind:key=option>{{option.Name}}</option>
          </b-form-select>
    </b-modal>
  </div>

</template>

<script>
import axios from 'axios'
import Treeselect from '@riophae/vue-treeselect'
import '@riophae/vue-treeselect/dist/vue-treeselect.css'
import {globalStore} from '../main.js'
export default {
  name: 'Switches',
  data: () => {
    return {
      form: {},
      spec_options: [],
      error: '',
      variables: [],
      variable: "",
      save_message: '',
      applied_spec: ''
    }
  },
  components: { Treeselect },
  beforeMount(){
    this.initialise();


  },
  beforeDestroy(){
    this.saveselection();
  },
  watch: {
    'variable': function(){
          this.form = {}
          this.spec_options = []
          this.save_message = ''
          this.load_current_specs();
          this.loadselection();
      }
  },
  methods: {
    initialise: function() {
      // first load the possible variables
      axios.get("http://localhost:5000/api/results/variables")
      .then((res) => {
        let variables = res.data.vars.specs_indicies;
        this.variables = variables;
        this.variable = variables[0].id
      });

   },
   load_current_specs: function() {
      axios.get(`http://localhost:5000/api/specs/var/${this.variable}/`)
      .then((res) => {
        this.form = res.data.vars
      }, (err) => {
        this.error = 'Failed to get specifications, the manager is either not running or encountered an error. (' + err.response.statusText + ')';
      })
      axios.get(`http://localhost:5000/api/specs/var/${this.variable}/option`)
       .then((res) => {
       this.spec_options = res.data.vars
       },(err) => {
        this.error = 'Failed to get specification options, the manager is either not running or encountered an error. (' + err.response.statusText + ')';
      })

    },
    save_current_specs: function() {
      this.save_message = "Wrting specifaction for " + this.variable
      let params = {"var": this.form,"name":this.variable}
      axios.get("http://localhost:5000/api/specs/save/",{params: params})
      .then((res) => {
        if (res.data.written == "Written"){
          this.save_message = "Specifaction successfully updated"
        }
        else{
          this.save_message = "Specification save failed please check file is not currently open"
        }

      }, (err) => {
        this.error = 'Failed to save specifications, the manager is either not running or encountered an error. (' + err.response.statusText + ')';
      })
    },
    apply_specs: function(){
      for (var key in this.form) {
        this.form[key] = this.applied_spec
      }
      this.applied_spec = ''
    },
    saveselection: function(){
      if (globalStore.spec_saved == false){

        globalStore.specs_variable = this.variable
        globalStore.spec_saved = true
      }
    },
    loadselection: function(){
      if (globalStore.spec_saved == true){
        this.variable = globalStore.specs_variable
        globalStore.spec_saved = false
      }
    }


  }
}
</script>

<!-- Add "scoped" attribute to limit CSS to this component only -->
<style scoped>
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
.box{
  background-color: #E5E5E5;
  margin-top: 2vh;
  margin-bottom: 5vh;

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
  font-weight: 400;
}
</style>
