import Vue from 'vue'
import App from './App.vue'
import router from './router'
import BootstrapVue from 'bootstrap-vue'
import * as taucharts from 'taucharts'
import Ripple from 'vue-ripple-directive'
import VueMoment from 'vue-moment'

import 'bootstrap/dist/css/bootstrap.css'
import 'bootstrap-vue/dist/bootstrap-vue.css'

import VueSSE from 'vue-sse';

// fontawesome
import { library } from '@fortawesome/fontawesome-svg-core'
import { faCopy, faTrashAlt, faEdit, faFileImport } from '@fortawesome/free-solid-svg-icons'
import { FontAwesomeIcon } from '@fortawesome/vue-fontawesome'

library.add(faCopy, faTrashAlt, faEdit, faFileImport)

Vue.component('font-awesome-icon', FontAwesomeIcon)

Vue.config.productionTip = false
Vue.use(BootstrapVue);
Vue.use(VueSSE);
Vue.use(VueMoment);
Vue.directive('ripple', Ripple);

export const globalStore = new Vue({
  data: {
    region_selection: [],
    pivot: [],
    table_fields: {},
    model_start_year: 2015,
    model_end_year: 2050,
    results_selected: {},
    saved: false,
    scen_region: null,
    scen_saved: false,
    spec_saved:false,
    specs_variable:null
  }
})

new Vue({
  router,
  render: h => h(App)
}).$mount('#app')
