<template>
    <nav class="navbar fixed-top navbar-expand-lg navbar-light" v-bind:class="{ dark: isDark }">
            <div class="col-2">
                <img v-if="isDark" src="/frontend/model-Icon-FTT-126px_with_text_PV.png" />
                <img v-else src="/frontend/model-Icon-FTT-126px_with_text_PV.png" />
            </div>
            <div class="col-1">
                <img v-if="isDark" src="/frontend/cambridge_logo_dark.png" />
                <img v-else src="/frontend/cambridge_logo_light.png" />
            </div>
            <div class="col-2"></div>
            <div id="navbar" class="col-7" v-if="!isActive('/exit')">
                <table>
                  <tr>
                    <td v-bind:class="{ active: isActive('/') }"><router-link to="/" class='nav-item'><a class="nav-link" :class="{ 'text-light': isDark }" v-b-tooltip.hover title="Home">HOME</a></router-link></td>
                                       <td v-bind:class="{ active: isActive('/run') }"><router-link to="/run" class='nav-item'><a class="nav-link" :class="{ 'text-light': isDark }" v-b-tooltip.hover title="Run the model">RUN</a></router-link></td>
                    <td v-bind:class="{ active: isActive('/results') }"><router-link to="/results" class='nav-item'><a class="nav-link" :class="{ 'text-light': isDark }" v-b-tooltip.hover title="View model results">RESULTS</a></router-link></td>
                    <td v-bind:class="{ active: isActive('/Extract') }"><router-link to="/Extract" class='nav-item'><a class="nav-link" :class="{ 'text-light': isDark }" v-b-tooltip.hover title="Extract multiple variables">EXTRACT</a></router-link></td> 
                    <td v-bind:class="{ active: isActive('/Gamma') }"><router-link to="/Gamma" class='nav-item'><a class="nav-link" :class="{ 'text-light': isDark }" v-b-tooltip.hover title="Update Gamma values">GAMMA</a></router-link></td> 
                    <td v-bind:class="{ active: isActive('/Metadata') }"><router-link to="/Metadata" class='nav-item'><a class="nav-link" :class="{ 'text-light': isDark }" v-b-tooltip.hover title="Display model classifications">CLASSIFICATIONS</a></router-link></td>
                    <td><a href='#' class="nav-link" :class="{ 'text-light': isDark }" v-b-tooltip.hover v-b-modal.modal-exit title="Exit application">EXIT</a></td>
                  </tr>
                </table>
            </div>
            <b-modal id="modal-exit" ref="modal_exit" title="Closing WEM" ok-title="Quit" @ok="exit_">
              Are you sure you want to close the World Energy Model interface?<br>
              If the modelling process is still running changes may be lost.
            </b-modal>
    </nav>
</template>

<script>
export default {
  name: 'Navigation',
  data (){
    return {
      isDark: false,
      currentPath: '/'
    }
  },
  methods: {
    isActive: function(path){
      return path == this.currentPath;
    },
    exit_: function(){
      this.currentPath = '/exit';
      this.$router.push('/exit')
    }
  },
  watch: {
    '$route' (to) {
      this.isDark = to.path == "/";
      this.currentPath = to.path;
    }
  }

}
</script>

<!-- Add "scoped" attribute to limit CSS to this component only -->
<style scoped lang="scss">
nav.dark{
    table{
      border-bottom: 1px solid #0B1F2C;
    }
}
table{
  width: 90%;
  border-bottom: 1px solid #E5E5E5;
  a{
    color: black;
  }
  a:hover{
    text-decoration: none;
  }
  tr td.active{
    border-bottom: 3px solid #3399FF;
  }
}
.navbar{
  background-color: white;
  &.dark{
    background-color: #40515A;
  }
}
#navbar a:hover{
  text-decoration: none;
}
.navbar-nav > li{
  padding-left:10px;
  padding-right:10px;
  font-size: 18px;
  font-weight: 900;
}
.navbar-header{
  padding-left: 3vw;
}
</style>
