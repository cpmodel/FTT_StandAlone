<template>
  <div id="app">
    <div id="wrapper" :style="{ backgroundImage: 'url(/frontend/' + bodybg + ')'}">
      <div id="content">
        <router-view/>
      </div>
    </div>
  </div>
</template>


<script>
export default {
  data: function() {
    return {
      bodybg: this.$route.path == '/' || this.$route.path == '/exit' ? '/front_background.png' : '',
    }
  },
  watch: {
    '$route' (to) {
      this.bodybg = to.path == "/" || this.$route.path == '/exit' ? '/front_background.png' : '';
    }
  }
}
</script>
<style lang='scss'>

body{
  overflow-x: hidden;
}
#wrapper{
  position: absolute;
  width: 100vw;
  height: 100vh;
  top: 0px;
  background-size:cover
}
#app {
  width: 100vw;
  height: 100vh;
  font-family: Helvetica;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  text-align: center;
  color: #2c3e50;
  overflow-x: hidden;
  #navbar{
    font-weight: 900;
  }
}
#content{
  margin-top: 10vh;
}
.footer{
  position: fixed;
  left: 0;
  bottom: 0;
  width: 100%;
  background-color: #6C757D;
  color: white;
  text-align: center;
  a, a:hover{
    color: white;
    text-decoration: none;
  }
}
// * {
//   border-radius: 0 !important;
// }
</style>
