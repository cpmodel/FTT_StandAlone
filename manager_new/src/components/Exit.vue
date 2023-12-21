<template>
  <div class="hello">
    <div class="container">
      <div id="title">
        <h1>Application terminated</h1>
      </div>
      <div id="intro_text">
        <p>You may close this window now.</p>
      </div>
    </div>
  </div>
</template>

<script>
import axios from 'axios'

export default {
  name: 'Exit',
  data: () => {
    return {}
  },
  mounted(){
    setTimeout(axios.get(`http://localhost:5000/api/exit`)
    .then((res) => {
      console.log('Application closed.')
      }, (err) => {
      this.error_message(err, "regions")
    }), 1000);
  }
}
</script>

<!-- Add "scoped" attribute to limit CSS to this component only -->
<style lang="scss">
.hello{
  h1{
    color: rgb(255, 255, 255);
    font-weight: 900;
    font-size: 72px;
    text-align: left;
    line-height: 70px;
  }
  p{
    color: rgb(255, 255, 255);
    text-align: left;
    font-size: 24px;
  }
  #intro_text{
    margin-top: 5vh;
    width: 70vw;
  }
  #title{
    margin-top: 45vh;
  }
}
</style>
