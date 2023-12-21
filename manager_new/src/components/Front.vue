<template>
  <div class="hello">
    <div class="container">
      <div id="title">
        <h1>FTT: Stand alone</h1>
      </div>
      <div id="intro_text">
        <p>Interface for standalone version of the suite for FTT models</p>
      </div>
    </div>
  </div>
</template>

<script>
export default {
  name: 'Front',
  data: () => {
    return {}
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
    color:  rgb(255, 255, 255);
    text-align: left;
    font-size: 24px;
  }
  #intro_text{
    margin-top: 5vh;
    margin-left: 1vh;
    width: 70vw;
  }
  #title{
    margin-top: 45vh;
  }
}
</style>
