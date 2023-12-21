<template>
  <div class='container'>
    <h1>Data table</h1>
    <b-button v-on:click="refresh()">Refresh</b-button>
    <h1>{{variable}}</h1>
    <br><br>
    <b-form-group>
     <div id=table_new ref="spreadsheet"></div>
    </b-form-group>
  </div>
</template>

<script>
import jexcel from 'jexcel'
import 'jexcel/dist/jexcel.css'

var changed = function(){
  console.log(this.data)
}

var options = {
  data:  JSON.parse(localStorage.pivot)['items'],
  columns:  JSON.parse(localStorage.table_fields),
  onchange: changed
} 

export default {
  name: 'Table_new',
  data: () => {
    return {
      data: JSON.parse(localStorage.pivot)['items'],
      columns: JSON.parse(localStorage.table_fields),
      variable: localStorage.variable
    }
  },
  components: {},
  methods: {
    refresh: function(event) {
      console.log('triggered');
      console.log(this.data)
      console.log(this.columns)
      this.$router.go()
    }
  },
  mounted: function () {
      console.log(this.data)
      console.log(this.columns)
      let spreadsheet = jexcel(this.$el, options)
      Object.assign(this, { spreadsheet })

  },
  watch:{
    data: function(){
      console.log(this.data)
      console.log(this.columns)
      options.data = this.data
      options.columns = this.columns
      let spreadsheet = jexcel(this.$el, options)
      Object.assign(this, { spreadsheet })

    }
  }
}
</script>

<style scoped lang="scss">
.jexcel_selectall{
  display: none
}
.jexcel_row{
  display: none
}
</style>
