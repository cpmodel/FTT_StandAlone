<template>
  <div class='container'>
    <h1>Data table</h1>
    <b-button v-on:click="refresh()">Refresh</b-button>
    <h1>{{variable}}</h1>
    <br><br>
    <b-form-group>
      <b-table striped bordered small hover :items="items" :fields="fields"></b-table>
    </b-form-group>
  </div>
</template>

<script>

export default {
  name: 'Table',
  data: () => {
    return {
      items: JSON.parse(localStorage.pivot)['items'],
      fields: JSON.parse(localStorage.table_fields),
      variable: localStorage.variable
    }
  },
  components: {},
  methods: {
    refresh: function(event) {
      this.$router.go()
    }
  }
}
</script>

<style scoped lang="scss">
</style>
