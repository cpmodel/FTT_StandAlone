<template>
  <div class="accordion" role="tablist">
    <b-card class="mb-1">
      <b-card-header header-tag="header" class="p-1" role="tab">
        <b-button block v-b-toggle.accordion-1 variant="info">Classifications</b-button>
      </b-card-header>
      <b-collapse id="accordion-1" visible accordion="my-accordion" role="tabpanel">
        <b-tabs card>
         <b-tab v-for="item in items" :title="`${item.name}`">
            <b-list-group-item v-for="title in item.title">
              {{title}}
            </b-list-group-item>
         </b-tab>
        </b-tabs>
      </b-collapse>
    </b-card>

    <b-card class="mb-1">
      <b-card-header header-tag="header" class="p-1" role="tab">
        <b-button block v-b-toggle.accordion-2 variant="info">Variables</b-button>
      </b-card-header>
      <b-collapse id="accordion-2" accordion="my-accordion" role="tabpanel">
        <div>
          <b-table striped hover :items="table" id="table"></b-table>
        </div>
      </b-collapse>
    </b-card>
  </div>
</template>

<script>
import axios from 'axios'
  export default {
    name: 'Metadata',
    data() {
      return {
        items:[],
        table:[]
      }
    },
    methods: {
       initialise: function() {
          // first load the possible counties and variables
      axios.get("http://localhost:5000/api/info/titles")
      .then((res) => {
        this.items = res.data;
        }, (err) => {
        this.error_message(err, "titles")
        })
      axios.get("http://localhost:5000/api/info/vars")
      .then((res) => {
        this.table = res.data.items;
        console.log(this.table)
        }, (err) => {
        this.error_message(err, "vars")
        })
       }
       
    },
    beforeMount(){

    this.initialise();
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
#table{
  overflow: scroll;
}
</style>
