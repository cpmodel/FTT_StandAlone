<template>
  <div class="home">
    <Navigation msg="Navigation" />
    <router-view/>
  </div>
</template>

<script>
// @ is an alias to /src
import Navigation from '@/components/Navigation.vue'

export default {
  name: 'home',
  components: {
    Navigation
  }
}
</script>
