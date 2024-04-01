import Vue from 'vue'
import Router from 'vue-router'
import Home from './views/Home.vue'
import Run from './components/Run.vue'
import Metadata from './components/Metadata.vue'
import Front from './components/Front.vue'
import Results from './components/Results.vue'
import Extract from './components/Extract.vue'
import Gamma from './components/Gamma.vue'
import Exit from './components/Exit.vue'

Vue.use(Router)

export default new Router({
  routes: [
    {
      path: '/',
      name: 'home',
      component: Home,
      children: [
        {
          path: 'run',
          component: Run
        },
        {
          path: 'metadata',
          component: Metadata
        },
        {
          path: '',
          component: Front
        },
        {
          path: 'results',
          component: Results
        },
        {
          path: 'Extract',
          component: Extract
        },
        {
          path: 'Gamma',
          component: Gamma
        },
        {
          path: 'exit',
          component: Exit
        }
      ]
    },
    {
      path: '/about',
      name: 'about',
      // route level code-splitting
      // this generates a separate chunk (about.[hash].js) for this route
      // which is lazy-loaded when the route is visited.
      component: () => import(/* webpackChunkName: "about" */ './views/About.vue'),
      children: [
        {
          path: 'run',
          component: Run
        }
      ]
    },
    // {
    //   name: 'table',
    //   component: Table,
    //   path: '/table'
    // },
  ]
})
