import './assets/main.css'

import { createApp } from 'vue'
import { createPinia } from 'pinia'
import Antd from 'ant-design-vue'
import 'ant-design-vue/dist/reset.css'
import axios from 'axios'

import App from './App.vue'
import router from './router'
import { useAuthStore } from './stores/auth'

// 配置axios
axios.defaults.baseURL = 'http://localhost:8080/api'
axios.defaults.timeout = 10000

// 请求拦截器
axios.interceptors.request.use(
  (config) => {
    console.log('📤 发出请求:', {
      method: config.method?.toUpperCase(),
      url: config.url,
      baseURL: config.baseURL,
      fullURL: (config.baseURL || '') + (config.url || ''),
      headers: config.headers,
      data: config.data
    })
    
    const token = localStorage.getItem('token')
    if (token) {
      config.headers.Authorization = `Bearer ${token}`
    }
    return config
  },
  (error) => {
    console.error('❌ 请求错误:', error)
    return Promise.reject(error)
  }
)

// 响应拦截器
axios.interceptors.response.use(
  (response) => {
    console.log('📥 收到响应:', {
      status: response.status,
      statusText: response.statusText,
      url: response.config.url,
      headers: response.headers,
      data: response.data
    })
    return response
  },
  (error) => {
    console.error('❌ 响应错误:', {
      message: error.message,
      code: error.code,
      response: error.response ? {
        status: error.response.status,
        statusText: error.response.statusText,
        headers: error.response.headers,
        data: error.response.data
      } : '无响应',
      config: {
        method: error.config?.method,
        url: error.config?.url,
        baseURL: error.config?.baseURL
      }
    })
    
    if (error.response?.status === 401) {
      const authStore = useAuthStore()
      authStore.clearToken()
      router.push('/login')
    }
    return Promise.reject(error)
  }
)

const app = createApp(App)
const pinia = createPinia()

app.use(pinia)
app.use(router)
app.use(Antd)

// 初始化认证状态
const authStore = useAuthStore()
authStore.initAuth()

app.mount('#app')
