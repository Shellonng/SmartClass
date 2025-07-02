import './assets/main.css'

import { createApp } from 'vue'
import { createPinia } from 'pinia'
import Antd from 'ant-design-vue'
import 'ant-design-vue/dist/reset.css'
import ElementPlus from 'element-plus'
import 'element-plus/dist/index.css'
import axios from 'axios'

import App from './App.vue'
import router from './router'
import { useAuthStore } from './stores/auth'

// 配置axios
axios.defaults.baseURL = 'http://localhost:8080'
axios.defaults.timeout = 10000
axios.defaults.withCredentials = true // 允许跨域请求携带凭证（cookies）

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
    
    // 确保请求包含cookie（用于Session认证）
    config.withCredentials = true
    
    // 从localStorage获取token并添加到请求头
    const token = localStorage.getItem('token')
    if (token && !config.headers['Authorization']) {
      config.headers['Authorization'] = `Bearer ${token}`
      console.log('请求自动添加token')
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
    
    // 检查并保存会话ID（如果存在）
    const setCookieHeader = response.headers['set-cookie']
    if (setCookieHeader) {
      console.log('服务器设置了Cookie:', setCookieHeader)
    }
    
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
    
    // 处理401未授权错误
    if (error.response && error.response.status === 401) {
      // 清除本地存储的认证信息
      localStorage.removeItem('token');
      localStorage.removeItem('user-token');
      localStorage.removeItem('userInfo');
      localStorage.removeItem('sessionId');
      
      // 重定向到登录页
      window.location.href = '/login';
    }
    return Promise.reject(error)
  }
)

const app = createApp(App)
const pinia = createPinia()

app.use(pinia)
app.use(router)
app.use(Antd)
app.use(ElementPlus)

// 初始化认证状态，并在完成后挂载应用
const initApp = async () => {
  const authStore = useAuthStore()
  
  try {
    // 检查是否有token
    const token = localStorage.getItem('token')
    if (token) {
      // 设置全局axios默认头部
      axios.defaults.headers.common['Authorization'] = `Bearer ${token}`
      console.log('启动时设置全局token')
    }
    
    // 初始化认证状态
    await authStore.init()
    console.log('认证初始化完成')
  } catch (error) {
    console.error('认证初始化失败:', error)
  } finally {
    // 无论认证是否成功，都挂载应用
    app.mount('#app')
  }
}

initApp()
