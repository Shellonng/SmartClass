import axios from 'axios'
import { ElMessage } from 'element-plus'
import router from '@/router'

// 定义API响应接口
export interface ApiResponse<T = any> {
  code: number;
  data: T;
  message?: string;
  success?: boolean;
}

// 创建axios实例
const service = axios.create({
  baseURL: '', // API基础URL
  timeout: 15000 // 请求超时时间
})

// 请求拦截器
service.interceptors.request.use(
  config => {
    // 从localStorage获取token
    const token = localStorage.getItem('token')
    if (token) {
      // 设置请求头Authorization
      config.headers['Authorization'] = `Bearer ${token}`
    }
    return config
  },
  error => {
    console.error('Request error:', error)
    return Promise.reject(error)
  }
)

// 响应拦截器
service.interceptors.response.use(
  response => {
    // 直接返回data对象（包含code、data、message等属性）
    return response.data as any
  },
  error => {
    console.error('Response error:', error)
    
    // 处理网络错误
    let message = error.message
    if (error.response) {
      switch (error.response.status) {
        case 401:
          message = '未授权，请重新登录'
          // 清除token并跳转到登录页
          localStorage.removeItem('token')
          router.push('/login')
          break
        case 403:
          message = '拒绝访问'
          break
        case 404:
          message = '请求错误，未找到该资源'
          break
        case 500:
          message = '服务器端错误'
          break
        default:
          message = `连接错误${error.response.status}`
      }
    } else {
      message = '网络连接异常，请检查您的网络'
    }
    
    ElMessage({
      message: message,
      type: 'error',
      duration: 5 * 1000
    })
    
    return Promise.reject(error)
  }
)

export default service 