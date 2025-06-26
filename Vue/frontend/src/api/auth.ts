import axios from 'axios'

// 认证相关接口
export interface LoginRequest {
  username: string
  password: string
  captcha?: string
  remember?: boolean
  role?: string
}

export interface LoginResponse {
  code: number
  message: string
  data: {
    token: string
    userInfo: {
      id: number
      username: string
      realName: string
      email: string
      role: string
      avatar?: string
    }
  }
}

export interface RegisterRequest {
  username: string
  password: string
  confirmPassword: string
  email: string
  realName: string
  role: string
  captcha?: string
}

// 登录
export const login = (data: LoginRequest) => {
  console.log('🚀 发送登录请求:')
  console.log('URL:', axios.defaults.baseURL + '/api/auth/login')
  console.log('数据:', data)
  console.log('Headers:', axios.defaults.headers)
  
  return axios.post<LoginResponse>('/api/auth/login', data)
}

// 注册
export const register = (data: RegisterRequest) => {
  return axios.post('/api/auth/register', data)
}

// 登出
export const logout = () => {
  return axios.post('/api/auth/logout')
}

// 获取用户信息
export const getUserInfo = () => {
  return axios.get('/api/auth/user-info')
}

// 刷新token
export const refreshToken = () => {
  return axios.post('/api/auth/refresh')
}

// 获取验证码
export const getCaptcha = () => {
  return axios.get('/api/auth/captcha')
}

// 修改密码
export const changePassword = (data: { oldPassword: string; newPassword: string }) => {
  return axios.post('/api/auth/change-password', data)
}

// 重置密码
export const resetPassword = (data: { email: string }) => {
  return axios.post('/api/auth/reset-password', data)
}