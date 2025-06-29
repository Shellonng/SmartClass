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
export const login = (data: {
  username: string
  password: string
  role?: string
}) => {
  return axios.post('/auth/login', data, {
    withCredentials: true  // 确保发送cookie
  })
}

// 注册
export const register = (data: {
  username: string
  realName: string
  email: string
  password: string
  confirmPassword: string
  role: string
  department?: string
  title?: string
}) => {
  return axios.post('/auth/register', data, {
    withCredentials: true
  })
}

// 登出
export const logout = () => {
  return axios.post('/auth/logout', {}, {
    withCredentials: true
  })
}

// 获取用户信息 - 移除token，完全基于Session
export const getUserInfo = () => {
  return axios.get('/auth/user-info', { 
    withCredentials: true  // 只依赖Session cookie
  })
}

// 刷新token
export const refreshToken = () => {
  const token = localStorage.getItem('user-token') || localStorage.getItem('token')
  const headers: Record<string, string> = {}
  
  if (token) {
    headers['Authorization'] = `Bearer ${token}`
  }
  
  return axios.post('/auth/refresh', {}, {
    headers,
    withCredentials: true
  })
}

// 获取验证码
export const getCaptcha = () => {
  return axios.get('/auth/captcha')
}

// 修改密码
export const changePassword = (data: {
  oldPassword: string
  newPassword: string
  confirmPassword: string
}) => {
  return axios.post('/auth/change-password', data, {
    withCredentials: true
  })
}

// 重置密码
export const resetPassword = (data: { email: string }) => {
  return axios.post('/auth/reset-password', data)
}