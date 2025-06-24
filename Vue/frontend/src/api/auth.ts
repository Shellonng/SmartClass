import axios from 'axios'

// 认证相关接口
export interface LoginRequest {
  username: string
  password: string
  captcha?: string
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
  return axios.post<LoginResponse>('/auth/login', data)
}

// 注册
export const register = (data: RegisterRequest) => {
  return axios.post('/auth/register', data)
}

// 登出
export const logout = () => {
  return axios.post('/auth/logout')
}

// 获取用户信息
export const getUserInfo = () => {
  return axios.get('/auth/userinfo')
}

// 刷新token
export const refreshToken = () => {
  return axios.post('/auth/refresh')
}

// 获取验证码
export const getCaptcha = () => {
  return axios.get('/auth/captcha')
}

// 修改密码
export const changePassword = (data: { oldPassword: string; newPassword: string }) => {
  return axios.post('/auth/change-password', data)
}

// 重置密码
export const resetPassword = (data: { email: string }) => {
  return axios.post('/auth/reset-password', data)
}