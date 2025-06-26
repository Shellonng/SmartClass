import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import router from '@/router'
import { login, logout, getUserInfo, type LoginRequest } from '@/api/auth'
import { message } from 'ant-design-vue'
import axios from 'axios'

export interface User {
  id: number
  username: string
  realName: string
  email: string
  role: 'teacher' | 'student'
  avatar?: string
}

interface LoginForm {
  username: string
  password: string
  captcha: string
  remember: boolean
  role: string
}

export const useAuthStore = defineStore('auth', () => {
  // 状态
  const token = ref<string | null>(localStorage.getItem('token'))
  const user = ref<User | null>(null)
  const isLoading = ref(false)

  // 计算属性
  const isAuthenticated = computed(() => !!token.value && !!user.value)
  const isTeacher = computed(() => user.value?.role === 'teacher')
  const isStudent = computed(() => user.value?.role === 'student')

  // 设置token
  const setToken = (newToken: string) => {
    token.value = newToken
    localStorage.setItem('token', newToken)
    // 设置axios默认header
    axios.defaults.headers.common['Authorization'] = `Bearer ${newToken}`
  }

  // 清除token
  const clearToken = () => {
    token.value = null
    user.value = null
    localStorage.removeItem('token')
    delete axios.defaults.headers.common['Authorization']
  }

  // 登录
  const loginUser = async (loginData: LoginRequest, skipRedirect = false) => {
    try {
      const response = await login(loginData)
      
      if (response.data.code === 200) {
        const { token, userInfo } = response.data.data
        
        // 保存token和用户信息
        setToken(token)
        user.value = userInfo as User
        
        message.success('登录成功')
        
        // 根据参数决定是否跳转
        if (!skipRedirect) {
          if (userInfo.role === 'student') {
            router.push('/student')
          } else if (userInfo.role === 'teacher') {
            router.push('/teacher')
          }
        }
        
        return { success: true, userInfo }
      } else {
        message.error(response.data.message || '登录失败')
        return {
          success: false,
          message: response.data.message || '登录失败'
        }
      }
    } catch (error: any) {
      const errorMessage = error.response?.data?.message || '登录失败，请检查网络连接'
      message.error(errorMessage)
      return {
        success: false,
        message: errorMessage
      }
    }
  }

  // 登出
  const logout = async () => {
    try {
      await axios.post('/api/auth/logout')
    } catch (error) {
      console.error('登出请求失败:', error)
    } finally {
      clearToken()
      localStorage.removeItem('remember')
    }
  }

  // 获取用户信息
  const fetchUserInfo = async () => {
    try {
      if (!token.value) return null
      
      const response = await axios.get('/api/auth/user-info')
      user.value = response.data.data
      return user.value
    } catch (error) {
      console.error('获取用户信息失败:', error)
      clearToken()
      return null
    }
  }

  // 刷新token
  const refreshToken = async () => {
    try {
      const response = await axios.post('/api/auth/refresh')
      const { token: newToken } = response.data.data
      setToken(newToken)
      return newToken
    } catch (error) {
      console.error('刷新token失败:', error)
      clearToken()
      throw error
    }
  }

  // 修改密码
  const changePassword = async (oldPassword: string, newPassword: string) => {
    try {
      await axios.post('/api/auth/change-password', {
        oldPassword,
        newPassword
      })
    } catch (error: any) {
      throw new Error(error.response?.data?.message || '修改密码失败')
    }
  }

  // 忘记密码
  const forgotPassword = async (username: string, email: string) => {
    try {
      await axios.post('/api/auth/forgot-password', {
        username,
        email
      })
    } catch (error: any) {
      throw new Error(error.response?.data?.message || '发送重置邮件失败')
    }
  }

  // 重置密码
  const resetPassword = async (token: string, newPassword: string) => {
    try {
      await axios.post('/api/auth/reset-password', {
        token,
        newPassword
      })
    } catch (error: any) {
      throw new Error(error.response?.data?.message || '重置密码失败')
    }
  }

  // 初始化认证状态
  const initAuth = async () => {
    if (token.value) {
      axios.defaults.headers.common['Authorization'] = `Bearer ${token.value}`
      await fetchUserInfo()
    }
  }

  return {
    // 状态
    token,
    user,
    isLoading,
    
    // 计算属性
    isAuthenticated,
    isTeacher,
    isStudent,
    
    // 方法
    loginUser,
    logout,
    fetchUserInfo,
    refreshToken,
    changePassword,
    forgotPassword,
    resetPassword,
    initAuth,
    setToken,
    clearToken
  }
})