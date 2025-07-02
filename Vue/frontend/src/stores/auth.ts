import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import router from '@/router'
import { login as loginApi, logout as logoutApi, getUserInfo } from '@/api/auth'
import { message } from 'ant-design-vue'
import axios from 'axios'

interface User {
  id: number
  username: string
  realName: string
  email: string
  role: string
  avatar?: string
}

interface LoginForm {
  username: string
  password: string
  captcha: string
  remember: boolean
  role: string
}

// 扩展登录响应接口，添加sessionId
interface LoginResponseData {
  token?: string
  sessionId?: string
  userInfo: {
    id: number
    username: string
    realName: string
    email: string
    role: string
    avatar?: string
  }
}

export const useAuthStore = defineStore('auth', () => {
  // 状态 - 移除token，只保留用户信息和sessionId
  const user = ref<User | null>(null)
  const isLoading = ref(false)
  const sessionId = ref<string | null>(localStorage.getItem('sessionId'))
  const token = ref<string | null>(localStorage.getItem('token'))

  // 计算属性
  const isAuthenticated = computed(() => !!user.value)
  const isTeacher = computed(() => user.value?.role === 'TEACHER')
  const isStudent = computed(() => user.value?.role === 'STUDENT')

  // 设置sessionId（用于调试目的，实际认证依赖cookie）
  const setSessionId = (newSessionId: string) => {
    sessionId.value = newSessionId
    localStorage.setItem('sessionId', newSessionId)
  }

  // 设置token
  const setToken = (newToken: string) => {
    token.value = newToken
    localStorage.setItem('token', newToken)
    // 立即设置到axios默认头部
    axios.defaults.headers.common['Authorization'] = `Bearer ${newToken}`
    console.log('Token已设置到axios默认头部')
  }

  // 设置用户信息
  const setUser = (userData: User) => {
    user.value = userData
    localStorage.setItem('userInfo', JSON.stringify(userData))
  }

  // 清除认证信息
  const clearAuth = () => {
    user.value = null
    sessionId.value = null
    token.value = null
    localStorage.removeItem('token')  // 清除旧的token
    localStorage.removeItem('user-token')  // 清除旧的token
    localStorage.removeItem('userInfo')
    localStorage.removeItem('sessionId')
    // 移除axios的Authorization头
    delete axios.defaults.headers.common['Authorization']
  }

  // 登录
  const login = async (credentials: { username: string; password: string; role?: string }) => {
    try {
      isLoading.value = true
      const response = await loginApi(credentials)
      
      if (response.data.success) {
        const { userInfo, sessionId: newSessionId, token: newToken } = response.data.data
        
        // 保存用户信息
        setUser(userInfo)
        
        // 保存sessionId
        if (newSessionId) {
          setSessionId(newSessionId)
        }
        
        // 保存token并设置到axios头部
        if (newToken) {
          setToken(newToken)
          // 额外确认token已正确设置
          console.log('登录成功，token已保存:', newToken.substring(0, 10) + '...')
        } else {
          console.warn('登录响应中没有token')
        }
        
        return { success: true, data: response.data.data }
      } else {
        return { success: false, message: response.data.message }
      }
    } catch (error: any) {
      console.error('登录失败:', error)
      return {
        success: false,
        message: error.response?.data?.message || '登录失败，请检查网络连接' 
      }
    } finally {
      isLoading.value = false
    }
  }

  // 登出
  const logout = async () => {
    try {
      await logoutApi()
    } catch (error) {
      console.error('登出请求失败:', error)
    } finally {
      clearAuth()
    }
  }

  // 获取用户信息
  const fetchUserInfo = async () => {
    try {
      const response = await getUserInfo()
      if (response.data.success) {
        setUser(response.data.data)
        return true
      } else {
        clearAuth()
        return false
      }
    } catch (error) {
      console.error('获取用户信息失败:', error)
      clearAuth()
      return false
    }
  }

  // 初始化 - 先验证session，再决定是否恢复用户信息
  const init = async () => {
    // 先检查token是否存在
    const savedToken = localStorage.getItem('token')
    if (savedToken) {
      // 如果token存在，设置到axios头部
      setToken(savedToken)
    }
    
    const savedUserInfo = localStorage.getItem('userInfo')
    if (savedUserInfo) {
      try {
        // 先从localStorage恢复用户信息
        const userInfo = JSON.parse(savedUserInfo)
        setUser(userInfo)
        
        // 然后验证session是否仍然有效
        const isValid = await fetchUserInfo()
        if (!isValid) {
          // 如果session无效，清除所有数据
          clearAuth()
        }
        } catch (error) {
        console.error('初始化用户信息失败:', error)
        clearAuth()
      }
    } else {
      // 如果没有保存的用户信息，尝试从session获取
      try {
        await fetchUserInfo()
      } catch (error) {
        console.error('从session获取用户信息失败:', error)
      }
    }
  }

  return {
    // 状态
    user,
    isLoading,
    sessionId,
    token,
    // 计算属性
    isAuthenticated,
    isTeacher,
    isStudent,
    // 方法
    login,
    logout,
    fetchUserInfo,
    init,
    setUser,
    setToken,
    clearAuth
  }
})