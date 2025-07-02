<template>
  <div class="login-page">
    <!-- 左侧背景区域 -->
    <div class="login-visual">
      <div class="visual-content">
        <div class="brand-section">
          <div class="logo">
            <img src="/logo.svg" alt="智慧课堂" class="logo-img" />
            <span class="logo-text">智慧课堂</span>
          </div>
          <h1 class="visual-title">开启智慧学习之旅</h1>
          <p class="visual-subtitle">
            汇聚全球优质教育资源<br />
            为每一位学习者提供个性化学习体验
          </p>
        </div>
        
        <div class="feature-highlights">
          <div class="feature-item">
            <div class="feature-icon">🎓</div>
            <div class="feature-text">
              <h4>优质课程</h4>
              <p>来自知名高校的精品课程</p>
            </div>
          </div>
          <div class="feature-item">
            <div class="feature-icon">🤖</div>
            <div class="feature-text">
              <h4>AI助学</h4>
              <p>智能推荐个性化学习路径</p>
            </div>
          </div>
          <div class="feature-item">
            <div class="feature-icon">📊</div>
            <div class="feature-text">
              <h4>学习分析</h4>
              <p>实时跟踪学习进度与效果</p>
            </div>
          </div>
        </div>
      </div>
    </div>
    
    <!-- 右侧登录表单区域 -->
    <div class="login-form-section">
      <div class="form-container">
        <div class="form-header">
          <h2 class="form-title">{{ pageTitle }}</h2>
          <p class="form-subtitle">{{ pageSubtitle }}</p>
        </div>

        <!-- 身份选择 - 只在登录模式显示 -->
        <div v-if="!isRegisterMode" class="role-selection">
          <div class="role-tabs">
            <div 
              class="role-tab" 
              :class="{ active: selectedRole === 'student' }"
              @click="selectRole('student')"
            >
              <BookOutlined />
              <span>学生登录</span>
            </div>
            <div 
              class="role-tab" 
              :class="{ active: selectedRole === 'teacher' }"
              @click="selectRole('teacher')"
            >
              <UserOutlined />
              <span>教师登录</span>
            </div>
          </div>
        </div>
          
        <!-- 登录表单 -->
        <a-form
          v-if="!isRegisterMode"
          :model="loginForm"
          :rules="loginRules"
          @finish="handleLogin"
          layout="vertical"
          class="login-form"
        >
          <a-form-item name="username" label="用户名">
            <a-input
              v-model:value="loginForm.username"
              size="large"
              placeholder="请输入用户名"
            >
              <template #prefix>
                <UserOutlined />
              </template>
            </a-input>
          </a-form-item>
          
          <a-form-item name="password" label="密码">
            <a-input-password
              v-model:value="loginForm.password"
              size="large"
              placeholder="请输入密码"
            >
              <template #prefix>
                <LockOutlined />
              </template>
            </a-input-password>
          </a-form-item>
          
          <div class="form-options">
            <a-checkbox v-model:checked="loginForm.remember">
              记住我
            </a-checkbox>
            <a @click="showForgotModal = true" class="forgot-link">忘记密码？</a>
          </div>
          
          <a-button 
            type="primary" 
            html-type="submit" 
            size="large" 
            block
            :loading="loading"
            class="login-btn"
          >
            {{ selectedRole === 'teacher' ? '教师登录' : '学生登录' }}
          </a-button>
        </a-form>

        <!-- 注册表单 -->
        <a-form
          v-else
          :model="registerForm"
          :rules="registerRules"
          @finish="handleRegister"
          layout="vertical"
          class="register-form"
        >
          <a-form-item name="realName" label="真实姓名">
            <a-input
              v-model:value="registerForm.realName"
              size="large"
              placeholder="请输入真实姓名"
            >
              <template #prefix>
                <UserOutlined />
              </template>
            </a-input>
          </a-form-item>

          <a-form-item name="username" label="用户名">
            <a-input
              v-model:value="registerForm.username"
              size="large"
              placeholder="请输入用户名"
            >
              <template #prefix>
                <UserOutlined />
              </template>
            </a-input>
          </a-form-item>

          <a-form-item name="email" label="邮箱">
            <a-input
              v-model:value="registerForm.email"
              size="large"
              placeholder="请输入邮箱地址"
            >
              <template #prefix>
                <MailOutlined />
              </template>
            </a-input>
          </a-form-item>
          
          <a-form-item name="password" label="密码">
            <a-input-password
              v-model:value="registerForm.password"
              size="large"
              placeholder="请输入密码"
            >
              <template #prefix>
                <LockOutlined />
              </template>
            </a-input-password>
          </a-form-item>

          <a-form-item name="confirmPassword" label="确认密码">
            <a-input-password
              v-model:value="registerForm.confirmPassword"
              size="large"
              placeholder="请再次输入密码"
            >
              <template #prefix>
                <LockOutlined />
              </template>
            </a-input-password>
          </a-form-item>

          <!-- 用户类型选择 -->
          <a-form-item name="role" label="用户角色">
            <a-radio-group v-model:value="registerForm.role" size="large">
              <a-radio-button value="STUDENT">学生</a-radio-button>
              <a-radio-button value="TEACHER">教师</a-radio-button>
            </a-radio-group>
          </a-form-item>
          
          <a-button 
            type="primary" 
            html-type="submit" 
            size="large" 
            block
            :loading="loading"
            class="register-btn"
          >
            立即注册
          </a-button>
        </a-form>
        
        <div class="form-footer">
          <p v-if="!isRegisterMode">
            还没有账户？ 
            <router-link to="/register" class="register-link">立即注册</router-link>
          </p>
          <p v-else>
            已有账户？ 
            <router-link to="/login" class="login-link">立即登录</router-link>
          </p>
        </div>
      </div>
      
      <!-- AI助手占位 -->
      <div class="ai-assistant">
        <a-button type="primary" shape="circle" size="large" class="ai-btn">
          <RobotOutlined />
        </a-button>
        <span class="ai-text">智能登录助手</span>
      </div>
    </div>
    
    <!-- 忘记密码弹窗 -->
    <a-modal
      v-model:open="showForgotModal"
      title="找回密码"
      @ok="handleForgotPassword"
    >
      <a-form :model="forgotForm" layout="vertical">
        <a-form-item label="用户名">
          <a-input v-model:value="forgotForm.username" placeholder="请输入用户名" />
        </a-form-item>
        <a-form-item label="邮箱">
          <a-input v-model:value="forgotForm.email" placeholder="请输入邮箱" />
        </a-form-item>
      </a-form>
    </a-modal>
  </div>
</template>

<script setup lang="ts">
import { ref, reactive, onMounted, computed } from 'vue'
import { useRouter, useRoute } from 'vue-router'
import { message } from 'ant-design-vue'
import {
  UserOutlined,
  BookOutlined,
  ArrowRightOutlined,
  ArrowLeftOutlined,
  LockOutlined,
  RobotOutlined,
  MailOutlined
} from '@ant-design/icons-vue'
import { useAuthStore } from '@/stores/auth'
import { register, login } from '@/api/auth'

const router = useRouter()
const route = useRoute()
const authStore = useAuthStore()

// 判断当前是登录还是注册模式
const isRegisterMode = computed(() => route.name === 'Register')
const pageTitle = computed(() => isRegisterMode.value ? '用户注册' : '欢迎回来')
const pageSubtitle = computed(() => 
  isRegisterMode.value ? '创建账户，开启学习之旅' : '请选择您的身份并登录账户'
)

// 选择的角色
const selectedRole = ref<'teacher' | 'student' | ''>('student')

// 登录表单
const loginForm = reactive({
  username: '',
  password: '',

  remember: false,
  role: ''
})

// 忘记密码表单
const forgotForm = reactive({
  username: '',
  email: ''
})

// 注册表单
const registerForm = reactive({
  username: '',
  password: '',
  confirmPassword: '',
  email: '',
  realName: '',
  role: 'STUDENT'
})

// 表单验证规则
const loginRules = {
  username: [{ required: true, message: '请输入用户名', trigger: 'blur' }],
  password: [{ required: true, message: '请输入密码', trigger: 'blur' }],
}

const registerRules = {
  realName: [{ required: true, message: '请输入真实姓名', trigger: 'blur' }],
  username: [{ required: true, message: '请输入用户名', trigger: 'blur' }],
  email: [
    { required: true, message: '请输入邮箱', trigger: 'blur' },
    { type: 'email', message: '请输入正确的邮箱格式', trigger: 'blur' }
  ],
  password: [
    { required: true, message: '请输入密码', trigger: 'blur' },
    { min: 6, message: '密码长度不能少于6位', trigger: 'blur' }
  ],
  confirmPassword: [
    { required: true, message: '请确认密码', trigger: 'blur' },
    {
      validator: (_rule: any, value: string) => {
        if (value !== registerForm.password) {
          return Promise.reject('两次输入的密码不一致')
        }
        return Promise.resolve()
      },
      trigger: 'blur'
    }
  ],
  role: [{ required: true, message: '请选择用户角色', trigger: 'change' }]
}

// 状态
const loading = ref(false)
const showForgotModal = ref(false)


// 选择角色
const selectRole = (role: 'teacher' | 'student') => {
  selectedRole.value = role
}



// 处理登录
const handleLogin = async () => {
  try {
    loading.value = true
    
    // 准备登录数据
    const loginData = {
      username: loginForm.username,
      password: loginForm.password,
      role: selectedRole.value.toUpperCase()
    }
    
    console.log('🚀 提交登录数据:', loginData)
    
    // 调用auth store的登录方法
    const result = await authStore.login(loginData)
    
    if (result.success) {
      message.success('登录成功')
      
      // 根据用户角色跳转到对应页面
      const userRole = result.data.userInfo.role
      console.log('登录成功，用户角色:', userRole)
      
      // 强制转为大写进行比较，确保角色匹配不区分大小写
      if (userRole.toUpperCase() === 'TEACHER') {
        console.log('跳转到教师端...')
        await router.push('/teacher/dashboard')
      } else if (userRole.toUpperCase() === 'STUDENT') {
        console.log('跳转到学生端...')  
        // 使用 replace 而不是 push，避免历史记录问题
        await router.replace('/student/dashboard')
      } else {
        console.log('跳转到首页...')
        await router.push('/')
      }
    } else {
      message.error(result.message || '登录失败')
    }
    
  } catch (error: any) {
    console.error('❌ 登录失败:', error)
    if (error.response?.data?.message) {
      message.error(error.response.data.message)
    } else {
      message.error(error.message || '登录失败，请检查用户名和密码')
    }
  } finally {
    loading.value = false
  }
}

// 处理忘记密码
const handleForgotPassword = () => {
  message.info('密码重置邮件已发送到您的邮箱')
}

// 处理注册
const handleRegister = async () => {
  try {
    loading.value = true
    
    // 准备注册数据
    const registerData = {
      username: registerForm.username,
      password: registerForm.password,
      confirmPassword: registerForm.confirmPassword,
      email: registerForm.email,
      realName: registerForm.realName,
      role: registerForm.role
    }
    
    // 调用注册API
    const response = await register(registerData)
    
    if (response.data.success) {
      message.success('注册成功！正在登录...')
      
      // 注册成功后自动登录
      const loginData = {
        username: registerForm.username,
        password: registerForm.password,
        role: registerForm.role
      }
      
      // 使用authStore登录
      const loginResult = await authStore.login(loginData)
      
      if (loginResult.success) {
        message.success('登录成功')
        
        // 根据用户角色跳转到对应页面
        const userRole = loginResult.data.userInfo.role
        // 强制转为大写进行比较，确保角色匹配不区分大小写
        if (userRole.toUpperCase() === 'TEACHER') {
          router.replace('/teacher/dashboard')
        } else if (userRole.toUpperCase() === 'STUDENT') {
          router.replace('/student/dashboard')
      } else {
          router.push('/')
        }
      } else {
        message.error('自动登录失败，请手动登录')
        router.push('/login') // 跳转回登录页面
      }
    } else {
      message.error(response.data.message || '注册失败')
    }
  } catch (error: any) {
    console.error('注册失败:', error)
    if (error.response?.data?.message) {
      message.error(error.response.data.message)
    } else {
      message.error('注册失败，请稍后重试')
    }
  } finally {
    loading.value = false
  }
}

// 跳转到注册页面
const goToRegister = () => {
  router.push('/register')
}


</script>

<style scoped>
.login-page {
  min-height: 100vh;
  display: flex;
  background: #ffffff;
}

/* 左侧视觉区域 */
.login-visual {
  flex: 1;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  display: flex;
  align-items: center;
  justify-content: center;
  position: relative;
  overflow: hidden;
}

.login-visual::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><defs><pattern id="grain" width="100" height="100" patternUnits="userSpaceOnUse"><circle cx="50" cy="50" r="1" fill="%23ffffff" opacity="0.1"/></pattern></defs><rect width="100" height="100" fill="url(%23grain)"/></svg>') repeat;
  pointer-events: none;
}

.visual-content {
  max-width: 480px;
  padding: 48px;
  position: relative;
  z-index: 1;
}

.brand-section {
  margin-bottom: 64px;
}

.logo {
  display: flex;
  align-items: center;
  gap: 12px;
  margin-bottom: 32px;
}

.logo-img {
  width: 40px;
  height: 40px;
}

.logo-text {
  font-size: 24px;
  font-weight: 600;
  color: white;
}

.visual-title {
  font-size: 3rem;
  font-weight: 700;
  line-height: 1.2;
  margin-bottom: 24px;
  background: linear-gradient(45deg, #ffffff, #e3f2fd);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}

.visual-subtitle {
  font-size: 1.125rem;
  line-height: 1.6;
  opacity: 0.9;
  margin: 0;
}

.feature-highlights {
  display: flex;
  flex-direction: column;
  gap: 24px;
}

.feature-item {
  display: flex;
  align-items: center;
  gap: 16px;
  padding: 20px;
  background: rgba(255, 255, 255, 0.1);
  border-radius: 16px;
  backdrop-filter: blur(10px);
  border: 1px solid rgba(255, 255, 255, 0.2);
}

.feature-icon {
  font-size: 32px;
  width: 48px;
  height: 48px;
  display: flex;
  align-items: center;
  justify-content: center;
  background: rgba(255, 255, 255, 0.2);
  border-radius: 12px;
}

.feature-text h4 {
  font-size: 16px;
  font-weight: 600;
  margin: 0 0 4px 0;
  color: white;
}

.feature-text p {
  font-size: 14px;
  margin: 0;
  opacity: 0.8;
}

/* 右侧表单区域 */
.login-form-section {
  flex: 1;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 48px;
  background: #fafafa;
}

.form-container {
  width: 100%;
  max-width: 400px;
}

.form-header {
  text-align: center;
  margin-bottom: 32px;
}

.form-title {
  font-size: 2rem;
  font-weight: 700;
  color: #333;
  margin-bottom: 8px;
}

.form-subtitle {
  color: #666;
  font-size: 16px;
  margin: 0;
}

/* 身份选择 */
.role-selection {
  margin-bottom: 32px;
}

.role-tabs {
  display: flex;
  background: #f0f0f0;
  border-radius: 12px;
  padding: 4px;
  gap: 4px;
}

.role-tab {
  flex: 1;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
  padding: 12px 16px;
  border-radius: 8px;
  cursor: pointer;
  transition: all 0.3s ease;
  font-weight: 500;
  color: #666;
}

.role-tab:hover {
  background: rgba(24, 144, 255, 0.1);
  color: #1890ff;
}

.role-tab.active {
  background: #1890ff;
  color: white;
  box-shadow: 0 2px 8px rgba(24, 144, 255, 0.3);
}

.step-content {
  animation: fadeInUp 0.5s ease-out;
}

.step-title {
  font-size: 24px;
  font-weight: 600;
  text-align: center;
  margin-bottom: 32px;
  color: #333;
}

.role-selector {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 16px;
  margin-bottom: 32px;
}

.role-card {
  padding: 24px 16px;
  border: 2px solid #e8e8e8;
  border-radius: 16px;
  text-align: center;
  cursor: pointer;
  transition: all 0.3s ease;
  background: #fafafa;
}

.role-card:hover {
  border-color: #1890ff;
  background: #f0f8ff;
  transform: translateY(-2px);
}

.role-card.active {
  border-color: #1890ff;
  background: linear-gradient(135deg, #e6f7ff 0%, #f0f8ff 100%);
  box-shadow: 0 4px 20px rgba(24, 144, 255, 0.2);
}

.role-icon {
  font-size: 32px;
  color: #1890ff;
  margin-bottom: 12px;
}

.role-card h3 {
  font-size: 18px;
  font-weight: 600;
  margin: 0 0 8px 0;
  color: #333;
}

.role-card p {
  font-size: 14px;
  color: #666;
  margin: 0;
  line-height: 1.4;
}

/* 表单样式增强 */
.login-form :deep(.ant-input) {
  height: 48px;
  border-radius: 12px;
  border: 1px solid #d9d9d9;
  font-size: 16px;
  transition: all 0.3s ease;
}

.login-form :deep(.ant-input:focus) {
  border-color: #1890ff;
  box-shadow: 0 0 0 2px rgba(24, 144, 255, 0.2);
}

.login-form :deep(.ant-input-password) {
  height: 48px;
  border-radius: 12px;
}

.login-form :deep(.ant-checkbox-wrapper) {
  font-size: 14px;
  color: #666;
}

.login-btn {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  border: none;
  transition: all 0.3s ease;
}

.login-btn:hover {
  transform: translateY(-2px);
  box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
}

.login-btn:active {
  transform: translateY(0);
}

.forgot-link {
  color: #1890ff;
  text-decoration: none;
  font-size: 14px;
  transition: opacity 0.3s ease;
}

.forgot-link:hover {
  opacity: 0.8;
  text-decoration: underline;
}

.form-footer {
  text-align: center;
  margin-top: 32px;
  padding-top: 24px;
  border-top: 1px solid #e8e8e8;
}

.register-link {
  color: #1890ff;
  text-decoration: none;
  font-weight: 500;
}

.register-link:hover {
  text-decoration: underline;
}

.continue-btn {
  width: 100%;
  height: 48px;
  font-size: 16px;
  font-weight: 600;
  border-radius: 12px;
}

.step-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 24px;
}

.back-btn {
  color: #666;
  font-size: 14px;
}

.selected-role {
  display: flex;
  align-items: center;
}

.role-badge {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 6px 12px;
  border-radius: 20px;
  font-size: 14px;
  font-weight: 500;
}

.role-badge.teacher {
  background: #e6f7ff;
  color: #1890ff;
}

.role-badge.student {
  background: #f6ffed;
  color: #52c41a;
}

.login-form {
  margin-top: 24px;
}



.form-options {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 24px;
}

.forgot-password {
  color: #1890ff;
  text-decoration: none;
}

.forgot-password:hover {
  text-decoration: underline;
}

.login-btn {
  height: 48px;
  font-size: 16px;
  font-weight: 600;
  border-radius: 12px;
}

.login-footer {
  text-align: center;
  margin-top: 24px;
  color: #666;
}

.login-footer a {
  color: #1890ff;
  text-decoration: none;
  cursor: pointer;
}

.login-footer a:hover {
  text-decoration: underline;
}

.ai-assistant {
  position: fixed;
  bottom: 32px;
  right: 32px;
  display: flex;
  align-items: center;
  gap: 12px;
  z-index: 3;
}

.ai-btn {
  width: 56px;
  height: 56px;
  font-size: 24px;
  box-shadow: 0 4px 20px rgba(24, 144, 255, 0.3);
}

.ai-text {
  background: rgba(255, 255, 255, 0.9);
  padding: 8px 16px;
  border-radius: 20px;
  font-size: 14px;
  color: #333;
  backdrop-filter: blur(10px);
}

@keyframes fadeInUp {
  from {
    opacity: 0;
    transform: translateY(20px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

/* 响应式设计 */
@media (max-width: 1024px) {
  .login-page {
    flex-direction: column;
  }
  
  .login-visual {
    min-height: 40vh;
  }
  
  .visual-content {
    padding: 32px;
    text-align: center;
  }
  
  .visual-title {
    font-size: 2.5rem;
  }
  
  .feature-highlights {
    flex-direction: row;
    justify-content: center;
    flex-wrap: wrap;
  }
  
  .feature-item {
    flex: 1;
    min-width: 200px;
    max-width: 300px;
  }
}

@media (max-width: 768px) {
  .login-form-section {
    padding: 32px 24px;
  }
  
  .visual-content {
    padding: 24px;
  }
  
  .visual-title {
    font-size: 2rem;
  }
  
  .feature-highlights {
    flex-direction: column;
  }
  
  .role-tabs {
    flex-direction: column;
  }
  
  .role-selector {
    grid-template-columns: 1fr;
  }
}

/* PC端大屏幕优化 */
@media (min-width: 1024px) {
  .step-title {
    font-size: 28px;
    margin-bottom: 32px;
  }
  
  .role-selector {
    gap: 24px;
    margin-bottom: 40px;
  }
  
  .role-card {
    padding: 32px 24px;
    min-height: 160px;
  }
  
  .role-icon {
    font-size: 40px;
    margin-bottom: 16px;
  }
  
  .role-card h3 {
    font-size: 20px;
    margin-bottom: 12px;
  }
  
  .role-card p {
    font-size: 16px;
  }
  
  .login-form .ant-form-item-label > label {
    font-size: 16px;
  }
  
  .login-form .ant-input-affix-wrapper,
  .login-form .ant-input {
    font-size: 16px;
    height: 48px;
  }
  
  .login-btn {
    height: 52px;
    font-size: 18px;
  }
}

/* 超大屏幕优化 */
@media (min-width: 1440px) {
  .role-selector {
    gap: 32px;
  }
  
  .role-card {
    padding: 40px 32px;
    min-height: 180px;
  }
  
  .role-icon {
    font-size: 48px;
  }
  
  .role-card h3 {
    font-size: 22px;
  }
  
  .role-card p {
    font-size: 18px;
  }
}


</style>