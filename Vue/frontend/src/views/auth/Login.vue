<template>
  <div class="login-container">
    <div class="login-background">
      <div class="background-overlay"></div>
    </div>
    
    <div class="login-content">
      <div class="login-card">
        <div class="login-header">
          <div class="logo">
            <img src="/logo.svg" alt="Education Platform" class="logo-img" />
            <h1 class="logo-text">智慧教育平台</h1>
          </div>
          <p class="subtitle">让学习更智能，让教育更高效</p>
        </div>

        <!-- 身份选择步骤 -->
        <div v-if="currentStep === 'role'" class="step-content">
          <h2 class="step-title">选择您的身份</h2>
          <div class="role-selector">
            <div 
              class="role-card" 
              :class="{ active: selectedRole === 'teacher' }"
              @click="selectRole('teacher')"
            >
              <div class="role-icon">
                <UserOutlined />
              </div>
              <h3>教师</h3>
              <p>管理课程、布置作业、批改成绩</p>
            </div>
            
            <div 
              class="role-card" 
              :class="{ active: selectedRole === 'student' }"
              @click="selectRole('student')"
            >
              <div class="role-icon">
                <BookOutlined />
              </div>
              <h3>学生</h3>
              <p>学习课程、完成作业、查看成绩</p>
            </div>
          </div>
          
          <a-button 
            type="primary" 
            size="large" 
            class="continue-btn"
            :disabled="!selectedRole"
            @click="nextStep"
          >
            继续
            <ArrowRightOutlined />
          </a-button>
        </div>

        <!-- 登录表单步骤 -->
        <div v-if="currentStep === 'login'" class="step-content">
          <div class="step-header">
            <a-button 
              type="text" 
              class="back-btn"
              @click="previousStep"
            >
              <ArrowLeftOutlined />
              返回
            </a-button>
            <div class="selected-role">
              <span class="role-badge" :class="selectedRole">
                <UserOutlined v-if="selectedRole === 'teacher'" />
                <BookOutlined v-if="selectedRole === 'student'" />
                {{ selectedRole === 'teacher' ? '教师' : '学生' }}
              </span>
            </div>
          </div>
          
          <h2 class="step-title">登录您的账户</h2>
          
          <a-form
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
              <a class="forgot-password" @click="showForgotModal = true">
                忘记密码？
              </a>
            </div>
            
            <a-form-item>
              <a-button 
                type="primary" 
                html-type="submit" 
                size="large" 
                class="login-btn"
                :loading="loading"
                block
              >
                登录
              </a-button>
            </a-form-item>
          </a-form>
          
          <div class="login-footer">
            <p>还没有账户？ <a @click="showRegisterModal = true">立即注册</a></p>
          </div>
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
    
    <!-- 注册弹窗 -->
    <a-modal
      v-model:open="showRegisterModal"
      title="用户注册"
      @ok="handleRegister"
    >
      <a-form :model="registerForm" layout="vertical">
        <a-form-item label="用户名">
          <a-input v-model:value="registerForm.username" placeholder="请输入用户名" />
        </a-form-item>
        <a-form-item label="密码">
          <a-input-password v-model:value="registerForm.password" placeholder="请输入密码" />
        </a-form-item>
        <a-form-item label="确认密码">
          <a-input-password v-model:value="registerForm.confirmPassword" placeholder="请确认密码" />
        </a-form-item>
        <a-form-item label="邮箱">
          <a-input v-model:value="registerForm.email" placeholder="请输入邮箱" />
        </a-form-item>
        <a-form-item label="真实姓名">
          <a-input v-model:value="registerForm.realName" placeholder="请输入真实姓名" />
        </a-form-item>
      </a-form>
    </a-modal>
  </div>
</template>

<script setup lang="ts">
import { ref, reactive, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { message } from 'ant-design-vue'
import {
  UserOutlined,
  BookOutlined,
  ArrowRightOutlined,
  ArrowLeftOutlined,
  LockOutlined,
  RobotOutlined
} from '@ant-design/icons-vue'
import { useAuthStore } from '@/stores/auth'
import type { LoginRequest } from '@/api/auth'

const router = useRouter()
const authStore = useAuthStore()

// 当前步骤
const currentStep = ref<'role' | 'login'>('role')

// 选择的角色
const selectedRole = ref<'teacher' | 'student' | ''>('')

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
  realName: ''
})

// 表单验证规则
const loginRules = {
  username: [{ required: true, message: '请输入用户名', trigger: 'blur' }],
  password: [{ required: true, message: '请输入密码', trigger: 'blur' }],

}

// 状态
const loading = ref(false)
const showForgotModal = ref(false)
const showRegisterModal = ref(false)


// 选择角色
const selectRole = (role: 'teacher' | 'student') => {
  selectedRole.value = role
}

// 下一步
const nextStep = () => {
  if (selectedRole.value) {
    loginForm.role = selectedRole.value
    currentStep.value = 'login'
  }
}

// 上一步
const previousStep = () => {
  currentStep.value = 'role'
}



// 处理登录
const handleLogin = async () => {
  try {
    loading.value = true
    
    // 准备登录数据
    const loginData: LoginRequest = {
      username: loginForm.username,
      password: loginForm.password,

    }
    
    // 调用登录接口
    const result = await authStore.loginUser(loginData)
    

    
  } catch (error: any) {
    message.error(error.message || '登录失败，请检查用户名和密码')
  } finally {
    loading.value = false
  }
}

// 处理忘记密码
const handleForgotPassword = () => {
  message.info('密码重置邮件已发送到您的邮箱')
  showForgotModal.value = false
}

// 处理注册
const handleRegister = () => {
  message.success('注册成功，请登录')
  showRegisterModal.value = false
}


</script>

<style scoped>
.login-container {
  position: relative;
  min-height: 100vh;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 20px;
  box-sizing: border-box;
}

.login-background {
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  z-index: 1;
}

.background-overlay {
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><defs><pattern id="grain" width="100" height="100" patternUnits="userSpaceOnUse"><circle cx="50" cy="50" r="1" fill="%23ffffff" opacity="0.1"/></pattern></defs><rect width="100" height="100" fill="url(%23grain)"/></svg>') repeat;
  opacity: 0.3;
}

.login-content {
  position: relative;
  z-index: 2;
  width: 100%;
  max-width: 480px;
  padding: 0;
  box-sizing: border-box;
}




.login-card {
  background: rgba(255, 255, 255, 0.95);
  backdrop-filter: blur(20px);
  border-radius: 24px;
  padding: 40px;
  box-shadow: 0 20px 60px rgba(0, 0, 0, 0.1);
  border: 1px solid rgba(255, 255, 255, 0.2);
}

.login-header {
  text-align: center;
  margin-bottom: 40px;
}

.logo {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 12px;
  margin-bottom: 16px;
}

.logo-img {
  width: 48px;
  height: 48px;
}

.logo-text {
  font-size: 28px;
  font-weight: 700;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  margin: 0;
}

.subtitle {
  color: #666;
  font-size: 16px;
  margin: 0;
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

/* PC端大屏幕优化 */
@media (min-width: 1024px) {
  .login-container {
    display: flex;
    align-items: center;
    justify-content: center;
    min-height: 100vh;
    padding: 20px;
  }
  
         .login-content {
    max-width: 800px;
    width: 100%;
    padding: 0;
  }
  
  .login-card {
    padding: 50px 60px;
    width: 100%;
    max-width: 700px;
    box-shadow: 0 20px 60px rgba(0, 0, 0, 0.1);
  }
  
  .logo-text {
    font-size: 32px;
  }
  
  .subtitle {
    font-size: 18px;
  }
  
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
  
  .continue-btn {
    height: 52px;
    font-size: 18px;
    margin-top: 8px;
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
  .login-content {
    max-width: 900px;
  }
  
  .login-card {
    padding: 60px 80px;
    max-width: 800px;
  }
  
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