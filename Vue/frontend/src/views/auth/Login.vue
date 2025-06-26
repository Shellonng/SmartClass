<template>
  <div class="login-container">
    <!-- 背景装饰 -->
    <div class="background-decoration">
      <div class="floating-shapes">
        <div class="shape shape-1"></div>
        <div class="shape shape-2"></div>
        <div class="shape shape-3"></div>
        <div class="shape shape-4"></div>
      </div>
    </div>

    <!-- 主要内容区 -->
    <div class="login-content">
      <!-- 左侧品牌介绍 -->
      <div class="brand-section">
        <div class="brand-header">
          <div class="logo-container">
            <img src="/logo.svg" alt="SmartClass" class="logo" />
            <h1 class="brand-name">SmartClass</h1>
          </div>
          <h2 class="brand-slogan">智慧教育 · 无界学习</h2>
          <p class="brand-description">
            连接全球优质教育资源，为每位学习者定制专属学习路径
          </p>
        </div>

        <div class="feature-showcase">
          <div class="feature-item" v-for="(feature, index) in features" :key="index">
            <div class="feature-icon">
              <component :is="feature.icon" />
            </div>
            <div class="feature-content">
              <h4>{{ feature.title }}</h4>
              <p>{{ feature.description }}</p>
            </div>
          </div>
        </div>

        <!-- 统计数据 -->
        <div class="stats-section">
          <div class="stat-item">
            <span class="stat-number">10K+</span>
            <span class="stat-label">注册用户</span>
          </div>
          <div class="stat-item">
            <span class="stat-number">500+</span>
            <span class="stat-label">精品课程</span>
          </div>
          <div class="stat-item">
            <span class="stat-number">95%</span>
            <span class="stat-label">满意度</span>
          </div>
        </div>
      </div>

      <!-- 右侧登录表单 -->
      <div class="form-section">
        <div class="form-container">
          <div class="form-header">
            <h3>欢迎回来</h3>
            <p>选择您的身份进行登录</p>
          </div>

          <!-- 角色切换 -->
          <div class="role-switcher">
            <div 
              class="role-option"
              :class="{ active: selectedRole === 'student' }"
              @click="selectedRole = 'student'"
            >
              <div class="role-icon">
                <BookOutlined />
              </div>
              <div class="role-info">
                <span class="role-title">学生登录</span>
                <span class="role-desc">探索知识的海洋</span>
              </div>
            </div>
            <div 
              class="role-option"
              :class="{ active: selectedRole === 'teacher' }"
              @click="selectedRole = 'teacher'"
            >
              <div class="role-icon">
                <UserOutlined />
              </div>
              <div class="role-info">
                <span class="role-title">教师登录</span>
                <span class="role-desc">传播智慧的力量</span>
              </div>
            </div>
          </div>

          <!-- 登录表单 -->
          <a-form
            :model="loginForm"
            :rules="loginRules"
            @finish="handleLogin"
            layout="vertical"
            class="login-form"
          >
            <a-form-item name="username">
              <a-input
                v-model:value="loginForm.username"
                size="large"
                placeholder="请输入用户名或邮箱"
                class="custom-input"
              >
                <template #prefix>
                  <UserOutlined class="input-icon" />
                </template>
              </a-input>
            </a-form-item>

            <a-form-item name="password">
              <a-input-password
                v-model:value="loginForm.password"
                size="large"
                placeholder="请输入密码"
                class="custom-input"
              >
                <template #prefix>
                  <LockOutlined class="input-icon" />
                </template>
              </a-input-password>
            </a-form-item>

            <div class="form-options">
              <a-checkbox v-model:checked="loginForm.remember">
                记住登录状态
              </a-checkbox>
              <a class="forgot-password" @click="showForgotModal = true">
                忘记密码？
              </a>
            </div>

            <a-button
              type="primary"
              html-type="submit"
              size="large"
              block
              :loading="loading"
              class="login-button"
            >
              <span v-if="!loading">
                {{ selectedRole === 'student' ? '学生登录' : '教师登录' }}
              </span>
              <span v-else>登录中...</span>
            </a-button>
          </a-form>

          <div class="form-footer">
            <p>还没有账户？ 
              <a class="register-link" @click="showRegisterModal = true">
                立即注册
              </a>
            </p>
          </div>

          <!-- 第三方登录 -->
          <div class="social-login">
            <div class="divider">
              <span>或使用以下方式登录</span>
            </div>
            <div class="social-buttons">
              <a-button class="social-btn wechat">
                <WechatOutlined />
                微信
              </a-button>
              <a-button class="social-btn qq">
                <QqOutlined />
                QQ
              </a-button>
            </div>
          </div>
        </div>

        <!-- AI助手 -->
        <div class="ai-assistant">
          <a-tooltip title="智能登录助手">
            <a-button
              type="primary"
              shape="circle"
              size="large"
              class="ai-button"
              @click="openAIAssistant"
            >
              <RobotOutlined />
            </a-button>
          </a-tooltip>
        </div>
      </div>
    </div>

    <!-- 忘记密码弹窗 -->
    <a-modal
      v-model:open="showForgotModal"
      title="找回密码"
      @ok="handleForgotPassword"
      class="custom-modal"
    >
      <a-form :model="forgotForm" layout="vertical">
        <a-form-item label="邮箱地址">
          <a-input
            v-model:value="forgotForm.email"
            placeholder="请输入注册时的邮箱地址"
          />
        </a-form-item>
      </a-form>
    </a-modal>

    <!-- 注册弹窗 -->
    <a-modal
      v-model:open="showRegisterModal"
      title="用户注册"
      @ok="handleRegister"
      width="500px"
      class="custom-modal"
    >
      <a-form :model="registerForm" layout="vertical">
        <a-form-item label="选择身份" required>
          <a-radio-group v-model:value="registerForm.role">
            <a-radio value="student">学生</a-radio>
            <a-radio value="teacher">教师</a-radio>
          </a-radio-group>
        </a-form-item>
        <a-form-item label="用户名" required>
          <a-input v-model:value="registerForm.username" placeholder="请输入用户名" />
        </a-form-item>
        <a-form-item label="邮箱" required>
          <a-input v-model:value="registerForm.email" placeholder="请输入邮箱地址" />
        </a-form-item>
        <a-form-item label="密码" required>
          <a-input-password v-model:value="registerForm.password" placeholder="请输入密码" />
        </a-form-item>
        <a-form-item label="确认密码" required>
          <a-input-password v-model:value="registerForm.confirmPassword" placeholder="请确认密码" />
        </a-form-item>
        <a-form-item label="真实姓名" required>
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
  LockOutlined,
  RobotOutlined,
  BulbOutlined,
  TeamOutlined,
  BarChartOutlined,
  WechatOutlined,
  QqOutlined
} from '@ant-design/icons-vue'
import { useAuthStore } from '@/stores/auth'
import type { LoginRequest } from '@/api/auth'

const router = useRouter()
const authStore = useAuthStore()

// 功能特色数据
const features = [
  {
    icon: 'BulbOutlined',
    title: 'AI智能教学',
    description: '个性化学习推荐，智能答疑辅导'
  },
  {
    icon: 'TeamOutlined',
    title: '互动课堂',
    description: '实时在线互动，提升学习体验'
  },
  {
    icon: 'BarChartOutlined',
    title: '学习分析',
    description: '数据驱动教学，精准把握学情'
  }
]

// 表单数据
const selectedRole = ref<'student' | 'teacher'>('student')
const loading = ref(false)
const showForgotModal = ref(false)
const showRegisterModal = ref(false)

const loginForm = reactive({
  username: '',
  password: '',
  remember: false
})

const forgotForm = reactive({
  email: ''
})

const registerForm = reactive({
  role: 'student',
  username: '',
  email: '',
  password: '',
  confirmPassword: '',
  realName: ''
})

// 表单验证规则
const loginRules = {
  username: [{ required: true, message: '请输入用户名', trigger: 'blur' }],
  password: [{ required: true, message: '请输入密码', trigger: 'blur' }]
}

// 处理登录
const handleLogin = async () => {
  try {
    loading.value = true
    
    const loginData: LoginRequest = {
      username: loginForm.username,
      password: loginForm.password
    }
    
    await authStore.loginUser(loginData)
    
    message.success('登录成功！')
    
    // 根据角色跳转
    if (selectedRole.value === 'student') {
      router.push('/student/dashboard')
    } else {
      router.push('/teacher/dashboard')
    }
    
  } catch (error: any) {
    message.error(error.message || '登录失败，请检查用户名和密码')
  } finally {
    loading.value = false
  }
}

// 处理忘记密码
const handleForgotPassword = () => {
  if (!forgotForm.email) {
    message.warning('请输入邮箱地址')
    return
  }
  
  message.success('密码重置邮件已发送，请查收')
  showForgotModal.value = false
  forgotForm.email = ''
}

// 处理注册
const handleRegister = () => {
  if (!registerForm.username || !registerForm.password || !registerForm.email || !registerForm.realName) {
    message.warning('请填写完整信息')
    return
  }
  
  if (registerForm.password !== registerForm.confirmPassword) {
    message.warning('两次输入的密码不一致')
    return
  }
  
  message.success('注册成功，请登录')
  showRegisterModal.value = false
  
  // 清空表单
  Object.assign(registerForm, {
    role: 'student',
    username: '',
    email: '',
    password: '',
    confirmPassword: '',
    realName: ''
  })
}

// 打开AI助手
const openAIAssistant = () => {
  message.info('AI助手功能即将上线，敬请期待！')
}

// 页面加载时的动画
onMounted(() => {
  // 添加页面加载动画
  const shapes = document.querySelectorAll('.shape')
  shapes.forEach((shape, index) => {
    setTimeout(() => {
      shape.classList.add('animate')
    }, index * 200)
  })
})
</script>

<style scoped>
.login-container {
  min-height: 100vh;
  position: relative;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  overflow: hidden;
}

/* 背景装饰 */
.background-decoration {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  pointer-events: none;
}

.floating-shapes {
  position: relative;
  width: 100%;
  height: 100%;
}

.shape {
  position: absolute;
  background: rgba(255, 255, 255, 0.1);
  border-radius: 50%;
  opacity: 0;
  animation: float 6s ease-in-out infinite;
  transition: opacity 0.5s ease;
}

.shape.animate {
  opacity: 1;
}

.shape-1 {
  width: 80px;
  height: 80px;
  top: 20%;
  left: 10%;
  animation-delay: 0s;
}

.shape-2 {
  width: 120px;
  height: 120px;
  top: 60%;
  left: 5%;
  animation-delay: 2s;
}

.shape-3 {
  width: 60px;
  height: 60px;
  top: 30%;
  right: 15%;
  animation-delay: 4s;
}

.shape-4 {
  width: 100px;
  height: 100px;
  bottom: 20%;
  right: 10%;
  animation-delay: 1s;
}

@keyframes float {
  0%, 100% {
    transform: translateY(0px) rotate(0deg);
  }
  50% {
    transform: translateY(-20px) rotate(180deg);
  }
}

/* 主要内容区 */
.login-content {
  position: relative;
  min-height: 100vh;
  display: flex;
  align-items: center;
  padding: 40px;
  max-width: 1400px;
  margin: 0 auto;
}

/* 左侧品牌区 */
.brand-section {
  flex: 1;
  padding-right: 80px;
  color: white;
}

.brand-header {
  margin-bottom: 60px;
}

.logo-container {
  display: flex;
  align-items: center;
  gap: 16px;
  margin-bottom: 24px;
}

.logo {
  width: 48px;
  height: 48px;
}

.brand-name {
  font-size: 32px;
  font-weight: 700;
  margin: 0;
  background: linear-gradient(45deg, #ffffff, #e3f2fd);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}

.brand-slogan {
  font-size: 28px;
  font-weight: 600;
  margin: 0 0 16px 0;
  opacity: 0.95;
}

.brand-description {
  font-size: 16px;
  line-height: 1.6;
  opacity: 0.8;
  margin: 0;
}

.feature-showcase {
  margin-bottom: 60px;
}

.feature-item {
  display: flex;
  align-items: center;
  gap: 20px;
  margin-bottom: 32px;
  padding: 24px;
  background: rgba(255, 255, 255, 0.1);
  border-radius: 16px;
  backdrop-filter: blur(10px);
  border: 1px solid rgba(255, 255, 255, 0.2);
  transition: all 0.3s ease;
}

.feature-item:hover {
  background: rgba(255, 255, 255, 0.15);
  transform: translateY(-2px);
}

.feature-icon {
  width: 56px;
  height: 56px;
  background: rgba(255, 255, 255, 0.2);
  border-radius: 16px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 24px;
}

.feature-content h4 {
  font-size: 18px;
  font-weight: 600;
  margin: 0 0 8px 0;
}

.feature-content p {
  font-size: 14px;
  margin: 0;
  opacity: 0.8;
}

.stats-section {
  display: flex;
  gap: 40px;
}

.stat-item {
  text-align: center;
}

.stat-number {
  display: block;
  font-size: 32px;
  font-weight: 700;
  margin-bottom: 8px;
}

.stat-label {
  font-size: 14px;
  opacity: 0.8;
}

/* 右侧表单区 */
.form-section {
  flex: 1;
  max-width: 480px;
  position: relative;
}

.form-container {
  background: white;
  border-radius: 24px;
  padding: 48px;
  box-shadow: 0 24px 64px rgba(0, 0, 0, 0.1);
  backdrop-filter: blur(10px);
}

.form-header {
  text-align: center;
  margin-bottom: 40px;
}

.form-header h3 {
  font-size: 28px;
  font-weight: 700;
  color: #333;
  margin: 0 0 8px 0;
}

.form-header p {
  color: #666;
  font-size: 16px;
  margin: 0;
}

/* 角色切换 */
.role-switcher {
  margin-bottom: 32px;
}

.role-option {
  display: flex;
  align-items: center;
  gap: 16px;
  padding: 20px;
  border: 2px solid #f0f0f0;
  border-radius: 16px;
  margin-bottom: 12px;
  cursor: pointer;
  transition: all 0.3s ease;
}

.role-option:last-child {
  margin-bottom: 0;
}

.role-option:hover {
  border-color: #d9d9d9;
  background: #fafafa;
}

.role-option.active {
  border-color: #1890ff;
  background: linear-gradient(135deg, #e6f7ff 0%, #f0f9ff 100%);
}

.role-icon {
  width: 48px;
  height: 48px;
  background: #f0f0f0;
  border-radius: 12px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 20px;
  color: #666;
  transition: all 0.3s ease;
}

.role-option.active .role-icon {
  background: #1890ff;
  color: white;
}

.role-info {
  flex: 1;
}

.role-title {
  display: block;
  font-size: 16px;
  font-weight: 600;
  color: #333;
  margin-bottom: 4px;
}

.role-desc {
  font-size: 14px;
  color: #666;
}

/* 表单样式 */
.login-form {
  margin-bottom: 32px;
}

.custom-input {
  border-radius: 12px;
  border: 1px solid #e0e0e0;
  transition: all 0.3s ease;
}

.custom-input:hover {
  border-color: #1890ff;
}

.custom-input:focus,
.custom-input:focus-within {
  border-color: #1890ff;
  box-shadow: 0 0 0 2px rgba(24, 144, 255, 0.2);
}

.input-icon {
  color: #999;
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
  font-size: 14px;
}

.forgot-password:hover {
  color: #40a9ff;
}

.login-button {
  height: 48px;
  border-radius: 12px;
  font-size: 16px;
  font-weight: 600;
  background: linear-gradient(135deg, #1890ff 0%, #36cfc9 100%);
  border: none;
  transition: all 0.3s ease;
}

.login-button:hover {
  background: linear-gradient(135deg, #40a9ff 0%, #5cdbd3 100%);
  transform: translateY(-1px);
  box-shadow: 0 8px 24px rgba(24, 144, 255, 0.3);
}

.form-footer {
  text-align: center;
  margin-bottom: 32px;
}

.form-footer p {
  color: #666;
  font-size: 14px;
  margin: 0;
}

.register-link {
  color: #1890ff;
  text-decoration: none;
  font-weight: 500;
}

.register-link:hover {
  color: #40a9ff;
}

/* 第三方登录 */
.social-login {
  text-align: center;
}

.divider {
  position: relative;
  margin: 32px 0;
  text-align: center;
}

.divider::before {
  content: '';
  position: absolute;
  top: 50%;
  left: 0;
  right: 0;
  height: 1px;
  background: #e0e0e0;
}

.divider span {
  background: white;
  padding: 0 16px;
  color: #999;
  font-size: 14px;
}

.social-buttons {
  display: flex;
  gap: 12px;
}

.social-btn {
  flex: 1;
  height: 44px;
  border-radius: 12px;
  font-size: 14px;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
}

.social-btn.wechat {
  background: #07c160;
  border-color: #07c160;
  color: white;
}

.social-btn.qq {
  background: #12b7f5;
  border-color: #12b7f5;
  color: white;
}

/* AI助手 */
.ai-assistant {
  position: absolute;
  bottom: 32px;
  right: 32px;
}

.ai-button {
  width: 56px;
  height: 56px;
  background: linear-gradient(135deg, #fa541c 0%, #faad14 100%);
  border: none;
  font-size: 24px;
  box-shadow: 0 8px 24px rgba(250, 84, 28, 0.3);
  animation: pulse 2s infinite;
}

@keyframes pulse {
  0% {
    box-shadow: 0 8px 24px rgba(250, 84, 28, 0.3);
  }
  50% {
    box-shadow: 0 8px 32px rgba(250, 84, 28, 0.5);
  }
  100% {
    box-shadow: 0 8px 24px rgba(250, 84, 28, 0.3);
  }
}

/* 弹窗样式 */
.custom-modal .ant-modal-content {
  border-radius: 16px;
  overflow: hidden;
}

.custom-modal .ant-modal-header {
  background: linear-gradient(135deg, #1890ff 0%, #36cfc9 100%);
  border: none;
  padding: 24px;
}

.custom-modal .ant-modal-title {
  color: white;
  font-size: 18px;
  font-weight: 600;
}

.custom-modal .ant-modal-body {
  padding: 32px 24px;
}

/* 响应式设计 */
@media (max-width: 1200px) {
  .login-content {
    flex-direction: column;
    text-align: center;
  }
  
  .brand-section {
    padding-right: 0;
    margin-bottom: 40px;
  }
  
  .stats-section {
    justify-content: center;
  }
}

@media (max-width: 768px) {
  .login-content {
    padding: 20px;
  }
  
  .form-container {
    padding: 32px 24px;
  }
  
  .feature-item {
    flex-direction: column;
    text-align: center;
  }
  
  .stats-section {
    flex-direction: column;
    gap: 20px;
  }
  
  .social-buttons {
    flex-direction: column;
  }
}
</style>