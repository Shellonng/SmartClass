<template>
  <div class="welcome-page">
    <!-- 顶部导航 -->
    <header class="header">
      <div class="header-container">
        <div class="logo-section">
          <img src="/logo.svg" alt="智慧课堂" class="logo" />
          <span class="brand-name">智慧课堂</span>
        </div>
        
        <nav class="nav-menu">
          <a href="#courses" class="nav-link">课程</a>
          <a href="#about" class="nav-link">关于我们</a>
          <a href="#help" class="nav-link">帮助中心</a>
        </nav>
        
        <div class="auth-buttons">
          <!-- 未登录状态 -->
          <template v-if="!authStore.isAuthenticated">
            <a-button @click="showLoginModal = true" class="login-btn">登录</a-button>
            <a-button type="primary" @click="showRegisterModal = true" class="register-btn">注册</a-button>
          </template>
          
          <!-- 已登录状态 -->
          <template v-else>
            <a-dropdown>
              <a-button class="user-menu-btn">
                <UserOutlined />
                {{ authStore.user?.realName || authStore.user?.username }}
                <DownOutlined />
              </a-button>
              <template #overlay>
                <a-menu>
                  <a-menu-item key="dashboard" @click="goToDashboard">
                    <DashboardOutlined />
                    仪表盘
                  </a-menu-item>
                  <a-menu-item key="profile">
                    <UserOutlined />
                    个人资料
                  </a-menu-item>
                  <a-menu-divider />
                  <a-menu-item key="logout" @click="handleLogout">
                    <LogoutOutlined />
                    退出登录
                  </a-menu-item>
                </a-menu>
              </template>
            </a-dropdown>
          </template>
        </div>
      </div>
    </header>

    <!-- 主横幅区域 -->
    <section class="hero-section">
      <div class="hero-container">
        <div class="hero-content">
          <h1 class="hero-title">
            开启智慧学习之旅
            <br>
            <span class="highlight">汇聚全球优质教育资源</span>
          </h1>
          <p class="hero-subtitle">
            智慧课堂是面向未来的在线学习平台，为学习者提供从高校课程到实战技能的在线教育服务
          </p>
          <div class="hero-actions">
            <a-button 
              type="primary" 
              size="large" 
              @click="authStore.isAuthenticated ? goToDashboard() : showLoginModal = true" 
              class="start-btn"
            >
              {{ authStore.isAuthenticated ? '进入学习' : '开始学习' }}
            </a-button>
            <a-button size="large" @click="scrollToCourses" class="explore-btn">
              浏览课程
            </a-button>
          </div>
        </div>
        
        <div class="hero-visual">
          <div class="visual-card">
            <div class="card-content">
              <div class="stats-grid">
                <div class="stat-item" v-for="stat in platformStats" :key="stat.label">
                  <div class="stat-number">{{ stat.value }}</div>
                  <div class="stat-label">{{ stat.label }}</div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>

    <!-- 特色功能区域 -->
    <section class="features-section">
      <div class="container">
        <h2 class="section-title">为什么选择智慧课堂</h2>
        <div class="features-grid">
          <div class="feature-card" v-for="feature in features" :key="feature.title">
            <div class="feature-icon">{{ feature.icon }}</div>
            <h3 class="feature-title">{{ feature.title }}</h3>
            <p class="feature-description">{{ feature.description }}</p>
            <ul class="feature-points">
              <li v-for="point in feature.points" :key="point">{{ point }}</li>
            </ul>
          </div>
        </div>
      </div>
    </section>

    <!-- 精品课程展示 -->
    <section id="courses" class="courses-section">
      <div class="container">
        <div class="section-header">
          <h2 class="section-title">精品课程推荐</h2>
          <a-button type="link" @click="goToCourses" class="view-all-btn">
            查看全部课程 <ArrowRightOutlined />
          </a-button>
        </div>
        
        <div class="courses-grid">
          <div class="course-card" v-for="course in featuredCourses" :key="course.id">
            <div class="course-image">
              <img :src="course.image || '/course-placeholder.jpg'" :alt="course.title" />
              <div class="course-overlay">
                <PlayCircleOutlined class="play-icon" />
              </div>
            </div>
            <div class="course-info">
              <div class="course-meta">
                <span class="university">{{ course.university }}</span>
                <span class="category">{{ course.category }}</span>
              </div>
              <h3 class="course-title">{{ course.title }}</h3>
              <p class="course-instructor">{{ course.instructor }}</p>
              <div class="course-stats">
                <span class="students-count">
                  <UserOutlined /> {{ formatNumber(course.students) }}人学习
                </span>
                <div class="rating">
                  <StarFilled v-for="i in 5" :key="i" class="star" />
                  <span class="rating-text">4.8</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>

    <!-- 合作伙伴 -->
    <section class="partners-section">
      <div class="container">
        <h2 class="section-title">合作伙伴</h2>
        <div class="partners-grid">
          <div class="partner-item" v-for="partner in partners" :key="partner.name">
            <img :src="partner.logo" :alt="partner.name" class="partner-logo" />
          </div>
        </div>
      </div>
    </section>

    <!-- 页脚 -->
    <footer class="footer">
      <div class="container">
        <div class="footer-content">
          <div class="footer-section">
            <div class="footer-logo">
              <img src="/logo.svg" alt="智慧课堂" class="logo" />
              <span class="brand-name">智慧课堂</span>
            </div>
            <p class="footer-description">
              致力于为学习者提供优质的在线教育服务，
              打造终身学习的智慧平台。
            </p>
          </div>
          
          <div class="footer-section">
            <h4 class="footer-title">课程分类</h4>
            <ul class="footer-links">
              <li><a href="#">计算机科学</a></li>
              <li><a href="#">数学</a></li>
              <li><a href="#">物理</a></li>
              <li><a href="#">语言学习</a></li>
            </ul>
          </div>
          
          <div class="footer-section">
            <h4 class="footer-title">帮助中心</h4>
            <ul class="footer-links">
              <li><a href="#">使用指南</a></li>
              <li><a href="#">常见问题</a></li>
              <li><a href="#">联系我们</a></li>
              <li><a href="#">意见反馈</a></li>
            </ul>
          </div>
          
          <div class="footer-section">
            <h4 class="footer-title">关注我们</h4>
            <div class="social-links">
              <a href="#" class="social-link">微信</a>
              <a href="#" class="social-link">微博</a>
              <a href="#" class="social-link">QQ群</a>
            </div>
          </div>
        </div>
        
        <div class="footer-bottom">
          <p>&copy; 2024 智慧课堂. 保留所有权利.</p>
        </div>
      </div>
    </footer>

    <!-- 登录弹窗 -->
    <a-modal
      v-model:open="showLoginModal"
      title=""
      :footer="null"
      :width="480"
      centered
      class="login-modal"
    >
      <div class="modal-content">
        <div class="modal-header">
          <h2 class="modal-title">欢迎回来</h2>
          <p class="modal-subtitle">请选择您的身份并登录账户</p>
        </div>

        <!-- 身份选择 -->
        <div class="role-selection">
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
            <a class="forgot-link">忘记密码？</a>
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
        
        <div class="form-footer">
          <p>还没有账户？ <a @click="showRegisterModal = true; showLoginModal = false" class="register-link">立即注册</a></p>
        </div>
      </div>
    </a-modal>

    <!-- 注册弹窗 -->
    <a-modal
      v-model:open="showRegisterModal"
      title=""
      :footer="null"
      :width="480"
      centered
      class="register-modal"
    >
      <div class="modal-content">
        <div class="modal-header">
          <h2 class="modal-title">创建账户</h2>
          <p class="modal-subtitle">加入智慧课堂，开启学习之旅</p>
        </div>

        <!-- 注册表单 -->
        <a-form
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
          <p>已有账户？ <a @click="showLoginModal = true; showRegisterModal = false" class="login-link">立即登录</a></p>
        </div>
      </div>
    </a-modal>
  </div>
</template>

<script setup lang="ts">
import { ref, reactive } from 'vue'
import { useRouter } from 'vue-router'
import { message } from 'ant-design-vue'
import {
  UserOutlined,
  ArrowRightOutlined,
  PlayCircleOutlined,
  StarFilled,
  BookOutlined,
  LockOutlined,
  MailOutlined,
  DownOutlined,
  DashboardOutlined,
  LogoutOutlined
} from '@ant-design/icons-vue'
import { useAuthStore } from '@/stores/auth'
import type { LoginRequest } from '@/api/auth'

const router = useRouter()
const authStore = useAuthStore()

// 弹窗状态
const showLoginModal = ref(false)
const showRegisterModal = ref(false)
const loading = ref(false)

// 登录表单
const loginForm = reactive({
  username: '',
  password: '',
  remember: false
})

// 注册表单
const registerForm = reactive({
  username: '',
  password: '',
  confirmPassword: '',
  email: '',
  realName: ''
})

// 选择的角色
const selectedRole = ref<'teacher' | 'student'>('student')

// 表单验证规则
const loginRules = {
  username: [{ required: true, message: '请输入用户名', trigger: 'blur' }],
  password: [{ required: true, message: '请输入密码', trigger: 'blur' }]
}

const registerRules = {
  username: [{ required: true, message: '请输入用户名', trigger: 'blur' }],
  password: [{ required: true, message: '请输入密码', trigger: 'blur' }],
  confirmPassword: [{ required: true, message: '请确认密码', trigger: 'blur' }],
  email: [{ required: true, message: '请输入邮箱', trigger: 'blur' }],
  realName: [{ required: true, message: '请输入真实姓名', trigger: 'blur' }]
}

// 平台统计数据
const platformStats = ref([
  { label: '优质课程', value: '5000+' },
  { label: '注册学员', value: '300万+' },
  { label: '合作高校', value: '200+' },
  { label: '认证证书', value: '50万+' }
])

// 特色功能
const features = ref([
  {
    icon: '🎓',
    title: '致力于汇聚高校优质课程',
    description: '与全国知名高校深度合作，提供最优质的教育资源',
    points: [
      '平台运行5000余门慕课',
      '为学习者提供学习认证证书',
      '打造随时随地学习的平台环境'
    ]
  },
  {
    icon: '🚀',
    title: '打造终身学习平台',
    description: '通过在线教育的方式提供敏捷教育方案',
    points: [
      '通过在线教育的方式提供敏捷教育方案',
      '通过体系化课程构建微专业服务体系',
      '打造服务于终身学习者的学习平台'
    ]
  },
  {
    icon: '💼',
    title: '构建职业技能培训体系',
    description: '联合知名企业，提供实用的职业技能培训',
    points: [
      '联合百度、京东、美团等企业深度合作',
      '融入实践实训环节，提供职业技能培训',
      '帮助学习者更好地应对职场挑战'
    ]
  }
])

// 精品课程
const featuredCourses = ref([
  {
    id: 1,
    title: '高等数学',
    instructor: '张教授',
    university: '清华大学',
    students: 15420,
    image: '',
    category: '数学'
  },
  {
    id: 2,
    title: '计算机程序设计基础',
    instructor: '李教授',
    university: '北京大学',
    students: 12350,
    image: '',
    category: '计算机'
  },
  {
    id: 3,
    title: '大学英语',
    instructor: '王教授',
    university: '复旦大学',
    students: 18900,
    image: '',
    category: '语言'
  },
  {
    id: 4,
    title: '线性代数',
    instructor: '赵教授',
    university: '中科大',
    students: 9800,
    image: '',
    category: '数学'
  }
])

// 合作伙伴
const partners = ref([
  { name: '清华大学', logo: '/partners/tsinghua.png' },
  { name: '北京大学', logo: '/partners/pku.png' },
  { name: '复旦大学', logo: '/partners/fudan.png' },
  { name: '中科大', logo: '/partners/ustc.png' }
])

// 方法
const selectRole = (role: 'teacher' | 'student') => {
  selectedRole.value = role
}

const handleLogin = async () => {
  try {
    loading.value = true
    
    const loginData: LoginRequest = {
      username: loginForm.username,
      password: loginForm.password
    }
    
    // 使用skipRedirect=true避免自动跳转
    const result = await authStore.loginUser(loginData, true)
    
    if (result.success) {
      showLoginModal.value = false
      message.success('登录成功！')
      // 悬浮窗登录成功后不跳转，保持在当前页面
      // 用户可以通过导航菜单或其他方式访问相应功能
    }
  } catch (error: any) {
    message.error(error.message || '登录失败，请检查用户名和密码')
  } finally {
    loading.value = false
  }
}

const handleRegister = async () => {
  try {
    loading.value = true
    
    if (registerForm.password !== registerForm.confirmPassword) {
      message.error('两次输入的密码不一致')
      return
    }
    
    // 这里应该调用注册API
    message.success('注册成功，请登录')
    showRegisterModal.value = false
    showLoginModal.value = true
  } catch (error: any) {
    message.error(error.message || '注册失败')
  } finally {
    loading.value = false
  }
}

const goToLogin = () => {
  router.push('/login')
}

const goToCourses = () => {
  router.push('/courses')
}

const scrollToCourses = () => {
  const element = document.getElementById('courses')
  if (element) {
    element.scrollIntoView({ behavior: 'smooth' })
  }
}

const formatNumber = (num: number) => {
  if (num >= 10000) {
    return (num / 10000).toFixed(1) + '万'
  }
  return num.toString()
}

// 跳转到仪表盘
const goToDashboard = () => {
  if (authStore.user?.role === 'teacher') {
    router.push('/teacher/dashboard')
  } else {
    router.push('/student/dashboard')
  }
}

// 处理退出登录
const handleLogout = async () => {
  try {
    await authStore.logout()
    message.success('已退出登录')
  } catch (error) {
    console.error('退出登录失败:', error)
  }
}
</script>

<style scoped>
.welcome-page {
  min-height: 100vh;
  background: #fff;
}

/* 顶部导航 */
.header {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  background: rgba(255, 255, 255, 0.95);
  backdrop-filter: blur(10px);
  border-bottom: 1px solid #f0f0f0;
  z-index: 1000;
  transition: all 0.3s ease;
}

.header-container {
  max-width: 1200px;
  margin: 0 auto;
  padding: 0 24px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  height: 64px;
}

.logo-section {
  display: flex;
  align-items: center;
  gap: 12px;
}

.logo {
  width: 32px;
  height: 32px;
}

.brand-name {
  font-size: 20px;
  font-weight: 600;
  color: #1890ff;
}

.nav-menu {
  display: flex;
  gap: 32px;
}

.nav-link {
  color: #666;
  text-decoration: none;
  font-weight: 500;
  transition: color 0.3s ease;
}

.nav-link:hover {
  color: #1890ff;
}

.auth-buttons {
  display: flex;
  gap: 12px;
}

.login-btn {
  border: none;
  background: transparent;
  color: #666;
}

.register-btn {
  background: #1890ff;
  border-color: #1890ff;
}

.user-menu-btn {
  border: 1px solid #d9d9d9;
  background: white;
  color: #333;
  display: flex;
  align-items: center;
  gap: 8px;
  font-weight: 500;
}

.user-menu-btn:hover {
  border-color: #1890ff;
  color: #1890ff;
}

/* 主横幅区域 */
.hero-section {
  padding: 120px 0 80px;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  position: relative;
  overflow: hidden;
}

.hero-section::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><defs><pattern id="grain" width="100" height="100" patternUnits="userSpaceOnUse"><circle cx="50" cy="50" r="1" fill="%23ffffff" opacity="0.1"/></pattern></defs><rect width="100" height="100" fill="url(%23grain)"/></svg>') repeat;
  opacity: 0.3;
}

.hero-container {
  max-width: 1200px;
  margin: 0 auto;
  padding: 0 24px;
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 60px;
  align-items: center;
  position: relative;
  z-index: 1;
}

.hero-title {
  font-size: 48px;
  font-weight: 700;
  line-height: 1.2;
  margin-bottom: 24px;
}

.highlight {
  color: #ffd700;
}

.hero-subtitle {
  font-size: 18px;
  line-height: 1.6;
  margin-bottom: 32px;
  opacity: 0.9;
}

.hero-actions {
  display: flex;
  gap: 16px;
}

.start-btn {
  background: #ffd700;
  border-color: #ffd700;
  color: #333;
  font-weight: 600;
}

.explore-btn {
  background: transparent;
  border: 2px solid white;
  color: white;
  font-weight: 600;
}

.visual-card {
  background: rgba(255, 255, 255, 0.1);
  backdrop-filter: blur(20px);
  border-radius: 16px;
  padding: 32px;
  border: 1px solid rgba(255, 255, 255, 0.2);
}

.stats-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 24px;
}

.stat-item {
  text-align: center;
}

.stat-number {
  font-size: 32px;
  font-weight: 700;
  color: #ffd700;
  margin-bottom: 8px;
}

.stat-label {
  font-size: 14px;
  opacity: 0.8;
}

/* 特色功能区域 */
.features-section {
  padding: 80px 0;
  background: #f8f9fa;
}

.container {
  max-width: 1200px;
  margin: 0 auto;
  padding: 0 24px;
}

.section-title {
  font-size: 36px;
  font-weight: 700;
  text-align: center;
  margin-bottom: 60px;
  color: #333;
}

.features-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
  gap: 40px;
}

.feature-card {
  background: white;
  padding: 40px;
  border-radius: 16px;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
  transition: transform 0.3s ease, box-shadow 0.3s ease;
}

.feature-card:hover {
  transform: translateY(-8px);
  box-shadow: 0 8px 40px rgba(0, 0, 0, 0.12);
}

.feature-icon {
  font-size: 48px;
  margin-bottom: 24px;
}

.feature-title {
  font-size: 20px;
  font-weight: 600;
  margin-bottom: 16px;
  color: #333;
}

.feature-description {
  color: #666;
  margin-bottom: 20px;
  line-height: 1.6;
}

.feature-points {
  list-style: none;
  padding: 0;
}

.feature-points li {
  padding: 8px 0;
  color: #666;
  position: relative;
  padding-left: 20px;
}

.feature-points li::before {
  content: '•';
  color: #1890ff;
  position: absolute;
  left: 0;
  font-weight: bold;
}

/* 课程展示区域 */
.courses-section {
  padding: 80px 0;
  background: white;
}

.section-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 60px;
}

.view-all-btn {
  color: #1890ff;
  font-weight: 500;
}

.courses-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
  gap: 32px;
}

.course-card {
  background: white;
  border-radius: 12px;
  overflow: hidden;
  box-shadow: 0 4px 16px rgba(0, 0, 0, 0.08);
  transition: transform 0.3s ease, box-shadow 0.3s ease;
  cursor: pointer;
}

.course-card:hover {
  transform: translateY(-4px);
  box-shadow: 0 8px 32px rgba(0, 0, 0, 0.12);
}

.course-image {
  position: relative;
  height: 180px;
  background: linear-gradient(45deg, #667eea, #764ba2);
  display: flex;
  align-items: center;
  justify-content: center;
}

.course-image img {
  width: 100%;
  height: 100%;
  object-fit: cover;
}

.course-overlay {
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  opacity: 0;
  transition: opacity 0.3s ease;
}

.course-card:hover .course-overlay {
  opacity: 1;
}

.play-icon {
  font-size: 48px;
  color: white;
}

.course-info {
  padding: 20px;
}

.course-meta {
  display: flex;
  justify-content: space-between;
  margin-bottom: 12px;
}

.university {
  color: #1890ff;
  font-size: 12px;
  font-weight: 500;
}

.category {
  background: #f0f0f0;
  color: #666;
  padding: 2px 8px;
  border-radius: 4px;
  font-size: 12px;
}

.course-title {
  font-size: 16px;
  font-weight: 600;
  margin-bottom: 8px;
  color: #333;
}

.course-instructor {
  color: #666;
  margin-bottom: 16px;
}

.course-stats {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.students-count {
  color: #666;
  font-size: 14px;
  display: flex;
  align-items: center;
  gap: 4px;
}

.rating {
  display: flex;
  align-items: center;
  gap: 4px;
}

.star {
  color: #ffd700;
  font-size: 12px;
}

.rating-text {
  color: #666;
  font-size: 14px;
}

/* 合作伙伴 */
.partners-section {
  padding: 60px 0;
  background: #f8f9fa;
}

.partners-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 32px;
  align-items: center;
}

.partner-item {
  text-align: center;
  padding: 20px;
  background: white;
  border-radius: 8px;
  transition: transform 0.3s ease;
}

.partner-item:hover {
  transform: scale(1.05);
}

.partner-logo {
  max-width: 120px;
  max-height: 60px;
  object-fit: contain;
}

/* 页脚 */
.footer {
  background: #001529;
  color: white;
  padding: 60px 0 20px;
}

.footer-content {
  display: grid;
  grid-template-columns: 2fr 1fr 1fr 1fr;
  gap: 40px;
  margin-bottom: 40px;
}

.footer-logo {
  display: flex;
  align-items: center;
  gap: 12px;
  margin-bottom: 16px;
}

.footer-logo .logo {
  width: 32px;
  height: 32px;
  filter: brightness(0) invert(1);
}

.footer-logo .brand-name {
  color: white;
}

.footer-description {
  color: #ccc;
  line-height: 1.6;
}

.footer-title {
  font-size: 16px;
  font-weight: 600;
  margin-bottom: 16px;
}

.footer-links {
  list-style: none;
  padding: 0;
}

.footer-links li {
  margin-bottom: 8px;
}

.footer-links a {
  color: #ccc;
  text-decoration: none;
  transition: color 0.3s ease;
}

.footer-links a:hover {
  color: #1890ff;
}

.social-links {
  display: flex;
  gap: 16px;
}

.social-link {
  color: #ccc;
  text-decoration: none;
  transition: color 0.3s ease;
}

.social-link:hover {
  color: #1890ff;
}

.footer-bottom {
  text-align: center;
  padding-top: 20px;
  border-top: 1px solid #333;
  color: #ccc;
}

/* 弹窗样式 - 采用玻璃拟态设计 */
:deep(.login-modal .ant-modal-content),
:deep(.register-modal .ant-modal-content) {
  border-radius: 32px;
  box-shadow: 
    0 32px 64px rgba(0, 0, 0, 0.12),
    0 0 0 1px rgba(255, 255, 255, 0.05),
    inset 0 1px 0 rgba(255, 255, 255, 0.1);
  border: 1px solid rgba(255, 255, 255, 0.18);
  background: linear-gradient(135deg, 
    rgba(255, 255, 255, 0.95) 0%, 
    rgba(255, 255, 255, 0.85) 100%);
  backdrop-filter: blur(40px) saturate(180%);
  padding: 0;
  overflow: hidden;
  position: relative;
}

:deep(.login-modal .ant-modal-content::before),
:deep(.register-modal .ant-modal-content::before) {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  height: 1px;
  background: linear-gradient(90deg, 
    transparent 0%, 
    rgba(255, 255, 255, 0.4) 50%, 
    transparent 100%);
}

:deep(.login-modal .ant-modal-body),
:deep(.register-modal .ant-modal-body) {
  padding: 0;
}

:deep(.login-modal .ant-modal-mask),
:deep(.register-modal .ant-modal-mask) {
  background: linear-gradient(135deg, 
    rgba(102, 126, 234, 0.1) 0%, 
    rgba(118, 75, 162, 0.15) 100%);
  backdrop-filter: blur(12px);
}

.modal-content {
  padding: 56px 48px 48px;
  background: transparent;
  position: relative;
}

.modal-content::before {
  content: '';
  position: absolute;
  top: 0;
  left: 50%;
  transform: translateX(-50%);
  width: 60px;
  height: 4px;
  background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
  border-radius: 2px;
  opacity: 0.6;
}

.modal-header {
  text-align: center;
  margin-bottom: 48px;
  position: relative;
}

.modal-title {
  font-size: 36px;
  font-weight: 800;
  margin-bottom: 16px;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  letter-spacing: -0.02em;
  line-height: 1.2;
}

.modal-subtitle {
  font-size: 17px;
  color: #64748b;
  margin: 0;
  font-weight: 500;
  opacity: 0.8;
  line-height: 1.5;
}

/* 身份选择样式 - Bento网格设计 */
.role-selection {
  margin-bottom: 40px;
}

.role-tabs {
  display: grid;
  grid-template-columns: 1fr 1fr;
  background: rgba(248, 250, 252, 0.8);
  border-radius: 20px;
  padding: 8px;
  gap: 8px;
  box-shadow: 
    inset 0 2px 8px rgba(0, 0, 0, 0.06),
    0 1px 3px rgba(0, 0, 0, 0.1);
  backdrop-filter: blur(10px);
  border: 1px solid rgba(255, 255, 255, 0.2);
}

.role-tab {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 12px;
  padding: 20px 24px;
  border-radius: 16px;
  cursor: pointer;
  transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
  font-weight: 600;
  font-size: 15px;
  color: #64748b;
  background: rgba(255, 255, 255, 0.7);
  position: relative;
  border: 1px solid rgba(255, 255, 255, 0.3);
  backdrop-filter: blur(10px);
}

.role-tab::before {
  content: '';
  position: absolute;
  inset: 0;
  border-radius: 16px;
  background: linear-gradient(135deg, 
    rgba(102, 126, 234, 0.1) 0%, 
    rgba(118, 75, 162, 0.1) 100%);
  opacity: 0;
  transition: opacity 0.3s ease;
}

.role-tab:hover {
  transform: translateY(-2px) scale(1.02);
  box-shadow: 
    0 8px 25px rgba(102, 126, 234, 0.15),
    0 3px 10px rgba(0, 0, 0, 0.1);
  color: #667eea;
  border-color: rgba(102, 126, 234, 0.2);
}

.role-tab:hover::before {
  opacity: 1;
}

.role-tab.active {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  box-shadow: 
    0 12px 30px rgba(102, 126, 234, 0.4),
    0 4px 15px rgba(0, 0, 0, 0.1),
    inset 0 1px 0 rgba(255, 255, 255, 0.2);
  transform: translateY(-2px) scale(1.02);
  border-color: transparent;
}

.role-tab.active::before {
  opacity: 0;
}

.role-tab.active::after {
  content: '';
  position: absolute;
  inset: 0;
  border-radius: 16px;
  background: linear-gradient(135deg, 
    rgba(255, 255, 255, 0.2) 0%, 
    rgba(255, 255, 255, 0.1) 50%,
    transparent 100%);
  pointer-events: none;
}

.role-tab .anticon {
  font-size: 18px;
  transition: transform 0.3s ease;
}

.role-tab:hover .anticon,
.role-tab.active .anticon {
  transform: scale(1.1);
}

/* 表单样式 - 现代化设计 */
.login-form,
.register-form {
  margin-top: 12px;
}

.login-form :deep(.ant-form-item),
.register-form :deep(.ant-form-item) {
  margin-bottom: 28px;
}

.login-form :deep(.ant-form-item-label),
.register-form :deep(.ant-form-item-label) {
  padding-bottom: 12px;
}

.login-form :deep(.ant-form-item-label > label),
.register-form :deep(.ant-form-item-label > label) {
  font-weight: 600;
  color: #374151;
  font-size: 15px;
  height: auto;
  position: relative;
}

.login-form :deep(.ant-form-item-label > label::after),
.register-form :deep(.ant-form-item-label > label::after) {
  content: '';
  position: absolute;
  bottom: -4px;
  left: 0;
  width: 0;
  height: 2px;
  background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
  transition: width 0.3s ease;
}

.login-form :deep(.ant-form-item:focus-within .ant-form-item-label > label::after),
.register-form :deep(.ant-form-item:focus-within .ant-form-item-label > label::after) {
  width: 30px;
}

.login-form :deep(.ant-input),
.register-form :deep(.ant-input) {
  height: 56px;
  border-radius: 18px;
  border: 2px solid rgba(229, 231, 235, 0.8);
  font-size: 16px;
  padding: 0 20px 0 52px;
  transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
  background: rgba(255, 255, 255, 0.9);
  backdrop-filter: blur(10px);
  box-shadow: 
    0 2px 8px rgba(0, 0, 0, 0.04),
    inset 0 1px 0 rgba(255, 255, 255, 0.5);
  position: relative;
}

.login-form :deep(.ant-input:hover),
.register-form :deep(.ant-input:hover) {
  border-color: rgba(102, 126, 234, 0.6);
  box-shadow: 
    0 4px 20px rgba(102, 126, 234, 0.12),
    inset 0 1px 0 rgba(255, 255, 255, 0.5);
  transform: translateY(-1px);
}

.login-form :deep(.ant-input:focus),
.register-form :deep(.ant-input:focus) {
  border-color: #667eea;
  box-shadow: 
    0 0 0 4px rgba(102, 126, 234, 0.1),
    0 8px 25px rgba(102, 126, 234, 0.15),
    inset 0 1px 0 rgba(255, 255, 255, 0.5);
  transform: translateY(-2px);
}

.login-form :deep(.ant-input-password),
.register-form :deep(.ant-input-password) {
  height: 56px;
  border-radius: 18px;
}

.login-form :deep(.ant-input-password .ant-input),
.register-form :deep(.ant-input-password .ant-input) {
  border: none;
  box-shadow: none;
  padding-left: 52px;
  background: transparent;
}

.login-form :deep(.ant-input-prefix),
.register-form :deep(.ant-input-prefix) {
  margin-right: 16px;
  color: #9ca3af;
  font-size: 18px;
  transition: all 0.3s ease;
}

.login-form :deep(.ant-form-item:focus-within .ant-input-prefix),
.register-form :deep(.ant-form-item:focus-within .ant-input-prefix) {
  color: #667eea;
  transform: scale(1.1);
}

.form-options {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin: 20px 0;
}

.login-form :deep(.ant-checkbox-wrapper),
.register-form :deep(.ant-checkbox-wrapper) {
  font-size: 14px;
  color: #6b7280;
  font-weight: 500;
  transition: all 0.3s ease;
}

.login-form :deep(.ant-checkbox-wrapper:hover),
.register-form :deep(.ant-checkbox-wrapper:hover) {
  color: #374151;
}

.login-form :deep(.ant-checkbox),
.register-form :deep(.ant-checkbox) {
  margin-right: 10px;
}

.login-form :deep(.ant-checkbox .ant-checkbox-inner),
.register-form :deep(.ant-checkbox .ant-checkbox-inner) {
  width: 18px;
  height: 18px;
  border-radius: 6px;
  border: 2px solid #d1d5db;
  transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}

.login-form :deep(.ant-checkbox:hover .ant-checkbox-inner),
.register-form :deep(.ant-checkbox:hover .ant-checkbox-inner) {
  border-color: #667eea;
  box-shadow: 0 0 0 2px rgba(102, 126, 234, 0.1);
}

.login-form :deep(.ant-checkbox-wrapper .ant-checkbox-checked .ant-checkbox-inner),
.register-form :deep(.ant-checkbox-wrapper .ant-checkbox-checked .ant-checkbox-inner) {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  border-color: #667eea;
  box-shadow: 0 2px 8px rgba(102, 126, 234, 0.3);
}

.login-form :deep(.ant-checkbox-checked .ant-checkbox-inner::after),
.register-form :deep(.ant-checkbox-checked .ant-checkbox-inner::after) {
  border-color: white;
  border-width: 2px;
}

.forgot-link {
  color: #667eea;
  text-decoration: none;
  font-size: 14px;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.3s ease;
  position: relative;
  padding: 4px 0;
}

.forgot-link::after {
  content: '';
  position: absolute;
  bottom: 0;
  left: 0;
  width: 0;
  height: 2px;
  background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
  transition: width 0.3s ease;
}

.forgot-link:hover {
  color: #5a67d8;
}

.forgot-link:hover::after {
  width: 100%;
}

/* 按钮样式 - 现代化设计 */
.login-btn,
.register-btn {
  width: 100%;
  height: 56px;
  border-radius: 18px;
  font-size: 16px;
  font-weight: 600;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  border: none;
  color: white;
  transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
  box-shadow: 
    0 6px 20px rgba(102, 126, 234, 0.25),
    inset 0 1px 0 rgba(255, 255, 255, 0.2);
  margin-top: 12px;
  position: relative;
  overflow: hidden;
}

.login-btn::before,
.register-btn::before {
  content: '';
  position: absolute;
  top: 0;
  left: -100%;
  width: 100%;
  height: 100%;
  background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent);
  transition: left 0.6s ease;
}

.login-btn:hover::before,
.register-btn:hover::before {
  left: 100%;
}

.login-btn:hover,
.register-btn:hover {
  transform: translateY(-3px);
  box-shadow: 
    0 12px 35px rgba(102, 126, 234, 0.35),
    inset 0 1px 0 rgba(255, 255, 255, 0.2);
  background: linear-gradient(135deg, #5a67d8 0%, #6b46c1 100%);
}

.login-btn:active,
.register-btn:active {
  transform: translateY(-1px);
  box-shadow: 
    0 6px 20px rgba(102, 126, 234, 0.25),
    inset 0 1px 0 rgba(255, 255, 255, 0.2);
  transition: all 0.1s ease;
}

/* 底部链接 - 现代化设计 */
.form-footer {
  text-align: center;
  margin-top: 32px;
  padding-top: 28px;
  border-top: 1px solid rgba(229, 231, 235, 0.8);
  color: #6b7280;
  font-size: 15px;
  position: relative;
}

.form-footer::before {
  content: '';
  position: absolute;
  top: 0;
  left: 50%;
  transform: translateX(-50%);
  width: 60px;
  height: 1px;
  background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
}

.form-footer p {
  margin: 0;
  color: #6b7280;
  font-size: 15px;
  font-weight: 400;
}

.register-link,
.login-link {
  color: #667eea;
  text-decoration: none;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.3s ease;
  margin-left: 8px;
  position: relative;
  padding: 2px 4px;
  border-radius: 4px;
}

.register-link::before,
.login-link::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
  border-radius: 4px;
  opacity: 0;
  transition: opacity 0.3s ease;
}

.register-link:hover,
.login-link:hover {
  color: #5a67d8;
  transform: translateY(-1px);
}

.register-link:hover::before,
.login-link:hover::before {
  opacity: 1;
}

/* 响应式设计 */
@media (max-width: 768px) {
  .header-container {
    padding: 0 16px;
  }
  
  .nav-menu {
    display: none;
  }
  
  .hero-container {
    grid-template-columns: 1fr;
    text-align: center;
    gap: 40px;
  }
  
  .hero-title {
    font-size: 36px;
  }
  
  .features-grid {
    grid-template-columns: 1fr;
  }
  
  .courses-grid {
    grid-template-columns: 1fr;
  }
  
  .footer-content {
    grid-template-columns: 1fr;
    gap: 32px;
  }
  
  .modal-content {
    padding: 24px;
  }
  
  :deep(.login-modal),
  :deep(.register-modal) {
    width: 90% !important;
    max-width: 400px;
  }
}
</style>