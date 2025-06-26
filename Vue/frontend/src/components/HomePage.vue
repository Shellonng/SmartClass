<script setup lang="ts">
import { ref, onMounted, computed } from 'vue'
import { useRouter } from 'vue-router'
import { useAuthStore } from '@/stores/auth'
import { 
  UserOutlined, 
  DownOutlined,
  PlayCircleOutlined,
  VideoCameraOutlined,
  BookOutlined,
  TrophyOutlined,
  TeamOutlined,
  StarFilled,
  ArrowRightOutlined,
  ClockCircleOutlined,
  FireOutlined
} from '@ant-design/icons-vue'

const router = useRouter()
const authStore = useAuthStore()

// 模拟课程数据
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

const categories = ref([
  { name: '计算机', count: 156, icon: '💻' },
  { name: '数学', count: 89, icon: '📊' },
  { name: '物理', count: 67, icon: '⚛️' },
  { name: '化学', count: 45, icon: '🧪' },
  { name: '语言', count: 123, icon: '🌍' },
  { name: '经济', count: 78, icon: '💰' }
])

const stats = ref([
  { label: '优质课程', value: '5000+', icon: '📚' },
  { label: '注册学员', value: '300万+', icon: '👥' },
  { label: '合作高校', value: '200+', icon: '🏫' },
  { label: '认证证书', value: '50万+', icon: '🏆' }
])

const goToCourses = () => {
  router.push('/courses')
}

const goToLogin = () => {
  router.push('/login')
}

const goToRegister = () => {
  router.push('/register')
}

const goToDashboard = () => {
  if (authStore.user?.role === 'teacher') {
    router.push('/teacher/dashboard')
  } else {
    router.push('/student/dashboard')
  }
}

const currentDate = computed(() => {
  return new Date().toLocaleDateString('zh-CN', {
    year: 'numeric',
    month: 'long',
    day: 'numeric',
    weekday: 'long'
  })
})
</script>

<template>
  <div class="home-page">
    <!-- 导航栏 -->
    <header class="main-header">
      <div class="header-container">
        <div class="header-left">
          <div class="logo">
            <img src="/logo.svg" alt="智慧课堂" class="logo-img" />
            <span class="logo-text">智慧课堂</span>
          </div>
          <nav class="main-nav">
            <a-menu mode="horizontal" class="nav-menu">
              <a-menu-item key="home">首页</a-menu-item>
              <a-menu-item key="courses" @click="goToCourses">课程</a-menu-item>
              <a-menu-item key="about">关于我们</a-menu-item>
            </a-menu>
          </nav>
        </div>
        <div class="header-right">
          <div v-if="authStore.isAuthenticated" class="user-info">
            <a-dropdown>
              <a-button type="text" class="user-btn">
                <UserOutlined />
                {{ authStore.user?.realName || '用户' }}
                <DownOutlined />
              </a-button>
              <template #overlay>
                <a-menu>
                  <a-menu-item @click="goToDashboard">个人中心</a-menu-item>
                  <a-menu-divider />
                  <a-menu-item @click="authStore.logout">退出登录</a-menu-item>
                </a-menu>
              </template>
            </a-dropdown>
          </div>
          <div v-else class="auth-buttons">
            <a-button @click="goToLogin">登录</a-button>
            <a-button type="primary" @click="goToRegister">注册</a-button>
          </div>
        </div>
      </div>
    </header>

    <!-- Hero Banner -->
    <section class="hero-banner">
      <div class="banner-container">
        <div class="banner-content">
          <div class="banner-text">
            <h1 class="banner-title">
              <span class="title-highlight">智慧课堂</span>
              <br />让学习更高效
            </h1>
            <p class="banner-subtitle">
              汇聚全球优质教育资源，提供个性化学习体验<br />
              助力每一位学习者实现知识梦想
            </p>
            <div class="banner-actions">
              <a-button type="primary" size="large" class="cta-button" @click="goToCourses">
                <PlayCircleOutlined />
                开始学习
              </a-button>
              <a-button size="large" class="secondary-button">
                <VideoCameraOutlined />
                观看介绍
              </a-button>
            </div>
            <div class="banner-stats">
              <div class="stat-item">
                <span class="stat-number">300万+</span>
                <span class="stat-label">学习者</span>
              </div>
              <div class="stat-item">
                <span class="stat-number">5000+</span>
                <span class="stat-label">优质课程</span>
              </div>
              <div class="stat-item">
                <span class="stat-number">200+</span>
                <span class="stat-label">合作高校</span>
              </div>
            </div>
          </div>
          <div class="banner-visual">
            <div class="visual-container">
              <div class="hero-image placeholder-image">
              <div class="placeholder-content">
                <BookOutlined style="font-size: 64px; color: #1890ff;" />
                <p>在线学习平台</p>
              </div>
            </div>
              <div class="floating-cards">
                <div class="floating-card card-1">
                  <BookOutlined />
                  <span>AI智能推荐</span>
                </div>
                <div class="floating-card card-2">
                  <TrophyOutlined />
                  <span>学习成就</span>
                </div>
                <div class="floating-card card-3">
                  <TeamOutlined />
                  <span>互动学习</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>

    <!-- 课程分类导航 -->
    <section class="category-nav">
      <div class="container">
        <div class="category-tabs">
          <div class="tab-item active">
            <span class="tab-icon">🔥</span>
            <span class="tab-text">热门推荐</span>
          </div>
          <div v-for="category in categories.slice(0, 5)" :key="category.name" class="tab-item">
            <span class="tab-icon">{{ category.icon }}</span>
            <span class="tab-text">{{ category.name }}</span>
          </div>
        </div>
      </div>
    </section>

    <!-- 精品课程推荐 -->
    <section class="featured-courses">
      <div class="container">
        <div class="section-header">
          <h2 class="section-title">精品课程推荐</h2>
          <p class="section-subtitle">来自知名高校的优质课程，助力您的学习之路</p>
        </div>
        
        <div class="courses-container">
          <div class="course-card" v-for="course in featuredCourses" :key="course.id">
            <div class="course-cover">
              <div v-if="!course.image" class="course-image placeholder-course-image">
                <div class="placeholder-content">
                  <BookOutlined style="font-size: 32px; color: #1890ff;" />
                  <span>{{ course.category }}</span>
                </div>
              </div>
              <img v-else :src="course.image" :alt="course.title" class="course-image" />
              <div class="course-overlay">
                <a-button type="primary" class="preview-btn">
                  <PlayCircleOutlined />
                  预览课程
                </a-button>
              </div>
              <div class="course-badge">{{ course.category }}</div>
            </div>
            
            <div class="course-info">
              <h3 class="course-title">{{ course.title }}</h3>
              <div class="course-meta">
                <div class="instructor-info">
                  <a-avatar size="small">{{ course.instructor.charAt(0) }}</a-avatar>
                  <span class="instructor-name">{{ course.instructor }}</span>
                  <span class="university">{{ course.university }}</span>
                </div>
              </div>
              
              <div class="course-stats">
                <div class="stat-item">
                  <UserOutlined />
                  <span>{{ course.students.toLocaleString() }}人学习</span>
                </div>
                <div class="stat-item">
                  <StarFilled />
                  <span>4.8</span>
                </div>
              </div>
              
              <div class="course-actions">
                <a-button type="primary" block>
                  立即学习
                </a-button>
              </div>
            </div>
          </div>
        </div>
        
        <div class="section-footer">
          <a-button size="large" @click="goToCourses">
            查看全部课程
            <ArrowRightOutlined />
          </a-button>
        </div>
      </div>
    </section>

    <!-- Categories Section -->
    <section class="categories-section">
      <div class="container">
        <div class="section-header">
          <h2>课程分类</h2>
          <p>涵盖多个学科领域，满足不同学习需求</p>
        </div>
        <div class="categories-grid">
          <div v-for="category in categories" :key="category.name" class="category-card">
            <div class="category-icon">{{ category.icon }}</div>
            <h3 class="category-name">{{ category.name }}</h3>
            <p class="category-count">{{ category.count }}门课程</p>
          </div>
        </div>
      </div>
    </section>

    <!-- Features Section -->
    <section class="features-section">
      <div class="container">
        <div class="section-header">
          <h2>为什么选择智慧课堂</h2>
        </div>
        <div class="features-grid">
          <div class="feature-item">
            <div class="feature-icon">🎓</div>
            <h3>优质课程资源</h3>
            <p>汇聚国内外知名高校优质课程，提供系统化学习体验</p>
          </div>
          <div class="feature-item">
            <div class="feature-icon">📱</div>
            <h3>随时随地学习</h3>
            <p>支持多终端学习，让您随时随地享受学习的乐趣</p>
          </div>
          <div class="feature-item">
            <div class="feature-icon">🏆</div>
            <h3>权威认证证书</h3>
            <p>完成课程学习可获得权威认证证书，提升个人竞争力</p>
          </div>
          <div class="feature-item">
            <div class="feature-icon">👨‍🏫</div>
            <h3>名师在线指导</h3>
            <p>知名教授在线答疑，提供专业的学习指导和建议</p>
          </div>
        </div>
      </div>
    </section>
  </div>
</template>

<style scoped>
.home-page {
  min-height: 100vh;
  background: #ffffff;
}

/* 导航栏样式 */
.main-header {
  background: #ffffff;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);
  position: sticky;
  top: 0;
  z-index: 1000;
}

.header-container {
  max-width: 1200px;
  margin: 0 auto;
  padding: 0 24px;
  display: flex;
  justify-content: space-between;
  align-items: center;
  height: 64px;
}

.header-left {
  display: flex;
  align-items: center;
  gap: 40px;
}

.logo {
  display: flex;
  align-items: center;
  gap: 12px;
}

.logo-img {
  width: 32px;
  height: 32px;
}

.logo-text {
  font-size: 20px;
  font-weight: 600;
  color: #1890ff;
}

.nav-menu {
  border-bottom: none;
  background: transparent;
}

.nav-menu .ant-menu-item {
  font-weight: 500;
  color: #666;
  border-bottom: 2px solid transparent;
}

.nav-menu .ant-menu-item:hover,
.nav-menu .ant-menu-item-selected {
  color: #1890ff;
  border-bottom-color: #1890ff;
}

.header-right {
  display: flex;
  align-items: center;
  gap: 16px;
}

.user-btn {
  display: flex;
  align-items: center;
  gap: 8px;
  font-weight: 500;
}

.auth-buttons {
  display: flex;
  gap: 12px;
}

/* Hero Banner */
.hero-banner {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  padding: 80px 0 120px;
  position: relative;
  overflow: hidden;
}

.hero-banner::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><defs><pattern id="grain" width="100" height="100" patternUnits="userSpaceOnUse"><circle cx="50" cy="50" r="1" fill="%23ffffff" opacity="0.1"/></pattern></defs><rect width="100" height="100" fill="url(%23grain)"/></svg>') repeat;
  pointer-events: none;
}

.banner-container {
  max-width: 1200px;
  margin: 0 auto;
  padding: 0 24px;
  position: relative;
  z-index: 1;
}

.banner-content {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 80px;
  align-items: center;
}

.banner-text {
  animation: slideInLeft 0.8s ease-out;
}

.banner-title {
  font-size: 3.5rem;
  font-weight: 700;
  line-height: 1.2;
  margin-bottom: 24px;
}

.title-highlight {
  background: linear-gradient(45deg, #ffffff, #e3f2fd);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}

.banner-subtitle {
  font-size: 1.25rem;
  line-height: 1.6;
  margin-bottom: 40px;
  opacity: 0.9;
}

.banner-actions {
  display: flex;
  gap: 16px;
  margin-bottom: 48px;
}

.cta-button {
  height: 48px;
  padding: 0 32px;
  font-size: 16px;
  border-radius: 24px;
  background: linear-gradient(45deg, #1890ff, #722ed1);
  border: none;
  box-shadow: 0 4px 15px rgba(24, 144, 255, 0.4);
  transition: all 0.3s ease;
}

.cta-button:hover {
  transform: translateY(-2px);
  box-shadow: 0 6px 20px rgba(24, 144, 255, 0.6);
}

/* 占位符图片样式 */
.placeholder-image {
  width: 100%;
  height: 400px;
  background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
  border: 2px dashed #1890ff;
  border-radius: 12px;
  display: flex;
  align-items: center;
  justify-content: center;
}

.placeholder-content {
  text-align: center;
  color: #1890ff;
}

.placeholder-content p {
  margin-top: 16px;
  font-size: 18px;
  font-weight: 500;
}

.placeholder-course-image {
  height: 200px;
  background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
  border: 1px solid #e2e8f0;
  border-radius: 8px;
  display: flex;
  align-items: center;
  justify-content: center;
  position: relative;
}

.placeholder-course-image .placeholder-content {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 8px;
}

.placeholder-course-image .placeholder-content span {
  font-size: 14px;
  font-weight: 500;
  color: #64748b;
}

.secondary-button {
  height: 48px;
  padding: 0 32px;
  font-size: 16px;
  border-radius: 24px;
  background: rgba(255, 255, 255, 0.1);
  color: white;
  border: 1px solid rgba(255, 255, 255, 0.2);
  backdrop-filter: blur(10px);
  transition: all 0.3s ease;
}

.secondary-button:hover {
  background: rgba(255, 255, 255, 0.2);
  transform: translateY(-2px);
}

.banner-stats {
  display: flex;
  gap: 32px;
}

.banner-stats .stat-item {
  text-align: center;
}

.stat-number {
  display: block;
  font-size: 1.5rem;
  font-weight: 700;
  margin-bottom: 4px;
}

.stat-label {
  font-size: 0.875rem;
  opacity: 0.8;
}

.banner-visual {
  position: relative;
  animation: slideInRight 0.8s ease-out;
}

.visual-container {
  position: relative;
}

.hero-image {
  width: 100%;
  height: auto;
  border-radius: 20px;
  box-shadow: 0 20px 40px rgba(0, 0, 0, 0.2);
  transition: transform 0.3s ease;
}

.hero-image:hover {
  transform: scale(1.02);
}

.floating-cards {
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  pointer-events: none;
}

.floating-card {
  position: absolute;
  background: rgba(255, 255, 255, 0.95);
  backdrop-filter: blur(10px);
  border-radius: 12px;
  padding: 12px 16px;
  display: flex;
  align-items: center;
  gap: 8px;
  color: #333;
  font-weight: 500;
  box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1);
  animation: float 3s ease-in-out infinite;
}

.card-1 {
  top: 20%;
  right: -10%;
  animation-delay: 0s;
}

.card-2 {
  bottom: 30%;
  left: -10%;
  animation-delay: 1s;
}

.card-3 {
  top: 60%;
  right: 10%;
  animation-delay: 2s;
}

/* 课程分类导航 */
.category-nav {
  background: #ffffff;
  padding: 24px 0;
  border-bottom: 1px solid #f0f0f0;
}

.category-tabs {
  display: flex;
  gap: 32px;
  justify-content: center;
  flex-wrap: wrap;
}

.tab-item {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 8px;
  padding: 16px 24px;
  border-radius: 12px;
  cursor: pointer;
  transition: all 0.3s ease;
  min-width: 100px;
}

.tab-item:hover,
.tab-item.active {
  background: #f0f7ff;
  color: #1890ff;
}

.tab-icon {
  font-size: 24px;
}

.tab-text {
  font-size: 14px;
  font-weight: 500;
}

/* 精品课程推荐 */
.featured-courses {
  padding: 80px 0;
  background: #fafafa;
}

.courses-container {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
  gap: 24px;
  margin-bottom: 48px;
}

.course-card {
  background: white;
  border-radius: 16px;
  overflow: hidden;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
  transition: all 0.3s ease;
  cursor: pointer;
}

.course-card:hover {
  transform: translateY(-4px);
  box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
}

.course-cover {
  position: relative;
  height: 200px;
  overflow: hidden;
}

.course-image {
  width: 100%;
  height: 100%;
  object-fit: cover;
  transition: transform 0.3s ease;
}

.course-card:hover .course-image {
  transform: scale(1.05);
}

.course-overlay {
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: rgba(0, 0, 0, 0.5);
  display: flex;
  align-items: center;
  justify-content: center;
  opacity: 0;
  transition: opacity 0.3s ease;
}

.course-card:hover .course-overlay {
  opacity: 1;
}

.preview-btn {
  border-radius: 20px;
  border: none;
  box-shadow: 0 4px 12px rgba(24, 144, 255, 0.3);
}

.course-badge {
  position: absolute;
  top: 12px;
  left: 12px;
  background: #1890ff;
  color: white;
  padding: 4px 12px;
  border-radius: 12px;
  font-size: 12px;
  font-weight: 500;
}

.course-info {
  padding: 20px;
}

.course-title {
  font-size: 18px;
  font-weight: 600;
  margin-bottom: 12px;
  color: #333;
  line-height: 1.4;
}

.course-meta {
  margin-bottom: 16px;
}

.instructor-info {
  display: flex;
  align-items: center;
  gap: 8px;
}

.instructor-name {
  font-weight: 500;
  color: #333;
}

.university {
  color: #666;
  font-size: 14px;
}

.course-stats {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 16px;
  color: #666;
  font-size: 14px;
}

.course-stats .stat-item {
  display: flex;
  align-items: center;
  gap: 4px;
}.course-actions .ant-btn {
   height: 40px;
   border-radius: 8px;
   font-weight: 500;
 }

/* 通用样式 */
.container {
  max-width: 1200px;
  margin: 0 auto;
  padding: 0 24px;
}

.section-header {
  text-align: center;
  margin-bottom: 48px;
}

.section-title {
  font-size: 2.5rem;
  font-weight: 700;
  color: #333;
  margin-bottom: 16px;
}

.section-subtitle {
  font-size: 1.125rem;
  color: #666;
  line-height: 1.6;
}

.section-footer {
  text-align: center;
  margin-top: 48px;
}

/* 动画效果 */
@keyframes slideInLeft {
  from {
    opacity: 0;
    transform: translateX(-50px);
  }
  to {
    opacity: 1;
    transform: translateX(0);
  }
}

@keyframes slideInRight {
  from {
    opacity: 0;
    transform: translateX(50px);
  }
  to {
    opacity: 1;
    transform: translateX(0);
  }
}

@keyframes float {
  0%, 100% {
    transform: translateY(0px);
  }
  50% {
    transform: translateY(-10px);
  }
}

@keyframes fadeInUp {
  from {
    opacity: 0;
    transform: translateY(30px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

/* 响应式设计 */
@media (max-width: 1024px) {
  .banner-content {
    grid-template-columns: 1fr;
    gap: 40px;
    text-align: center;
  }
  
  .banner-title {
    font-size: 2.5rem;
  }
  
  .courses-container {
    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
  }
}

@media (max-width: 768px) {
  .header-container {
    padding: 0 16px;
  }
  
  .header-left {
    gap: 20px;
  }
  
  .main-nav {
    display: none;
  }
  
  .banner-title {
    font-size: 2rem;
  }
  
  .banner-actions {
    flex-direction: column;
    align-items: center;
  }
  
  .banner-stats {
    justify-content: center;
    gap: 20px;
  }
  
  .category-tabs {
    gap: 16px;
  }
  
  .tab-item {
    padding: 12px 16px;
    min-width: 80px;
  }
  
  .courses-container {
    grid-template-columns: 1fr;
  }
  
  .floating-cards {
    display: none;
  }
}

@media (max-width: 480px) {
  .container {
    padding: 0 16px;
  }
  
  .banner-title {
    font-size: 1.75rem;
  }
  
  .section-title {
    font-size: 2rem;
  }
  
  .course-info {
    padding: 16px;
  }
}

/* Stats Section */
.stats-section {
  padding: 60px 20px;
  background: #f8f9fa;
}

.container {
  max-width: 1200px;
  margin: 0 auto;
}

.stats-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
  gap: 40px;
}

.stat-item {
  text-align: center;
  padding: 30px;
  background: white;
  border-radius: 12px;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
}

.stat-icon {
  font-size: 3rem;
  margin-bottom: 15px;
}

.stat-value {
  font-size: 2.5rem;
  font-weight: bold;
  color: #1890ff;
  margin-bottom: 10px;
}

.stat-label {
  font-size: 1.1rem;
  color: #666;
}

/* Courses Section */
.courses-section {
  padding: 80px 20px;
}

.section-header {
  text-align: center;
  margin-bottom: 60px;
}

.section-header h2 {
  font-size: 2.5rem;
  margin-bottom: 15px;
  color: #333;
}

.section-header p {
  font-size: 1.1rem;
  color: #666;
}

.courses-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
  gap: 30px;
  margin-bottom: 50px;
}

.course-card {
  background: white;
  border-radius: 12px;
  overflow: hidden;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
  transition: transform 0.3s ease, box-shadow 0.3s ease;
  cursor: pointer;
}

.course-card:hover {
  transform: translateY(-5px);
  box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
}

.course-image {
  position: relative;
  height: 200px;
  overflow: hidden;
}

.course-image img {
  width: 100%;
  height: 100%;
  object-fit: cover;
}

.course-category {
  position: absolute;
  top: 15px;
  left: 15px;
  background: #1890ff;
  color: white;
  padding: 5px 12px;
  border-radius: 20px;
  font-size: 0.9rem;
}

.course-content {
  padding: 20px;
}

.course-title {
  font-size: 1.3rem;
  font-weight: bold;
  margin-bottom: 10px;
  color: #333;
}

.course-instructor {
  color: #666;
  margin-bottom: 15px;
}

.course-stats {
  display: flex;
  align-items: center;
  color: #999;
  font-size: 0.9rem;
}

.section-footer {
  text-align: center;
}

/* Categories Section */
.categories-section {
  padding: 80px 20px;
  background: #f8f9fa;
}

.categories-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 30px;
}

.category-card {
  background: white;
  padding: 40px 20px;
  border-radius: 12px;
  text-align: center;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
  transition: transform 0.3s ease;
  cursor: pointer;
}

.category-card:hover {
  transform: translateY(-3px);
}

.category-icon {
  font-size: 3rem;
  margin-bottom: 20px;
}

.category-name {
  font-size: 1.3rem;
  font-weight: bold;
  margin-bottom: 10px;
  color: #333;
}

.category-count {
  color: #666;
}

/* Features Section */
.features-section {
  padding: 80px 20px;
}

.features-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
  gap: 40px;
}

.feature-item {
  text-align: center;
  padding: 40px 20px;
}

.feature-icon {
  font-size: 4rem;
  margin-bottom: 20px;
}

.feature-item h3 {
  font-size: 1.5rem;
  margin-bottom: 15px;
  color: #333;
}

.feature-item p {
  color: #666;
  line-height: 1.6;
}

/* Responsive Design */
@media (max-width: 768px) {
  .hero-content {
    grid-template-columns: 1fr;
    text-align: center;
  }
  
  .hero-title {
    font-size: 2.5rem;
  }
  
  .hero-buttons {
    justify-content: center;
  }
  
  .stats-grid,
  .courses-grid,
  .categories-grid,
  .features-grid {
    grid-template-columns: 1fr;
  }
}
</style>