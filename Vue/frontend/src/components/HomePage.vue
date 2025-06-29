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
    university: '中科院',
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
  { label: '注册学员', value: '300万', icon: '👥' },
  { label: '合作高校', value: '200+', icon: '🏫' },
  { label: '认证证书', value: '50万', icon: '🏆' }
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
  if (authStore.user?.role === 'TEACHER') {
    router.push('/teacher/dashboard')
  } else if (authStore.user?.role === 'STUDENT') {
    router.push('/student/dashboard') 
  } else {
    router.push('/')
  }
}

const handleLogout = async () => {
  await authStore.logout()
  router.push('/')
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
                  <a-menu-item @click="handleLogout">退出登录</a-menu-item>
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
                <span class="stat-number">300万</span>
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
  height: 32px;
  width: auto;
}

.logo-text {
  font-size: 1.5rem;
  font-weight: bold;
  color: #1890ff;
}

.main-nav .nav-menu {
  border: none;
  background: transparent;
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
}

.auth-buttons {
  display: flex;
  gap: 12px;
}

/* Hero Banner */
.hero-banner {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  padding: 80px 20px;
  min-height: 600px;
  display: flex;
  align-items: center;
}

.banner-container {
  max-width: 1200px;
  margin: 0 auto;
  width: 100%;
}

.banner-content {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 60px;
  align-items: center;
}

.banner-title {
  font-size: 3.5rem;
  font-weight: bold;
  margin-bottom: 24px;
  line-height: 1.2;
}

.title-highlight {
  color: #ffd700;
}

.banner-subtitle {
  font-size: 1.2rem;
  margin-bottom: 32px;
  line-height: 1.6;
  opacity: 0.9;
}

.banner-actions {
  display: flex;
  gap: 16px;
  margin-bottom: 40px;
}

.cta-button, .secondary-button {
  height: 48px;
  padding: 0 24px;
  font-size: 1.1rem;
  border-radius: 6px;
}

.secondary-button {
  background: rgba(255, 255, 255, 0.1);
  border: 1px solid rgba(255, 255, 255, 0.3);
  color: white;
}

.banner-stats {
  display: flex;
  gap: 40px;
}

.stat-item {
  text-align: center;
}

.stat-number {
  display: block;
  font-size: 2rem;
  font-weight: bold;
  color: #ffd700;
  margin-bottom: 8px;
}

.stat-label {
  font-size: 1rem;
  opacity: 0.8;
}

.banner-visual {
  position: relative;
}

.visual-container {
  position: relative;
  height: 400px;
}

.hero-image {
  width: 100%;
  height: 300px;
  background: rgba(255, 255, 255, 0.1);
  border-radius: 12px;
  display: flex;
  align-items: center;
  justify-content: center;
  backdrop-filter: blur(10px);
}

.placeholder-content {
  text-align: center;
  color: white;
}

.placeholder-content p {
  margin-top: 16px;
  font-size: 1.1rem;
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
  background: rgba(255, 255, 255, 0.9);
  padding: 12px 16px;
  border-radius: 8px;
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 0.9rem;
  color: #333;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
}

.card-1 {
  top: 20px;
  right: 20px;
}

.card-2 {
  bottom: 80px;
  left: 20px;
}

.card-3 {
  top: 50%;
  right: -20px;
}

/* 课程分类导航 */
.category-nav {
  background: #f8f9fa;
  padding: 40px 20px;
}

.container {
  max-width: 1200px;
  margin: 0 auto;
}

.category-tabs {
  display: flex;
  gap: 20px;
  justify-content: center;
  flex-wrap: wrap;
}

.tab-item {
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 20px;
  background: white;
  border-radius: 12px;
  cursor: pointer;
  transition: all 0.3s ease;
  min-width: 120px;
}

.tab-item:hover,
.tab-item.active {
  background: #1890ff;
  color: white;
  transform: translateY(-2px);
}

.tab-icon {
  font-size: 2rem;
  margin-bottom: 8px;
}

.tab-text {
  font-size: 0.9rem;
  font-weight: 500;
}

/* 精品课程推荐 */
.featured-courses {
  padding: 80px 20px;
}

.section-header {
  text-align: center;
  margin-bottom: 60px;
}

.section-title {
  font-size: 2.5rem;
  margin-bottom: 16px;
  color: #333;
}

.section-subtitle {
  font-size: 1.1rem;
  color: #666;
}

.courses-container {
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

.course-cover {
  position: relative;
  height: 200px;
  overflow: hidden;
}

.course-image,
.placeholder-course-image {
  width: 100%;
  height: 100%;
  object-fit: cover;
}

.placeholder-course-image {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  color: white;
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

.course-badge {
  position: absolute;
  top: 15px;
  left: 15px;
  background: #1890ff;
  color: white;
  padding: 5px 12px;
  border-radius: 20px;
  font-size: 0.9rem;
}

.course-info {
  padding: 20px;
}

.course-title {
  font-size: 1.3rem;
  font-weight: bold;
  margin-bottom: 12px;
  color: #333;
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
  font-size: 0.9rem;
}

.course-stats {
  display: flex;
  align-items: center;
  gap: 20px;
  margin-bottom: 16px;
  color: #999;
  font-size: 0.9rem;
}

.course-stats .stat-item {
  display: flex;
  align-items: center;
  gap: 4px;
}

.course-actions {
  margin-top: 16px;
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

/* 响应式设计 */
@media (max-width: 1024px) {
  .banner-content {
    grid-template-columns: 1fr;
    text-align: center;
    gap: 40px;
  }
  
  .banner-title {
    font-size: 2.5rem;
  }
}

@media (max-width: 768px) {
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
    gap: 10px;
  }
  
  .tab-item {
    min-width: 100px;
    padding: 15px;
  }
  
  .courses-container,
  .categories-grid,
  .features-grid {
    grid-template-columns: 1fr;
  }
}
</style>
