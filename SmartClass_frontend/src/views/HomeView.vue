<script setup lang="ts">
import { ref } from 'vue'
import { useRouter } from 'vue-router'

const router = useRouter()

// 功能卡片数据
const features = ref([
  { id: 1, title: '课程管理', icon: '📚', desc: '创建与管理课程内容' },
  { id: 2, title: '在线考试', icon: '📝', desc: '创建试卷与自动评分' },
  { id: 3, title: '学生管理', icon: '👨‍🎓', desc: '管理学生信息与分组' },
  { id: 4, title: '实时互动', icon: '💬', desc: '课堂问答与讨论区' },
  { id: 5, title: '资源中心', icon: '📁', desc: '教学资料共享库' },
  { id: 6, title: '数据统计', icon: '📊', desc: '学习数据分析报表' },
  { id: 7, title: 'AI讲师', icon: '🤖', desc: 'AI大模型课程辅导' }
])

// 最新通知数据
const notifications = ref([
  { id: 1, text: '系统将于本周五凌晨进行维护升级', time: '2小时前' },
  { id: 2, text: '新增AI作业批改功能已上线', time: '1天前' },
  { id: 3, text: '暑期教师培训计划已发布', time: '3天前' }
])

const goto = (page:string) => {
  router.push(page)
}

</script>

<template>
  <div class="home-container">
    <!-- 顶部导航栏 -->
    <header class="header">
      <div class="logo">智慧课堂系统</div>
      <nav class="nav-menu">
        <a href="#" class="active">首页</a>
        <a href="#">课程中心</a>
        <a href="#">考试系统</a>
        <a href="#">成绩分析</a>
        <a href="#">资源库</a>
      </nav>
      <div class="user-actions">
        <button class="notification-btn">🔔</button>
        <div class="user-info">
          <span class="avatar">👤</span>
          <a @click="goto('/personal')">张老师</a>
        </div>
      </div>
    </header>

    <!-- 主内容区 -->
    <main class="main-content">
      <div class="welcome-section">
        <h1>欢迎使用智慧课堂系统</h1>
        <p class="subtitle">让教学更智能，让学习更高效</p>
        
        <div class="quick-actions">
          <button class="primary-btn">创建新课程</button>
          <button class="secondary-btn">导入学生名单</button>
          <button class="secondary-btn">查看教学日历</button>
        </div>
      </div>

      <!-- 功能卡片网格 -->
      <div class="features-grid">
        <div v-for="feature in features" :key="feature.id" class="feature-card">
          <div class="feature-icon">{{ feature.icon }}</div>
          <h3>{{ feature.title }}</h3>
          <p>{{ feature.desc }}</p>
        </div>
      </div>

      <!-- 通知和统计区域 -->
      <div class="dashboard-section">
        <div class="notifications">
          <h2>最新通知</h2>
          <ul>
            <li v-for="note in notifications" :key="note.id">
              <span class="note-text">{{ note.text }}</span>
              <span class="note-time">{{ note.time }}</span>
            </li>
          </ul>
        </div>
        
        <div class="stats">
          <h2>课堂数据</h2>
          <div class="stats-grid">
            <div class="stat-item">
              <div class="stat-value">24</div>
              <div class="stat-label">进行中课程</div>
            </div>
            <div class="stat-item">
              <div class="stat-value">183</div>
              <div class="stat-label">学生总数</div>
            </div>
            <div class="stat-item">
              <div class="stat-value">96%</div>
              <div class="stat-label">到课率</div>
            </div>
          </div>
        </div>
      </div>
    </main>

    <!-- 页脚 -->
    <footer class="footer">
      <p>智慧课堂系统 © 2025 版权所有 | 教育科技赋能未来</p>
      <div class="footer-links">
        <a href="#"@click="goto('/about')">关于我们</a>
        <a href="#"@click="goto('/about')">帮助中心</a>
        <a href="#">隐私政策</a>
        <a href="#">联系我们</a>
      </div>
    </footer>
  </div>
</template>

<style scoped>
.home-container {
  display: flex;
  flex-direction: column;
  min-height: 100vh;
  background-color: #f5f7fa;
  font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}

/* 顶部导航栏样式 */
.header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 0 2rem;
  height: 60px;
  background-color: #fff;
  box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
  position: sticky;
  top: 0;
  z-index: 100;
}

.logo {
  font-size: 1.2rem;
  font-weight: bold;
  color: #409EFF;
}

.nav-menu {
  display: flex;
  gap: 2rem;
}

.nav-menu a {
  text-decoration: none;
  color: #606266;
  padding: 0.5rem 0;
  transition: color 0.3s;
}

.nav-menu a.active,
.nav-menu a:hover {
  color: #409EFF;
  border-bottom: 2px solid #409EFF;
}

.user-actions {
  display: flex;
  align-items: center;
  gap: 1.5rem;
}

.notification-btn {
  background: none;
  border: none;
  font-size: 1.2rem;
  cursor: pointer;
  color: #606266;
}

.user-info {
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

.avatar {
  background-color: #e6f7ff;
  border-radius: 50%;
  width: 30px;
  height: 30px;
  display: flex;
  justify-content: center;
  align-items: center;
}

/* 主内容区样式 */
.main-content {
  flex: 1;
  padding: 2rem;
  max-width: 1200px;
  margin: 0 auto;
  width: 100%;
}

.welcome-section {
  text-align: center;
  margin-bottom: 3rem;
  padding: 2rem;
  background: linear-gradient(135deg, #ecf5ff 0%, #ffffff 100%);
  border-radius: 12px;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
}

.welcome-section h1 {
  font-size: 2.5rem;
  margin-bottom: 0.5rem;
  color: #303133;
}

.subtitle {
  font-size: 1.2rem;
  color: #606266;
  margin-bottom: 2rem;
}

.quick-actions {
  display: flex;
  justify-content: center;
  gap: 1rem;
  margin-top: 1.5rem;
}

.primary-btn, .secondary-btn {
  padding: 0.8rem 1.5rem;
  border-radius: 6px;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.3s;
  border: none;
}

.primary-btn {
  background-color: #409EFF;
  color: white;
}

.primary-btn:hover {
  background-color: #66b1ff;
  transform: translateY(-2px);
}

.secondary-btn {
  background-color: white;
  color: #409EFF;
  border: 1px solid #409EFF;
}

.secondary-btn:hover {
  background-color: #ecf5ff;
}

/* 功能卡片样式 */
.features-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
  gap: 1.5rem;
  margin-bottom: 3rem;
}

.feature-card {
  background: white;
  border-radius: 10px;
  padding: 1.5rem;
  text-align: center;
  box-shadow: 0 4px 8px rgba(0, 0, 0, 0.05);
  transition: transform 0.3s, box-shadow 0.3s;
}

.feature-card:hover {
  transform: translateY(-5px);
  box-shadow: 0 6px 16px rgba(0, 0, 0, 0.1);
}

.feature-icon {
  font-size: 2.5rem;
  margin-bottom: 1rem;
}

.feature-card h3 {
  margin: 0.5rem 0;
  color: #303133;
}

.feature-card p {
  color: #606266;
  font-size: 0.95rem;
  margin: 0;
}

/* 通知和统计区域 */
.dashboard-section {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 2rem;
  margin-top: 2rem;
}

.notifications, .stats {
  background: white;
  border-radius: 10px;
  padding: 1.5rem;
  box-shadow: 0 4px 8px rgba(0, 0, 0, 0.05);
}

.notifications h2, .stats h2 {
  margin-top: 0;
  color: #303133;
  padding-bottom: 0.8rem;
  border-bottom: 1px solid #ebeef5;
}

.notifications ul {
  list-style: none;
  padding: 0;
  margin: 0;
}

.notifications li {
  padding: 0.8rem 0;
  border-bottom: 1px solid #f5f7fa;
  display: flex;
  justify-content: space-between;
}

.note-text {
  color: #606266;
}

.note-time {
  color: #909399;
  font-size: 0.85rem;
}

.stats-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 1rem;
}

.stat-item {
  text-align: center;
  padding: 1rem;
  background-color: #f5f7fa;
  border-radius: 8px;
}

.stat-value {
  font-size: 2rem;
  font-weight: bold;
  color: #409EFF;
}

.stat-label {
  color: #909399;
  font-size: 0.9rem;
}

/* 页脚样式 */
.footer {
  text-align: center;
  padding: 1.5rem;
  background-color: white;
  color: #909399;
  font-size: 0.9rem;
  border-top: 1px solid #ebeef5;
}

.footer-links {
  margin-top: 0.5rem;
}

.footer-links a {
  color: #606266;
  text-decoration: none;
  margin: 0 0.8rem;
  transition: color 0.3s;
}

.footer-links a:hover {
  color: #409EFF;
}
</style>