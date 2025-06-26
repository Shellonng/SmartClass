<template>
  <div class="student-dashboard">
    <!-- 页面头部 -->
    <div class="dashboard-header">
      <div class="welcome-section">
        <div class="user-greeting">
          <div class="avatar-container">
            <a-avatar :size="64" :src="userStore.user?.avatar || ''" class="user-avatar">
              <template #icon>
                <UserOutlined />
              </template>
            </a-avatar>
            <div class="online-indicator"></div>
          </div>
          
          <div class="greeting-content">
            <h1 class="greeting-title">
              {{ getGreeting() }}，{{ userStore.user?.realName || '同学' }}！
            </h1>
            <p class="greeting-subtitle">
              {{ formatDate(new Date()) }} · 今天也要加油学习哦
            </p>
          </div>
        </div>
        
        <div class="quick-actions">
          <a-button 
            type="primary" 
            size="large" 
            @click="$router.push('/student/courses')"
            class="action-btn primary"
          >
            <BookOutlined />
            我的课程
          </a-button>
          <a-button 
            size="large" 
            @click="$router.push('/student/assignments')"
            class="action-btn secondary"
          >
            <EditOutlined />
            作业中心
          </a-button>
          <a-button 
            size="large" 
            @click="openAIAssistant"
            class="action-btn ai"
          >
            <RobotOutlined />
            AI助手
          </a-button>
        </div>
      </div>
      
      <!-- 学习概览卡片 -->
      <div class="stats-overview">
        <div class="stat-card study-time">
          <div class="stat-icon">
            <ClockCircleOutlined />
          </div>
          <div class="stat-content">
            <div class="stat-value">{{ todayStudyTime }}h</div>
            <div class="stat-label">今日学习</div>
            <div class="stat-trend">+15% 较昨日</div>
          </div>
        </div>
        
        <div class="stat-card assignments">
          <div class="stat-icon">
            <CheckCircleOutlined />
          </div>
          <div class="stat-content">
            <div class="stat-value">{{ completedAssignments }}/{{ totalAssignments }}</div>
            <div class="stat-label">本周作业</div>
            <div class="stat-trend">完成率 {{ assignmentCompletionRate }}%</div>
          </div>
        </div>
        
        <div class="stat-card grade">
          <div class="stat-icon">
            <TrophyOutlined />
          </div>
          <div class="stat-content">
            <div class="stat-value">{{ averageGrade }}</div>
            <div class="stat-label">平均成绩</div>
            <div class="stat-trend">排名 {{ ranking }}</div>
          </div>
        </div>
        
        <div class="stat-card streak">
          <div class="stat-icon">
            <FireOutlined />
          </div>
          <div class="stat-content">
            <div class="stat-value">{{ studyStreak }}</div>
            <div class="stat-label">连续学习天数</div>
            <div class="stat-trend">再接再厉！</div>
          </div>
        </div>
      </div>
    </div>

    <!-- 主要内容区域 -->
    <div class="dashboard-content">
      <a-row :gutter="[24, 24]">
        <!-- 左侧主要内容 -->
        <a-col :lg="16" :md="24" :sm="24">
          <!-- 待完成作业 -->
          <div class="content-card assignments-card">
            <div class="card-header">
              <div class="header-left">
                <h3 class="card-title">
                  <ClockCircleOutlined class="title-icon" />
                  待完成作业
                  <a-badge :count="pendingAssignments.length" class="title-badge" />
                </h3>
                <p class="card-subtitle">抓紧时间完成作业，不要拖延哦</p>
              </div>
              <a-button type="link" @click="$router.push('/student/assignments')">
                查看全部 <ArrowRightOutlined />
              </a-button>
            </div>
            
            <div class="assignments-list">
              <div 
                v-for="assignment in pendingAssignments.slice(0, 3)" 
                :key="assignment.id"
                class="assignment-item"
                :class="{ urgent: assignment.isUrgent }"
                @click="openAssignment(assignment)"
              >
                <div class="assignment-left">
                  <div class="assignment-subject" :style="{ backgroundColor: assignment.subjectColor }">
                    {{ assignment.subject }}
                  </div>
                  <div class="assignment-content">
                    <h4 class="assignment-title">{{ assignment.title }}</h4>
                    <p class="assignment-desc">{{ assignment.description }}</p>
                  </div>
                </div>
                
                <div class="assignment-right">
                  <div class="assignment-deadline" :class="{ urgent: assignment.isUrgent }">
                    <ClockCircleOutlined />
                    <span>{{ formatDeadline(assignment.deadline) }}</span>
                  </div>
                  <a-button 
                    type="primary" 
                    size="small"
                    :class="{ 'urgent-btn': assignment.isUrgent }"
                  >
                    {{ assignment.isUrgent ? '立即完成' : '开始作业' }}
                  </a-button>
                </div>
              </div>
              
              <div v-if="pendingAssignments.length === 0" class="empty-state">
                <CheckCircleOutlined class="empty-icon" />
                <h4>太棒了！</h4>
                <p>暂时没有待完成的作业</p>
              </div>
            </div>
          </div>
          
          <!-- 学习进度 -->
          <div class="content-card progress-card">
            <div class="card-header">
              <div class="header-left">
                <h3 class="card-title">
                  <BarChartOutlined class="title-icon" />
                  学习进度
                </h3>
                <p class="card-subtitle">追踪你的学习轨迹</p>
              </div>
              <a-select v-model:value="progressTimeRange" size="small">
                <a-select-option value="week">本周</a-select-option>
                <a-select-option value="month">本月</a-select-option>
                <a-select-option value="semester">本学期</a-select-option>
              </a-select>
            </div>
            
            <div class="progress-content">
              <!-- 学习目标进度 -->
              <div class="progress-item">
                <div class="progress-header">
                  <span class="progress-label">学习目标完成情况</span>
                  <span class="progress-percentage">{{ weeklyProgress }}%</span>
                </div>
                <a-progress 
                  :percent="weeklyProgress" 
                  :stroke-color="getProgressColor(weeklyProgress)"
                  :show-info="false"
                  class="progress-bar"
                />
                <div class="progress-details">
                  <span>已完成 {{ completedHours }}h / 目标 {{ targetHours }}h</span>
                </div>
              </div>
              
              <!-- 各科目进度 -->
              <div class="subjects-progress">
                <div 
                  v-for="subject in subjectsProgress" 
                  :key="subject.name"
                  class="subject-progress"
                >
                  <div class="subject-info">
                    <div class="subject-icon" :style="{ backgroundColor: subject.color }">
                      {{ subject.name.charAt(0) }}
                    </div>
                    <div class="subject-details">
                      <span class="subject-name">{{ subject.name }}</span>
                      <span class="subject-progress-text">{{ subject.progress }}%</span>
                    </div>
                  </div>
                  <a-progress 
                    :percent="subject.progress" 
                    :stroke-color="subject.color"
                    :show-info="false"
                    size="small"
                  />
                </div>
              </div>
            </div>
          </div>
        </a-col>
        
        <!-- 右侧侧边栏 -->
        <a-col :lg="8" :md="24" :sm="24">
          <!-- 最近成绩 -->
          <div class="content-card grades-card">
            <div class="card-header">
              <div class="header-left">
                <h3 class="card-title">
                  <TrophyOutlined class="title-icon" />
                  最近成绩
                </h3>
              </div>
              <a-button type="link" size="small" @click="$router.push('/student/grades')">
                查看更多
              </a-button>
            </div>
            
            <div class="grades-list">
              <div 
                v-for="grade in recentGrades" 
                :key="grade.id"
                class="grade-item"
              >
                <div class="grade-left">
                  <div class="grade-subject" :style="{ backgroundColor: grade.subjectColor }">
                    {{ grade.subject.charAt(0) }}
                  </div>
                  <div class="grade-info">
                    <span class="grade-assignment">{{ grade.assignment }}</span>
                    <span class="grade-subject-name">{{ grade.subject }}</span>
                  </div>
                </div>
                <div class="grade-right">
                  <div class="grade-score" :class="getGradeClass(grade.score)">
                    {{ grade.score }}
                  </div>
                  <div class="grade-date">{{ formatRelativeTime(grade.date) }}</div>
                </div>
              </div>
            </div>
          </div>
          
          <!-- AI学习建议 -->
          <div class="content-card ai-card">
            <div class="card-header">
              <h3 class="card-title">
                <RobotOutlined class="title-icon ai-icon" />
                AI学习建议
              </h3>
            </div>
            
            <div class="ai-suggestions">
              <div class="ai-suggestion" v-for="suggestion in aiSuggestions" :key="suggestion.id">
                <div class="suggestion-icon">
                  <component :is="suggestion.icon" />
                </div>
                <div class="suggestion-content">
                  <h4>{{ suggestion.title }}</h4>
                  <p>{{ suggestion.description }}</p>
                </div>
              </div>
              
              <a-button type="primary" block class="ai-chat-btn" @click="openAIAssistant">
                <RobotOutlined />
                与AI助手对话
              </a-button>
            </div>
          </div>
          
          <!-- 今日计划 -->
          <div class="content-card schedule-card">
            <div class="card-header">
              <h3 class="card-title">
                <CalendarOutlined class="title-icon" />
                今日计划
              </h3>
            </div>
            
            <div class="schedule-list">
              <div 
                v-for="item in todaySchedule" 
                :key="item.id"
                class="schedule-item"
                :class="{ completed: item.completed, current: item.isCurrent }"
              >
                <div class="schedule-time">
                  <div class="time-dot" :class="{ completed: item.completed, current: item.isCurrent }"></div>
                  <span class="time-text">{{ item.time }}</span>
                </div>
                <div class="schedule-content">
                  <h4>{{ item.title }}</h4>
                  <p>{{ item.description }}</p>
                </div>
                <a-checkbox 
                  v-model:checked="item.completed"
                  @change="updateScheduleItem(item)"
                  class="schedule-checkbox"
                />
              </div>
            </div>
          </div>
        </a-col>
      </a-row>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, reactive, computed, onMounted } from 'vue'
import { message } from 'ant-design-vue'
import {
  UserOutlined,
  BookOutlined,
  EditOutlined,
  RobotOutlined,
  ClockCircleOutlined,
  CheckCircleOutlined,
  TrophyOutlined,
  FireOutlined,
  ArrowRightOutlined,
  BarChartOutlined,
  CalendarOutlined,
  BulbOutlined,
  AimOutlined,
  LineChartOutlined
} from '@ant-design/icons-vue'
import { useAuthStore } from '@/stores/auth'

const userStore = useAuthStore()

// 统计数据
const todayStudyTime = ref(3.5)
const completedAssignments = ref(8)
const totalAssignments = ref(10)
const averageGrade = ref(92)
const ranking = ref('前5%')
const studyStreak = ref(7)

// 计算属性
const assignmentCompletionRate = computed(() => 
  Math.round((completedAssignments.value / totalAssignments.value) * 100)
)

// 进度数据
const progressTimeRange = ref('week')
const weeklyProgress = ref(75)
const completedHours = ref(15)
const targetHours = ref(20)

// 各科目进度
const subjectsProgress = ref([
  { name: '数学', progress: 85, color: '#1890ff' },
  { name: '英语', progress: 78, color: '#52c41a' },
  { name: '物理', progress: 92, color: '#722ed1' },
  { name: '化学', progress: 68, color: '#fa8c16' }
])

// 待完成作业
const pendingAssignments = ref([
  {
    id: 1,
    title: '数学函数综合练习',
    description: '完成第三章函数基础练习题，包括一次函数和二次函数的综合应用',
    subject: '数学',
    subjectColor: '#1890ff',
    deadline: new Date('2024-01-20T23:59:59'),
    isUrgent: true
  },
  {
    id: 2,
    title: '英语阅读理解训练',
    description: '阅读指定文章并完成相关理解题目，提升阅读能力',
    subject: '英语',
    subjectColor: '#52c41a',
    deadline: new Date('2024-01-22T23:59:59'),
    isUrgent: false
  },
  {
    id: 3,
    title: '物理实验报告',
    description: '撰写光学实验的详细报告，包括数据分析和结论',
    subject: '物理',
    subjectColor: '#722ed1',
    deadline: new Date('2024-01-25T23:59:59'),
    isUrgent: false
  }
])

// 最近成绩
const recentGrades = ref([
  {
    id: 1,
    subject: '数学',
    assignment: '期中考试',
    score: 92,
    subjectColor: '#1890ff',
    date: new Date('2024-01-15')
  },
  {
    id: 2,
    subject: '英语',
    assignment: '单元测试',
    score: 88,
    subjectColor: '#52c41a',
    date: new Date('2024-01-12')
  },
  {
    id: 3,
    subject: '物理',
    assignment: '实验报告',
    score: 95,
    subjectColor: '#722ed1',
    date: new Date('2024-01-10')
  }
])

// AI学习建议
const aiSuggestions = ref([
  {
    id: 1,
    icon: 'BulbOutlined',
    title: '学习建议',
    description: '建议加强英语语法练习，可以提升整体成绩'
  },
  {
    id: 2,
    icon: 'AimOutlined',
    title: '学习计划',
    description: '制定数学复习计划，准备下周的章节测试'
  },
  {
    id: 3,
    icon: 'LineChartOutlined',
    title: '进度分析',
    description: '本周学习效率较高，建议保持当前学习节奏'
  }
])

// 今日计划
const todaySchedule = ref([
  {
    id: 1,
    time: '09:00',
    title: '数学课',
    description: '函数图像与性质',
    completed: true,
    isCurrent: false
  },
  {
    id: 2,
    time: '14:00',
    title: '完成英语作业',
    description: '阅读理解练习',
    completed: false,
    isCurrent: true
  },
  {
    id: 3,
    time: '16:00',
    title: '物理实验',
    description: '光学实验操作',
    completed: false,
    isCurrent: false
  },
  {
    id: 4,
    time: '19:00',
    title: '复习总结',
    description: '整理今日学习笔记',
    completed: false,
    isCurrent: false
  }
])

// 方法函数
const getGreeting = () => {
  const hour = new Date().getHours()
  if (hour < 12) return '早上好'
  if (hour < 18) return '下午好'
  return '晚上好'
}

const formatDate = (date: Date) => {
  return date.toLocaleDateString('zh-CN', {
    year: 'numeric',
    month: 'long',
    day: 'numeric',
    weekday: 'long'
  })
}

const formatDeadline = (deadline: Date) => {
  const now = new Date()
  const diffTime = deadline.getTime() - now.getTime()
  const diffDays = Math.ceil(diffTime / (1000 * 60 * 60 * 24))
  
  if (diffDays < 0) return '已截止'
  if (diffDays === 0) return '今天截止'
  if (diffDays === 1) return '明天截止'
  return `${diffDays}天后截止`
}

const formatRelativeTime = (date: Date) => {
  const now = new Date()
  const diffTime = now.getTime() - date.getTime()
  const diffDays = Math.floor(diffTime / (1000 * 60 * 60 * 24))
  
  if (diffDays === 0) return '今天'
  if (diffDays === 1) return '昨天'
  if (diffDays < 7) return `${diffDays}天前`
  return date.toLocaleDateString('zh-CN')
}

const getGradeClass = (score: number) => {
  if (score >= 90) return 'excellent'
  if (score >= 80) return 'good'
  if (score >= 70) return 'average'
  return 'poor'
}

const getProgressColor = (progress: number) => {
  if (progress >= 80) return '#52c41a'
  if (progress >= 60) return '#faad14'
  return '#ff4d4f'
}

const openAssignment = (assignment: any) => {
  message.info(`即将打开作业：${assignment.title}`)
  // 这里可以跳转到作业详情页
}

const openAIAssistant = () => {
  message.info('AI助手功能即将上线，敬请期待！')
}

const updateScheduleItem = (item: any) => {
  message.success(`已${item.completed ? '完成' : '取消完成'}：${item.title}`)
}

// 页面初始化
onMounted(() => {
  // 这里可以调用API获取数据
  console.log('学生Dashboard初始化完成')
})
</script>

<style scoped>
.student-dashboard {
  min-height: 100vh;
  background: #f5f7fa;
  padding: 24px;
}

/* 页面头部 */
.dashboard-header {
  margin-bottom: 24px;
}

.welcome-section {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  border-radius: 16px;
  padding: 32px;
  color: white;
  margin-bottom: 24px;
  position: relative;
  overflow: hidden;
}

.welcome-section::before {
  content: '';
  position: absolute;
  top: -50%;
  right: -50%;
  width: 200%;
  height: 200%;
  background: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><circle cx="50" cy="50" r="2" fill="white" opacity="0.1"/></svg>') repeat;
  animation: float 20s linear infinite;
}

@keyframes float {
  0% { transform: translate(0, 0) rotate(0deg); }
  100% { transform: translate(-50px, -50px) rotate(360deg); }
}

.user-greeting {
  display: flex;
  align-items: center;
  gap: 24px;
  margin-bottom: 32px;
  position: relative;
  z-index: 2;
}

.avatar-container {
  position: relative;
}

.user-avatar {
  background: rgba(255, 255, 255, 0.2);
  backdrop-filter: blur(10px);
  border: 3px solid rgba(255, 255, 255, 0.3);
}

.online-indicator {
  position: absolute;
  bottom: 4px;
  right: 4px;
  width: 16px;
  height: 16px;
  background: #52c41a;
  border-radius: 50%;
  border: 3px solid white;
}

.greeting-content {
  flex: 1;
}

.greeting-title {
  font-size: 32px;
  font-weight: 700;
  margin: 0 0 8px 0;
  background: linear-gradient(45deg, #ffffff, #e3f2fd);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}

.greeting-subtitle {
  font-size: 16px;
  margin: 0;
  opacity: 0.9;
}

.quick-actions {
  display: flex;
  gap: 16px;
  position: relative;
  z-index: 2;
}

.action-btn {
  height: 48px;
  padding: 0 24px;
  border-radius: 12px;
  font-weight: 600;
  display: flex;
  align-items: center;
  gap: 8px;
  transition: all 0.3s ease;
}

.action-btn.primary {
  background: rgba(255, 255, 255, 0.2);
  border-color: rgba(255, 255, 255, 0.3);
  color: white;
  backdrop-filter: blur(10px);
}

.action-btn.primary:hover {
  background: rgba(255, 255, 255, 0.3);
  transform: translateY(-2px);
}

.action-btn.secondary,
.action-btn.ai {
  background: rgba(255, 255, 255, 0.1);
  border-color: rgba(255, 255, 255, 0.2);
  color: white;
  backdrop-filter: blur(10px);
}

.action-btn.secondary:hover,
.action-btn.ai:hover {
  background: rgba(255, 255, 255, 0.2);
  transform: translateY(-2px);
}

/* 统计概览卡片 */
.stats-overview {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
  gap: 20px;
}

.stat-card {
  background: white;
  border-radius: 16px;
  padding: 24px;
  display: flex;
  align-items: center;
  gap: 20px;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.06);
  border: 1px solid #f0f0f0;
  transition: all 0.3s ease;
}

.stat-card:hover {
  transform: translateY(-4px);
  box-shadow: 0 8px 30px rgba(0, 0, 0, 0.12);
}

.stat-icon {
  width: 64px;
  height: 64px;
  border-radius: 16px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 24px;
  color: white;
}

.stat-card.study-time .stat-icon {
  background: linear-gradient(135deg, #1890ff, #36cfc9);
}

.stat-card.assignments .stat-icon {
  background: linear-gradient(135deg, #52c41a, #73d13d);
}

.stat-card.grade .stat-icon {
  background: linear-gradient(135deg, #faad14, #ffc53d);
}

.stat-card.streak .stat-icon {
  background: linear-gradient(135deg, #ff4d4f, #ff7875);
}

.stat-content {
  flex: 1;
}

.stat-value {
  font-size: 28px;
  font-weight: 700;
  color: #333;
  margin-bottom: 4px;
}

.stat-label {
  font-size: 14px;
  color: #666;
  margin-bottom: 4px;
}

.stat-trend {
  font-size: 12px;
  color: #52c41a;
  font-weight: 500;
}

/* 主要内容区域 */
.dashboard-content {
  margin-top: 24px;
}

.content-card {
  background: white;
  border-radius: 16px;
  padding: 24px;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.06);
  border: 1px solid #f0f0f0;
  margin-bottom: 24px;
  transition: all 0.3s ease;
}

.content-card:hover {
  box-shadow: 0 8px 30px rgba(0, 0, 0, 0.12);
}

.card-header {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  margin-bottom: 24px;
}

.header-left {
  flex: 1;
}

.card-title {
  font-size: 20px;
  font-weight: 600;
  color: #333;
  margin: 0 0 8px 0;
  display: flex;
  align-items: center;
  gap: 8px;
}

.title-icon {
  color: #1890ff;
}

.title-icon.ai-icon {
  background: linear-gradient(135deg, #fa541c, #faad14);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}

.title-badge {
  margin-left: 8px;
}

.card-subtitle {
  color: #666;
  font-size: 14px;
  margin: 0;
}

/* 作业列表 */
.assignments-list {
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.assignment-item {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 20px;
  border: 1px solid #f0f0f0;
  border-radius: 12px;
  cursor: pointer;
  transition: all 0.3s ease;
}

.assignment-item:hover {
  border-color: #1890ff;
  background: #fafbff;
  transform: translateY(-2px);
}

.assignment-item.urgent {
  border-color: #ff4d4f;
  background: #fff2f0;
}

.assignment-left {
  display: flex;
  align-items: center;
  gap: 16px;
  flex: 1;
}

.assignment-subject {
  padding: 4px 12px;
  border-radius: 20px;
  color: white;
  font-size: 12px;
  font-weight: 500;
  white-space: nowrap;
}

.assignment-content {
  flex: 1;
  min-width: 0;
}

.assignment-title {
  font-size: 16px;
  font-weight: 600;
  color: #333;
  margin: 0 0 4px 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.assignment-desc {
  font-size: 14px;
  color: #666;
  margin: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.assignment-right {
  display: flex;
  flex-direction: column;
  align-items: flex-end;
  gap: 8px;
}

.assignment-deadline {
  display: flex;
  align-items: center;
  gap: 4px;
  font-size: 12px;
  color: #666;
}

.assignment-deadline.urgent {
  color: #ff4d4f;
  font-weight: 500;
}

.urgent-btn {
  background: #ff4d4f !important;
  border-color: #ff4d4f !important;
}

/* 学习进度 */
.progress-content {
  display: flex;
  flex-direction: column;
  gap: 24px;
}

.progress-item {
  padding: 20px;
  background: #fafbff;
  border-radius: 12px;
  border: 1px solid #e6f7ff;
}

.progress-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 12px;
}

.progress-label {
  font-size: 16px;
  font-weight: 500;
  color: #333;
}

.progress-percentage {
  font-size: 18px;
  font-weight: 600;
  color: #1890ff;
}

.progress-bar {
  margin-bottom: 8px;
}

.progress-details {
  font-size: 12px;
  color: #666;
}

.subjects-progress {
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.subject-progress {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.subject-info {
  display: flex;
  align-items: center;
  gap: 12px;
}

.subject-icon {
  width: 32px;
  height: 32px;
  border-radius: 8px;
  color: white;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 14px;
  font-weight: 600;
}

.subject-details {
  display: flex;
  justify-content: space-between;
  align-items: center;
  flex: 1;
}

.subject-name {
  font-size: 14px;
  font-weight: 500;
  color: #333;
}

.subject-progress-text {
  font-size: 14px;
  color: #666;
}

/* 成绩列表 */
.grades-list {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.grade-item {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 16px;
  border: 1px solid #f0f0f0;
  border-radius: 8px;
  transition: all 0.3s ease;
}

.grade-item:hover {
  background: #fafafa;
}

.grade-left {
  display: flex;
  align-items: center;
  gap: 12px;
  flex: 1;
}

.grade-subject {
  width: 32px;
  height: 32px;
  border-radius: 6px;
  color: white;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 14px;
  font-weight: 600;
}

.grade-info {
  display: flex;
  flex-direction: column;
  gap: 2px;
}

.grade-assignment {
  font-size: 14px;
  font-weight: 500;
  color: #333;
}

.grade-subject-name {
  font-size: 12px;
  color: #666;
}

.grade-right {
  display: flex;
  flex-direction: column;
  align-items: flex-end;
  gap: 4px;
}

.grade-score {
  font-size: 18px;
  font-weight: 600;
  padding: 4px 8px;
  border-radius: 6px;
}

.grade-score.excellent {
  color: #52c41a;
  background: #f6ffed;
}

.grade-score.good {
  color: #1890ff;
  background: #e6f7ff;
}

.grade-score.average {
  color: #faad14;
  background: #fffbe6;
}

.grade-score.poor {
  color: #ff4d4f;
  background: #fff2f0;
}

.grade-date {
  font-size: 12px;
  color: #999;
}

/* AI建议 */
.ai-suggestions {
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.ai-suggestion {
  display: flex;
  align-items: flex-start;
  gap: 12px;
  padding: 16px;
  background: linear-gradient(135deg, #fff7e6, #fff2e8);
  border-radius: 12px;
  border: 1px solid #ffe7ba;
}

.suggestion-icon {
  width: 32px;
  height: 32px;
  background: linear-gradient(135deg, #fa541c, #faad14);
  border-radius: 8px;
  color: white;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 14px;
}

.suggestion-content h4 {
  font-size: 14px;
  font-weight: 600;
  color: #333;
  margin: 0 0 4px 0;
}

.suggestion-content p {
  font-size: 12px;
  color: #666;
  margin: 0;
  line-height: 1.4;
}

.ai-chat-btn {
  background: linear-gradient(135deg, #fa541c, #faad14);
  border: none;
  height: 40px;
  border-radius: 8px;
  font-weight: 500;
}

/* 今日计划 */
.schedule-list {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.schedule-item {
  display: flex;
  align-items: center;
  gap: 16px;
  padding: 16px;
  border-radius: 8px;
  transition: all 0.3s ease;
}

.schedule-item:hover {
  background: #fafafa;
}

.schedule-item.current {
  background: #e6f7ff;
  border: 1px solid #91d5ff;
}

.schedule-item.completed {
  opacity: 0.6;
}

.schedule-time {
  display: flex;
  align-items: center;
  gap: 8px;
  min-width: 80px;
}

.time-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: #d9d9d9;
}

.time-dot.current {
  background: #1890ff;
  box-shadow: 0 0 0 3px rgba(24, 144, 255, 0.2);
}

.time-dot.completed {
  background: #52c41a;
}

.time-text {
  font-size: 12px;
  color: #666;
  font-weight: 500;
}

.schedule-content {
  flex: 1;
  min-width: 0;
}

.schedule-content h4 {
  font-size: 14px;
  font-weight: 500;
  color: #333;
  margin: 0 0 4px 0;
}

.schedule-content p {
  font-size: 12px;
  color: #666;
  margin: 0;
}

.schedule-checkbox {
  margin-left: auto;
}

/* 空状态 */
.empty-state {
  text-align: center;
  padding: 40px 20px;
  color: #999;
}

.empty-icon {
  font-size: 48px;
  color: #52c41a;
  margin-bottom: 16px;
}

.empty-state h4 {
  font-size: 16px;
  color: #333;
  margin: 0 0 8px 0;
}

.empty-state p {
  font-size: 14px;
  color: #666;
  margin: 0;
}

/* 响应式设计 */
@media (max-width: 1200px) {
  .stats-overview {
    grid-template-columns: repeat(2, 1fr);
  }
  
  .user-greeting {
    flex-direction: column;
    text-align: center;
    gap: 16px;
  }
  
  .quick-actions {
    justify-content: center;
    flex-wrap: wrap;
  }
}

@media (max-width: 768px) {
  .student-dashboard {
    padding: 16px;
  }
  
  .welcome-section {
    padding: 24px 20px;
  }
  
  .greeting-title {
    font-size: 24px;
  }
  
  .stats-overview {
    grid-template-columns: 1fr;
  }
  
  .assignment-item {
    flex-direction: column;
    align-items: flex-start;
    gap: 12px;
  }
  
  .assignment-left {
    width: 100%;
  }
  
  .assignment-right {
    width: 100%;
    flex-direction: row;
    align-items: center;
    justify-content: space-between;
  }
  
  .quick-actions {
    flex-direction: column;
    gap: 12px;
  }
  
  .action-btn {
    width: 100%;
    justify-content: center;
  }
}

@media (max-width: 480px) {
  .card-header {
    flex-direction: column;
    align-items: flex-start;
    gap: 12px;
  }
  
  .stat-card {
    flex-direction: column;
    text-align: center;
    gap: 12px;
  }
  
  .greeting-content {
    text-align: center;
  }
}
</style>