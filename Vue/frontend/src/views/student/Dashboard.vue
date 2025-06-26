<template>
  <div class="student-dashboard">
    <!-- 顶部欢迎区域 -->
    <div class="dashboard-header">
      <div class="welcome-section">
        <!-- 装饰性背景元素 -->
        <div class="bg-decoration">
          <div class="floating-shape shape-1"></div>
          <div class="floating-shape shape-2"></div>
          <div class="floating-shape shape-3"></div>
          <div class="floating-shape shape-4"></div>
        </div>
        
        <div class="welcome-content">
          <div class="user-info">
            <div class="avatar-container">
              <a-avatar :size="80" :src="userStore.user?.avatar || ''" class="user-avatar">
                <template #icon>
                  <UserOutlined />
                </template>
              </a-avatar>
              <div class="online-status">
                <div class="status-dot"></div>
                <span class="status-text">在线学习</span>
              </div>
            </div>
            
            <div class="greeting-text">
              <h1 class="greeting-title">
                {{ getGreeting() }}，{{ userStore.user?.realName || '同学' }}！
              </h1>
              <p class="greeting-subtitle">
                {{ formatDate(new Date()) }} · 让我们一起创造美好的学习时光
              </p>
              <div class="achievement-badge">
                <TrophyOutlined />
                <span>连续学习 {{ studyStreak }} 天</span>
              </div>
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
              <span>我的课程</span>
            </a-button>
            <a-button 
              size="large" 
              @click="$router.push('/student/assignments')"
              class="action-btn secondary"
            >
              <EditOutlined />
              <span>作业中心</span>
            </a-button>
            <a-button 
              size="large" 
              @click="openAIAssistant"
              class="action-btn ai"
            >
              <RobotOutlined />
              <span>AI学习助手</span>
            </a-button>
          </div>
        </div>
      </div>
      
      <!-- 学习数据统计 -->
      <div class="stats-overview">
        <div class="stat-card study-time">
          <div class="stat-header">
            <div class="stat-icon">
              <ClockCircleOutlined />
            </div>
            <div class="stat-meta">
              <span class="stat-label">今日学习时长</span>
              <div class="stat-trend positive">
                <ArrowUpOutlined />
                <span>+15%</span>
              </div>
            </div>
          </div>
          <div class="stat-value">{{ todayStudyTime }}<span class="unit">小时</span></div>
          <div class="stat-progress">
            <a-progress 
              :percent="(todayStudyTime / 8) * 100" 
              :stroke-color="{ '0%': '#667eea', '100%': '#764ba2' }"
              :show-info="false"
              :stroke-width="6"
            />
            <span class="progress-text">目标：8小时</span>
          </div>
        </div>
        
        <div class="stat-card assignments">
          <div class="stat-header">
            <div class="stat-icon">
              <CheckCircleOutlined />
            </div>
            <div class="stat-meta">
              <span class="stat-label">本周作业</span>
              <div class="stat-trend positive">
                <span>{{ assignmentCompletionRate }}%</span>
              </div>
            </div>
          </div>
          <div class="stat-value">{{ completedAssignments }}<span class="unit">/{{ totalAssignments }}</span></div>
          <div class="stat-description">已完成 {{ completedAssignments }} 项作业</div>
        </div>
        
        <div class="stat-card grade">
          <div class="stat-header">
            <div class="stat-icon">
              <TrophyOutlined />
            </div>
            <div class="stat-meta">
              <span class="stat-label">平均成绩</span>
              <div class="stat-trend positive">
                <span>{{ ranking }}</span>
              </div>
            </div>
          </div>
          <div class="stat-value">{{ averageGrade }}<span class="unit">分</span></div>
          <div class="stat-description">班级排名前5%</div>
        </div>
        
        <div class="stat-card streak">
          <div class="stat-header">
            <div class="stat-icon">
              <FireOutlined />
            </div>
            <div class="stat-meta">
              <span class="stat-label">学习打卡</span>
              <div class="stat-trend">
                <span>坚持中</span>
              </div>
            </div>
          </div>
          <div class="stat-value">{{ studyStreak }}<span class="unit">天</span></div>
          <div class="stat-description">连续学习天数</div>
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
          
          <!-- AI学习助手 -->
          <div class="content-card ai-card">
            <div class="card-header">
              <div class="header-left">
                <h3 class="card-title">
                  <RobotOutlined class="title-icon ai-icon" />
                  AI学习助手
                </h3>
                <p class="card-subtitle">个性化学习建议与智能分析</p>
              </div>
              <div class="ai-status">
                <div class="status-indicator online"></div>
                <span class="status-text">在线</span>
              </div>
            </div>
            
            <div class="ai-suggestions">
              <div class="ai-suggestion" v-for="suggestion in aiSuggestions" :key="suggestion.id">
                <div class="suggestion-icon">
                  <component :is="suggestion.icon" />
                </div>
                <div class="suggestion-content">
                  <h4>{{ suggestion.title }}</h4>
                  <p>{{ suggestion.description }}</p>
                  <div class="suggestion-actions">
                    <a-button type="link" size="small">了解更多</a-button>
                  </div>
                </div>
              </div>
              
              <div class="ai-chat-section">
                <div class="chat-preview">
                  <div class="chat-avatar">
                    <RobotOutlined />
                  </div>
                  <div class="chat-message">
                    <p>你好！我是你的AI学习助手，有什么可以帮助你的吗？</p>
                  </div>
                </div>
                <a-button type="primary" block class="ai-chat-btn" @click="openAIAssistant">
                  <RobotOutlined />
                  开始对话
                </a-button>
              </div>
            </div>
          </div>
          
          <!-- 今日学习计划 -->
          <div class="content-card schedule-card">
            <div class="card-header">
              <div class="header-left">
                <h3 class="card-title">
                  <CalendarOutlined class="title-icon" />
                  今日学习计划
                </h3>
                <p class="card-subtitle">合理安排时间，高效学习</p>
              </div>
              <div class="schedule-progress">
                <a-progress 
                  type="circle" 
                  :percent="scheduleCompletionRate" 
                  :width="40"
                  :stroke-color="{ '0%': '#667eea', '100%': '#764ba2' }"
                />
              </div>
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
                  <div class="schedule-tags" v-if="item.tags">
                    <a-tag 
                      v-for="tag in item.tags" 
                      :key="tag" 
                      size="small"
                      :color="getTagColor(tag)"
                    >
                      {{ tag }}
                    </a-tag>
                  </div>
                </div>
                <div class="schedule-actions">
                  <a-checkbox 
                    v-model:checked="item.completed"
                    @change="updateScheduleItem(item)"
                    class="schedule-checkbox"
                  />
                  <a-button 
                    v-if="item.isCurrent && !item.completed" 
                    type="primary" 
                    size="small"
                    @click="startTask(item)"
                  >
                    开始
                  </a-button>
                </div>
              </div>
            </div>
            
            <div class="schedule-summary">
              <div class="summary-item">
                <span class="summary-label">已完成</span>
                <span class="summary-value">{{ completedTasks }}/{{ todaySchedule.length }}</span>
              </div>
              <div class="summary-item">
                <span class="summary-label">预计用时</span>
                <span class="summary-value">{{ totalEstimatedTime }}h</span>
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
  ArrowUpOutlined,
  BarChartOutlined,
  CalendarOutlined,
  BulbOutlined,
  AimOutlined,
  LineChartOutlined,
  StarOutlined,
  HeartOutlined,
  ThunderboltOutlined
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
    title: '个性化学习建议',
    description: '根据你的学习数据分析，建议加强英语语法练习，重点关注时态和语态的运用'
  },
  {
    id: 2,
    icon: 'AimOutlined',
    title: '智能学习规划',
    description: '为你制定了数学复习计划，建议每天练习30分钟函数题型，准备下周测试'
  },
  {
    id: 3,
    icon: 'LineChartOutlined',
    title: '学习效率分析',
    description: '本周学习效率提升15%，建议保持当前学习节奏，适当增加难题练习'
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
    isCurrent: false,
    tags: ['必修', '重点'],
    estimatedTime: 1.5
  },
  {
    id: 2,
    time: '14:00',
    title: '完成英语作业',
    description: '阅读理解练习',
    completed: false,
    isCurrent: true,
    tags: ['作业', '紧急'],
    estimatedTime: 1
  },
  {
    id: 3,
    time: '16:00',
    title: '物理实验',
    description: '光学实验操作',
    completed: false,
    isCurrent: false,
    tags: ['实验', '选修'],
    estimatedTime: 2
  },
  {
    id: 4,
    time: '19:00',
    title: '复习总结',
    description: '整理今日学习笔记',
    completed: false,
    isCurrent: false,
    tags: ['复习'],
    estimatedTime: 0.5
  }
])

// 计算属性
const scheduleCompletionRate = computed(() => {
  const completed = todaySchedule.value.filter(item => item.completed).length
  return Math.round((completed / todaySchedule.value.length) * 100)
})

const completedTasks = computed(() => 
  todaySchedule.value.filter(item => item.completed).length
)

const totalEstimatedTime = computed(() => 
  todaySchedule.value.reduce((total, item) => total + item.estimatedTime, 0)
)

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

const getTagColor = (tag: string) => {
  const colorMap: { [key: string]: string } = {
    '必修': 'blue',
    '选修': 'green',
    '重点': 'red',
    '作业': 'orange',
    '紧急': 'volcano',
    '实验': 'purple',
    '复习': 'geekblue'
  }
  return colorMap[tag] || 'default'
}

const startTask = (item: any) => {
  message.info(`开始执行任务：${item.title}`)
  // 这里可以跳转到具体的学习页面
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
  margin-bottom: 32px;
}

.welcome-section {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
  border-radius: 24px;
  padding: 40px;
  color: white;
  margin-bottom: 32px;
  position: relative;
  overflow: hidden;
  box-shadow: 0 20px 40px rgba(102, 126, 234, 0.3);
}

/* 装饰性背景元素 */
.bg-decoration {
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  pointer-events: none;
  overflow: hidden;
}

.floating-shape {
  position: absolute;
  border-radius: 50%;
  background: rgba(255, 255, 255, 0.1);
  backdrop-filter: blur(10px);
}

.shape-1 {
  width: 120px;
  height: 120px;
  top: -60px;
  right: -60px;
  animation: float-1 20s ease-in-out infinite;
}

.shape-2 {
  width: 80px;
  height: 80px;
  top: 50%;
  right: 10%;
  animation: float-2 15s ease-in-out infinite reverse;
}

.shape-3 {
  width: 60px;
  height: 60px;
  bottom: -30px;
  left: 20%;
  animation: float-3 18s ease-in-out infinite;
}

.shape-4 {
  width: 40px;
  height: 40px;
  top: 20%;
  left: -20px;
  animation: float-4 12s ease-in-out infinite reverse;
}

@keyframes float-1 {
  0%, 100% { transform: translate(0, 0) rotate(0deg); }
  33% { transform: translate(-20px, -20px) rotate(120deg); }
  66% { transform: translate(20px, -10px) rotate(240deg); }
}

@keyframes float-2 {
  0%, 100% { transform: translate(0, 0) scale(1); }
  50% { transform: translate(-15px, 15px) scale(1.1); }
}

@keyframes float-3 {
  0%, 100% { transform: translate(0, 0) rotate(0deg); }
  50% { transform: translate(10px, -20px) rotate(180deg); }
}

@keyframes float-4 {
  0%, 100% { transform: translate(0, 0) scale(1); }
  25% { transform: translate(15px, -10px) scale(0.9); }
  75% { transform: translate(-10px, 15px) scale(1.1); }
}

.welcome-content {
  position: relative;
  z-index: 2;
}

.user-info {
  display: flex;
  align-items: center;
  gap: 32px;
  margin-bottom: 40px;
}

.avatar-container {
  position: relative;
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 12px;
}

.user-avatar {
  background: rgba(255, 255, 255, 0.15);
  backdrop-filter: blur(20px);
  border: 4px solid rgba(255, 255, 255, 0.2);
  box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
}

.online-status {
  display: flex;
  align-items: center;
  gap: 6px;
  background: rgba(255, 255, 255, 0.15);
  backdrop-filter: blur(10px);
  padding: 4px 12px;
  border-radius: 20px;
  font-size: 12px;
  font-weight: 500;
}

.status-dot {
  width: 8px;
  height: 8px;
  background: #52c41a;
  border-radius: 50%;
  animation: pulse 2s ease-in-out infinite;
}

@keyframes pulse {
  0%, 100% { opacity: 1; transform: scale(1); }
  50% { opacity: 0.7; transform: scale(1.2); }
}

.status-text {
  color: rgba(255, 255, 255, 0.9);
}

.greeting-text {
  flex: 1;
}

.greeting-title {
  font-size: 36px;
  font-weight: 800;
  margin: 0 0 12px 0;
  background: linear-gradient(45deg, #ffffff, #f0f8ff);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  text-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}

.greeting-subtitle {
  font-size: 16px;
  margin: 0 0 16px 0;
  opacity: 0.9;
  line-height: 1.5;
}

.achievement-badge {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  background: rgba(255, 255, 255, 0.15);
  backdrop-filter: blur(10px);
  padding: 8px 16px;
  border-radius: 20px;
  font-size: 14px;
  font-weight: 600;
  border: 1px solid rgba(255, 255, 255, 0.2);
}

.quick-actions {
  display: flex;
  gap: 20px;
  position: relative;
  z-index: 2;
}

.action-btn {
  height: 56px;
  padding: 0 28px;
  border-radius: 16px;
  font-weight: 600;
  font-size: 15px;
  display: flex;
  align-items: center;
  gap: 10px;
  transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
  position: relative;
  overflow: hidden;
}

.action-btn::before {
  content: '';
  position: absolute;
  top: 0;
  left: -100%;
  width: 100%;
  height: 100%;
  background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent);
  transition: left 0.5s ease;
}

.action-btn:hover::before {
  left: 100%;
}

.action-btn.primary {
  background: rgba(255, 255, 255, 0.25);
  border: 2px solid rgba(255, 255, 255, 0.3);
  color: white;
  backdrop-filter: blur(20px);
  box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
}

.action-btn.primary:hover {
  background: rgba(255, 255, 255, 0.35);
  transform: translateY(-3px);
  box-shadow: 0 12px 40px rgba(0, 0, 0, 0.15);
}

.action-btn.secondary {
  background: rgba(255, 255, 255, 0.15);
  border: 2px solid rgba(255, 255, 255, 0.2);
  color: white;
  backdrop-filter: blur(20px);
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
}

.action-btn.secondary:hover {
  background: rgba(255, 255, 255, 0.25);
  transform: translateY(-3px);
  box-shadow: 0 8px 32px rgba(0, 0, 0, 0.15);
}

.action-btn.ai {
  background: linear-gradient(135deg, rgba(255, 255, 255, 0.2), rgba(255, 255, 255, 0.1));
  border: 2px solid rgba(255, 255, 255, 0.25);
  color: white;
  backdrop-filter: blur(20px);
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
}

.action-btn.ai:hover {
  background: linear-gradient(135deg, rgba(255, 255, 255, 0.3), rgba(255, 255, 255, 0.2));
  transform: translateY(-3px);
  box-shadow: 0 8px 32px rgba(0, 0, 0, 0.15);
}

/* 学习数据统计 */
.stats-overview {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
  gap: 24px;
}

.stat-card {
  background: white;
  border-radius: 20px;
  padding: 28px;
  box-shadow: 0 8px 40px rgba(0, 0, 0, 0.08);
  border: 1px solid rgba(0, 0, 0, 0.05);
  transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
  position: relative;
  overflow: hidden;
}

.stat-card::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  height: 4px;
  background: linear-gradient(90deg, #667eea, #764ba2);
  transform: scaleX(0);
  transform-origin: left;
  transition: transform 0.3s ease;
}

.stat-card:hover {
  transform: translateY(-8px);
  box-shadow: 0 20px 60px rgba(0, 0, 0, 0.15);
}

.stat-card:hover::before {
  transform: scaleX(1);
}

.stat-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 20px;
}

.stat-icon {
  width: 56px;
  height: 56px;
  border-radius: 16px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 24px;
  color: white;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);
}

.stat-card.study-time .stat-icon {
  background: linear-gradient(135deg, #667eea, #764ba2);
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

.stat-meta {
  text-align: right;
}

.stat-label {
  font-size: 14px;
  color: #666;
  font-weight: 500;
  margin-bottom: 4px;
}

.stat-trend {
  display: flex;
  align-items: center;
  gap: 4px;
  font-size: 12px;
  font-weight: 600;
  padding: 4px 8px;
  border-radius: 12px;
  background: rgba(82, 196, 26, 0.1);
  color: #52c41a;
}

.stat-trend.positive {
  background: rgba(82, 196, 26, 0.1);
  color: #52c41a;
}

.stat-value {
  font-size: 32px;
  font-weight: 800;
  color: #333;
  margin-bottom: 8px;
  line-height: 1;
}

.stat-value .unit {
  font-size: 16px;
  font-weight: 500;
  color: #999;
  margin-left: 4px;
}

.stat-description {
  font-size: 13px;
  color: #999;
  margin-bottom: 16px;
}

.stat-progress {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.progress-text {
  font-size: 12px;
  color: #666;
  text-align: right;
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

/* AI学习助手 */
.ai-status {
  display: flex;
  align-items: center;
  gap: 8px;
}

.status-indicator {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: #52c41a;
  animation: pulse 2s ease-in-out infinite;
}

.status-indicator.online {
  background: #52c41a;
}

.status-text {
  font-size: 12px;
  color: #52c41a;
  font-weight: 500;
}

.ai-suggestions {
  display: flex;
  flex-direction: column;
  gap: 20px;
}

.ai-suggestion {
  display: flex;
  align-items: flex-start;
  gap: 16px;
  padding: 20px;
  background: linear-gradient(135deg, #f6f9ff, #e8f4f8);
  border-radius: 16px;
  border: 1px solid #e6f7ff;
  transition: all 0.3s ease;
}

.ai-suggestion:hover {
  transform: translateY(-2px);
  box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1);
}

.suggestion-icon {
  width: 40px;
  height: 40px;
  background: linear-gradient(135deg, #667eea, #764ba2);
  border-radius: 12px;
  color: white;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 16px;
  box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
}

.suggestion-content {
  flex: 1;
}

.suggestion-content h4 {
  font-size: 15px;
  font-weight: 600;
  color: #333;
  margin: 0 0 8px 0;
}

.suggestion-content p {
  font-size: 13px;
  color: #666;
  margin: 0 0 12px 0;
  line-height: 1.5;
}

.suggestion-actions {
  display: flex;
  gap: 8px;
}

.ai-chat-section {
  background: linear-gradient(135deg, #f0f8ff, #e6f7ff);
  border-radius: 16px;
  padding: 20px;
  border: 1px solid #b3e0ff;
}

.chat-preview {
  display: flex;
  align-items: flex-start;
  gap: 12px;
  margin-bottom: 16px;
}

.chat-avatar {
  width: 36px;
  height: 36px;
  background: linear-gradient(135deg, #667eea, #764ba2);
  border-radius: 50%;
  color: white;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 16px;
  box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
}

.chat-message {
  background: white;
  border-radius: 12px;
  padding: 12px 16px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
  position: relative;
}

.chat-message::before {
  content: '';
  position: absolute;
  left: -8px;
  top: 12px;
  width: 0;
  height: 0;
  border-top: 8px solid transparent;
  border-bottom: 8px solid transparent;
  border-right: 8px solid white;
}

.chat-message p {
  font-size: 14px;
  color: #333;
  margin: 0;
  line-height: 1.4;
}

.ai-chat-btn {
  background: linear-gradient(135deg, #667eea, #764ba2);
  border: none;
  height: 44px;
  border-radius: 12px;
  font-weight: 600;
  font-size: 15px;
  box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
  transition: all 0.3s ease;
}

.ai-chat-btn:hover {
  transform: translateY(-2px);
  box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
}

/* 今日学习计划 */
.schedule-progress {
  display: flex;
  align-items: center;
  gap: 8px;
}

.schedule-list {
  display: flex;
  flex-direction: column;
  gap: 16px;
  margin-bottom: 20px;
}

.schedule-item {
  display: flex;
  align-items: center;
  gap: 16px;
  padding: 20px;
  border-radius: 12px;
  border: 1px solid #f0f0f0;
  transition: all 0.3s ease;
  background: white;
}

.schedule-item:hover {
  border-color: #d9d9d9;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
}

.schedule-item.current {
  background: linear-gradient(135deg, #e6f7ff, #f0f8ff);
  border-color: #91d5ff;
  box-shadow: 0 4px 20px rgba(24, 144, 255, 0.15);
}

.schedule-item.completed {
  background: #f6ffed;
  border-color: #b7eb8f;
  opacity: 0.8;
}

.schedule-time {
  display: flex;
  align-items: center;
  gap: 12px;
  min-width: 90px;
}

.time-dot {
  width: 12px;
  height: 12px;
  border-radius: 50%;
  background: #d9d9d9;
  transition: all 0.3s ease;
}

.time-dot.current {
  background: #1890ff;
  box-shadow: 0 0 0 4px rgba(24, 144, 255, 0.2);
  animation: pulse 2s ease-in-out infinite;
}

.time-dot.completed {
  background: #52c41a;
  box-shadow: 0 0 0 4px rgba(82, 196, 26, 0.2);
}

.time-text {
  font-size: 13px;
  color: #666;
  font-weight: 600;
}

.schedule-content {
  flex: 1;
  min-width: 0;
}

.schedule-content h4 {
  font-size: 15px;
  font-weight: 600;
  color: #333;
  margin: 0 0 6px 0;
}

.schedule-content p {
  font-size: 13px;
  color: #666;
  margin: 0 0 8px 0;
  line-height: 1.4;
}

.schedule-tags {
  display: flex;
  gap: 6px;
  flex-wrap: wrap;
}

.schedule-actions {
  display: flex;
  align-items: center;
  gap: 12px;
}

.schedule-checkbox {
  transform: scale(1.1);
}

.schedule-summary {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 16px;
  background: linear-gradient(135deg, #f6f9ff, #e8f4f8);
  border-radius: 12px;
  border: 1px solid #e6f7ff;
}

.summary-item {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 4px;
}

.summary-label {
  font-size: 12px;
  color: #666;
  font-weight: 500;
}

.summary-value {
  font-size: 16px;
  font-weight: 600;
  color: #333;
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
  
  .user-info {
    flex-direction: column;
    text-align: center;
    gap: 20px;
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
    padding: 32px 24px;
    border-radius: 20px;
  }
  
  .greeting-title {
    font-size: 28px;
  }
  
  .stats-overview {
    grid-template-columns: 1fr;
    gap: 16px;
  }
  
  .stat-card {
    padding: 20px;
  }
  
  .stat-value {
    font-size: 28px;
  }
  
  .assignment-item {
    flex-direction: column;
    align-items: flex-start;
    gap: 16px;
    padding: 16px;
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
    height: 48px;
    font-size: 14px;
  }
  
  .content-card {
    padding: 20px;
    margin-bottom: 20px;
  }
  
  .ai-suggestion {
    padding: 16px;
  }
  
  .chat-preview {
    flex-direction: column;
    align-items: center;
    text-align: center;
    gap: 12px;
  }
}

@media (max-width: 480px) {
  .student-dashboard {
    padding: 12px;
  }
  
  .welcome-section {
    padding: 24px 20px;
    border-radius: 16px;
  }
  
  .greeting-title {
    font-size: 24px;
  }
  
  .user-avatar {
    width: 64px !important;
    height: 64px !important;
  }
  
  .card-header {
    flex-direction: column;
    align-items: flex-start;
    gap: 12px;
  }
  
  .stat-header {
    flex-direction: column;
    align-items: flex-start;
    gap: 8px;
  }
  
  .stat-meta {
    text-align: left;
  }
  
  .stat-value {
    font-size: 24px;
  }
  
  .greeting-text {
    text-align: center;
  }
  
  .achievement-badge {
    align-self: center;
  }
  
  .floating-shape {
    display: none;
  }
  
  .ai-chat-section {
    padding: 16px;
  }
  
  .suggestion-icon {
    width: 32px;
    height: 32px;
    font-size: 14px;
  }
}
</style>