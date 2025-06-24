<template>
  <div class="student-dashboard">
    <div class="dashboard-header">
      <div class="welcome-section">
        <h1 class="welcome-title">
          你好，{{ authStore.user?.realName || '同学' }}！
        </h1>
        <p class="welcome-subtitle">
          今天是 {{ currentDate }}，继续加油学习吧！
        </p>
      </div>
      
      <div class="study-progress">
        <div class="progress-item">
          <div class="progress-label">今日学习时长</div>
          <div class="progress-value">{{ todayStudyTime }}小时</div>
        </div>
        <div class="progress-item">
          <div class="progress-label">本周完成作业</div>
          <div class="progress-value">{{ weeklyTasks }}/{{ totalWeeklyTasks }}</div>
        </div>
      </div>
    </div>

    <div class="dashboard-content">
      <!-- 统计卡片 -->
      <div class="stats-grid">
        <div class="stat-card pending-card">
          <div class="stat-icon">
            <ClockCircleOutlined />
          </div>
          <div class="stat-info">
            <div class="stat-number">{{ stats.pendingTasks }}</div>
            <div class="stat-label">待完成作业</div>
          </div>
          <div class="stat-action">
            <a-button type="link" @click="$router.push('/student/tasks?status=pending')">
              立即完成
            </a-button>
          </div>
        </div>
        
        <div class="stat-card completed-card">
          <div class="stat-icon">
            <CheckCircleOutlined />
          </div>
          <div class="stat-info">
            <div class="stat-number">{{ stats.completedTasks }}</div>
            <div class="stat-label">已完成作业</div>
          </div>
        </div>
        
        <div class="stat-card grade-card">
          <div class="stat-icon">
            <TrophyOutlined />
          </div>
          <div class="stat-info">
            <div class="stat-number">{{ stats.averageGrade }}</div>
            <div class="stat-label">平均成绩</div>
          </div>
        </div>
        
        <div class="stat-card resource-card">
          <div class="stat-icon">
            <BookOutlined />
          </div>
          <div class="stat-info">
            <div class="stat-number">{{ stats.studyResources }}</div>
            <div class="stat-label">学习资源</div>
          </div>
        </div>
      </div>

      <!-- 主要内容区域 -->
      <div class="main-content">
        <!-- 左侧内容 -->
        <div class="left-content">
          <!-- 待完成作业 -->
          <div class="content-card">
            <div class="card-header">
              <h3>
                <ClockCircleOutlined />
                待完成作业
              </h3>
              <a-button type="link" @click="$router.push('/student/tasks')">
                查看全部
                <ArrowRightOutlined />
              </a-button>
            </div>
            
            <div class="task-list">
              <div 
                v-for="task in pendingTasks" 
                :key="task.id"
                class="task-item"
                :class="{ urgent: task.isUrgent }"
                @click="$router.push(`/student/tasks/${task.id}`)"
              >
                <div class="task-priority">
                  <div class="priority-dot" :class="task.priority"></div>
                </div>
                <div class="task-info">
                  <div class="task-title">{{ task.title }}</div>
                  <div class="task-meta">
                    {{ task.subject }} · {{ task.teacher }}
                  </div>
                  <div class="task-deadline">
                    <ClockCircleOutlined />
                    截止：{{ formatDeadline(task.deadline) }}
                  </div>
                </div>
                <div class="task-action">
                  <a-button type="primary" size="small">
                    开始作业
                  </a-button>
                </div>
              </div>
              
              <div v-if="pendingTasks.length === 0" class="empty-state">
                <CheckCircleOutlined />
                <p>太棒了！暂时没有待完成的作业</p>
              </div>
            </div>
          </div>
          
          <!-- 最近成绩 -->
          <div class="content-card">
            <div class="card-header">
              <h3>
                <BarChartOutlined />
                最近成绩
              </h3>
              <a-button type="link" @click="$router.push('/student/grades')">
                查看详情
                <ArrowRightOutlined />
              </a-button>
            </div>
            
            <div class="grade-list">
              <div 
                v-for="grade in recentGrades" 
                :key="grade.id"
                class="grade-item"
              >
                <div class="grade-subject">
                  <div class="subject-icon" :style="{ background: grade.color }">
                    {{ grade.subject.charAt(0) }}
                  </div>
                  <div class="subject-info">
                    <div class="subject-name">{{ grade.subject }}</div>
                    <div class="assignment-name">{{ grade.assignment }}</div>
                  </div>
                </div>
                <div class="grade-score">
                  <div class="score" :class="getGradeClass(grade.score)">{{ grade.score }}</div>
                  <div class="score-date">{{ formatDate(grade.date) }}</div>
                </div>
              </div>
            </div>
          </div>
        </div>
        
        <!-- 右侧内容 -->
        <div class="right-content">
          <!-- 学习计划 -->
          <div class="content-card">
            <div class="card-header">
              <h3>
                <CalendarOutlined />
                今日学习计划
              </h3>
              <a-button type="link" size="small">
                <SettingOutlined />
              </a-button>
            </div>
            
            <div class="study-plan">
              <div 
                v-for="plan in todayPlans" 
                :key="plan.id"
                class="plan-item"
                :class="{ completed: plan.completed }"
              >
                <a-checkbox 
                  v-model:checked="plan.completed"
                  @change="updatePlan(plan)"
                />
                <div class="plan-content">
                  <div class="plan-title">{{ plan.title }}</div>
                  <div class="plan-time">{{ plan.time }}</div>
                </div>
              </div>
            </div>
            
            <div class="plan-progress">
              <div class="progress-label">
                今日完成度：{{ completedPlansCount }}/{{ todayPlans.length }}
              </div>
              <a-progress 
                :percent="Math.round((completedPlansCount / todayPlans.length) * 100)" 
                size="small"
              />
            </div>
          </div>
          
          <!-- 学习资源推荐 -->
          <div class="content-card">
            <div class="card-header">
              <h3>
                <BookOutlined />
                推荐资源
              </h3>
            </div>
            
            <div class="resource-list">
              <div 
                v-for="resource in recommendedResources" 
                :key="resource.id"
                class="resource-item"
                @click="openResource(resource)"
              >
                <div class="resource-icon">
                  <component :is="resource.icon" />
                </div>
                <div class="resource-info">
                  <div class="resource-title">{{ resource.title }}</div>
                  <div class="resource-desc">{{ resource.description }}</div>
                </div>
                <div class="resource-badge">
                  <a-tag :color="resource.type === 'video' ? 'blue' : 'green'">
                    {{ resource.type === 'video' ? '视频' : '文档' }}
                  </a-tag>
                </div>
              </div>
            </div>
          </div>
          
          <!-- AI学习助手 -->
          <div class="content-card ai-card">
            <div class="card-header">
              <h3>
                <RobotOutlined />
                AI学习助手
              </h3>
            </div>
            
            <div class="ai-features">
              <div class="ai-feature" @click="openAIFeature('analysis')">
                <div class="feature-icon">
                  <BarChartOutlined />
                </div>
                <div class="feature-content">
                  <div class="feature-title">学情分析</div>
                  <div class="feature-desc">分析学习情况，提供改进建议</div>
                </div>
              </div>
              
              <div class="ai-feature" @click="openAIFeature('tutor')">
                <div class="feature-icon">
                  <BulbOutlined />
                </div>
                <div class="feature-content">
                  <div class="feature-title">智能答疑</div>
                  <div class="feature-desc">24小时在线解答学习问题</div>
                </div>
              </div>
            </div>
            
            <a-button type="primary" block class="ai-action-btn">
              开启AI助手
            </a-button>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, reactive, computed, onMounted } from 'vue'
import { message } from 'ant-design-vue'
import {
  ClockCircleOutlined,
  CheckCircleOutlined,
  TrophyOutlined,
  BookOutlined,
  ArrowRightOutlined,
  BarChartOutlined,
  CalendarOutlined,
  SettingOutlined,
  RobotOutlined,
  BulbOutlined,
  PlayCircleOutlined,
  FileTextOutlined
} from '@ant-design/icons-vue'
import { useAuthStore } from '@/stores/auth'

const authStore = useAuthStore()

// 当前日期
const currentDate = computed(() => {
  return new Date().toLocaleDateString('zh-CN', {
    year: 'numeric',
    month: 'long',
    day: 'numeric',
    weekday: 'long'
  })
})

// 学习进度数据
const todayStudyTime = ref(2.5)
const weeklyTasks = ref(8)
const totalWeeklyTasks = ref(10)

// 统计数据
const stats = reactive({
  pendingTasks: 3,
  completedTasks: 15,
  averageGrade: 85,
  studyResources: 24
})

// 待完成作业
const pendingTasks = ref([
  {
    id: 1,
    title: '数学函数练习题',
    subject: '数学',
    teacher: '张老师',
    deadline: new Date('2024-01-20T23:59:59'),
    priority: 'high',
    isUrgent: true
  },
  {
    id: 2,
    title: '英语阅读理解',
    subject: '英语',
    teacher: '李老师',
    deadline: new Date('2024-01-22T23:59:59'),
    priority: 'medium',
    isUrgent: false
  },
  {
    id: 3,
    title: '物理实验报告',
    subject: '物理',
    teacher: '王老师',
    deadline: new Date('2024-01-25T23:59:59'),
    priority: 'low',
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
    date: new Date('2024-01-15'),
    color: '#1890ff'
  },
  {
    id: 2,
    subject: '英语',
    assignment: '单元测试',
    score: 88,
    date: new Date('2024-01-12'),
    color: '#52c41a'
  },
  {
    id: 3,
    subject: '物理',
    assignment: '实验报告',
    score: 85,
    date: new Date('2024-01-10'),
    color: '#722ed1'
  }
])

// 今日学习计划
const todayPlans = ref([
  {
    id: 1,
    title: '复习数学函数',
    time: '09:00-10:00',
    completed: true
  },
  {
    id: 2,
    title: '完成英语作业',
    time: '14:00-15:00',
    completed: false
  },
  {
    id: 3,
    title: '预习物理新课',
    time: '19:00-20:00',
    completed: false
  },
  {
    id: 4,
    title: '整理笔记',
    time: '20:00-20:30',
    completed: false
  }
])

// 推荐资源
const recommendedResources = ref([
  {
    id: 1,
    title: '函数图像变换',
    description: '详细讲解函数图像的平移和伸缩',
    type: 'video',
    icon: 'PlayCircleOutlined'
  },
  {
    id: 2,
    title: '英语语法总结',
    description: '高中英语语法知识点汇总',
    type: 'document',
    icon: 'FileTextOutlined'
  },
  {
    id: 3,
    title: '物理实验指导',
    description: '实验操作步骤和注意事项',
    type: 'document',
    icon: 'FileTextOutlined'
  }
])

// 计算已完成计划数量
const completedPlansCount = computed(() => {
  return todayPlans.value.filter(plan => plan.completed).length
})

// 格式化截止时间
const formatDeadline = (deadline: Date) => {
  const now = new Date()
  const diff = deadline.getTime() - now.getTime()
  const days = Math.ceil(diff / (1000 * 60 * 60 * 24))
  
  if (days < 0) {
    return '已过期'
  } else if (days === 0) {
    return '今天截止'
  } else if (days === 1) {
    return '明天截止'
  } else {
    return `${days}天后截止`
  }
}

// 格式化日期
const formatDate = (date: Date) => {
  return date.toLocaleDateString('zh-CN', {
    month: 'short',
    day: 'numeric'
  })
}

// 获取成绩等级样式
const getGradeClass = (score: number) => {
  if (score >= 90) return 'excellent'
  if (score >= 80) return 'good'
  if (score >= 70) return 'average'
  return 'poor'
}

// 更新学习计划
const updatePlan = (plan: any) => {
  message.success(plan.completed ? '计划已完成' : '计划已标记为未完成')
}

// 打开资源
const openResource = (resource: any) => {
  message.info(`正在打开：${resource.title}`)
}

// 打开AI功能
const openAIFeature = (feature: string) => {
  if (feature === 'analysis') {
    message.info('正在生成学情分析报告...')
  } else if (feature === 'tutor') {
    message.info('正在启动智能答疑助手...')
  }
}

onMounted(() => {
  // 加载数据
})
</script>

<style scoped>
.student-dashboard {
  padding: 24px;
  background: #f5f5f5;
  min-height: 100vh;
}

.dashboard-header {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  margin-bottom: 32px;
  background: white;
  padding: 32px;
  border-radius: 16px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);
}

.welcome-section {
  flex: 1;
}

.welcome-title {
  font-size: 32px;
  font-weight: 700;
  color: #333;
  margin: 0 0 8px 0;
}

.welcome-subtitle {
  font-size: 16px;
  color: #666;
  margin: 0;
}

.study-progress {
  display: flex;
  gap: 32px;
}

.progress-item {
  text-align: center;
}

.progress-label {
  font-size: 14px;
  color: #666;
  margin-bottom: 8px;
}

.progress-value {
  font-size: 24px;
  font-weight: 700;
  color: #1890ff;
}

.stats-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
  gap: 24px;
  margin-bottom: 32px;
}

.stat-card {
  background: white;
  padding: 24px;
  border-radius: 16px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);
  display: flex;
  align-items: center;
  gap: 16px;
  transition: transform 0.2s ease;
  position: relative;
  overflow: hidden;
}

.stat-card:hover {
  transform: translateY(-2px);
}

.stat-card::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  height: 4px;
}

.pending-card::before {
  background: #ff7875;
}

.completed-card::before {
  background: #52c41a;
}

.grade-card::before {
  background: #faad14;
}

.resource-card::before {
  background: #1890ff;
}

.stat-icon {
  width: 56px;
  height: 56px;
  border-radius: 12px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 24px;
  color: white;
}

.pending-card .stat-icon {
  background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%);
}

.completed-card .stat-icon {
  background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
}

.grade-card .stat-icon {
  background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
}

.resource-card .stat-icon {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
}

.stat-info {
  flex: 1;
}

.stat-number {
  font-size: 32px;
  font-weight: 700;
  color: #333;
  line-height: 1;
  margin-bottom: 4px;
}

.stat-label {
  font-size: 14px;
  color: #666;
}

.stat-action {
  margin-left: auto;
}

.main-content {
  display: grid;
  grid-template-columns: 2fr 1fr;
  gap: 24px;
}

.content-card {
  background: white;
  border-radius: 16px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);
  margin-bottom: 24px;
  overflow: hidden;
}

.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 24px 24px 0 24px;
  margin-bottom: 20px;
}

.card-header h3 {
  font-size: 18px;
  font-weight: 600;
  color: #333;
  margin: 0;
  display: flex;
  align-items: center;
  gap: 8px;
}

.task-list,
.grade-list {
  padding: 0 24px 24px 24px;
}

.task-item {
  display: flex;
  align-items: center;
  gap: 16px;
  padding: 16px;
  border-radius: 12px;
  cursor: pointer;
  transition: all 0.2s ease;
  margin-bottom: 12px;
  border: 1px solid #f0f0f0;
}

.task-item:hover {
  background: #f8f9fa;
  border-color: #1890ff;
}

.task-item.urgent {
  border-color: #ff7875;
  background: #fff2f0;
}

.task-priority {
  display: flex;
  align-items: center;
}

.priority-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
}

.priority-dot.high {
  background: #ff7875;
}

.priority-dot.medium {
  background: #faad14;
}

.priority-dot.low {
  background: #52c41a;
}

.task-info {
  flex: 1;
}

.task-title {
  font-size: 16px;
  font-weight: 600;
  color: #333;
  margin-bottom: 4px;
}

.task-meta {
  font-size: 14px;
  color: #666;
  margin-bottom: 4px;
}

.task-deadline {
  font-size: 12px;
  color: #999;
  display: flex;
  align-items: center;
  gap: 4px;
}

.empty-state {
  text-align: center;
  padding: 40px 20px;
  color: #999;
}

.empty-state .anticon {
  font-size: 48px;
  margin-bottom: 16px;
  color: #52c41a;
}

.grade-item {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 16px;
  border-radius: 12px;
  background: #fafafa;
  margin-bottom: 12px;
}

.grade-subject {
  display: flex;
  align-items: center;
  gap: 12px;
}

.subject-icon {
  width: 40px;
  height: 40px;
  border-radius: 8px;
  display: flex;
  align-items: center;
  justify-content: center;
  color: white;
  font-weight: 600;
}

.subject-name {
  font-size: 16px;
  font-weight: 600;
  color: #333;
}

.assignment-name {
  font-size: 14px;
  color: #666;
}

.grade-score {
  text-align: right;
}

.score {
  font-size: 24px;
  font-weight: 700;
  margin-bottom: 4px;
}

.score.excellent {
  color: #52c41a;
}

.score.good {
  color: #1890ff;
}

.score.average {
  color: #faad14;
}

.score.poor {
  color: #ff7875;
}

.score-date {
  font-size: 12px;
  color: #999;
}

.study-plan {
  padding: 0 24px;
}

.plan-item {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 12px 0;
  border-bottom: 1px solid #f0f0f0;
}

.plan-item:last-child {
  border-bottom: none;
}

.plan-item.completed .plan-title {
  text-decoration: line-through;
  color: #999;
}

.plan-content {
  flex: 1;
}

.plan-title {
  font-size: 14px;
  color: #333;
  margin-bottom: 4px;
}

.plan-time {
  font-size: 12px;
  color: #999;
}

.plan-progress {
  padding: 16px 24px 24px 24px;
  border-top: 1px solid #f0f0f0;
}

.progress-label {
  font-size: 14px;
  color: #666;
  margin-bottom: 8px;
}

.resource-list {
  padding: 0 24px 24px 24px;
}

.resource-item {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 16px;
  border-radius: 12px;
  cursor: pointer;
  transition: background-color 0.2s ease;
  margin-bottom: 8px;
}

.resource-item:hover {
  background: #f8f9fa;
}

.resource-icon {
  width: 40px;
  height: 40px;
  border-radius: 8px;
  background: #f0f8ff;
  display: flex;
  align-items: center;
  justify-content: center;
  color: #1890ff;
  font-size: 18px;
}

.resource-info {
  flex: 1;
}

.resource-title {
  font-size: 14px;
  font-weight: 600;
  color: #333;
  margin-bottom: 4px;
}

.resource-desc {
  font-size: 12px;
  color: #666;
  line-height: 1.4;
}

.ai-card {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
}

.ai-card .card-header h3 {
  color: white;
}

.ai-features {
  padding: 0 24px;
}

.ai-feature {
  display: flex;
  gap: 12px;
  padding: 16px;
  background: rgba(255, 255, 255, 0.1);
  border-radius: 12px;
  margin-bottom: 12px;
  cursor: pointer;
  transition: background-color 0.2s ease;
}

.ai-feature:hover {
  background: rgba(255, 255, 255, 0.2);
}

.feature-icon {
  width: 40px;
  height: 40px;
  border-radius: 8px;
  background: rgba(255, 255, 255, 0.2);
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 18px;
}

.feature-title {
  font-size: 14px;
  font-weight: 600;
  margin-bottom: 4px;
}

.feature-desc {
  font-size: 12px;
  opacity: 0.8;
  line-height: 1.4;
}

.ai-action-btn {
  margin: 0 24px 24px 24px;
  background: rgba(255, 255, 255, 0.2);
  border: 1px solid rgba(255, 255, 255, 0.3);
  color: white;
}

.ai-action-btn:hover {
  background: rgba(255, 255, 255, 0.3);
  border-color: rgba(255, 255, 255, 0.5);
}

@media (max-width: 1200px) {
  .main-content {
    grid-template-columns: 1fr;
  }
}


</style>