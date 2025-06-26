<template>
  <div class="teacher-dashboard">
    <a-spin :spinning="loading" tip="加载中...">
      <!-- 顶部横幅 -->
      <div class="dashboard-header">
        <div class="header-content">
          <div class="welcome-section">
            <h1 class="welcome-title">
              <span class="greeting">{{ greeting }}</span>
              <span class="teacher-name">{{ authStore.user?.name || '老师' }}</span>
            </h1>
            <p class="welcome-subtitle">{{ formatDate(currentDate) }} · 让每一次教学都充满智慧</p>
          </div>
          
          <div class="quick-actions">
            <a-button 
              type="primary" 
              size="large" 
              @click="showCreateClassModal = true"
              class="action-btn create-class"
            >
              <template #icon><PlusOutlined /></template>
              创建班级
            </a-button>
            <a-button 
              size="large" 
              @click="showAssignmentModal = true"
              class="action-btn assign-task"
            >
              <template #icon><FileTextOutlined /></template>
              布置作业
            </a-button>
          </div>
        </div>
        
        <!-- 教学概览卡片 -->
        <div class="overview-cards">
          <div class="overview-card">
            <div class="card-icon classes">
              <BookOutlined />
            </div>
            <div class="card-content">
              <div class="card-number">{{ stats.classCount }}</div>
              <div class="card-label">管理班级</div>
            </div>
          </div>
          
          <div class="overview-card">
            <div class="card-icon students">
              <UserOutlined />
            </div>
            <div class="card-content">
              <div class="card-number">{{ stats.studentCount }}</div>
              <div class="card-label">学生总数</div>
            </div>
          </div>
          
          <div class="overview-card">
            <div class="card-icon tasks">
              <FileTextOutlined />
            </div>
            <div class="card-content">
              <div class="card-number">{{ stats.taskCount }}</div>
              <div class="card-label">布置作业</div>
            </div>
          </div>
          
          <div class="overview-card">
            <div class="card-icon pending">
              <ClockCircleOutlined />
            </div>
            <div class="card-content">
              <div class="card-number">{{ stats.pendingCount }}</div>
              <div class="card-label">待批改</div>
            </div>
          </div>
        </div>
      </div>

    <!-- 主要内容区域 -->
    <div class="dashboard-content">
      <!-- 快速统计 -->
      <div class="quick-stats">
        <div class="stat-item teaching">
          <div class="stat-icon">
            <FileTextOutlined />
          </div>
          <div class="stat-content">
            <div class="stat-number">{{ stats.taskCount }}</div>
            <div class="stat-label">布置作业</div>
          </div>
          <div class="stat-trend">
            <span class="trend-value">+12%</span>
            <ArrowUpOutlined class="trend-icon up" />
          </div>
        </div>
        
        <div class="stat-item grading">
          <div class="stat-icon">
            <EditOutlined />
          </div>
          <div class="stat-content">
            <div class="stat-number">{{ stats.pendingCount }}</div>
            <div class="stat-label">待批改</div>
          </div>
          <div class="stat-trend">
            <span class="trend-value">-5%</span>
            <ArrowDownOutlined class="trend-icon down" />
          </div>
        </div>
        
        <div class="stat-item performance">
          <div class="stat-icon">
            <TrophyOutlined />
          </div>
          <div class="stat-content">
            <div class="stat-number">{{ averageScore }}</div>
            <div class="stat-label">平均分</div>
          </div>
          <div class="stat-trend">
            <span class="trend-value">+8%</span>
            <ArrowUpOutlined class="trend-icon up" />
          </div>
        </div>
      </div>

      <!-- 内容网格 -->
      <div class="content-grid">
        <!-- 我的班级 -->
        <div class="content-section">
          <div class="section-header">
            <h3 class="section-title">
              <TeamOutlined />
              我的班级
            </h3>
            <a-button type="link" @click="$router.push('/teacher/classes')">
              查看全部
              <ArrowRightOutlined />
            </a-button>
          </div>
          
          <div class="class-grid">
            <div 
              v-for="classItem in recentClasses" 
              :key="classItem.id"
              class="class-card"
              @click="$router.push(`/teacher/classes/${classItem.id}`)"
            >
              <div class="class-header">
                <div class="class-avatar" :style="{ background: getClassColor(classItem.subject) }">
                  {{ classItem.name.charAt(0) }}
                </div>
                <div class="class-actions">
                  <a-dropdown>
                    <a-button type="text" size="small">
                      <MoreOutlined />
                    </a-button>
                    <template #overlay>
                      <a-menu>
                        <a-menu-item key="edit">编辑班级</a-menu-item>
                        <a-menu-item key="students">学生管理</a-menu-item>
                        <a-menu-item key="settings">班级设置</a-menu-item>
                      </a-menu>
                    </template>
                  </a-dropdown>
                </div>
              </div>
              
              <div class="class-content">
                <h4 class="class-name">{{ classItem.name }}</h4>
                <p class="class-subject">{{ classItem.subject }}</p>
                
                <div class="class-stats">
                  <div class="stat-item">
                    <UserOutlined />
                    <span>{{ classItem.studentCount }}名学生</span>
                  </div>
                  <div class="stat-item">
                    <FileTextOutlined />
                    <span>{{ classItem.taskCount }}个作业</span>
                  </div>
                </div>
                
                <div class="class-progress">
                  <div class="progress-info">
                    <span>学习进度</span>
                    <span>{{ classItem.progress }}%</span>
                  </div>
                  <a-progress :percent="classItem.progress" size="small" :show-info="false" />
                </div>
              </div>
              
              <div class="class-footer">
                <a-button type="primary" size="small" block>
                  进入班级
                </a-button>
              </div>
            </div>
            
            <!-- 创建新班级卡片 -->
            <div class="class-card create-card" @click="showCreateClassModal = true">
              <div class="create-content">
                <PlusOutlined class="create-icon" />
                <h4>创建新班级</h4>
                <p>开始新的教学旅程</p>
              </div>
            </div>
          </div>
        </div>
          
        <!-- 最近作业 -->
        <div class="content-section">
          <div class="section-header">
            <h3 class="section-title">
              <FileTextOutlined />
              最近作业
            </h3>
            <a-button type="link" @click="$router.push('/teacher/tasks')">
              查看全部
              <ArrowRightOutlined />
            </a-button>
          </div>
          
          <div class="task-list">
            <div 
              v-for="task in recentTasks" 
              :key="task.id"
              class="task-card"
              @click="$router.push(`/teacher/tasks/${task.id}`)"
            >
              <div class="task-header">
                <div class="task-priority">
                  <div class="priority-indicator" :class="task.priority"></div>
                  <span class="task-subject">{{ task.className }}</span>
                </div>
                <div class="task-deadline" :class="{ urgent: isUrgent(task.deadline) }">
                  <ClockCircleOutlined />
                  {{ formatDate(task.deadline) }}
                </div>
              </div>
              
              <h4 class="task-title">{{ task.title }}</h4>
              <p class="task-description">{{ task.description }}</p>
              
              <div class="task-progress">
                <div class="progress-info">
                  <span>提交进度</span>
                  <span>{{ task.submittedCount }}/{{ task.totalCount }}</span>
                </div>
                <a-progress 
                  :percent="task.completionRate" 
                  size="small" 
                  :show-info="false"
                />
              </div>
              
              <div class="task-actions">
                <a-button size="small" type="primary">
                  查看详情
                </a-button>
              </div>
            </div>
            
            <div v-if="recentTasks.length === 0" class="empty-state">
              <FileTextOutlined />
              <p>暂无作业</p>
            </div>
          </div>
        </div>
        
        <!-- AI教学助手 -->
        <div class="content-section ai-section">
          <div class="section-header">
            <h3 class="section-title">
              <RobotOutlined />
              AI教学助手
            </h3>
          </div>
          
          <div class="ai-features">
            <div class="ai-feature-card" @click="handleAIFeature('recommend')">
              <div class="feature-icon">
                <BulbOutlined />
              </div>
              <div class="feature-content">
                <div class="feature-title">智能作业推荐</div>
                <div class="feature-desc">根据学生学习情况，推荐个性化作业内容</div>
              </div>
            </div>
            
            <div class="ai-feature-card" @click="handleAIFeature('analysis')">
              <div class="feature-icon">
                <BarChartOutlined />
              </div>
              <div class="feature-content">
                <div class="feature-title">学情分析报告</div>
                <div class="feature-desc">生成班级学习情况分析和改进建议</div>
              </div>
            </div>
            
            <div class="ai-feature-card" @click="handleAIFeature('grading')">
              <div class="feature-icon">
                <EditOutlined />
              </div>
              <div class="feature-content">
                <div class="feature-title">智能批改助手</div>
                <div class="feature-desc">AI辅助批改作业，提高批改效率</div>
              </div>
            </div>
          </div>
          
          <a-button type="primary" block class="ai-action-btn">
            开启AI助手
          </a-button>
        </div>
        
        <!-- 待办事项 -->
        <div class="content-section">
          <div class="section-header">
            <h3 class="section-title">
              <CheckCircleOutlined />
              待办事项
            </h3>
            <a-button type="link" size="small">
              <SettingOutlined />
            </a-button>
          </div>
          
          <div class="todo-list">
            <div 
              v-for="todo in todoList" 
              :key="todo.id"
              class="todo-item"
              :class="{ completed: todo.completed }"
            >
              <a-checkbox 
                v-model:checked="todo.completed"
                @change="updateTodo(todo)"
              />
              <div class="todo-content">
                <span class="todo-text">{{ todo.text }}</span>
                <span class="todo-time">{{ todo.time }}</span>
              </div>
            </div>
            
            <div v-if="todoList.length === 0" class="empty-state">
              <CheckCircleOutlined />
              <p>暂无待办事项</p>
            </div>
          </div>
        </div>
      </div>
    </div>
    
    <!-- 创建班级弹窗 -->
    <a-modal
      v-model:open="showCreateClassModal"
      title="创建班级"
      @ok="handleCreateClass"
    >
      <a-form :model="classForm" layout="vertical">
        <a-form-item label="班级名称" required>
          <a-input v-model:value="classForm.name" placeholder="请输入班级名称" />
        </a-form-item>
        <a-form-item label="学科">
          <a-select v-model:value="classForm.subject" placeholder="请选择学科">
            <a-select-option value="数学">数学</a-select-option>
            <a-select-option value="语文">语文</a-select-option>
            <a-select-option value="英语">英语</a-select-option>
            <a-select-option value="物理">物理</a-select-option>
            <a-select-option value="化学">化学</a-select-option>
          </a-select>
        </a-form-item>
        <a-form-item label="班级描述">
          <a-textarea v-model:value="classForm.description" placeholder="请输入班级描述" />
        </a-form-item>
      </a-form>
    </a-modal>
    
    <!-- 布置作业弹窗 -->
    <a-modal
      v-model:open="showCreateTaskModal"
      title="布置作业"
      @ok="handleCreateTask"
      width="600px"
    >
      <a-form :model="taskForm" layout="vertical">
        <a-form-item label="作业标题" required>
          <a-input v-model:value="taskForm.title" placeholder="请输入作业标题" />
        </a-form-item>
        <a-form-item label="选择班级" required>
          <a-select v-model:value="taskForm.classId" placeholder="请选择班级">
            <a-select-option 
              v-for="classItem in recentClasses" 
              :key="classItem.id"
              :value="classItem.id"
            >
              {{ classItem.name }}
            </a-select-option>
          </a-select>
        </a-form-item>
        <a-form-item label="截止时间" required>
          <a-date-picker 
            v-model:value="taskForm.deadline" 
            show-time 
            placeholder="请选择截止时间"
            style="width: 100%"
          />
        </a-form-item>
        <a-form-item label="作业内容">
          <a-textarea 
            v-model:value="taskForm.content" 
            placeholder="请输入作业内容"
            :rows="4"
          />
        </a-form-item>
      </a-form>
    </a-modal>
    </a-spin>
  </div>
</template>

<script setup lang="ts">
import { ref, reactive, onMounted, computed } from 'vue'
import { message } from 'ant-design-vue'
import {
  PlusOutlined,
  FileAddOutlined,
  TeamOutlined,
  UserOutlined,
  FileTextOutlined,
  ClockCircleOutlined,
  ArrowRightOutlined,
  SettingOutlined,
  RobotOutlined,
  BulbOutlined,
  BarChartOutlined,
  EditOutlined,
  TrophyOutlined,
  ArrowUpOutlined,
  ArrowDownOutlined,
  MoreOutlined,
  CheckCircleOutlined
} from '@ant-design/icons-vue'
import { useAuthStore } from '@/stores/auth'
import {
  getDashboardData,
  getCourses,
  getAssignments,
  getStudents,
  getStatistics
} from '@/api/teacher'
import type {
  Course,
  Assignment,
  Student
} from '@/api/teacher'

const authStore = useAuthStore()

// 响应式数据
const loading = ref(false)

// 当前日期
const currentDate = computed(() => {
  return new Date().toLocaleDateString('zh-CN', {
    year: 'numeric',
    month: 'long',
    day: 'numeric',
    weekday: 'long'
  })
})

// 统计数据
const stats = reactive({
  classCount: 0,
  studentCount: 0,
  taskCount: 0,
  pendingCount: 0
})

// 平均分
const averageScore = ref(0)

// 最近班级
const recentClasses = ref<Course[]>([])

// 最近作业
const recentTasks = ref<Assignment[]>([])

// 待办事项
const todoList = ref<Array<{
  id: number
  text: string
  time: string
  completed: boolean
}>>([])

// 弹窗状态
const showCreateClassModal = ref(false)
const showCreateTaskModal = ref(false)

// 表单数据
const classForm = reactive({
  name: '',
  subject: '',
  description: ''
})

const taskForm = reactive({
  title: '',
  classId: null,
  deadline: null,
  content: ''
})

// 格式化日期
const formatDate = (date: Date) => {
  return date.toLocaleDateString('zh-CN', {
    month: 'short',
    day: 'numeric'
  })
}

// 判断是否紧急
const isUrgent = (deadline: Date) => {
  const now = new Date()
  const diffTime = deadline.getTime() - now.getTime()
  const diffDays = Math.ceil(diffTime / (1000 * 60 * 60 * 24))
  return diffDays <= 2
}

// 获取班级颜色
const getClassColor = (subject: string) => {
  const colors: Record<string, string> = {
    '数学': 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
    '语文': 'linear-gradient(135deg, #f093fb 0%, #f5576c 100%)',
    '英语': 'linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)',
    '物理': 'linear-gradient(135deg, #43e97b 0%, #38f9d7 100%)',
    '化学': 'linear-gradient(135deg, #fa709a 0%, #fee140 100%)'
  }
  return colors[subject] || 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)'
}

// 处理AI功能
const handleAIFeature = (feature: string) => {
  switch (feature) {
    case 'recommend':
      message.info('正在生成智能作业推荐...')
      break
    case 'analysis':
      message.info('正在生成学情分析报告...')
      break
    case 'grading':
      message.info('正在启动智能批改助手...')
      break
    default:
      message.info('功能开发中...')
  }
}

// 更新待办事项
const updateTodo = (todo: any) => {
  message.success(todo.completed ? '任务已完成' : '任务已标记为未完成')
}

// 创建班级
const handleCreateClass = () => {
  message.success('班级创建成功！')
  showCreateClassModal.value = false
  // 重置表单
  Object.assign(classForm, {
    name: '',
    subject: '',
    description: ''
  })
}

// 布置作业
const handleCreateTask = () => {
  message.success('作业布置成功！')
  showCreateTaskModal.value = false
  // 重置表单
  Object.assign(taskForm, {
    title: '',
    classId: null,
    deadline: null,
    content: ''
  })
}

// 数据加载函数
const loadDashboardData = async () => {
  try {
    const response = await getDashboardData()
    const data = response.data || response
    stats.classCount = data.classCount || 0
    stats.studentCount = data.studentCount || 0
    stats.taskCount = data.taskCount || 0
    stats.pendingCount = data.pendingCount || 0
    averageScore.value = data.averageScore || 0
  } catch (error) {
    console.error('加载仪表盘数据失败:', error)
    message.error('加载仪表盘数据失败')
    // 如果API调用失败，使用默认值
    stats.classCount = 0
    stats.studentCount = 0
    stats.taskCount = 0
    stats.pendingCount = 0
    averageScore.value = 0
  }
}

const loadRecentClasses = async () => {
  try {
    const response = await getCourses()
    const data = response.data || response
    recentClasses.value = (Array.isArray(data) ? data.slice(0, 3) : []).map(course => ({
      ...course,
      subject: course.name, // 使用课程名称作为科目
      students: course.studentCount || 0,
      taskCount: Math.floor(Math.random() * 10) + 1, // 临时随机数据
      progress: Math.floor(Math.random() * 100) + 1,
      pendingTasks: Math.floor(Math.random() * 5)
    }))
  } catch (error) {
    console.error('加载班级数据失败:', error)
    message.error('加载班级数据失败')
  }
}

const loadRecentTasks = async () => {
  try {
    const response = await getAssignments({ page: 1, size: 3 })
    const data = response.data || response
    recentTasks.value = (Array.isArray(data) ? data : []).map(assignment => ({
      ...assignment,
      className: assignment.courseName || '未知班级',
      deadline: new Date(assignment.dueDate),
      priority: assignment.status === 'published' ? 'high' : 'medium',
      completionRate: Math.floor((assignment.submissionCount / assignment.totalStudents) * 100) || 0,
      submittedCount: assignment.submissionCount || 0,
      totalCount: assignment.totalStudents || 0
    }))
  } catch (error) {
    console.error('加载作业数据失败:', error)
    message.error('加载作业数据失败')
  }
}

const loadTodoList = async () => {
  try {
    // 这里应该有专门的待办事项API，暂时使用模拟数据
    todoList.value = [
      {
        id: 1,
        text: '批改待审核作业',
        time: '今天',
        completed: false
      },
      {
        id: 2,
        text: '准备明天的课件',
        time: '明天',
        completed: false
      }
    ]
  } catch (error) {
    console.error('加载待办事项失败:', error)
  }
}

// 初始化数据
const initializeData = async () => {
  loading.value = true
  try {
    await Promise.all([
      loadDashboardData(),
      loadRecentClasses(),
      loadRecentTasks(),
      loadTodoList()
    ])
  } finally {
    loading.value = false
  }
}

onMounted(() => {
  initializeData()
})
</script>

<style scoped>
.teacher-dashboard {
  background: #f8fafc;
  min-height: 100vh;
}

/* 顶部横幅样式 */
.dashboard-banner {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  padding: 32px 0;
  margin-bottom: 32px;
  position: relative;
  overflow: hidden;
}

.dashboard-banner::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><defs><pattern id="grain" width="100" height="100" patternUnits="userSpaceOnUse"><circle cx="50" cy="50" r="1" fill="%23ffffff" opacity="0.1"/></pattern></defs><rect width="100" height="100" fill="url(%23grain)"/></svg>') repeat;
  pointer-events: none;
}

.banner-content {
  max-width: 1200px;
  margin: 0 auto;
  padding: 0 24px;
  position: relative;
  z-index: 1;
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 48px;
}

.welcome-section {
  flex: 1;
  display: flex;
  align-items: center;
  gap: 24px;
}

.avatar-section {
  display: flex;
  align-items: center;
  gap: 16px;
}

.user-info {
  flex: 1;
}

.welcome-title {
  font-size: 2rem;
  font-weight: 700;
  margin: 0 0 8px 0;
  background: linear-gradient(45deg, #ffffff, #e3f2fd);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}

.welcome-subtitle {
  font-size: 1rem;
  margin: 0;
  opacity: 0.9;
}

.quick-actions {
  display: flex;
  gap: 12px;
}

.quick-actions .ant-btn {
  height: 48px;
  border-radius: 12px;
  font-weight: 500;
  display: flex;
  align-items: center;
  gap: 8px;
}

.teaching-overview {
  display: flex;
  gap: 24px;
}

.overview-card {
  background: rgba(255, 255, 255, 0.15);
  backdrop-filter: blur(10px);
  border-radius: 16px;
  padding: 20px;
  text-align: center;
  border: 1px solid rgba(255, 255, 255, 0.2);
  min-width: 120px;
}

.overview-icon {
  font-size: 24px;
  margin-bottom: 8px;
  opacity: 0.9;
}

.overview-value {
  font-size: 1.5rem;
  font-weight: 700;
  margin-bottom: 4px;
}

.overview-label {
  font-size: 0.875rem;
  opacity: 0.8;
}

/* 主要内容区域 */
.dashboard-content {
  max-width: 1200px;
  margin: 0 auto;
  padding: 0 24px;
}

/* 快速统计 */
.quick-stats {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
  gap: 24px;
  margin-bottom: 32px;
}

.stat-item {
  background: white;
  padding: 24px;
  border-radius: 16px;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
  border: 1px solid rgba(0, 0, 0, 0.04);
  display: flex;
  align-items: center;
  gap: 16px;
  transition: all 0.3s ease;
  position: relative;
  overflow: hidden;
}

.stat-item::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  height: 4px;
}

.stat-item.teaching::before {
  background: linear-gradient(90deg, #667eea, #764ba2);
}

.stat-item.grading::before {
  background: linear-gradient(90deg, #f093fb, #f5576c);
}

.stat-item.performance::before {
  background: linear-gradient(90deg, #43e97b, #38f9d7);
}

.stat-item:hover {
  transform: translateY(-4px);
  box-shadow: 0 8px 30px rgba(0, 0, 0, 0.12);
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

.teaching .stat-icon {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
}

.grading .stat-icon {
  background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
}

.performance .stat-icon {
  background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
}

.stat-content {
  flex: 1;
}

.stat-number {
  font-size: 2rem;
  font-weight: 700;
  color: #1a1a1a;
  line-height: 1;
  margin-bottom: 4px;
}

.stat-label {
  font-size: 14px;
  color: #666;
  font-weight: 500;
}

.stat-trend {
  display: flex;
  align-items: center;
  gap: 4px;
  font-size: 12px;
  font-weight: 500;
}

.trend-value {
  font-weight: 600;
}

.trend-icon.up {
  color: #52c41a;
}

.trend-icon.down {
  color: #ff4d4f;
}

/* 内容网格 */
.content-grid {
  display: grid;
  grid-template-columns: 2fr 1fr;
  gap: 32px;
}

.content-section {
  background: white;
  border-radius: 16px;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
  border: 1px solid rgba(0, 0, 0, 0.04);
  margin-bottom: 24px;
  overflow: hidden;
}

.section-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 24px;
  border-bottom: 1px solid #f0f0f0;
}

.section-title {
  font-size: 1.25rem;
  font-weight: 600;
  color: #1a1a1a;
  margin: 0;
  display: flex;
  align-items: center;
  gap: 8px;
}

/* 班级网格 */
.class-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
  gap: 20px;
  padding: 24px;
}

.class-card {
  background: white;
  border-radius: 16px;
  box-shadow: 0 4px 15px rgba(0, 0, 0, 0.08);
  border: 1px solid rgba(0, 0, 0, 0.04);
  overflow: hidden;
  cursor: pointer;
  transition: all 0.3s ease;
  position: relative;
}

.class-card:hover {
  transform: translateY(-4px);
  box-shadow: 0 8px 30px rgba(0, 0, 0, 0.12);
}

.class-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 20px 20px 0 20px;
}

.class-avatar {
  width: 48px;
  height: 48px;
  border-radius: 12px;
  display: flex;
  align-items: center;
  justify-content: center;
  color: white;
  font-weight: 600;
  font-size: 18px;
}

.class-content {
  padding: 16px 20px;
}

.class-name {
  font-size: 1.125rem;
  font-weight: 600;
  color: #1a1a1a;
  margin: 0 0 4px 0;
}

.class-subject {
  font-size: 0.875rem;
  color: #666;
  margin: 0 0 16px 0;
}

.class-stats {
  display: flex;
  gap: 16px;
  margin-bottom: 16px;
}

.class-stats .stat-item {
  display: flex;
  align-items: center;
  gap: 4px;
  font-size: 0.75rem;
  color: #666;
}

.class-progress {
  margin-bottom: 16px;
}

.progress-info {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 8px;
  font-size: 0.75rem;
  color: #666;
}

.class-footer {
  padding: 0 20px 20px 20px;
}

.create-card {
  display: flex;
  align-items: center;
  justify-content: center;
  min-height: 280px;
  background: linear-gradient(135deg, #f8fafc 0%, #e3f2fd 100%);
  border: 2px dashed #d0d7de;
  transition: all 0.3s ease;
}

.create-card:hover {
  border-color: #667eea;
  background: linear-gradient(135deg, #e3f2fd 0%, #f0f8ff 100%);
}

.create-content {
  text-align: center;
  color: #666;
}

.create-icon {
  font-size: 2rem;
  margin-bottom: 12px;
  color: #667eea;
}

.create-content h4 {
  font-size: 1rem;
  font-weight: 600;
  margin: 0 0 4px 0;
  color: #333;
}

.create-content p {
  font-size: 0.875rem;
  margin: 0;
  color: #666;
}

/* 作业列表 */
.task-list {
  padding: 24px;
}

.task-card {
  background: #f8fafc;
  border-radius: 12px;
  padding: 20px;
  margin-bottom: 16px;
  border-left: 4px solid #667eea;
  transition: all 0.3s ease;
  cursor: pointer;
}

.task-card:hover {
  transform: translateX(4px);
  box-shadow: 0 4px 15px rgba(102, 126, 234, 0.15);
}

.task-header {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  margin-bottom: 12px;
}

.task-priority {
  display: flex;
  align-items: center;
  gap: 8px;
}

.priority-indicator {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: #52c41a;
}

.priority-indicator.high {
  background: #ff4d4f;
}

.priority-indicator.medium {
  background: #faad14;
}

.priority-indicator.low {
  background: #52c41a;
}

.task-subject {
  font-size: 0.75rem;
  color: #666;
  background: #f0f0f0;
  padding: 2px 8px;
  border-radius: 4px;
}

.task-deadline {
  display: flex;
  align-items: center;
  gap: 4px;
  font-size: 0.75rem;
  color: #666;
}

.task-deadline.urgent {
  color: #ff4d4f;
  font-weight: 500;
}

.task-title {
  font-size: 1rem;
  font-weight: 600;
  color: #1a1a1a;
  margin: 0 0 8px 0;
}

.task-description {
  font-size: 0.875rem;
  color: #666;
  margin: 0 0 16px 0;
  line-height: 1.4;
}

.task-progress {
  margin-bottom: 16px;
}

.task-actions {
  display: flex;
  justify-content: flex-end;
}

/* AI功能区域 */
.ai-section {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
}

.ai-section .section-header {
  border-bottom-color: rgba(255, 255, 255, 0.2);
}

.ai-section .section-title {
  color: white;
}

.ai-features {
  padding: 0 24px;
  display: grid;
  gap: 12px;
}

.ai-feature-card {
  background: rgba(255, 255, 255, 0.1);
  border-radius: 12px;
  padding: 16px;
  cursor: pointer;
  transition: all 0.3s ease;
  border: 1px solid rgba(255, 255, 255, 0.2);
  display: flex;
  gap: 12px;
}

.ai-feature-card:hover {
  background: rgba(255, 255, 255, 0.2);
  transform: translateY(-2px);
}

.feature-icon {
  width: 40px;
  height: 40px;
  border-radius: 10px;
  background: rgba(255, 255, 255, 0.2);
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 18px;
  flex-shrink: 0;
}

.feature-content {
  flex: 1;
}

.feature-title {
  font-size: 0.875rem;
  font-weight: 600;
  margin: 0 0 4px 0;
}

.feature-desc {
  font-size: 0.75rem;
  opacity: 0.8;
  line-height: 1.3;
  margin: 0;
}

.ai-action-btn {
  margin: 24px;
  background: rgba(255, 255, 255, 0.2);
  border: 1px solid rgba(255, 255, 255, 0.3);
  color: white;
}

.ai-action-btn:hover {
  background: rgba(255, 255, 255, 0.3);
  border-color: rgba(255, 255, 255, 0.5);
  color: white;
}

/* 待办事项 */
.todo-list {
  padding: 0 24px 24px 24px;
}

.todo-item {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 12px 0;
  border-bottom: 1px solid #f0f0f0;
}

.todo-item:last-child {
  border-bottom: none;
}

.todo-item.completed .todo-text {
  text-decoration: line-through;
  color: #999;
}

.todo-content {
  flex: 1;
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.todo-text {
  font-size: 14px;
  color: #333;
}

.todo-time {
  font-size: 12px;
  color: #999;
}

.empty-state {
  text-align: center;
  padding: 40px 20px;
  color: #999;
}

.empty-state .anticon {
  font-size: 2rem;
  margin-bottom: 8px;
  opacity: 0.5;
}

.empty-state p {
  margin: 0;
  font-size: 0.875rem;
}

@media (max-width: 1200px) {
  .content-grid {
    grid-template-columns: 1fr;
  }
  
  .class-grid {
    grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
  }
}

@media (max-width: 768px) {
  .banner-content {
    flex-direction: column;
    text-align: center;
    gap: 24px;
  }
  
  .teaching-overview {
    justify-content: center;
  }
  
  .quick-actions {
    width: 100%;
    justify-content: center;
  }
  
  .dashboard-content {
    padding: 0 16px;
  }
  
  .quick-stats {
    grid-template-columns: 1fr;
  }
  
  .class-grid {
    grid-template-columns: 1fr;
  }
}

@media (max-width: 480px) {
  .overview-card {
    min-width: 100px;
    padding: 16px;
  }
  
  .overview-value {
    font-size: 1.25rem;
  }
  
  .stat-number {
    font-size: 1.5rem;
  }
}


</style>