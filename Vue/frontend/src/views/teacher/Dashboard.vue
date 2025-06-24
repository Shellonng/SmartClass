<template>
  <div class="teacher-dashboard">
    <div class="dashboard-header">
      <div class="welcome-section">
        <h1 class="welcome-title">
          欢迎回来，{{ authStore.user?.realName || '老师' }}！
        </h1>
        <p class="welcome-subtitle">
          今天是 {{ currentDate }}，祝您工作愉快
        </p>
      </div>
      
      <div class="quick-actions">
        <a-button type="primary" size="large" @click="showCreateClassModal = true">
          <PlusOutlined />
          创建班级
        </a-button>
        
        <a-button size="large" @click="showCreateTaskModal = true">
          <FileAddOutlined />
          布置作业
        </a-button>
      </div>
    </div>

    <div class="dashboard-content">
      <!-- 统计卡片 -->
      <div class="stats-grid">
        <div class="stat-card">
          <div class="stat-icon class-icon">
            <TeamOutlined />
          </div>
          <div class="stat-info">
            <div class="stat-number">{{ stats.classCount }}</div>
            <div class="stat-label">管理班级</div>
          </div>
        </div>
        
        <div class="stat-card">
          <div class="stat-icon student-icon">
            <UserOutlined />
          </div>
          <div class="stat-info">
            <div class="stat-number">{{ stats.studentCount }}</div>
            <div class="stat-label">学生总数</div>
          </div>
        </div>
        
        <div class="stat-card">
          <div class="stat-icon task-icon">
            <FileTextOutlined />
          </div>
          <div class="stat-info">
            <div class="stat-number">{{ stats.taskCount }}</div>
            <div class="stat-label">布置作业</div>
          </div>
        </div>
        
        <div class="stat-card">
          <div class="stat-icon pending-icon">
            <ClockCircleOutlined />
          </div>
          <div class="stat-info">
            <div class="stat-number">{{ stats.pendingCount }}</div>
            <div class="stat-label">待批改</div>
          </div>
        </div>
      </div>

      <!-- 主要内容区域 -->
      <div class="main-content">
        <!-- 左侧内容 -->
        <div class="left-content">
          <!-- 最近班级 -->
          <div class="content-card">
            <div class="card-header">
              <h3>我的班级</h3>
              <a-button type="link" @click="$router.push('/teacher/classes')">
                查看全部
                <ArrowRightOutlined />
              </a-button>
            </div>
            
            <div class="class-list">
              <div 
                v-for="classItem in recentClasses" 
                :key="classItem.id"
                class="class-item"
                @click="$router.push(`/teacher/classes/${classItem.id}`)"
              >
                <div class="class-avatar">
                  {{ classItem.name.charAt(0) }}
                </div>
                <div class="class-info">
                  <div class="class-name">{{ classItem.name }}</div>
                  <div class="class-meta">
                    {{ classItem.studentCount }}名学生 · {{ classItem.subject }}
                  </div>
                </div>
                <div class="class-status">
                  <a-badge :count="classItem.pendingTasks" />
                </div>
              </div>
            </div>
          </div>
          
          <!-- 最近作业 -->
          <div class="content-card">
            <div class="card-header">
              <h3>最近作业</h3>
              <a-button type="link" @click="$router.push('/teacher/tasks')">
                查看全部
                <ArrowRightOutlined />
              </a-button>
            </div>
            
            <div class="task-list">
              <div 
                v-for="task in recentTasks" 
                :key="task.id"
                class="task-item"
                @click="$router.push(`/teacher/tasks/${task.id}`)"
              >
                <div class="task-icon">
                  <FileTextOutlined />
                </div>
                <div class="task-info">
                  <div class="task-title">{{ task.title }}</div>
                  <div class="task-meta">
                    {{ task.className }} · 截止：{{ formatDate(task.deadline) }}
                  </div>
                </div>
                <div class="task-progress">
                  <a-progress 
                    :percent="task.completionRate" 
                    size="small" 
                    :show-info="false"
                  />
                  <span class="progress-text">
                    {{ task.submittedCount }}/{{ task.totalCount }}
                  </span>
                </div>
              </div>
            </div>
          </div>
        </div>
        
        <!-- 右侧内容 -->
        <div class="right-content">
          <!-- 待办事项 -->
          <div class="content-card">
            <div class="card-header">
              <h3>待办事项</h3>
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
                <span class="todo-text">{{ todo.text }}</span>
                <span class="todo-time">{{ todo.time }}</span>
              </div>
            </div>
          </div>
          
          <!-- AI助手 -->
          <div class="content-card ai-card">
            <div class="card-header">
              <h3>
                <RobotOutlined />
                AI教学助手
              </h3>
            </div>
            
            <div class="ai-suggestions">
              <div class="ai-suggestion">
                <div class="suggestion-icon">
                  <BulbOutlined />
                </div>
                <div class="suggestion-content">
                  <div class="suggestion-title">智能作业推荐</div>
                  <div class="suggestion-desc">
                    根据学生学习情况，推荐个性化作业内容
                  </div>
                </div>
              </div>
              
              <div class="ai-suggestion">
                <div class="suggestion-icon">
                  <BarChartOutlined />
                </div>
                <div class="suggestion-content">
                  <div class="suggestion-title">学情分析报告</div>
                  <div class="suggestion-desc">
                    生成班级学习情况分析和改进建议
                  </div>
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
  BarChartOutlined
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

// 统计数据
const stats = reactive({
  classCount: 5,
  studentCount: 156,
  taskCount: 23,
  pendingCount: 12
})

// 最近班级
const recentClasses = ref([
  {
    id: 1,
    name: '高一(1)班',
    subject: '数学',
    studentCount: 45,
    pendingTasks: 3
  },
  {
    id: 2,
    name: '高一(2)班',
    subject: '数学',
    studentCount: 43,
    pendingTasks: 1
  },
  {
    id: 3,
    name: '高二(1)班',
    subject: '数学',
    studentCount: 42,
    pendingTasks: 2
  }
])

// 最近作业
const recentTasks = ref([
  {
    id: 1,
    title: '函数与方程练习',
    className: '高一(1)班',
    deadline: new Date('2024-01-20'),
    completionRate: 75,
    submittedCount: 34,
    totalCount: 45
  },
  {
    id: 2,
    title: '三角函数应用',
    className: '高一(2)班',
    deadline: new Date('2024-01-22'),
    completionRate: 60,
    submittedCount: 26,
    totalCount: 43
  }
])

// 待办事项
const todoList = ref([
  {
    id: 1,
    text: '批改高一(1)班数学作业',
    time: '今天',
    completed: false
  },
  {
    id: 2,
    text: '准备明天的课件',
    time: '明天',
    completed: false
  },
  {
    id: 3,
    text: '家长会材料准备',
    time: '本周',
    completed: true
  }
])

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

onMounted(() => {
  // 加载数据
})
</script>

<style scoped>
.teacher-dashboard {
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

.quick-actions {
  display: flex;
  gap: 12px;
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
}

.stat-card:hover {
  transform: translateY(-2px);
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

.class-icon {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
}

.student-icon {
  background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
}

.task-icon {
  background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
}

.pending-icon {
  background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
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
}

.class-list,
.task-list {
  padding: 0 24px 24px 24px;
}

.class-item,
.task-item {
  display: flex;
  align-items: center;
  gap: 16px;
  padding: 16px;
  border-radius: 12px;
  cursor: pointer;
  transition: background-color 0.2s ease;
  margin-bottom: 8px;
}

.class-item:hover,
.task-item:hover {
  background: #f8f9fa;
}

.class-avatar {
  width: 48px;
  height: 48px;
  border-radius: 12px;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  display: flex;
  align-items: center;
  justify-content: center;
  color: white;
  font-weight: 600;
  font-size: 18px;
}

.task-icon {
  width: 48px;
  height: 48px;
  border-radius: 12px;
  background: #f0f8ff;
  display: flex;
  align-items: center;
  justify-content: center;
  color: #1890ff;
  font-size: 20px;
}

.class-info,
.task-info {
  flex: 1;
}

.class-name,
.task-title {
  font-size: 16px;
  font-weight: 600;
  color: #333;
  margin-bottom: 4px;
}

.class-meta,
.task-meta {
  font-size: 14px;
  color: #666;
}

.task-progress {
  display: flex;
  flex-direction: column;
  align-items: flex-end;
  gap: 4px;
  min-width: 100px;
}

.progress-text {
  font-size: 12px;
  color: #666;
}

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

.todo-text {
  flex: 1;
  font-size: 14px;
  color: #333;
}

.todo-time {
  font-size: 12px;
  color: #999;
}

.ai-card {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
}

.ai-card .card-header h3 {
  color: white;
  display: flex;
  align-items: center;
  gap: 8px;
}

.ai-suggestions {
  padding: 0 24px;
}

.ai-suggestion {
  display: flex;
  gap: 12px;
  padding: 16px;
  background: rgba(255, 255, 255, 0.1);
  border-radius: 12px;
  margin-bottom: 12px;
}

.suggestion-icon {
  width: 40px;
  height: 40px;
  border-radius: 8px;
  background: rgba(255, 255, 255, 0.2);
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 18px;
}

.suggestion-title {
  font-size: 14px;
  font-weight: 600;
  margin-bottom: 4px;
}

.suggestion-desc {
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