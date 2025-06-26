<template>
  <div class="student-assignments">
    <!-- 页面头部 -->
    <div class="page-header">
      <div class="header-content">
        <h1 class="page-title">
          <FileTextOutlined />
          作业中心
        </h1>
        <p class="page-description">高效管理学习任务，提升学习效率</p>
      </div>
      <div class="header-stats">
        <div class="stat-item urgent">
          <div class="stat-number">{{ urgentCount }}</div>
          <div class="stat-label">紧急作业</div>
        </div>
        <div class="stat-item todo">
          <div class="stat-number">{{ todoCount }}</div>
          <div class="stat-label">待完成</div>
        </div>
        <div class="stat-item completed">
          <div class="stat-number">{{ completedCount }}</div>
          <div class="stat-label">已完成</div>
        </div>
      </div>
    </div>

    <!-- 智能提醒卡片 -->
    <div class="reminder-cards" v-if="urgentAssignments.length > 0">
      <div class="reminder-card urgent" v-for="assignment in urgentAssignments" :key="assignment.id">
        <div class="reminder-icon">
          <ClockCircleOutlined />
        </div>
        <div class="reminder-content">
          <div class="reminder-title">紧急提醒</div>
          <div class="reminder-text">
            《{{ assignment.title }}》还有 {{ getTimeRemaining(assignment.deadline) }} 截止
          </div>
        </div>
        <div class="reminder-actions">
          <a-button type="primary" size="small" @click="goToAssignment(assignment.id)">
            立即完成
          </a-button>
        </div>
      </div>
    </div>

    <!-- 功能导航 -->
    <div class="function-tabs">
      <a-tabs v-model:activeKey="activeTab" size="large" @change="handleTabChange">
        <a-tab-pane key="all" tab="全部作业">
          <template #tab>
            <span>
              <FileTextOutlined />
              全部作业 ({{ totalAssignments }})
            </span>
          </template>
        </a-tab-pane>
        <a-tab-pane key="todo" tab="待完成">
          <template #tab>
            <span>
              <ClockCircleOutlined />
              待完成 ({{ todoCount }})
            </span>
          </template>
        </a-tab-pane>
        <a-tab-pane key="completed" tab="已完成">
          <template #tab>
            <span>
              <CheckCircleOutlined />
              已完成 ({{ completedCount }})
            </span>
          </template>
        </a-tab-pane>
        <a-tab-pane key="overdue" tab="已逾期">
          <template #tab>
            <span>
              <ExclamationCircleOutlined />
              已逾期 ({{ overdueCount }})
            </span>
          </template>
        </a-tab-pane>
      </a-tabs>
    </div>

    <!-- 筛选和搜索 -->
    <div class="filter-section">
      <div class="filter-left">
        <a-input-search
          v-model:value="searchKeyword"
          placeholder="搜索作业标题、课程名称..."
          style="width: 300px"
          @search="handleSearch"
        />
        <a-select
          v-model:value="courseFilter"
          placeholder="选择课程"
          style="width: 150px"
          allow-clear
          @change="handleFilter"
        >
          <a-select-option value="">全部课程</a-select-option>
          <a-select-option v-for="course in courses" :key="course.id" :value="course.id">
            {{ course.name }}
          </a-select-option>
        </a-select>
        <a-select
          v-model:value="typeFilter"
          placeholder="作业类型"
          style="width: 120px"
          allow-clear
          @change="handleFilter"
        >
          <a-select-option value="">全部类型</a-select-option>
          <a-select-option value="homework">课后作业</a-select-option>
          <a-select-option value="exam">课堂测验</a-select-option>
          <a-select-option value="project">项目作业</a-select-option>
          <a-select-option value="report">实验报告</a-select-option>
        </a-select>
      </div>
      <div class="filter-right">
        <a-radio-group v-model:value="sortBy" @change="handleSort">
          <a-radio-button value="deadline">按截止时间</a-radio-button>
          <a-radio-button value="priority">按优先级</a-radio-button>
          <a-radio-button value="createTime">按发布时间</a-radio-button>
        </a-radio-group>
        <a-button @click="refreshData" :loading="loading">
          <ReloadOutlined />
          刷新
        </a-button>
      </div>
    </div>

    <!-- 批量操作栏 -->
    <div class="batch-actions" v-if="selectedRowKeys.length > 0">
      <div class="batch-info">
        已选择 {{ selectedRowKeys.length }} 项作业
      </div>
      <div class="batch-buttons">
        <a-button @click="batchMarkAsRead">
          <EyeOutlined />
          标记已读
        </a-button>
        <a-button type="primary" @click="batchSubmit" :disabled="!canBatchSubmit">
          <UploadOutlined />
          批量提交
        </a-button>
        <a-button @click="clearSelection">
          清除选择
        </a-button>
      </div>
    </div>

    <!-- 作业列表 -->
    <div class="assignments-content">
      <a-spin :spinning="loading" tip="加载中...">
        <div class="assignments-list" v-if="filteredAssignments.length > 0">
          <div 
            v-for="assignment in filteredAssignments" 
            :key="assignment.id"
            class="assignment-card"
            :class="{
              urgent: isUrgent(assignment),
              overdue: isOverdue(assignment),
              completed: assignment.status === 'completed',
              selected: selectedRowKeys.includes(assignment.id)
            }"
            @click="selectAssignment(assignment.id)"
          >
            <!-- 选择框 -->
            <div class="assignment-selector">
              <a-checkbox 
                :checked="selectedRowKeys.includes(assignment.id)"
                @change="(e) => handleSelectChange(assignment.id, e.target.checked)"
                @click.stop
              />
            </div>

            <!-- 作业状态标识 -->
            <div class="assignment-status">
              <div class="status-indicator" :class="assignment.status">
                <CheckCircleOutlined v-if="assignment.status === 'completed'" />
                <ClockCircleOutlined v-else-if="assignment.status === 'pending'" />
                <ExclamationCircleOutlined v-else-if="assignment.status === 'overdue'" />
                <EditOutlined v-else />
              </div>
            </div>

            <!-- 作业信息 -->
            <div class="assignment-info">
              <div class="assignment-header">
                <h3 class="assignment-title">{{ assignment.title }}</h3>
                <div class="assignment-meta">
                  <a-tag :color="getTypeColor(assignment.type)" size="small">
                    {{ getTypeText(assignment.type) }}
                  </a-tag>
                  <a-tag :color="getPriorityColor(assignment.priority)" size="small">
                    {{ getPriorityText(assignment.priority) }}
                  </a-tag>
                </div>
              </div>
              
              <div class="assignment-details">
                <div class="detail-item">
                  <BookOutlined />
                  <span class="course-name">{{ assignment.courseName }}</span>
                  <span class="teacher-name">{{ assignment.teacherName }}</span>
                </div>
                
                <div class="detail-item">
                  <CalendarOutlined />
                  <span>发布时间：{{ formatDate(assignment.createTime) }}</span>
                </div>
                
                <div class="detail-item deadline" :class="{ urgent: isUrgent(assignment), overdue: isOverdue(assignment) }">
                  <ClockCircleOutlined />
                  <span>截止时间：{{ formatDate(assignment.deadline) }}</span>
                  <span class="time-remaining">({{ getTimeRemaining(assignment.deadline) }})</span>
                </div>

                <div class="assignment-description">
                  {{ assignment.description }}
                </div>
              </div>

              <!-- 提交信息 -->
              <div class="submission-info" v-if="assignment.submissionTime">
                <div class="submission-status completed">
                  <CheckCircleOutlined />
                  <span>已于 {{ formatDate(assignment.submissionTime) }} 提交</span>
                  <span v-if="assignment.score !== null" class="score">
                    得分：{{ assignment.score }}/{{ assignment.totalScore }}
                  </span>
                </div>
              </div>
            </div>

            <!-- 作业操作 -->
            <div class="assignment-actions">
              <a-button-group>
                <a-button size="small" @click.stop="viewAssignment(assignment.id)">
                  <EyeOutlined />
                  查看
                </a-button>
                <a-button 
                  v-if="assignment.status !== 'completed'"
                  type="primary" 
                  size="small"
                  @click.stop="startAssignment(assignment.id)"
                >
                  <EditOutlined />
                  {{ assignment.status === 'draft' ? '继续完成' : '开始作业' }}
                </a-button>
                <a-button 
                  v-if="assignment.status === 'completed'"
                  size="small"
                  @click.stop="viewSubmission(assignment.id)"
                >
                  <FileSearchOutlined />
                  查看提交
                </a-button>
                <a-dropdown :trigger="['click']" @click.stop>
                  <a-button size="small">
                    <MoreOutlined />
                  </a-button>
                  <template #overlay>
                    <a-menu>
                      <a-menu-item key="download" @click="downloadAssignment(assignment.id)">
                        <DownloadOutlined />
                        下载附件
                      </a-menu-item>
                      <a-menu-item key="remind" @click="setReminder(assignment.id)">
                        <BellOutlined />
                        设置提醒
                      </a-menu-item>
                      <a-menu-divider />
                      <a-menu-item key="feedback" @click="viewFeedback(assignment.id)">
                        <MessageOutlined />
                        查看反馈
                      </a-menu-item>
                    </a-menu>
                  </template>
                </a-dropdown>
              </a-button-group>
            </div>
          </div>
        </div>

        <!-- 空状态 -->
        <div v-else class="empty-state">
          <a-empty 
            :image="Empty.PRESENTED_IMAGE_SIMPLE"
            :description="getEmptyDescription()"
          >
            <a-button type="primary" @click="refreshData">
              <ReloadOutlined />
              刷新数据
            </a-button>
          </a-empty>
        </div>
      </a-spin>

      <!-- 分页 -->
      <div class="pagination-wrapper" v-if="filteredAssignments.length > 0">
        <a-pagination
          v-model:current="currentPage"
          v-model:page-size="pageSize"
          :total="totalAssignments"
          :show-size-changer="true"
          :show-quick-jumper="true"
          :show-total="(total, range) => `共 ${total} 项作业，当前显示 ${range[0]}-${range[1]} 项`"
          @change="handlePageChange"
        />
      </div>
    </div>

    <!-- 设置提醒弹窗 -->
    <a-modal
      v-model:open="reminderModalVisible"
      title="设置作业提醒"
      :width="500"
      @ok="handleSetReminder"
      @cancel="handleCancelReminder"
    >
      <div class="reminder-form">
        <a-form ref="reminderFormRef" :model="reminderForm" layout="vertical">
          <a-form-item label="提醒时间">
            <a-radio-group v-model:value="reminderForm.type">
              <a-radio value="before24h">截止前24小时</a-radio>
              <a-radio value="before12h">截止前12小时</a-radio>
              <a-radio value="before6h">截止前6小时</a-radio>
              <a-radio value="before1h">截止前1小时</a-radio>
              <a-radio value="custom">自定义时间</a-radio>
            </a-radio-group>
          </a-form-item>
          <a-form-item v-if="reminderForm.type === 'custom'" label="自定义时间">
            <a-date-picker
              v-model:value="reminderForm.customTime"
              show-time
              format="YYYY-MM-DD HH:mm:ss"
              placeholder="选择提醒时间"
              style="width: 100%"
            />
          </a-form-item>
          <a-form-item label="提醒方式">
            <a-checkbox-group v-model:value="reminderForm.methods">
              <a-checkbox value="notification">浏览器通知</a-checkbox>
              <a-checkbox value="email">邮件提醒</a-checkbox>
              <a-checkbox value="sms">短信提醒</a-checkbox>
            </a-checkbox-group>
          </a-form-item>
          <a-form-item label="提醒内容">
            <a-textarea 
              v-model:value="reminderForm.content"
              placeholder="自定义提醒内容（可选）"
              :rows="3"
            />
          </a-form-item>
        </a-form>
      </div>
    </a-modal>
  </div>
</template>

<script setup lang="ts">
import { ref, reactive, computed, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { message, Empty } from 'ant-design-vue'
import dayjs from 'dayjs'
import relativeTime from 'dayjs/plugin/relativeTime'
import 'dayjs/locale/zh-cn'
import {
  FileTextOutlined,
  ClockCircleOutlined,
  CheckCircleOutlined,
  ExclamationCircleOutlined,
  EditOutlined,
  EyeOutlined,
  UploadOutlined,
  ReloadOutlined,
  BookOutlined,
  CalendarOutlined,
  FileSearchOutlined,
  MoreOutlined,
  DownloadOutlined,
  BellOutlined,
  MessageOutlined
} from '@ant-design/icons-vue'

dayjs.extend(relativeTime)
dayjs.locale('zh-cn')

interface Assignment {
  id: number
  title: string
  description: string
  courseName: string
  teacherName: string
  type: 'homework' | 'exam' | 'project' | 'report'
  priority: 'high' | 'medium' | 'low'
  status: 'pending' | 'draft' | 'completed' | 'overdue'
  createTime: string
  deadline: string
  submissionTime?: string
  score?: number
  totalScore: number
  attachments: string[]
  requirements: string[]
  isRead: boolean
}

interface Course {
  id: number
  name: string
}

const router = useRouter()

// 页面状态
const loading = ref(false)
const activeTab = ref('all')
const searchKeyword = ref('')
const courseFilter = ref('')
const typeFilter = ref('')
const sortBy = ref('deadline')
const currentPage = ref(1)
const pageSize = ref(10)
const selectedRowKeys = ref<number[]>([])
const reminderModalVisible = ref(false)
const currentReminderAssignment = ref<number | null>(null)

// 表单数据
const reminderForm = reactive({
  type: 'before24h',
  customTime: null,
  methods: ['notification'],
  content: ''
})

// 模拟数据
const courses = ref<Course[]>([
  { id: 1, name: '高等数学' },
  { id: 2, name: '大学英语' },
  { id: 3, name: 'C++程序设计' },
  { id: 4, name: '线性代数' }
])

const assignments = ref<Assignment[]>([
  {
    id: 1,
    title: '第三章函数极限习题',
    description: '完成教材第三章后的习题1-15，要求写出详细解题过程，注意格式规范。',
    courseName: '高等数学',
    teacherName: '张教授',
    type: 'homework',
    priority: 'high',
    status: 'pending',
    createTime: '2024-01-15 14:30:00',
    deadline: '2024-01-22 23:59:59',
    totalScore: 100,
    attachments: ['习题详解.pdf', '参考答案.docx'],
    requirements: ['手写解答过程', '拍照上传', '格式清晰'],
    isRead: true
  }
])

// 计算属性
const totalAssignments = computed(() => assignments.value.length)
const todoCount = computed(() => assignments.value.filter(a => a.status === 'pending' || a.status === 'draft').length)
const completedCount = computed(() => assignments.value.filter(a => a.status === 'completed').length)
const overdueCount = computed(() => assignments.value.filter(a => a.status === 'overdue').length)
const urgentCount = computed(() => urgentAssignments.value.length)

const urgentAssignments = computed(() => {
  return assignments.value.filter(assignment => {
    if (assignment.status === 'completed' || assignment.status === 'overdue') return false
    const deadline = dayjs(assignment.deadline)
    const now = dayjs()
    const hoursLeft = deadline.diff(now, 'hour')
    return hoursLeft <= 24 && hoursLeft >= 0
  })
})

const filteredAssignments = computed(() => {
  let filtered = assignments.value

  // 按标签筛选
  if (activeTab.value !== 'all') {
    switch (activeTab.value) {
      case 'todo':
        filtered = filtered.filter(a => a.status === 'pending' || a.status === 'draft')
        break
      case 'completed':
        filtered = filtered.filter(a => a.status === 'completed')
        break
      case 'overdue':
        filtered = filtered.filter(a => a.status === 'overdue')
        break
    }
  }

  // 搜索筛选
  if (searchKeyword.value) {
    const keyword = searchKeyword.value.toLowerCase()
    filtered = filtered.filter(a => 
      a.title.toLowerCase().includes(keyword) ||
      a.courseName.toLowerCase().includes(keyword) ||
      a.description.toLowerCase().includes(keyword)
    )
  }

  // 课程筛选
  if (courseFilter.value) {
    const course = courses.value.find(c => c.id === courseFilter.value)
    if (course) {
      filtered = filtered.filter(a => a.courseName === course.name)
    }
  }

  // 类型筛选
  if (typeFilter.value) {
    filtered = filtered.filter(a => a.type === typeFilter.value)
  }

  // 排序
  filtered.sort((a, b) => {
    switch (sortBy.value) {
      case 'deadline':
        return new Date(a.deadline).getTime() - new Date(b.deadline).getTime()
      case 'priority':
        const priorityOrder = { high: 3, medium: 2, low: 1 }
        return priorityOrder[b.priority] - priorityOrder[a.priority]
      case 'createTime':
        return new Date(b.createTime).getTime() - new Date(a.createTime).getTime()
      default:
        return 0
    }
  })

  return filtered.slice((currentPage.value - 1) * pageSize.value, currentPage.value * pageSize.value)
})

const canBatchSubmit = computed(() => {
  return selectedRowKeys.value.some(id => {
    const assignment = assignments.value.find(a => a.id === id)
    return assignment && (assignment.status === 'pending' || assignment.status === 'draft')
  })
})

// 方法
const handleTabChange = (key: string) => {
  activeTab.value = key
  currentPage.value = 1
  selectedRowKeys.value = []
}

const handleSearch = () => {
  currentPage.value = 1
}

const handleFilter = () => {
  currentPage.value = 1
}

const handleSort = () => {
  currentPage.value = 1
}

const handlePageChange = () => {
  selectedRowKeys.value = []
}

const handleSelectChange = (id: number, checked: boolean) => {
  if (checked) {
    selectedRowKeys.value.push(id)
  } else {
    const index = selectedRowKeys.value.indexOf(id)
    if (index > -1) {
      selectedRowKeys.value.splice(index, 1)
    }
  }
}

const selectAssignment = (id: number) => {
  const index = selectedRowKeys.value.indexOf(id)
  if (index > -1) {
    selectedRowKeys.value.splice(index, 1)
  } else {
    selectedRowKeys.value.push(id)
  }
}

const clearSelection = () => {
  selectedRowKeys.value = []
}

const refreshData = async () => {
  loading.value = true
  try {
    await new Promise(resolve => setTimeout(resolve, 1000))
    message.success('数据刷新成功')
  } catch (error) {
    message.error('刷新失败，请重试')
  } finally {
    loading.value = false
  }
}

const batchMarkAsRead = () => {
  selectedRowKeys.value.forEach(id => {
    const assignment = assignments.value.find(a => a.id === id)
    if (assignment) {
      assignment.isRead = true
    }
  })
  message.success(`已标记 ${selectedRowKeys.value.length} 项作业为已读`)
  selectedRowKeys.value = []
}

const batchSubmit = () => {
  const submittableCount = selectedRowKeys.value.filter(id => {
    const assignment = assignments.value.find(a => a.id === id)
    return assignment && (assignment.status === 'pending' || assignment.status === 'draft')
  }).length
  
  if (submittableCount === 0) {
    message.warning('选中的作业中没有可提交的项目')
    return
  }
  
  message.success(`批量提交 ${submittableCount} 项作业成功`)
  selectedRowKeys.value = []
}

const goToAssignment = (id: number) => {
  router.push(`/student/assignments/${id}`)
}

const viewAssignment = (id: number) => {
  router.push(`/student/assignments/${id}`)
}

const startAssignment = (id: number) => {
  const assignment = assignments.value.find(a => a.id === id)
  if (assignment) {
    assignment.isRead = true
    message.success(`开始作业：${assignment.title}`)
    router.push(`/student/assignments/${id}/submit`)
  }
}

const viewSubmission = (id: number) => {
  router.push(`/student/assignments/${id}/submission`)
}

const downloadAssignment = (id: number) => {
  message.success('下载已开始')
}

const setReminder = (id: number) => {
  currentReminderAssignment.value = id
  reminderModalVisible.value = true
}

const viewFeedback = (id: number) => {
  router.push(`/student/assignments/${id}/feedback`)
}

const handleSetReminder = () => {
  message.success('提醒设置成功')
  reminderModalVisible.value = false
  Object.assign(reminderForm, {
    type: 'before24h',
    customTime: null,
    methods: ['notification'],
    content: ''
  })
}

const handleCancelReminder = () => {
  reminderModalVisible.value = false
  Object.assign(reminderForm, {
    type: 'before24h',
    customTime: null,
    methods: ['notification'],
    content: ''
  })
}

// 工具方法
const isUrgent = (assignment: Assignment) => {
  if (assignment.status === 'completed' || assignment.status === 'overdue') return false
  const deadline = dayjs(assignment.deadline)
  const now = dayjs()
  const hoursLeft = deadline.diff(now, 'hour')
  return hoursLeft <= 24 && hoursLeft >= 0
}

const isOverdue = (assignment: Assignment) => {
  return assignment.status === 'overdue'
}

const getTimeRemaining = (deadline: string) => {
  const now = dayjs()
  const deadlineTime = dayjs(deadline)
  
  if (deadlineTime.isBefore(now)) {
    return '已逾期'
  }
  
  const diff = deadlineTime.diff(now)
  const days = Math.floor(diff / (1000 * 60 * 60 * 24))
  const hours = Math.floor((diff % (1000 * 60 * 60 * 24)) / (1000 * 60 * 60))
  const minutes = Math.floor((diff % (1000 * 60 * 60)) / (1000 * 60))
  
  if (days > 0) {
    return `${days}天${hours}小时`
  } else if (hours > 0) {
    return `${hours}小时${minutes}分钟`
  } else {
    return `${minutes}分钟`
  }
}

const formatDate = (date: string) => {
  return dayjs(date).format('YYYY-MM-DD HH:mm')
}

const getTypeText = (type: string) => {
  const typeMap = {
    homework: '课后作业',
    exam: '课堂测验',
    project: '项目作业',
    report: '实验报告'
  }
  return typeMap[type as keyof typeof typeMap] || '未知类型'
}

const getTypeColor = (type: string) => {
  const colorMap = {
    homework: 'blue',
    exam: 'red',
    project: 'purple',
    report: 'green'
  }
  return colorMap[type as keyof typeof colorMap] || 'default'
}

const getPriorityText = (priority: string) => {
  const priorityMap = {
    high: '高优先级',
    medium: '中优先级',
    low: '低优先级'
  }
  return priorityMap[priority as keyof typeof priorityMap] || '未知优先级'
}

const getPriorityColor = (priority: string) => {
  const colorMap = {
    high: 'red',
    medium: 'orange',
    low: 'green'
  }
  return colorMap[priority as keyof typeof colorMap] || 'default'
}

const getEmptyDescription = () => {
  switch (activeTab.value) {
    case 'todo':
      return '暂无待完成的作业'
    case 'completed':
      return '暂无已完成的作业'
    case 'overdue':
      return '暂无逾期的作业'
    default:
      return '暂无作业数据'
  }
}

// 页面初始化
onMounted(() => {
  console.log('作业中心页面初始化完成')
  refreshData()
})
</script>

<style scoped>
.student-assignments {
  padding: 24px;
  min-height: 100vh;
  background: #f5f7fa;
}

/* 页面头部 */
.page-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 24px;
  padding: 32px;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  border-radius: 20px;
  color: white;
  box-shadow: 0 12px 40px rgba(102, 126, 234, 0.3);
}

.header-content {
  flex: 1;
}

.page-title {
  font-size: 28px;
  font-weight: 700;
  margin: 0 0 8px 0;
  display: flex;
  align-items: center;
  gap: 12px;
}

.page-description {
  font-size: 16px;
  margin: 0;
  opacity: 0.9;
}

.header-stats {
  display: flex;
  gap: 32px;
}

.stat-item {
  text-align: center;
}

.stat-number {
  font-size: 24px;
  font-weight: 700;
  margin-bottom: 4px;
}

.stat-label {
  font-size: 12px;
  opacity: 0.8;
}

.stat-item.urgent .stat-number {
  color: #ff4d4f;
}

.stat-item.todo .stat-number {
  color: #faad14;
}

.stat-item.completed .stat-number {
  color: #52c41a;
}

/* 智能提醒卡片 */
.reminder-cards {
  margin-bottom: 24px;
}

.reminder-card {
  background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%);
  border-radius: 16px;
  padding: 20px;
  margin-bottom: 12px;
  display: flex;
  align-items: center;
  gap: 16px;
  color: white;
  box-shadow: 0 8px 25px rgba(255, 154, 158, 0.3);
}

.reminder-card.urgent {
  background: linear-gradient(135deg, #ff6b6b 0%, #ffa500 100%);
}

.reminder-icon {
  font-size: 24px;
  background: rgba(255, 255, 255, 0.2);
  width: 48px;
  height: 48px;
  border-radius: 12px;
  display: flex;
  align-items: center;
  justify-content: center;
}

.reminder-content {
  flex: 1;
}

.reminder-title {
  font-size: 16px;
  font-weight: 600;
  margin-bottom: 4px;
}

.reminder-text {
  font-size: 14px;
  opacity: 0.9;
}

.reminder-actions .ant-btn {
  background: rgba(255, 255, 255, 0.2);
  border-color: rgba(255, 255, 255, 0.3);
  color: white;
}

.reminder-actions .ant-btn:hover {
  background: rgba(255, 255, 255, 0.3);
}

/* 功能导航 */
.function-tabs {
  background: white;
  border-radius: 16px;
  padding: 0 24px;
  margin-bottom: 24px;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.06);
}

.function-tabs :deep(.ant-tabs-tab) {
  padding: 16px 0;
  font-weight: 500;
}

/* 筛选区域 */
.filter-section {
  background: white;
  border-radius: 16px;
  padding: 20px 24px;
  margin-bottom: 24px;
  display: flex;
  justify-content: space-between;
  align-items: center;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.06);
  gap: 16px;
}

.filter-left {
  display: flex;
  gap: 12px;
  align-items: center;
}

.filter-right {
  display: flex;
  gap: 12px;
  align-items: center;
}

/* 批量操作 */
.batch-actions {
  background: #e6f7ff;
  border: 1px solid #91d5ff;
  border-radius: 12px;
  padding: 16px 24px;
  margin-bottom: 24px;
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.batch-info {
  color: #1890ff;
  font-weight: 500;
}

.batch-buttons {
  display: flex;
  gap: 12px;
}

/* 作业列表 */
.assignments-content {
  background: white;
  border-radius: 16px;
  padding: 24px;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.06);
}

.assignments-list {
  margin-bottom: 24px;
}

.assignment-card {
  background: white;
  border: 1px solid #f0f0f0;
  border-radius: 12px;
  padding: 20px;
  margin-bottom: 16px;
  display: flex;
  gap: 16px;
  transition: all 0.3s ease;
  cursor: pointer;
  position: relative;
}

.assignment-card:hover {
  border-color: #1890ff;
  box-shadow: 0 8px 25px rgba(24, 144, 255, 0.1);
  transform: translateY(-2px);
}

.assignment-card.selected {
  border-color: #1890ff;
  background: #f6ffed;
}

.assignment-card.urgent {
  border-left: 4px solid #ff4d4f;
}

.assignment-card.overdue {
  border-left: 4px solid #ff7875;
  background: #fff2f0;
}

.assignment-card.completed {
  border-left: 4px solid #52c41a;
}

.assignment-selector {
  display: flex;
  align-items: flex-start;
  padding-top: 2px;
}

.assignment-status {
  display: flex;
  align-items: flex-start;
  padding-top: 2px;
}

.status-indicator {
  width: 32px;
  height: 32px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  color: white;
  font-size: 14px;
}

.status-indicator.pending {
  background: #faad14;
}

.status-indicator.draft {
  background: #1890ff;
}

.status-indicator.completed {
  background: #52c41a;
}

.status-indicator.overdue {
  background: #ff4d4f;
}

.assignment-info {
  flex: 1;
}

.assignment-header {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  margin-bottom: 12px;
}

.assignment-title {
  font-size: 18px;
  font-weight: 600;
  color: #333;
  margin: 0;
  line-height: 1.4;
}

.assignment-meta {
  display: flex;
  gap: 8px;
  align-items: center;
}

.assignment-details {
  margin-bottom: 16px;
}

.detail-item {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 8px;
  font-size: 14px;
  color: #666;
}

.detail-item:last-child {
  margin-bottom: 0;
}

.course-name {
  font-weight: 500;
  color: #1890ff;
}

.teacher-name {
  color: #999;
}

.detail-item.deadline {
  font-weight: 500;
}

.detail-item.deadline.urgent {
  color: #ff4d4f;
}

.detail-item.deadline.overdue {
  color: #ff7875;
}

.time-remaining {
  font-weight: 600;
}

.assignment-description {
  font-size: 14px;
  color: #666;
  line-height: 1.5;
  margin-top: 12px;
  padding: 12px;
  background: #fafafa;
  border-radius: 8px;
}

.submission-info {
  margin-top: 16px;
  padding: 12px;
  border-radius: 8px;
}

.submission-status {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 14px;
}

.submission-status.completed {
  color: #52c41a;
  background: #f6ffed;
}

.score {
  margin-left: auto;
  font-weight: 600;
  color: #1890ff;
}

.assignment-actions {
  display: flex;
  align-items: flex-start;
  padding-top: 2px;
}

/* 分页 */
.pagination-wrapper {
  display: flex;
  justify-content: center;
  margin-top: 24px;
}

/* 空状态 */
.empty-state {
  text-align: center;
  padding: 60px 20px;
}

/* 提醒表单 */
.reminder-form {
  padding: 20px 0;
}

/* 响应式设计 */
@media (max-width: 1200px) {
  .filter-section {
    flex-direction: column;
    align-items: stretch;
    gap: 16px;
  }
  
  .filter-left,
  .filter-right {
    justify-content: space-between;
  }
}

@media (max-width: 768px) {
  .student-assignments {
    padding: 16px;
  }
  
  .page-header {
    flex-direction: column;
    gap: 16px;
    text-align: center;
    padding: 24px;
  }
  
  .header-stats {
    gap: 20px;
  }
  
  .assignment-card {
    flex-direction: column;
    gap: 12px;
  }
  
  .assignment-header {
    flex-direction: column;
    gap: 8px;
  }
  
  .batch-actions {
    flex-direction: column;
    gap: 12px;
    text-align: center;
  }
  
  .filter-left {
    flex-direction: column;
    gap: 8px;
  }
  
  .filter-right {
    flex-direction: column;
    gap: 8px;
  }
}
</style> 