<template>
  <div class="learning-records">
    <div class="page-header">
      <h1 class="page-title">学习记录</h1>
      <p class="page-description">记录您的学习轨迹，追踪学习进度</p>
    </div>

    <div class="content-wrapper">
      <div class="stats-cards">
        <div class="stat-card">
          <div class="stat-icon">
            <ClockCircleOutlined />
          </div>
          <div class="stat-content">
            <div class="stat-value">{{ totalHours }}</div>
            <div class="stat-label">总学习时长</div>
          </div>
        </div>
        <div class="stat-card">
          <div class="stat-icon">
            <ReadOutlined />
          </div>
          <div class="stat-content">
            <div class="stat-value">{{ completedCourses }}</div>
            <div class="stat-label">已完成课程</div>
          </div>
        </div>
        <div class="stat-card">
          <div class="stat-icon">
            <FileTextOutlined />
          </div>
          <div class="stat-content">
            <div class="stat-value">{{ completedAssignments }}</div>
            <div class="stat-label">已完成作业</div>
          </div>
        </div>
        <div class="stat-card">
          <div class="stat-icon">
            <TrophyOutlined />
          </div>
          <div class="stat-content">
            <div class="stat-value">{{ averageScore }}</div>
            <div class="stat-label">平均成绩</div>
          </div>
        </div>
      </div>

      <div class="chart-section">
        <div class="section-header">
          <h2>学习时长趋势</h2>
          <a-radio-group v-model:value="timeRange" @change="handleTimeRangeChange">
            <a-radio-button value="week">本周</a-radio-button>
            <a-radio-button value="month">本月</a-radio-button>
            <a-radio-button value="semester">本学期</a-radio-button>
          </a-radio-group>
        </div>
        <div class="chart-container">
          <div class="chart-placeholder">
            <p>学习时长趋势图表</p>
            <p>（实际项目中应使用ECharts等图表库）</p>
          </div>
        </div>
      </div>

      <div class="records-section">
        <div class="section-header">
          <h2>学习记录</h2>
          <div class="filter-controls">
            <a-select
              v-model:value="courseFilter"
              placeholder="选择课程"
              style="width: 180px"
              @change="handleFilter"
            >
              <a-select-option value="">全部课程</a-select-option>
              <a-select-option v-for="course in courses" :key="course.id" :value="course.id">
                {{ course.title }}
              </a-select-option>
            </a-select>
            <a-select
              v-model:value="activityFilter"
              placeholder="活动类型"
              style="width: 150px"
              @change="handleFilter"
            >
              <a-select-option value="">全部类型</a-select-option>
              <a-select-option value="video">视频学习</a-select-option>
              <a-select-option value="quiz">测验</a-select-option>
              <a-select-option value="assignment">作业</a-select-option>
              <a-select-option value="exam">考试</a-select-option>
              <a-select-option value="resource">资源阅读</a-select-option>
            </a-select>
            <a-range-picker
              v-model:value="dateRange"
              format="YYYY-MM-DD"
              @change="handleDateRangeChange"
            />
          </div>
        </div>
        
        <a-spin :spinning="loading">
          <a-table
            :columns="columns"
            :data-source="records"
            :pagination="pagination"
            @change="handleTableChange"
            :row-key="record => record.id"
          >
            <template #bodyCell="{ column, record }">
              <template v-if="column.key === 'time'">
                {{ formatDateTime(record.time) }}
              </template>
              
              <template v-else-if="column.key === 'duration'">
                {{ formatDuration(record.duration) }}
              </template>
              
              <template v-else-if="column.key === 'activityType'">
                <a-tag :color="getActivityColor(record.activityType)">
                  {{ getActivityText(record.activityType) }}
                </a-tag>
              </template>
              
              <template v-else-if="column.key === 'progress'">
                <a-progress
                  :percent="record.progress"
                  size="small"
                  :status="record.progress === 100 ? 'success' : 'active'"
                />
              </template>
              
              <template v-else-if="column.key === 'score'">
                <span v-if="record.score !== null">{{ record.score }}</span>
                <span v-else>-</span>
              </template>
              
              <template v-else-if="column.key === 'action'">
                <a-button type="link" size="small" @click="viewDetail(record)">
                  查看详情
                </a-button>
              </template>
            </template>
          </a-table>
        </a-spin>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { message } from 'ant-design-vue'
import dayjs from 'dayjs'
import { 
  ClockCircleOutlined,
  ReadOutlined,
  FileTextOutlined,
  TrophyOutlined
} from '@ant-design/icons-vue'

// 统计数据
const totalHours = ref<string>('76.5')
const completedCourses = ref<number>(3)
const completedAssignments = ref<number>(24)
const averageScore = ref<string>('85.6')

// 筛选条件
const timeRange = ref<string>('week')
const courseFilter = ref<string>('')
const activityFilter = ref<string>('')
const dateRange = ref<any>(null)

// 课程列表
const courses = ref([
  { id: 1, title: '计算机组成原理' },
  { id: 2, title: 'Java程序设计' },
  { id: 3, title: '微观经济学原理' }
])

// 表格配置
const columns = [
  {
    title: '时间',
    dataIndex: 'time',
    key: 'time',
    sorter: true
  },
  {
    title: '课程',
    dataIndex: 'courseName',
    key: 'courseName'
  },
  {
    title: '活动类型',
    dataIndex: 'activityType',
    key: 'activityType',
    filters: [
      { text: '视频学习', value: 'video' },
      { text: '测验', value: 'quiz' },
      { text: '作业', value: 'assignment' },
      { text: '考试', value: 'exam' },
      { text: '资源阅读', value: 'resource' }
    ]
  },
  {
    title: '内容',
    dataIndex: 'content',
    key: 'content'
  },
  {
    title: '学习时长',
    dataIndex: 'duration',
    key: 'duration',
    sorter: true
  },
  {
    title: '进度',
    dataIndex: 'progress',
    key: 'progress',
    sorter: true
  },
  {
    title: '得分',
    dataIndex: 'score',
    key: 'score',
    sorter: true
  },
  {
    title: '操作',
    key: 'action'
  }
]

// 分页
const pagination = ref({
  current: 1,
  pageSize: 10,
  total: 0,
  showSizeChanger: true,
  showQuickJumper: true
})

// 数据
const loading = ref<boolean>(false)
const records = ref<any[]>([])

// 加载学习记录
const loadRecords = async () => {
  try {
    loading.value = true
    
    // 模拟API调用
    await new Promise(resolve => setTimeout(resolve, 1000))
    
    // 模拟数据
    records.value = [
      {
        id: 1,
        time: '2025-07-02 14:30:00',
        courseName: '计算机组成原理',
        courseId: 1,
        activityType: 'video',
        content: '第1章：计算机系统概述',
        duration: 45,
        progress: 100,
        score: null
      },
      {
        id: 2,
        time: '2025-07-02 15:20:00',
        courseName: '计算机组成原理',
        courseId: 1,
        activityType: 'quiz',
        content: '第1章测验',
        duration: 15,
        progress: 100,
        score: 90
      },
      {
        id: 3,
        time: '2025-07-01 10:15:00',
        courseName: 'Java程序设计',
        courseId: 2,
        activityType: 'assignment',
        content: '第3章作业：面向对象编程',
        duration: 60,
        progress: 100,
        score: 85
      },
      {
        id: 4,
        time: '2025-07-01 16:45:00',
        courseName: '微观经济学原理',
        courseId: 3,
        activityType: 'resource',
        content: '阅读材料：需求曲线分析',
        duration: 30,
        progress: 75,
        score: null
      },
      {
        id: 5,
        time: '2025-06-30 09:30:00',
        courseName: 'Java程序设计',
        courseId: 2,
        activityType: 'exam',
        content: '期中考试',
        duration: 120,
        progress: 100,
        score: 88
      }
    ]
    
    pagination.value.total = 42 // 模拟总数据量
    
  } catch (error) {
    console.error('加载学习记录失败:', error)
    message.error('加载学习记录失败')
  } finally {
    loading.value = false
  }
}

// 处理时间范围变化
const handleTimeRangeChange = () => {
  console.log('时间范围变更为:', timeRange.value)
  // 实际应重新加载图表数据
}

// 处理筛选
const handleFilter = () => {
  pagination.value.current = 1
  loadRecords()
}

// 处理日期范围变化
const handleDateRangeChange = (dates: any) => {
  console.log('日期范围变更为:', dates)
  loadRecords()
}

// 处理表格变化
const handleTableChange = (pag: any, filters: any, sorter: any) => {
  pagination.value.current = pag.current
  pagination.value.pageSize = pag.pageSize
  console.log('排序:', sorter)
  console.log('筛选:', filters)
  loadRecords()
}

// 查看详情
const viewDetail = (record: any) => {
  message.info(`查看记录详情: ${record.id}`)
}

// 格式化日期时间
const formatDateTime = (date: string): string => {
  return dayjs(date).format('YYYY-MM-DD HH:mm')
}

// 格式化学习时长
const formatDuration = (minutes: number): string => {
  if (minutes < 60) {
    return `${minutes}分钟`
  }
  const hours = Math.floor(minutes / 60)
  const remainMinutes = minutes % 60
  return remainMinutes > 0 ? `${hours}小时${remainMinutes}分钟` : `${hours}小时`
}

// 获取活动类型文本
const getActivityText = (type: string): string => {
  const typeMap: Record<string, string> = {
    'video': '视频学习',
    'quiz': '测验',
    'assignment': '作业',
    'exam': '考试',
    'resource': '资源阅读'
  }
  return typeMap[type] || '未知类型'
}

// 获取活动类型颜色
const getActivityColor = (type: string): string => {
  const colorMap: Record<string, string> = {
    'video': 'blue',
    'quiz': 'green',
    'assignment': 'orange',
    'exam': 'red',
    'resource': 'purple'
  }
  return colorMap[type] || 'default'
}

onMounted(() => {
  loadRecords()
})
</script>

<style scoped>
.learning-records {
  padding: 24px;
}

.page-header {
  margin-bottom: 24px;
}

.page-title {
  font-size: 24px;
  font-weight: 600;
  margin: 0 0 8px 0;
}

.page-description {
  color: #666;
  margin: 0;
}

.content-wrapper {
  background: white;
  border-radius: 8px;
  padding: 24px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);
}

.stats-cards {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 20px;
  margin-bottom: 24px;
}

.stat-card {
  background: #f9f9f9;
  border-radius: 8px;
  padding: 20px;
  display: flex;
  align-items: center;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
}

.stat-icon {
  font-size: 24px;
  color: #1890ff;
  margin-right: 16px;
  padding: 12px;
  background: rgba(24, 144, 255, 0.1);
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
}

.stat-content {
  flex: 1;
}

.stat-value {
  font-size: 24px;
  font-weight: 600;
  color: #333;
  margin-bottom: 4px;
}

.stat-label {
  color: #666;
  font-size: 14px;
}

.section-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 16px;
}

.section-header h2 {
  font-size: 18px;
  margin: 0;
}

.chart-section {
  margin-bottom: 32px;
}

.chart-container {
  height: 300px;
  background: #f9f9f9;
  border-radius: 8px;
  display: flex;
  align-items: center;
  justify-content: center;
  border: 1px dashed #d9d9d9;
}

.chart-placeholder {
  text-align: center;
  color: #999;
}

.records-section {
  margin-bottom: 24px;
}

.filter-controls {
  display: flex;
  gap: 16px;
}

@media (max-width: 768px) {
  .stats-cards {
    grid-template-columns: repeat(2, 1fr);
  }
  
  .section-header {
    flex-direction: column;
    align-items: flex-start;
    gap: 16px;
  }
  
  .filter-controls {
    flex-direction: column;
    width: 100%;
  }
}
</style> 