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
            <div class="stat-value">{{ formatDurationHours(statistics.totalDuration) }}</div>
            <div class="stat-label">总学习时长</div>
          </div>
        </div>
        <div class="stat-card">
          <div class="stat-icon">
            <ReadOutlined />
          </div>
          <div class="stat-content">
            <div class="stat-value">{{ statistics.totalLearningDays || 0 }}</div>
            <div class="stat-label">学习天数</div>
          </div>
        </div>
        <div class="stat-card">
          <div class="stat-icon">
            <FileTextOutlined />
          </div>
          <div class="stat-content">
            <div class="stat-value">{{ statistics.completedSections || 0 }}</div>
            <div class="stat-label">已完成章节</div>
          </div>
        </div>
        <div class="stat-card">
          <div class="stat-icon">
            <TrophyOutlined />
          </div>
          <div class="stat-content">
            <div class="stat-value">{{ statistics.viewedResources || 0 }}</div>
            <div class="stat-label">查看资源数</div>
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
        <div class="chart-container" id="dailyChart"></div>
      </div>
      
      <div class="charts-row">
        <div class="chart-section half-width">
          <div class="section-header">
            <h2>章节学习时长分布</h2>
          </div>
          <div class="chart-container" id="sectionChart"></div>
        </div>
        
        <div class="chart-section half-width">
          <div class="section-header">
            <h2>资源类型学习时长分布</h2>
          </div>
          <div class="chart-container" id="resourceTypeChart"></div>
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
                {{ formatDateTime(record.startTime) }}
              </template>
              
              <template v-else-if="column.key === 'duration'">
                {{ formatDuration(record.duration) }}
              </template>
              
              <template v-else-if="column.key === 'activityType'">
                <a-tag :color="getActivityColor(record.resourceType)">
                  {{ getActivityText(record.resourceType) }}
                </a-tag>
              </template>
              
              <template v-else-if="column.key === 'progress'">
                <a-progress
                  :percent="record.progress"
                  size="small"
                  :status="record.progress === 100 ? 'success' : 'active'"
                />
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
import { ref, reactive, onMounted, onUnmounted } from 'vue'
import { message } from 'ant-design-vue'
import dayjs from 'dayjs'
import * as echarts from 'echarts'
import { 
  ClockCircleOutlined,
  ReadOutlined,
  FileTextOutlined,
  TrophyOutlined
} from '@ant-design/icons-vue'
import { getLearningRecords, getLearningStatistics } from '@/api/learningRecord'

// 统计数据
const statistics = reactive({
  totalDuration: 0,
  totalLearningDays: 0,
  avgDailyDuration: 0,
  completedSections: 0,
  totalSections: 0,
  viewedResources: 0,
  dailyDurations: [],
  sectionDistribution: [],
  resourceTypeDistribution: []
})

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

// 图表实例
let dailyChart: echarts.ECharts | null = null;
let sectionChart: echarts.ECharts | null = null;
let resourceTypeChart: echarts.ECharts | null = null;

// 表格配置
const columns = [
  {
    title: '时间',
    dataIndex: 'startTime',
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
    dataIndex: 'resourceType',
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

// 加载学习统计数据
const loadStatistics = async () => {
  try {
    loading.value = true
    
    // 计算日期范围
    let startDate = dayjs().subtract(7, 'day').format('YYYY-MM-DD')
    let endDate = dayjs().format('YYYY-MM-DD')
    
    if (timeRange.value === 'month') {
      startDate = dayjs().subtract(30, 'day').format('YYYY-MM-DD')
    } else if (timeRange.value === 'semester') {
      startDate = dayjs().subtract(120, 'day').format('YYYY-MM-DD')
    }
    
    const courseId = courseFilter.value ? Number(courseFilter.value) : 1 // 默认使用第一个课程
    
    const response = await getLearningStatistics(courseId, startDate, endDate)
    
    if (response && response.data) {
      Object.assign(statistics, response.data)
      renderCharts()
    }
    
  } catch (error) {
    console.error('加载学习统计数据失败:', error)
    message.error('加载学习统计数据失败')
  } finally {
    loading.value = false
  }
}

// 加载学习记录
const loadRecords = async () => {
  try {
    loading.value = true
    
    const courseId = courseFilter.value ? Number(courseFilter.value) : undefined
    
    const response = await getLearningRecords(courseId)
    
    if (response && response.data) {
      records.value = response.data
      pagination.value.total = records.value.length
    } else {
      // 如果后端暂未实现或测试阶段，使用模拟数据
      records.value = [
        {
          id: 1,
          startTime: '2025-07-02 14:30:00',
          endTime: '2025-07-02 15:15:00',
          courseName: '计算机组成原理',
          courseId: 1,
          resourceType: 'video',
          content: '第1章：计算机系统概述',
          duration: 2700,
          progress: 100
        },
        {
          id: 2,
          startTime: '2025-07-02 15:20:00',
          endTime: '2025-07-02 15:35:00',
          courseName: '计算机组成原理',
          courseId: 1,
          resourceType: 'quiz',
          content: '第1章测验',
          duration: 900,
          progress: 100
        },
        {
          id: 3,
          startTime: '2025-07-01 10:15:00',
          endTime: '2025-07-01 11:15:00',
          courseName: 'Java程序设计',
          courseId: 2,
          resourceType: 'assignment',
          content: '第3章作业：面向对象编程',
          duration: 3600,
          progress: 100
        },
        {
          id: 4,
          startTime: '2025-07-01 16:45:00',
          endTime: '2025-07-01 17:15:00',
          courseName: '微观经济学原理',
          courseId: 3,
          resourceType: 'resource',
          content: '阅读材料：需求曲线分析',
          duration: 1800,
          progress: 75
        },
        {
          id: 5,
          startTime: '2025-06-30 09:30:00',
          endTime: '2025-06-30 11:30:00',
          courseName: 'Java程序设计',
          courseId: 2,
          resourceType: 'exam',
          content: '期中考试',
          duration: 7200,
          progress: 100
        }
      ]
      pagination.value.total = records.value.length
    }
  } catch (error) {
    console.error('加载学习记录失败:', error)
    message.error('加载学习记录失败')
  } finally {
    loading.value = false
  }
}

// 初始化图表
const initCharts = () => {
  // 每日学习时长图表
  dailyChart = echarts.init(document.getElementById('dailyChart'))
  
  // 章节学习时长分布图表
  sectionChart = echarts.init(document.getElementById('sectionChart'))
  
  // 资源类型学习时长分布图表
  resourceTypeChart = echarts.init(document.getElementById('resourceTypeChart'))
  
  // 窗口大小变化时自动调整图表大小
  window.addEventListener('resize', () => {
    dailyChart?.resize()
    sectionChart?.resize()
    resourceTypeChart?.resize()
  })
}

// 渲染图表
const renderCharts = () => {
  renderDailyChart()
  renderSectionChart()
  renderResourceTypeChart()
}

// 渲染每日学习时长图表
const renderDailyChart = () => {
  if (!dailyChart) return
  
  const days = statistics.dailyDurations.map((item: any) => item.date)
  const durations = statistics.dailyDurations.map((item: any) => Math.round(item.duration / 60)) // 转换为分钟
  
  const option = {
    title: {
      text: '每日学习时长'
    },
    tooltip: {
      trigger: 'axis',
      formatter: function(params: any) {
        const data = params[0].data
        return `${params[0].name}: ${formatDuration(data)}`
      }
    },
    xAxis: {
      type: 'category',
      data: days,
      axisLabel: {
        rotate: 45
      }
    },
    yAxis: {
      type: 'value',
      name: '时长(分钟)'
    },
    series: [
      {
        name: '学习时长',
        type: 'bar',
        data: durations,
        itemStyle: {
          color: '#1890ff'
        }
      }
    ]
  }
  
  dailyChart.setOption(option)
}

// 渲染章节学习时长分布图表
const renderSectionChart = () => {
  if (!sectionChart) return
  
  const data = statistics.sectionDistribution.map((item: any) => ({
    name: item.section_title,
    value: Math.round(item.duration / 60) // 转换为分钟
  }))
  
  const option = {
    title: {
      text: '章节学习时长分布'
    },
    tooltip: {
      trigger: 'item',
      formatter: function(params: any) {
        return `${params.name}: ${formatDuration(params.value * 60)}`
      }
    },
    series: [
      {
        name: '章节学习时长',
        type: 'pie',
        radius: '65%',
        center: ['50%', '50%'],
        data: data,
        emphasis: {
          itemStyle: {
            shadowBlur: 10,
            shadowOffsetX: 0,
            shadowColor: 'rgba(0, 0, 0, 0.5)'
          }
        }
      }
    ]
  }
  
  sectionChart.setOption(option)
}

// 渲染资源类型学习时长分布图表
const renderResourceTypeChart = () => {
  if (!resourceTypeChart) return
  
  const typeNameMap: Record<string, string> = {
    'video': '视频',
    'quiz': '测验',
    'assignment': '作业',
    'exam': '考试',
    'resource': '资源'
  }
  
  const data = statistics.resourceTypeDistribution.map((item: any) => ({
    name: typeNameMap[item.resource_type] || item.resource_type,
    value: Math.round(item.duration / 60) // 转换为分钟
  }))
  
  const option = {
    title: {
      text: '资源类型学习时长分布'
    },
    tooltip: {
      trigger: 'item',
      formatter: function(params: any) {
        return `${params.name}: ${formatDuration(params.value * 60)}`
      }
    },
    series: [
      {
        name: '资源类型',
        type: 'pie',
        radius: '65%',
        center: ['50%', '50%'],
        data: data,
        emphasis: {
          itemStyle: {
            shadowBlur: 10,
            shadowOffsetX: 0,
            shadowColor: 'rgba(0, 0, 0, 0.5)'
          }
        }
      }
    ]
  }
  
  resourceTypeChart.setOption(option)
}

// 处理时间范围变化
const handleTimeRangeChange = () => {
  loadStatistics()
}

// 处理筛选
const handleFilter = () => {
  pagination.value.current = 1
  loadRecords()
  loadStatistics()
}

// 处理日期范围变化
const handleDateRangeChange = (dates: any) => {
  loadRecords()
}

// 处理表格变化
const handleTableChange = (pag: any, filters: any, sorter: any) => {
  pagination.value.current = pag.current
  pagination.value.pageSize = pag.pageSize
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

// 格式化学习时长（秒转为小时分钟）
const formatDuration = (seconds: number): string => {
  if (seconds < 60) {
    return `${seconds}秒`
  }
  const minutes = Math.floor(seconds / 60)
  if (minutes < 60) {
    return `${minutes}分钟`
  }
  const hours = Math.floor(minutes / 60)
  const remainMinutes = minutes % 60
  return remainMinutes > 0 ? `${hours}小时${remainMinutes}分钟` : `${hours}小时`
}

// 格式化学习时长（秒转为小时，保留1位小数）
const formatDurationHours = (seconds: number): string => {
  const hours = seconds / 3600
  return hours.toFixed(1)
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

// 清理图表
onUnmounted(() => {
  dailyChart?.dispose()
  sectionChart?.dispose()
  resourceTypeChart?.dispose()
  window.removeEventListener('resize', () => {})
})

onMounted(() => {
  loadRecords()
  initCharts()
  loadStatistics()
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
  border-radius: 8px;
  overflow: hidden;
}

.charts-row {
  display: flex;
  gap: 24px;
  margin-bottom: 32px;
}

.half-width {
  flex: 1;
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
  
  .charts-row {
    flex-direction: column;
  }
}
</style> 