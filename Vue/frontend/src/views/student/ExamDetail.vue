<template>
  <div class="exam-detail-page">
    <a-spin :spinning="loading">
      <div class="page-header">
        <a-button class="back-btn" type="link" @click="goBack">
          <arrow-left-outlined /> 返回
        </a-button>
      </div>

      <div v-if="exam" class="exam-detail-container">
        <div class="exam-header">
          <h1 class="exam-title">{{ exam.title }}</h1>
          <div class="exam-meta">
            <a-tag color="orange">考试</a-tag>
            <a-tag :color="getStatusColor(exam.status)">{{ getStatusText(exam.status) }}</a-tag>
          </div>
        </div>

        <div class="exam-info">
          <div class="info-item">
            <clock-circle-outlined /> 开始时间：{{ formatDateTime(exam.startTime) }}
          </div>
          <div class="info-item">
            <calendar-outlined /> 截止时间：{{ formatDateTime(exam.endTime) }}
          </div>
          <div class="info-item" v-if="exam.timeLimit">
            <hourglass-outlined /> 时间限制：{{ exam.timeLimit }} 分钟
          </div>
          <div class="info-item" v-if="exam.totalScore">
            <trophy-outlined /> 总分值：{{ exam.totalScore }} 分
          </div>
        </div>

        <div class="exam-description">
          <div class="section-title">考试说明</div>
          <div class="description-content">{{ exam.description || '暂无说明' }}</div>
        </div>

        <div class="exam-rules">
          <div class="section-title">考试须知</div>
          <div class="rules-content">
            <ol>
              <li>考试时间严格按照规定的开始和结束时间，超时系统将自动提交。</li>
              <li>考试过程中请勿刷新页面或关闭浏览器，否则可能导致作答数据丢失。</li>
              <li>考试开始后计时器将自动启动，请合理安排答题时间。</li>
              <li>提交后将无法重新进入考试，请确认所有题目都已作答后再提交。</li>
            </ol>
          </div>
        </div>

        <div class="action-area">
          <a-button 
            type="primary" 
            size="large" 
            :disabled="!canStart" 
            @click="startExam"
            danger
          >
            开始考试
          </a-button>
          <div v-if="!canStart" class="cannot-start-tip">
            {{ startDisabledReason }}
          </div>
        </div>
      </div>
      
      <a-empty v-else description="未找到考试信息" />
    </a-spin>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { message } from 'ant-design-vue'
import { 
  ArrowLeftOutlined, 
  ClockCircleOutlined, 
  CalendarOutlined, 
  HourglassOutlined,
  TrophyOutlined
} from '@ant-design/icons-vue'
import assignmentApi from '@/api/assignment'
import dayjs from 'dayjs'

const route = useRoute()
const router = useRouter()
const loading = ref(true)
const exam = ref<any>(null)
const examId = ref<number>(Number(route.params.id) || 0)

// 加载考试详情
const loadExamDetail = async () => {
  try {
    loading.value = true
    // 使用相同的API，因为考试也是Assignment表中的数据
    const response = await assignmentApi.getStudentAssignmentDetail(examId.value)
    
    if (response.code === 200 && response.data) {
      exam.value = response.data.assignment || {}
      
      // 根据当前时间和截止时间判断状态
      const now = new Date()
      if (exam.value.endTime && now > new Date(exam.value.endTime)) {
        exam.value.status = 'completed' // 已截止
      } else if (exam.value.startTime && now < new Date(exam.value.startTime)) {
        exam.value.status = 'pending' // 未开始
      } else {
        exam.value.status = 'in_progress' // 进行中
      }
    } else {
      message.error(response.message || '获取考试详情失败')
    }
  } catch (error) {
    console.error('获取考试详情失败:', error)
    message.error('获取考试详情失败')
  } finally {
    loading.value = false
  }
}

// 判断是否可以开始考试
const canStart = computed(() => {
  if (!exam.value) return false
  
  const now = new Date()
  const startTime = exam.value.startTime ? new Date(exam.value.startTime) : null
  const endTime = exam.value.endTime ? new Date(exam.value.endTime) : null
  
  // 考试必须在开始时间之后才能进入
  if (startTime && now < startTime) {
    return false
  }
  
  // 如果已经截止，不能开始
  if (endTime && now > endTime) {
    return false
  }
  
  return true
})

// 不能开始的原因
const startDisabledReason = computed(() => {
  if (!exam.value) return ''
  
  const now = new Date()
  const startTime = exam.value.startTime ? new Date(exam.value.startTime) : null
  const endTime = exam.value.endTime ? new Date(exam.value.endTime) : null
  
  if (startTime && now < startTime) {
    return `考试将于 ${formatDateTime(exam.value.startTime)} 开始，请届时参加`
  }
  
  if (endTime && now > endTime) {
    return '该考试已截止，无法参加'
  }
  
  return ''
})

// 开始考试
const startExam = () => {
  if (!canStart.value) {
    message.warning(startDisabledReason.value)
    return
  }
  
  // 跳转到考试页面
  router.push(`/student/exams/${examId.value}/do`)
}

// 返回上一页
const goBack = () => {
  router.back()
}

// 格式化日期时间
const formatDateTime = (date: string | Date) => {
  if (!date) return '未设置'
  return dayjs(date).format('YYYY-MM-DD HH:mm')
}

// 获取模式文本
const getModeText = (mode: string) => {
  const modeMap: Record<string, string> = {
    'question': '答题模式',
    'file': '文件提交'
  }
  return modeMap[mode] || ''
}

// 获取模式颜色
const getModeColor = (mode: string) => {
  const colorMap: Record<string, string> = {
    'question': 'purple',
    'file': 'cyan'
  }
  return colorMap[mode] || 'default'
}

// 获取状态文本
const getStatusText = (status: string) => {
  const statusMap: Record<string, string> = {
    'pending': '未开始',
    'in_progress': '进行中',
    'completed': '已截止'
  }
  return statusMap[status] || '未知状态'
}

// 获取状态颜色
const getStatusColor = (status: string) => {
  const colorMap: Record<string, string> = {
    'pending': 'gold',
    'in_progress': 'green',
    'completed': 'red'
  }
  return colorMap[status] || 'default'
}

onMounted(() => {
  loadExamDetail()
})
</script>

<style scoped>
.exam-detail-page {
  padding: 24px;
  max-width: 1000px;
  margin: 0 auto;
}

.page-header {
  margin-bottom: 24px;
}

.back-btn {
  font-size: 16px;
  padding: 0;
}

.exam-detail-container {
  background-color: #fff;
  border-radius: 8px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
  padding: 24px;
}

.exam-header {
  margin-bottom: 24px;
  border-bottom: 1px solid #f0f0f0;
  padding-bottom: 16px;
}

.exam-title {
  font-size: 24px;
  font-weight: 500;
  margin-bottom: 16px;
}

.exam-meta {
  display: flex;
  gap: 8px;
}

.exam-info {
  margin-bottom: 24px;
}

.info-item {
  margin-bottom: 8px;
  font-size: 14px;
  color: #666;
}

.section-title {
  font-size: 18px;
  font-weight: 500;
  margin-bottom: 16px;
}

.exam-description {
  margin-bottom: 24px;
}

.description-content {
  background-color: #f9f9f9;
  padding: 16px;
  border-radius: 4px;
  white-space: pre-line;
}

.exam-rules {
  margin-bottom: 32px;
  border: 1px solid #ffe58f;
  background-color: #fffbe6;
  padding: 16px;
  border-radius: 4px;
}

.rules-content ol {
  margin-left: 20px;
  padding-left: 0;
}

.rules-content li {
  margin-bottom: 8px;
}

.action-area {
  text-align: center;
  margin-top: 32px;
  padding-top: 24px;
  border-top: 1px solid #f0f0f0;
}

.cannot-start-tip {
  margin-top: 16px;
  color: #ff4d4f;
  font-size: 14px;
}
</style> 