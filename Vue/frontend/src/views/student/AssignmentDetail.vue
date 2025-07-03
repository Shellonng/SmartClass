<template>
  <div class="assignment-detail-page">
    <a-spin :spinning="loading">
      <div class="page-header">
        <a-button class="back-btn" type="link" @click="goBack">
          <arrow-left-outlined /> 返回
        </a-button>
      </div>

      <div v-if="assignment" class="assignment-detail-container">
        <div class="assignment-header">
          <h1 class="assignment-title">{{ assignment.title }}</h1>
          <div class="assignment-meta">
            <a-tag :color="getTypeColor(assignment.type)">{{ getTypeText(assignment.type) }}</a-tag>
            <a-tag :color="getModeColor(assignment.mode)">{{ getModeText(assignment.mode) }}</a-tag>
            <a-tag :color="getStatusColor(assignment.status)">{{ getStatusText(assignment.status) }}</a-tag>
          </div>
        </div>

        <div class="assignment-info">
          <div class="info-item">
            <clock-circle-outlined /> 开始时间：{{ formatDateTime(assignment.startTime) }}
          </div>
          <div class="info-item">
            <calendar-outlined /> 截止时间：{{ formatDateTime(assignment.endTime) }}
          </div>
          <div class="info-item" v-if="assignment.timeLimit">
            <hourglass-outlined /> 时间限制：{{ assignment.timeLimit }} 分钟
          </div>
        </div>

        <div class="assignment-description">
          <div class="section-title">作业说明</div>
          <div class="description-content">{{ assignment.description || '暂无说明' }}</div>
        </div>

        <div class="action-area">
          <a-button 
            type="primary" 
            size="large" 
            :disabled="!canStart" 
            @click="startAssignment"
          >
            开始{{ assignment.type === 'exam' ? '考试' : '作业' }}
          </a-button>
          <div v-if="!canStart" class="cannot-start-tip">
            {{ startDisabledReason }}
          </div>
        </div>
      </div>
      
      <a-empty v-else description="未找到作业信息" />
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
  HourglassOutlined
} from '@ant-design/icons-vue'
import assignmentApi from '@/api/assignment'
import dayjs from 'dayjs'

const route = useRoute()
const router = useRouter()
const loading = ref(true)
const assignment = ref<any>(null)
const assignmentId = ref<number>(Number(route.params.id) || 0)

// 加载作业详情
const loadAssignmentDetail = async () => {
  try {
    loading.value = true
    const response = await assignmentApi.getStudentAssignmentDetail(assignmentId.value)
    console.log('接收到的作业详情数据:', response)
    
    // 解析API返回的数据结构
    if (response && response.code === 200 && response.data) {
      assignment.value = response.data.assignment
      console.log('解析后的作业信息:', assignment.value)
      
      // 根据当前时间和截止时间判断状态
      const now = new Date()
      if (assignment.value.endTime && now > new Date(assignment.value.endTime)) {
        assignment.value.status = 'completed' // 已截止
      } else if (assignment.value.startTime && now < new Date(assignment.value.startTime)) {
        assignment.value.status = 'pending' // 未开始
      } else {
        assignment.value.status = 'in_progress' // 进行中
      }
    } else {
      message.error('获取作业详情失败: ' + (response?.message || '未知错误'))
    }
  } catch (error) {
    console.error('获取作业详情失败:', error)
    message.error('获取作业详情失败: ' + (error instanceof Error ? error.message : '未知错误'))
  } finally {
    loading.value = false
  }
}

// 判断是否可以开始作业/考试
const canStart = computed(() => {
  if (!assignment.value) return false
  
  const now = new Date()
  const startTime = assignment.value.startTime ? new Date(assignment.value.startTime) : null
  const endTime = assignment.value.endTime ? new Date(assignment.value.endTime) : null
  
  // 对于考试，在开始时间之前不能进入
  if (assignment.value.type === 'exam' && startTime && now < startTime) {
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
  if (!assignment.value) return ''
  
  const now = new Date()
  const startTime = assignment.value.startTime ? new Date(assignment.value.startTime) : null
  const endTime = assignment.value.endTime ? new Date(assignment.value.endTime) : null
  
  if (assignment.value.type === 'exam' && startTime && now < startTime) {
    return `考试将于 ${formatDateTime(assignment.value.startTime)} 开始，请届时参加`
  }
  
  if (endTime && now > endTime) {
    return '该作业/考试已截止，无法参加'
  }
  
  return ''
})

// 开始作业/考试
const startAssignment = () => {
  if (!canStart.value) {
    message.warning(startDisabledReason.value)
    return
  }
  
  const type = assignment.value.type
  const mode = assignment.value.mode
  
  console.log('开始作业/考试:', type, mode)
  console.log('作业完整信息:', assignment.value)
  
  try {
    // 构建完整的URL
    const baseUrl = window.location.origin
    
    if (type === 'homework') {
      if (mode === 'question') {
        // 答题型作业 - 统一使用/student/exams/:id/do路径
        const url = `${baseUrl}/student/exams/${assignmentId.value}/do`
        console.log('跳转到答题页面:', url)
        // 延迟跳转，确保页面准备就绪
        message.loading('正在准备答题页面...', 1)
        setTimeout(() => {
          try {
            router.push(`/student/exams/${assignmentId.value}/do`)
          } catch (err) {
            console.error('Router导航失败，使用location:', err)
            window.location.href = url
          }
        }, 500)
      } else if (mode === 'file') {
        // 文件上传型作业
        const url = `${baseUrl}/student/assignments/file/${assignmentId.value}/submit`
        console.log('跳转到文件上传页面:', url)
        // 延迟跳转，确保页面准备就绪
        message.loading('正在准备上传页面...', 1)
        setTimeout(() => {
          try {
            router.push(`/student/assignments/file/${assignmentId.value}/submit`)
          } catch (err) {
            console.error('Router导航失败，使用location:', err)
            window.location.href = url
          }
        }, 500)
      } else {
        // 未知模式，默认按答题型处理
        console.warn(`未知的作业模式: ${mode}，默认按答题型处理`)
        const url = `${baseUrl}/student/exams/${assignmentId.value}/do`
        message.loading('正在准备答题页面...', 1)
        setTimeout(() => {
          try {
            router.push(`/student/exams/${assignmentId.value}/do`)
          } catch (err) {
            console.error('Router导航失败，使用location:', err)
            window.location.href = url
          }
        }, 500)
      }
    } else if (type === 'exam') {
      // 考试路径
      const url = `${baseUrl}/student/exams/${assignmentId.value}/do`
      console.log('跳转到考试页面:', url)
      message.loading('正在准备考试页面...', 1)
      setTimeout(() => {
        try {
          router.push(`/student/exams/${assignmentId.value}/do`)
        } catch (err) {
          console.error('Router导航失败，使用location:', err)
          window.location.href = url
        }
      }, 500)
    } else {
      // 未知类型
      console.error(`不支持的任务类型: ${type}`)
      message.error(`不支持的任务类型: ${type}`)
    }
  } catch (error) {
    console.error('路由跳转失败:', error)
    message.error('跳转失败，请刷新页面后重试')
  }
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

// 获取类型文本
const getTypeText = (type: string) => {
  const typeMap: Record<string, string> = {
    'homework': '作业',
    'exam': '考试'
  }
  return typeMap[type] || '未知类型'
}

// 获取类型颜色
const getTypeColor = (type: string) => {
  const colorMap: Record<string, string> = {
    'homework': 'blue',
    'exam': 'orange'
  }
  return colorMap[type] || 'default'
}

// 获取模式文本
const getModeText = (mode: string) => {
  const modeMap: Record<string, string> = {
    'question': '答题模式',
    'file': '文件提交'
  }
  return modeMap[mode] || '未知模式'
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
  loadAssignmentDetail().then(() => {
    if (assignment.value) {
      console.log('检测到作业类型:', assignment.value.type, '模式:', assignment.value.mode)
      // 不再自动跳转，等待用户点击"开始作业"或"开始考试"按钮
    }
  })
})
</script>

<style scoped>
.assignment-detail-page {
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

.assignment-detail-container {
  background-color: #fff;
  border-radius: 8px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
  padding: 24px;
}

.assignment-header {
  margin-bottom: 24px;
  border-bottom: 1px solid #f0f0f0;
  padding-bottom: 16px;
}

.assignment-title {
  font-size: 24px;
  font-weight: 500;
  margin-bottom: 16px;
}

.assignment-meta {
  display: flex;
  gap: 8px;
}

.assignment-info {
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

.assignment-description {
  margin-bottom: 32px;
}

.description-content {
  background-color: #f9f9f9;
  padding: 16px;
  border-radius: 4px;
  white-space: pre-line;
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