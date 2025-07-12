<template>
  <div class="file-submission-page">
    <div class="page-header">
      <a-button class="back-btn" type="link" @click="goBack">
        <arrow-left-outlined /> 返回
      </a-button>
      <h1>{{ assignment?.title || '文件提交作业' }}</h1>
    </div>

    <a-spin :spinning="loading">
      <div v-if="assignment" class="submission-container">
        <div class="assignment-info">
          <div class="info-item">
            <clock-circle-outlined /> 开始时间：{{ formatDateTime(assignment.startTime) }}
          </div>
          <div class="info-item">
            <calendar-outlined /> 截止时间：{{ formatDateTime(assignment.endTime) }}
          </div>
        </div>

        <div class="assignment-description">
          <div class="section-title">作业说明</div>
          <div class="description-content">{{ assignment.description || '按照规范提交实验报告文档' }}</div>
        </div>

        <div class="upload-section">
          <div class="section-title">文件上传</div>
          <div class="upload-area">
            <a-upload-dragger
              :fileList="fileList"
              :beforeUpload="beforeUpload"
              :multiple="false"
              @change="handleChange"
              :showUploadList="true"
              :maxCount="1"
            >
              <p class="ant-upload-drag-icon">
                <inbox-outlined />
              </p>
              <p class="ant-upload-text">点击或拖拽文件到此区域上传</p>
              <p class="ant-upload-hint">
                支持单个文件上传，{{ allowedFileTypesText }}
              </p>
            </a-upload-dragger>
          </div>
        </div>

        <div class="submission-actions">
          <a-button 
            type="primary" 
            size="large" 
            :disabled="!canSubmit" 
            :loading="submitting"
            @click="submitAssignment"
          >
            提交作业
          </a-button>
          <div v-if="!canSubmit && fileList.length === 0" class="submission-tip">
            请先上传文件
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
  InboxOutlined
} from '@ant-design/icons-vue'
import assignmentApi from '@/api/assignment'
import dayjs from 'dayjs'

const route = useRoute()
const router = useRouter()
const loading = ref(true)
const submitting = ref(false)
const assignment = ref<any>(null)
const assignmentId = ref<number>(Number(route.params.id) || 0)
const fileList = ref<any[]>([])

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

// 允许上传的文件类型
const allowedFileTypes = computed(() => {
  if (!assignment.value || !assignment.value.allowedFileTypes) {
    return ['.pdf', '.doc', '.docx', '.zip']
  }
  return assignment.value.allowedFileTypes
})

// 文件类型文本展示
const allowedFileTypesText = computed(() => {
  return `支持的文件类型: ${allowedFileTypes.value.join(', ')}`
})

// 检查文件类型是否允许
const isFileTypeAllowed = (fileName: string) => {
  const fileExtension = '.' + fileName.split('.').pop()?.toLowerCase()
  return allowedFileTypes.value.includes(fileExtension)
}

// 上传前校验
const beforeUpload = (file: File) => {
  // 检查文件类型
  if (!isFileTypeAllowed(file.name)) {
    message.error(`不支持的文件类型，请上传${allowedFileTypes.value.join('、')}格式的文件`)
    return false
  }
  
  // 检查文件大小，限制为20MB
  const isLessThan20M = file.size / 1024 / 1024 < 20
  if (!isLessThan20M) {
    message.error('文件大小不能超过20MB')
    return false
  }
  
  return false // 阻止自动上传，改为手动提交
}

// 处理文件变化
const handleChange = (info: any) => {
  let newFileList = [...info.fileList]
  
  // 只保留最后一个文件
  newFileList = newFileList.slice(-1)
  
  // 更新文件状态
  newFileList = newFileList.map(file => {
    if (file.response) {
      file.url = file.response.url
    }
    return file
  })
  
  fileList.value = newFileList
}

// 是否可以提交
const canSubmit = computed(() => {
  return fileList.value.length > 0
})

// 提交作业
const submitAssignment = async () => {
  if (!canSubmit.value) {
    message.warning('请先上传文件')
    return
  }
  
  try {
    submitting.value = true
    
    const formData = new FormData()
    formData.append('file', fileList.value[0].originFileObj)
    
    const response = await assignmentApi.submitAssignmentFile(assignmentId.value, formData)
    
    if (response && response.code === 200) {
      message.success('作业提交成功')
      // 延迟跳转回作业详情页
      setTimeout(() => {
        router.push(`/student/assignments/${assignmentId.value}`)
      }, 1500)
    } else {
      message.error('作业提交失败: ' + (response?.message || '未知错误'))
    }
  } catch (error) {
    console.error('提交作业失败:', error)
    message.error('提交作业失败: ' + (error instanceof Error ? error.message : '未知错误'))
  } finally {
    submitting.value = false
  }
}

// 返回上一页
const goBack = () => {
  router.push(`/student/assignments/${assignmentId.value}`)
}

// 格式化日期时间
const formatDateTime = (date: string | Date) => {
  if (!date) return '未设置'
  return dayjs(date).format('YYYY-MM-DD HH:mm')
}

onMounted(() => {
  loadAssignmentDetail()
})
</script>

<style scoped>
.file-submission-page {
  padding: 24px;
  max-width: 1000px;
  margin: 0 auto;
}

.page-header {
  margin-bottom: 24px;
  display: flex;
  align-items: center;
}

.back-btn {
  font-size: 16px;
  padding: 0;
  margin-right: 16px;
}

.page-header h1 {
  margin: 0;
  font-size: 24px;
}

.submission-container {
  background-color: #fff;
  border-radius: 8px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
  padding: 24px;
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

.upload-section {
  margin-bottom: 32px;
}

.upload-area {
  padding: 16px 0;
}

.submission-actions {
  text-align: center;
  margin-top: 32px;
  padding-top: 24px;
  border-top: 1px solid #f0f0f0;
}

.submission-tip {
  margin-top: 16px;
  color: #ff4d4f;
  font-size: 14px;
}
</style> 