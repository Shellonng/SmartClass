<template>
  <div class="file-submit-page">
    <a-spin :spinning="loading">
      <div class="page-header">
        <a-button class="back-btn" type="link" @click="goBack">
          <arrow-left-outlined /> 返回
        </a-button>
      </div>

      <div v-if="assignment" class="file-submit-container">
        <div class="assignment-header">
          <h1 class="assignment-title">{{ assignment.title }}</h1>
          <div class="assignment-meta">
            <a-tag color="blue">文件提交作业</a-tag>
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
        </div>

        <div class="assignment-description">
          <div class="section-title">作业说明</div>
          <div class="description-content">{{ assignment.description || '暂无说明' }}</div>
        </div>

        <div class="file-upload-section">
          <div class="section-title">文件上传</div>
          <div class="upload-area">
            <a-upload-dragger
              v-model:fileList="fileList"
              :beforeUpload="beforeUpload"
              :multiple="false"
              :maxCount="1"
              @change="handleChange"
            >
              <p class="ant-upload-drag-icon">
                <inbox-outlined />
              </p>
              <p class="ant-upload-text">点击或拖拽文件到此区域上传</p>
              <p class="ant-upload-hint">
                支持单个文件上传，请确保文件符合要求
              </p>
            </a-upload-dragger>
          </div>
          
          <div class="upload-tips">
            <ul>
              <li>支持的文件格式：.doc, .docx, .pdf, .zip, .rar, .7z</li>
              <li>文件大小不超过50MB</li>
              <li>文件名请勿包含特殊字符</li>
            </ul>
          </div>
        </div>

        <div class="submission-section" v-if="previousSubmission">
          <div class="section-title">历史提交记录</div>
          <div class="submission-record">
            <div class="record-item">
              <div class="record-info">
                <div class="record-filename">{{ previousSubmission.fileName }}</div>
                <div class="record-time">提交时间：{{ formatDateTime(previousSubmission.submitTime) }}</div>
              </div>
              <div class="record-actions">
                <a-button type="link" @click="downloadFile(previousSubmission.fileUrl)">
                  <download-outlined /> 下载
                </a-button>
              </div>
            </div>
          </div>
        </div>

        <div class="action-area">
          <a-button 
            type="primary" 
            size="large" 
            :disabled="!canSubmit" 
            :loading="submitting"
            @click="submitAssignment"
          >
            提交作业
          </a-button>
          <div v-if="!canSubmit && !fileList.length" class="submit-tip">
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
import { message, Upload } from 'ant-design-vue'
import { 
  ArrowLeftOutlined, 
  ClockCircleOutlined, 
  CalendarOutlined, 
  InboxOutlined,
  DownloadOutlined
} from '@ant-design/icons-vue'
import assignmentApi from '@/api/assignment'
import dayjs from 'dayjs'
import type { UploadProps } from 'ant-design-vue'

const route = useRoute()
const router = useRouter()
const loading = ref(true)
const submitting = ref(false)
const assignment = ref<any>(null)
const assignmentId = ref<number>(Number(route.params.id) || 0)
const fileList = ref<any[]>([])
const previousSubmission = ref<any>(null)

// 判断是否可以提交
const canSubmit = computed(() => {
  return fileList.value.length > 0
})

// 加载作业详情
const loadAssignmentDetail = async () => {
  try {
    loading.value = true
    const response = await assignmentApi.getStudentAssignmentDetail(assignmentId.value)
    assignment.value = response
    
    // 根据当前时间和截止时间判断状态
    const now = new Date()
    if (assignment.value.endTime && now > new Date(assignment.value.endTime)) {
      assignment.value.status = 'completed' // 已截止
    } else if (assignment.value.startTime && now < new Date(assignment.value.startTime)) {
      assignment.value.status = 'pending' // 未开始
    } else {
      assignment.value.status = 'in_progress' // 进行中
    }
    
    // 获取历史提交记录
    try {
      const submissionResponse = await assignmentApi.getStudentSubmission(assignmentId.value)
      if (submissionResponse) {
        previousSubmission.value = submissionResponse
      }
    } catch (error) {
      console.error('获取历史提交记录失败:', error)
    }
  } catch (error) {
    console.error('获取作业详情失败:', error)
    message.error('获取作业详情失败')
  } finally {
    loading.value = false
  }
}

// 上传前验证
const beforeUpload: UploadProps['beforeUpload'] = (file) => {
  // 检查文件类型
  const validTypes = [
    'application/msword',
    'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
    'application/pdf',
    'application/zip',
    'application/x-rar-compressed',
    'application/x-7z-compressed'
  ]
  const isValidType = validTypes.includes(file.type)
  if (!isValidType) {
    message.error('只支持 .doc, .docx, .pdf, .zip, .rar, .7z 格式的文件!')
  }
  
  // 检查文件大小
  const isLt50M = file.size / 1024 / 1024 < 50
  if (!isLt50M) {
    message.error('文件大小不能超过50MB!')
  }
  
  return isValidType && isLt50M ? true : Upload.LIST_IGNORE
}

// 处理上传状态变化
const handleChange = (info: any) => {
  fileList.value = info.fileList.slice(-1)
}

// 下载文件
const downloadFile = (url: string) => {
  if (!url) {
    message.error('文件链接无效')
    return
  }
  
  window.open(url, '_blank')
}

// 提交作业
const submitAssignment = async () => {
  if (!canSubmit.value) {
    message.warning('请先上传文件')
    return
  }
  
  if (assignment.value.status === 'completed') {
    message.warning('作业已截止，无法提交')
    return
  }
  
  try {
    submitting.value = true
    
    // 创建FormData
    const formData = new FormData()
    formData.append('file', fileList.value[0].originFileObj)
    formData.append('assignmentId', assignmentId.value.toString())
    
    // 调用API提交文件
    await assignmentApi.submitAssignmentFile(assignmentId.value, formData)
    
    message.success('作业提交成功')
    
    // 刷新页面或重新加载数据
    loadAssignmentDetail()
  } catch (error) {
    console.error('提交作业失败:', error)
    message.error('提交作业失败，请重试')
  } finally {
    submitting.value = false
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
  loadAssignmentDetail()
})
</script>

<style scoped>
.file-submit-page {
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

.file-submit-container {
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

.file-upload-section {
  margin-bottom: 32px;
}

.upload-area {
  margin-bottom: 16px;
}

.upload-tips {
  color: #666;
  font-size: 14px;
}

.upload-tips ul {
  padding-left: 20px;
  margin: 0;
}

.submission-section {
  margin-bottom: 32px;
  border: 1px solid #f0f0f0;
  border-radius: 4px;
  padding: 16px;
}

.record-item {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.record-filename {
  font-weight: 500;
  margin-bottom: 4px;
}

.record-time {
  font-size: 12px;
  color: #999;
}

.action-area {
  text-align: center;
  margin-top: 32px;
  padding-top: 24px;
  border-top: 1px solid #f0f0f0;
}

.submit-tip {
  margin-top: 16px;
  color: #ff4d4f;
  font-size: 14px;
}
</style> 