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
            <calendar-outlined /> 开始时间: {{ formatDateTime(assignment.startTime) || '未设置' }}
          </div>
          <div class="info-item">
            <clock-circle-outlined /> 截止时间: {{ formatDateTime(assignment.endTime) || '未设置' }}
          </div>
        </div>

        <div class="assignment-description">
          <div class="section-title">作业说明</div>
          <div class="description-content">{{ assignment.description || '暂无说明' }}</div>
        </div>

        <div class="file-upload-section" v-if="!isSubmitted">
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
        </div>

        <div class="submission-section" v-if="isSubmitted">
          <div class="section-title">已提交</div>
          <div class="submission-record">
            <div class="record-item">
              <div class="record-info">
                <div class="record-filename">{{ previousSubmission.fileName }}</div>
                <div class="record-time">提交时间：{{ formatDateTime(previousSubmission.submitTime) }}</div>
              </div>
            </div>
          </div>
        </div>

        <div class="action-area" v-if="!isSubmitted">
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
  InboxOutlined
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

// 判断是否已提交
const isSubmitted = computed(() => {
  return previousSubmission.value && previousSubmission.value.status === 1
})

// 判断是否可以提交
const canSubmit = computed(() => {
  return fileList.value.length > 0 && !isSubmitted.value
})

// 加载作业详情
const loadAssignmentDetail = async () => {
  try {
    loading.value = true
    const response = await assignmentApi.getStudentAssignmentDetail(assignmentId.value)
    
    if (response && response.code === 200 && response.data) {
      // 确保正确获取作业信息
      assignment.value = response.data.assignment
      console.log('作业信息:', assignment.value)
      
      // 如果有提交记录，设置previousSubmission
      if (response.data.submission) {
        previousSubmission.value = response.data.submission
        console.log('提交记录:', previousSubmission.value)
      }
      
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

// 上传前验证
const beforeUpload: UploadProps['beforeUpload'] = (file) => {
  // 检查文件类型
  const isAcceptedType = /\.(jpg|jpeg|png|gif|pdf|doc|docx|xls|xlsx|ppt|pptx|zip|rar|txt)$/i.test(file.name);
  if (!isAcceptedType) {
    message.error('只支持常见文档、图片和压缩文件格式!');
    return Upload.LIST_IGNORE;
  }
  
  // 检查文件大小
  const isLt50M = file.size / 1024 / 1024 < 50;
  if (!isLt50M) {
    message.error('文件大小不能超过50MB!');
    return Upload.LIST_IGNORE;
  }
  
  console.log('文件通过验证:', file.name, file.type, file.size);
  return isAcceptedType && isLt50M ? true : Upload.LIST_IGNORE;
}

// 处理上传状态变化
const handleChange = (info: any) => {
  fileList.value = info.fileList.slice(-1)
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
    
    // 确保文件有效
    if (!fileList.value[0] || !fileList.value[0].originFileObj) {
      message.error('无效的文件，请重新上传')
      submitting.value = false
      return
    }
    
    // 创建FormData
    const formData = new FormData()
    formData.append('file', fileList.value[0].originFileObj)
    
    const fileName = fileList.value[0].name
    console.log('开始提交文件:', assignmentId.value, fileName)
    
    // 调用API提交文件
    const response = await assignmentApi.submitAssignmentFile(assignmentId.value, formData)
    
    if (response && response.code === 200) {
      message.success('作业提交成功')
      
      // 创建本地提交记录对象（如果不存在）
      if (!previousSubmission.value) {
        previousSubmission.value = {
          assignmentId: assignmentId.value,
          status: 1,  // 设置为已提交状态
          submitTime: new Date(),
          fileName: fileName
        }
      } else {
        // 更新现有提交记录
        previousSubmission.value.status = 1
        previousSubmission.value.submitTime = new Date()
        previousSubmission.value.fileName = fileName
      }
      
      // 清空文件列表
      fileList.value = []
      
      // 重新加载作业详情
      await loadAssignmentDetail()
    } else {
      console.error('提交失败响应:', response)
      message.error('提交失败: ' + (response?.message || '未知错误'))
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
  console.log('FileSubmit组件加载，任务ID:', route.params.id)
  assignmentId.value = Number(route.params.id) || 0
  console.log('设置作业ID:', assignmentId.value)
  
  // 确保加载作业详情
  if (assignmentId.value > 0) {
    loadAssignmentDetail()
  } else {
    message.error('无效的作业ID')
    console.error('无效的作业ID:', route.params)
  }
})
</script>

<style scoped>
.file-submit-page {
  padding: 24px;
  max-width: 800px;
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

.submission-section {
  margin-bottom: 32px;
  border: 1px solid #f0f0f0;
  border-radius: 4px;
  padding: 16px;
  background-color: #f6ffed;
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