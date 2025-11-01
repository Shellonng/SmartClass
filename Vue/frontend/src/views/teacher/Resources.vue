<template>
  <div class="teacher-resources">
    <div class="page-header">
      <h1>资源管理</h1>
      <a-button type="primary" @click="showUploadModal">
          <UploadOutlined />
          上传资源
        </a-button>
    </div>
    
    <div class="resources-content">
      <a-spin :spinning="loading">
        <a-empty v-if="resources.length === 0" description="暂无资源" />
        <a-table 
          v-else 
          :dataSource="resources" 
          :columns="columns" 
          :pagination="pagination"
          :rowKey="record => record.id"
          @change="handleTableChange"
        >
          <!-- 资源名称 -->
          <template #bodyCell="{ column, record }">
            <template v-if="column.dataIndex === 'name'">
              <div class="resource-name">
                <div class="file-icon">
                  <FileOutlined v-if="record.fileType === 'txt'" />
                  <FilePdfOutlined v-else-if="record.fileType === 'pdf'" />
                  <FileWordOutlined v-else-if="['doc', 'docx'].includes(record.fileType)" />
                  <FilePptOutlined v-else-if="['ppt', 'pptx'].includes(record.fileType)" />
                  <FileExcelOutlined v-else-if="['xls', 'xlsx'].includes(record.fileType)" />
                  <FileZipOutlined v-else-if="['zip', 'rar', '7z'].includes(record.fileType)" />
                  <FileImageOutlined v-else-if="['jpg', 'jpeg', 'png', 'gif'].includes(record.fileType)" />
                  <FileOutlined v-else />
                </div>
                <span class="file-name">{{ record.name }}</span>
              </div>
            </template>

            <!-- 文件类型 -->
            <template v-else-if="column.dataIndex === 'fileType'">
              <a-tag :color="getFileTypeColor(record.fileType)">{{ record.fileType.toUpperCase() }}</a-tag>
            </template>

            <!-- 操作 -->
            <template v-else-if="column.dataIndex === 'action'">
              <div class="resource-actions">
                <a-tooltip title="预览" v-if="canPreview(record.fileType)">
                  <a-button type="link" @click="previewResource(record)">
                    <EyeOutlined />
                  </a-button>
                </a-tooltip>
                <a-tooltip title="下载">
                  <a-button type="link" @click="downloadResource(record)">
                    <DownloadOutlined />
                  </a-button>
                </a-tooltip>
                <a-tooltip title="删除">
                  <a-popconfirm
                    title="确定要删除这个资源吗？"
                    @confirm="deleteResource(record.id)"
                    ok-text="确定"
                    cancel-text="取消"
                  >
                    <a-button type="link" danger>
                      <DeleteOutlined />
                    </a-button>
                  </a-popconfirm>
                </a-tooltip>
              </div>
            </template>
          </template>
        </a-table>
      </a-spin>
    </div>
    
    <!-- 上传资源弹窗 -->
    <a-modal
      v-model:open="uploadModalVisible"
      title="上传资源"
      :maskClosable="false"
      @ok="handleUpload"
      :okButtonProps="{ loading: uploading }"
      :okText="uploading ? '上传中...' : '上传'"
    >
      <a-form :model="uploadForm" layout="vertical">
        <a-form-item label="课程" required>
          <a-select v-model:value="uploadForm.courseId" placeholder="请选择课程">
            <a-select-option v-for="course in courses" :key="course.id" :value="course.id">{{ course.name }}</a-select-option>
          </a-select>
        </a-form-item>
        <a-form-item label="资源名称" required>
          <a-input v-model:value="uploadForm.name" placeholder="请输入资源名称" />
        </a-form-item>
        <a-form-item label="资源描述">
          <a-textarea v-model:value="uploadForm.description" placeholder="请输入资源描述" :rows="4" />
        </a-form-item>
        <a-form-item label="选择文件" required>
          <a-upload
            v-model:fileList="fileList"
            :beforeUpload="beforeUpload"
            :multiple="false"
            :maxCount="1"
            @remove="handleRemove"
          >
            <a-button>
              <UploadOutlined />
              选择文件
            </a-button>
          </a-upload>
          <div class="upload-tip">
            支持的文件类型：PDF、Word、Excel、PPT、图片、压缩包等
          </div>
        </a-form-item>
      </a-form>
    </a-modal>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { message } from 'ant-design-vue'
import {
  UploadOutlined,
  DownloadOutlined,
  DeleteOutlined,
  EyeOutlined,
  FileOutlined,
  FilePdfOutlined,
  FileWordOutlined,
  FilePptOutlined,
  FileExcelOutlined,
  FileZipOutlined,
  FileImageOutlined
} from '@ant-design/icons-vue'
import request from '@/utils/request'
import type { ApiResponse } from '@/utils/request'
import type { CourseResource } from '@/api/course'
import {
  getTeacherCourseResources,
  uploadCourseResource,
  deleteCourseResource,
  downloadResourceDirectly,
  getUserResources,
  getResourcePreviewUrl
} from '@/api/course'
import { getTeacherCourses } from '@/api/teacher'
import { formatDate } from '@/utils/date'

// 资源列表状态
const resources = ref<CourseResource[]>([])
const courses = ref<any[]>([])
const loading = ref(false)
const pagination = ref({
  current: 1,
  pageSize: 10,
  total: 0,
})

// 上传资源相关状态
const uploadModalVisible = ref(false)
const uploading = ref(false)
const fileList = ref<any[]>([])
const uploadForm = ref({
  courseId: undefined as number | undefined,
  name: '',
  description: ''
})

// 表格列定义
const columns = [
  {
    title: '资源名称',
    dataIndex: 'name',
    key: 'name',
    ellipsis: true,
    width: '25%'
  },
  {
    title: '所属课程',
    dataIndex: 'courseName',
    key: 'courseName',
    width: '15%'
  },
  {
    title: '类型',
    dataIndex: 'fileType',
    key: 'fileType',
    width: '10%'
  },
  {
    title: '大小',
    dataIndex: 'fileSize',
    key: 'fileSize',
    width: '10%',
    render: (text: number, record: CourseResource) => record.formattedSize || formatFileSize(text)
  },
  {
    title: '上传者',
    dataIndex: 'uploadUserName',
    key: 'uploadUserName',
    width: '10%'
  },
  {
    title: '上传时间',
    dataIndex: 'createTime',
    key: 'createTime',
    width: '15%'
  },
  {
    title: '操作',
    dataIndex: 'action',
    key: 'action',
    width: '15%'
  }
]

onMounted(() => {
  fetchResources()
  fetchCourses()
})

// 获取资源列表
const fetchResources = async () => {
  loading.value = true
  try {
    // 使用新的API获取当前用户上传的所有资源
    const response = await getUserResources(
      pagination.value.current, 
      pagination.value.pageSize
    )
    
    console.log('获取用户资源响应:', response)
    
    if (response && response.data) {
      // 处理API返回的数据
      const data = response.data
      
      if (data.code === 200) {
        resources.value = data.data.records || []
        pagination.value.total = data.data.total || 0
        
        // 处理文件大小格式化
        resources.value.forEach(resource => {
          if (!resource.formattedSize && resource.fileSize) {
            resource.formattedSize = formatFileSize(resource.fileSize)
          }
        })
      } else {
        message.error(data.message || '获取资源列表失败')
      }
    } else {
      message.error('获取资源列表失败')
    }
  } catch (error) {
    console.error('获取资源列表失败:', error)
    message.error('获取资源列表失败，请稍后再试')
  } finally {
    loading.value = false
  }
}

// 获取课程列表
const fetchCourses = async () => {
  try {
    const response = await getTeacherCourses()
    
    if (response && response.data) {
      // 处理API返回的数据
      let coursesData: any[] = []
      
      // 根据返回数据结构处理
      if (Array.isArray(response.data)) {
        coursesData = response.data
      } else if (response.data.records || response.data.content || response.data.list) {
        coursesData = response.data.records || response.data.content || response.data.list
      } else if (response.data.code === 200 && response.data.data) {
        if (Array.isArray(response.data.data)) {
          coursesData = response.data.data
        } else if (response.data.data.records || response.data.data.content || response.data.data.list) {
          coursesData = response.data.data.records || response.data.data.content || response.data.data.list
        }
      }
      
      // 格式化课程数据
      courses.value = coursesData.map((course: any) => ({
        id: course.id,
        name: course.title || course.courseName || '未命名课程'
      }))
    } else {
      message.error('获取课程列表失败')
    }
  } catch (error) {
    console.error('获取课程列表失败:', error)
    message.error('获取课程列表失败')
  }
}

// 表格分页变化
const handleTableChange = (pag: any) => {
  pagination.value.current = pag.current
  pagination.value.pageSize = pag.pageSize
  fetchResources()
}

// 显示上传弹窗
const showUploadModal = () => {
  uploadForm.value = {
    courseId: undefined,
    name: '',
    description: ''
  }
  fileList.value = []
  uploadModalVisible.value = true
}

// 文件上传前处理
const beforeUpload = (file: File) => {
  if (!uploadForm.value.name) {
    const fileName = file.name
    const lastDotIndex = fileName.lastIndexOf('.')
    if (lastDotIndex > 0) {
      uploadForm.value.name = fileName.substring(0, lastDotIndex)
    } else {
      uploadForm.value.name = fileName
    }
  }
  return false // 阻止自动上传
}

// 移除文件
const handleRemove = () => {
  fileList.value = []
}

// 处理上传
const handleUpload = async () => {
  if (!uploadForm.value.courseId) {
    message.error('请选择课程')
    return
  }
  
  if (fileList.value.length === 0) {
    message.error('请选择要上传的文件')
    return
  }
  
  if (!uploadForm.value.name) {
    message.error('请输入资源名称')
    return
  }
  
  uploading.value = true
  try {
    const file = fileList.value[0].originFileObj
    const response = await uploadCourseResource(
      uploadForm.value.courseId,
      file,
      uploadForm.value.name,
      uploadForm.value.description
    )
    
    if (response.data && response.data.code === 200) {
      message.success('资源上传成功')
      uploadModalVisible.value = false
      fetchResources()
    } else {
      message.error(response.data?.message || '资源上传失败')
    }
  } catch (error) {
    console.error('资源上传失败:', error)
    message.error('资源上传失败')
  } finally {
    uploading.value = false
  }
}

// 删除资源
const deleteResource = async (id: number) => {
  try {
    const response = await deleteCourseResource(id)
    if (response.data && response.data.code === 200) {
      message.success('资源删除成功')
      fetchResources()
    } else {
      message.error(response.data?.message || '资源删除失败')
    }
  } catch (error) {
    console.error('资源删除失败:', error)
    message.error('资源删除失败')
  }
}

// 下载资源
const downloadResource = async (resource: CourseResource) => {
  try {
    console.log('开始下载资源:', resource.name)
    message.loading({ content: '正在下载文件...', key: 'download' })
    
    // 构建下载URL
    const baseUrl = import.meta.env.VITE_API_BASE_URL || ''
    const downloadUrl = `${baseUrl}/api/teacher/resources/${resource.id}/download`
    
    // 创建一个临时链接并模拟点击下载
    const link = document.createElement('a')
    link.href = downloadUrl
    link.setAttribute('download', `${resource.name}.${resource.fileType}`)
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
    
    message.success({ content: '下载成功', key: 'download' })
  } catch (error) {
    console.error('下载资源失败:', error)
    message.error({ content: '下载失败，请重试', key: 'download' })
  }
}

// 预览资源
const previewResource = (resource: CourseResource) => {
  try {
    console.log('预览资源:', resource.name)
    
    // 构建完整的预览URL
    const baseUrl = import.meta.env.VITE_API_BASE_URL || ''
    const previewUrl = `${baseUrl}/api/teacher/resources/${resource.id}/preview`
    
    console.log('预览URL:', previewUrl)
    
    // 根据文件类型选择预览方式
    if (['jpg', 'jpeg', 'png', 'gif'].includes(resource.fileType.toLowerCase())) {
      // 图片预览
      // TODO: 使用图片预览组件
      window.open(previewUrl, '_blank')
    } else if (resource.fileType.toLowerCase() === 'pdf') {
      // PDF预览
      window.open(previewUrl, '_blank')
    } else if (['mp4', 'webm'].includes(resource.fileType.toLowerCase())) {
      // 视频预览
      // TODO: 使用视频预览组件
      window.open(previewUrl, '_blank')
    } else if (['mp3', 'wav', 'ogg'].includes(resource.fileType.toLowerCase())) {
      // 音频预览
      // TODO: 使用音频预览组件
      window.open(previewUrl, '_blank')
    } else if (['doc', 'docx'].includes(resource.fileType.toLowerCase())) {
      // Word文档 - 使用Google Docs Viewer
      const encodedUrl = encodeURIComponent(previewUrl)
      window.open(`https://docs.google.com/viewer?url=${encodedUrl}&embedded=true`, '_blank')
    } else if (['ppt', 'pptx'].includes(resource.fileType.toLowerCase())) {
      // PowerPoint - 使用Google Docs Viewer
      const encodedUrl = encodeURIComponent(previewUrl)
      window.open(`https://docs.google.com/viewer?url=${encodedUrl}&embedded=true`, '_blank')
    } else if (['xls', 'xlsx'].includes(resource.fileType.toLowerCase())) {
      // Excel - 使用Google Docs Viewer
      const encodedUrl = encodeURIComponent(previewUrl)
      window.open(`https://docs.google.com/viewer?url=${encodedUrl}&embedded=true`, '_blank')
    } else {
      // 其他类型 - 直接下载
      message.info('此文件类型不支持在线预览，将为您下载')
      downloadResource(resource)
    }
  } catch (error) {
    console.error('预览资源失败:', error)
    message.error('预览失败，请稍后重试')
  }
}

// 检查是否可以预览
const canPreview = (fileType: string): boolean => {
  const supportedTypes = ['pdf', 'jpg', 'jpeg', 'png', 'gif', 'mp4', 'mp3']
  return supportedTypes.includes(fileType.toLowerCase())
}

// 获取文件类型颜色
const getFileTypeColor = (fileType: string): string => {
  const type = fileType.toLowerCase()
  if (type === 'pdf') return 'red'
  if (['doc', 'docx'].includes(type)) return 'blue'
  if (['ppt', 'pptx'].includes(type)) return 'orange'
  if (['xls', 'xlsx'].includes(type)) return 'green'
  if (['zip', 'rar', '7z'].includes(type)) return 'purple'
  if (['jpg', 'jpeg', 'png', 'gif'].includes(type)) return 'cyan'
  if (['mp4', 'mp3'].includes(type)) return 'magenta'
  return 'default'
}

// 格式化文件大小
const formatFileSize = (size: number): string => {
  if (size === null || size === undefined) return '-'
  if (size < 1024) return size + ' B'
  if (size < 1024 * 1024) return (size / 1024).toFixed(2) + ' KB'
  if (size < 1024 * 1024 * 1024) return (size / (1024 * 1024)).toFixed(2) + ' MB'
  return (size / (1024 * 1024 * 1024)).toFixed(2) + ' GB'
}
</script>

<style scoped>
.teacher-resources {
  padding: 24px;
  background-color: #fff;
}

.page-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 24px;
}

.page-header h1 {
  margin: 0;
  font-size: 24px;
  font-weight: 500;
}

.resources-content {
  margin-top: 20px;
}

.resource-name {
  display: flex;
  align-items: center;
}

.file-icon {
  margin-right: 8px;
  font-size: 16px;
}

.file-name {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.resource-actions {
  display: flex;
  gap: 8px;
}

.upload-tip {
  margin-top: 8px;
  color: #999;
  font-size: 12px;
}
</style> 