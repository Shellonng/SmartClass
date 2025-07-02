<template>
  <div class="resource-management">
    <div class="resource-header">
      <h2>资料管理</h2>
      <a-button type="primary" @click="showUploadModal">
        <UploadOutlined />
        上传资料
      </a-button>
    </div>

    <div class="resource-content">
      <a-spin :spinning="loading">
        <a-empty v-if="resources.length === 0" description="暂无资料" />
        
        <a-table
          v-else
          :dataSource="resources"
          :columns="columns"
          :pagination="pagination"
          :rowKey="(record: CourseResource) => record.id"
          @change="handleTableChange"
        >
          <!-- 资源名称 -->
          <template #bodyCell="{ column, record }: { column: any, record: CourseResource }">
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

            <!-- 文件大小 -->
            <template v-else-if="column.dataIndex === 'fileSize'">
              {{ record.formattedSize }}
            </template>

            <!-- 上传时间 -->
            <template v-else-if="column.dataIndex === 'createTime'">
              {{ formatDate(record.createTime) }}
            </template>

            <!-- 操作 -->
            <template v-else-if="column.dataIndex === 'action'">
              <div class="resource-actions">
                <a-tooltip title="预览" v-if="canPreview(record.fileType)">
                  <a-button type="link" @click="showResourcePreview(record)">
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
                    @confirm="deleteResource(record)"
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
      title="上传资料"
      :maskClosable="false"
      @ok="handleUpload"
      :okButtonProps="{ loading: uploading }"
      :okText="uploading ? '上传中...' : '上传'"
    >
      <a-form :model="uploadForm" layout="vertical">
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
            <template #itemRender="{ file }">
              <div class="upload-item">
                <div class="upload-item-info">
                  <div class="file-icon">
                    <FileOutlined v-if="getFileType(file.name) === 'txt'" />
                    <FilePdfOutlined v-else-if="getFileType(file.name) === 'pdf'" />
                    <FileWordOutlined v-else-if="['doc', 'docx'].includes(getFileType(file.name))" />
                    <FilePptOutlined v-else-if="['ppt', 'pptx'].includes(getFileType(file.name))" />
                    <FileExcelOutlined v-else-if="['xls', 'xlsx'].includes(getFileType(file.name))" />
                    <FileZipOutlined v-else-if="['zip', 'rar', '7z'].includes(getFileType(file.name))" />
                    <FileImageOutlined v-else-if="['jpg', 'jpeg', 'png', 'gif'].includes(getFileType(file.name))" />
                    <FileOutlined v-else />
                  </div>
                  <div class="file-details">
                    <div class="file-name">{{ file.name }}</div>
                    <div class="file-size">{{ formatFileSize(file.size) }}</div>
                  </div>
                </div>
                <a-button type="text" @click="handleRemove">
                  <DeleteOutlined />
                </a-button>
              </div>
            </template>
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
import { ref, onMounted, computed, watch, defineProps } from 'vue'
import { useRoute } from 'vue-router'
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
import { formatDate } from '@/utils/date'
import type { CourseResource } from '@/api/course'
import {
  getTeacherCourseResources,
  getTeacherCourseResourcesPage,
  uploadCourseResource,
  deleteCourseResource,
  getResourceDownloadUrl,
  getResourcePreviewUrl,
  downloadResourceDirectly
} from '@/api/course'

const props = defineProps<{
  courseId?: number
}>()

const route = useRoute()
const courseId = computed(() => {
  // 优先使用传入的courseId，如果没有则尝试从路由获取
  if (props.courseId !== undefined) {
    console.log('CourseResources - 使用props传入的courseId:', props.courseId)
    return props.courseId
  }
  
  // 从路由参数获取courseId
  const idParam = route.params.courseId
  const idStr = typeof idParam === 'string' ? idParam : Array.isArray(idParam) ? idParam[0] : ''
  const id = parseInt(idStr)
  
  console.log('CourseResources - 从路由获取courseId:', {
    routeParams: route.params,
    idParam,
    idStr,
    id,
    isNaN: isNaN(id),
    final: isNaN(id) ? -1 : id
  })
  
  return isNaN(id) ? -1 : id
})

// 资源列表状态
const resources = ref<CourseResource[]>([])
const loading = ref(false)
const pagination = ref({
  current: 1,
  pageSize: 10,
  total: 0,
  onChange: (page: number, pageSize: number) => {
    pagination.value.current = page
    pagination.value.pageSize = pageSize
    fetchResources()
  }
})

// 表格列定义
const columns = [
  {
    title: '资源名称',
    dataIndex: 'name',
    key: 'name',
    ellipsis: true,
    width: '30%'
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
    width: '10%'
  },
  {
    title: '上传者',
    dataIndex: 'uploadUserName',
    key: 'uploadUserName',
    width: '15%'
  },
  {
    title: '上传时间',
    dataIndex: 'createTime',
    key: 'createTime',
    width: '20%'
  },
  {
    title: '操作',
    dataIndex: 'action',
    key: 'action',
    width: '15%'
  }
]

// 上传资源相关状态
const uploadModalVisible = ref(false)
const uploading = ref(false)
const fileList = ref<any[]>([])
const uploadForm = ref({
  name: '',
  description: ''
})

// 生命周期钩子
onMounted(() => {
  fetchResources()
})

// 监听路由变化
watch(
  () => route.params.courseId,
  (newId) => {
    if (newId) {
      fetchResources()
    }
  }
)

// 获取资源列表
const fetchResources = async () => {
  // 检查课程ID是否有效
  if (courseId.value <= 0) {
    message.error('无效的课程ID')
    console.error('无效的课程ID:', courseId.value)
    return
  }

  console.log('开始获取课程资源，课程ID:', courseId.value)
  loading.value = true
  try {
    const { data } = await getTeacherCourseResourcesPage(
      courseId.value,
      pagination.value.current,
      pagination.value.pageSize
    )
    
    console.log('获取课程资源响应:', data)
    
    if (data.code === 200) {
      resources.value = data.data.records
      pagination.value.total = data.data.total
      console.log('成功获取课程资源:', resources.value)
    } else {
      message.error(data.message || '获取资源列表失败')
      console.error('获取资源列表失败:', data)
    }
  } catch (error) {
    console.error('获取资源列表失败:', error)
    message.error('获取资源列表失败')
  } finally {
    loading.value = false
  }
}

// 显示上传弹窗
const showUploadModal = () => {
  uploadForm.value.name = ''
  uploadForm.value.description = ''
  fileList.value = []
  uploadModalVisible.value = true
}

// 文件上传前检查
const beforeUpload = (file: File) => {
  // 自动设置资源名称为文件名（不包含扩展名）
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

// 处理文件上传
const handleUpload = async () => {
  if (fileList.value.length === 0) {
    message.error('请选择要上传的文件')
    return
  }
  
  if (!uploadForm.value.name) {
    message.error('请输入资源名称')
    return
  }
  
  // 检查课程ID是否有效
  if (courseId.value <= 0) {
    message.error('无效的课程ID')
    return
  }
  
  uploading.value = true
  
  try {
    const file = fileList.value[0].originFileObj
    const { data } = await uploadCourseResource(
      courseId.value,
      file,
      uploadForm.value.name,
      uploadForm.value.description
    )
    
    if (data.code === 200) {
      message.success('资源上传成功')
      uploadModalVisible.value = false
      fetchResources()
    } else {
      message.error(data.message || '资源上传失败')
    }
  } catch (error) {
    console.error('资源上传失败:', error)
    message.error('资源上传失败')
  } finally {
    uploading.value = false
  }
}

// 删除资源
const deleteResource = async (resource: CourseResource) => {
  try {
    const { data } = await deleteCourseResource(resource.id)
    
    if (data.code === 200) {
      message.success('资源删除成功')
      fetchResources()
    } else {
      message.error(data.message || '资源删除失败')
    }
  } catch (error) {
    console.error('资源删除失败:', error)
    message.error('资源删除失败')
  }
}

// 下载资源
const downloadResource = async (resource: CourseResource | null) => {
  if (!resource) return
  
  console.log('下载资源:', {
    resourceId: resource.id,
    resourceName: resource.name
  })
  
  try {
    // 显示加载状态
    message.loading({ content: '正在下载文件...', key: 'download' })
    
    // 使用直接下载API
    const response = await downloadResourceDirectly(resource.id)
    
    // 创建Blob对象
    const blob = new Blob([response.data], { 
      type: response.headers['content-type'] || 'application/octet-stream' 
    })
    
    // 创建下载链接
    const url = window.URL.createObjectURL(blob)
    const link = document.createElement('a')
    link.href = url
    link.download = `${resource.name}.${resource.fileType}`
    document.body.appendChild(link)
    link.click()
    
    // 清理
    window.URL.revokeObjectURL(url)
    document.body.removeChild(link)
    
    // 显示成功消息
    message.success({ content: '下载成功', key: 'download' })
  } catch (error) {
    console.error('下载资源失败:', error)
    message.error({ content: '下载失败，请重试', key: 'download' })
  }
}

// 预览资源
const showResourcePreview = async (resource: CourseResource) => {
  // 构建完整的预览URL
  const baseUrl = 'http://localhost:8080'
  const previewUrl = `${baseUrl}${getResourcePreviewUrl(resource.id)}`
  
  console.log('预览资源:', {
    resourceId: resource.id,
    resourceName: resource.name,
    resourceType: resource.fileType,
    previewUrl: previewUrl
  })
  
  // 直接在新窗口打开预览
  window.open(previewUrl, '_blank');
}

// 检查是否可以预览
const canPreview = (fileType: string) => {
  const supportedTypes = ['pdf', 'jpg', 'jpeg', 'png', 'gif', 'mp4', 'mp3']
  return supportedTypes.includes(fileType.toLowerCase())
}

// 获取文件类型颜色
const getFileTypeColor = (fileType: string) => {
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

// 获取文件类型
const getFileType = (fileName: string) => {
  const lastDotIndex = fileName.lastIndexOf('.')
  if (lastDotIndex > 0) {
    return fileName.substring(lastDotIndex + 1).toLowerCase()
  }
  return ''
}

// 格式化文件大小
const formatFileSize = (size: number) => {
  if (size === 0) return '0 B'
  
  const units = ['B', 'KB', 'MB', 'GB', 'TB']
  const digitGroups = Math.floor(Math.log10(size) / Math.log10(1024))
  
  return `${(size / Math.pow(1024, digitGroups)).toFixed(2)} ${units[digitGroups]}`
}

// 表格变化事件
const handleTableChange = (pag: any) => {
  pagination.value.current = pag.current
  pagination.value.pageSize = pag.pageSize
  fetchResources()
}
</script>

<style scoped>
.resource-management {
  padding: 24px;
  background-color: #fff;
  border-radius: 8px;
}

.resource-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 24px;
}

.resource-header h2 {
  margin: 0;
  font-size: 20px;
  font-weight: 600;
}

.resource-content {
  background-color: #fff;
}

.resource-name {
  display: flex;
  align-items: center;
  gap: 8px;
}

.file-icon {
  font-size: 18px;
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
  font-size: 12px;
  color: #999;
  margin-top: 8px;
}

.upload-item {
  display: flex;
  justify-content: space-between;
  align-items: center;
  width: 100%;
}

.upload-item-info {
  display: flex;
  align-items: center;
  gap: 8px;
  flex: 1;
  overflow: hidden;
}

.file-details {
  flex: 1;
  overflow: hidden;
}

.file-size {
  font-size: 12px;
  color: #999;
}
</style> 