<template>
  <div class="favorites">
    <a-row class="favorites-header">
      <a-col :span="24">
        <h2>我的收藏</h2>
        <p>这里包含了您收藏的所有学习资源</p>
      </a-col>
    </a-row>

    <a-spin :spinning="loading">
      <div v-if="favoriteResources.length === 0 && !loading" class="empty-resources">
        <a-empty description="暂无收藏资源" />
        <div class="empty-action">
          <a-button type="primary" @click="goToResourceLibrary">
            去资源库浏览
          </a-button>
        </div>
      </div>
      
      <a-row :gutter="[16, 16]" v-else>
        <a-col :xs="24" :sm="12" :md="8" :lg="6" v-for="resource in favoriteResources" :key="resource.id">
          <a-card hoverable class="resource-card">
            <template #cover>
              <div class="resource-icon">
                <file-pdf-outlined v-if="resource.fileType === 'pdf'" class="pdf-icon" />
                <file-word-outlined v-else-if="resource.fileType === 'doc' || resource.fileType === 'docx'" class="word-icon" />
                <file-ppt-outlined v-else-if="resource.fileType === 'ppt' || resource.fileType === 'pptx'" class="ppt-icon" />
                <file-excel-outlined v-else-if="resource.fileType === 'xls' || resource.fileType === 'xlsx'" class="excel-icon" />
                <video-camera-outlined v-else-if="resource.fileType === 'mp4' || resource.fileType === 'mov'" class="video-icon" />
                <picture-outlined v-else-if="['jpg', 'jpeg', 'png', 'gif'].includes(resource.fileType)" class="image-icon" />
                <file-zip-outlined v-else-if="resource.fileType === 'zip' || resource.fileType === 'rar'" class="zip-icon" />
                <file-outlined v-else class="file-icon" />
              </div>
            </template>
            <a-card-meta :title="resource.name">
              <template #description>
                <div class="resource-meta">
                  <div class="resource-course">
                    <span>所属课程：</span>
                    <span class="course-name">{{ getCourseNameById(resource.courseId) }}</span>
                  </div>
                  <div class="resource-info">
                    <span class="resource-size">{{ resource.formattedSize || formatFileSize(resource.fileSize) }}</span>
                    <span class="resource-type">{{ resource.fileType.toUpperCase() }}</span>
                  </div>
                  <div class="resource-download">
                    <download-outlined /> <span>{{ resource.downloadCount || 0 }}</span>
                  </div>
                </div>
              </template>
            </a-card-meta>
            <div class="card-actions">
              <a-button type="primary" size="small" @click="handlePreview(resource)">
                <template #icon><eye-outlined /></template>
                预览
              </a-button>
              <a-button type="default" size="small" @click="handleDownload(resource)">
                <template #icon><download-outlined /></template>
                下载
              </a-button>
              <a-button type="danger" size="small" @click="handleUnfavorite(resource)">
                <template #icon><star-filled /></template>
                取消收藏
              </a-button>
            </div>
          </a-card>
        </a-col>
      </a-row>
    </a-spin>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { message } from 'ant-design-vue'
import { 
  FilePdfOutlined, 
  FileWordOutlined, 
  FilePptOutlined, 
  FileExcelOutlined, 
  FileZipOutlined,
  FileOutlined, 
  PictureOutlined,
  VideoCameraOutlined,
  DownloadOutlined,
  EyeOutlined,
  StarFilled
} from '@ant-design/icons-vue'
import { getAllStudentResources, downloadResourceDirectly, previewResourceDirectly, getEnrolledCourses } from '@/api/course'

// 定义资源类型接口
interface Resource {
  id: number
  courseId: number
  name: string
  fileType: string
  fileSize: number
  fileUrl: string
  description?: string
  downloadCount: number
  formattedSize: string
  uploadUserId: number
  uploadUserName?: string
  createTime: string
}

interface Course {
  id: number
  title: string
}

// 状态变量
const router = useRouter()
const loading = ref(true)
const favoriteResources = ref<Resource[]>([])
const courseMap = ref<Map<number, string>>(new Map())

// 模拟获取收藏的资源 (未来需要替换为实际API)
const fetchFavoriteResources = async () => {
  loading.value = true
  try {
    // 这里应该调用获取收藏资源的API
    // 当前使用getAllStudentResources模拟，后续可以添加专门的收藏API
    const data = await getAllStudentResources()
    // 模拟收藏了前3个资源
    favoriteResources.value = data.slice(0, 3)
  } catch (error) {
    console.error('获取收藏资源失败:', error)
    message.error('获取收藏列表失败，请稍后再试')
  } finally {
    loading.value = false
  }
}

// 获取学生已选课程
const fetchCourses = async () => {
  try {
    const courses = await getEnrolledCourses()
    
    // 构建课程ID到课程名称的映射
    courses.forEach(course => {
      courseMap.value.set(course.id, course.title)
    })
  } catch (error) {
    console.error('获取课程列表失败:', error)
  }
}

// 根据课程ID获取课程名称
const getCourseNameById = (id: number): string => {
  return courseMap.value.get(id) || '未知课程'
}

// 格式化文件大小
const formatFileSize = (bytes: number): string => {
  if (!bytes) return '0 B'
  const units = ['B', 'KB', 'MB', 'GB', 'TB']
  let i = 0
  while (bytes >= 1024 && i < units.length - 1) {
    bytes /= 1024
    i++
  }
  return `${bytes.toFixed(2)} ${units[i]}`
}

// 资源预览
const handlePreview = (resource: Resource) => {
  try {
    previewResourceDirectly(resource.id)
  } catch (error) {
    console.error('预览资源失败:', error)
    message.error('预览资源失败，请稍后再试')
  }
}

// 资源下载
const handleDownload = (resource: Resource) => {
  try {
    downloadResourceDirectly(resource.id)
    message.success('资源下载中')
  } catch (error) {
    console.error('下载资源失败:', error)
    message.error('下载资源失败，请稍后再试')
  }
}

// 取消收藏
const handleUnfavorite = (resource: Resource) => {
  // 这里应该调用取消收藏API
  message.success(`已取消收藏：${resource.name}`)
  // 从列表中移除
  favoriteResources.value = favoriteResources.value.filter(item => item.id !== resource.id)
}

// 跳转到资源库
const goToResourceLibrary = () => {
  router.push('/student/resources/library')
}

// 页面加载完成后获取资源
onMounted(() => {
  fetchCourses()
  fetchFavoriteResources()
})
</script> 

<style scoped>
.favorites {
  padding: 20px;
}

.favorites-header {
  margin-bottom: 24px;
}

.favorites-header h2 {
  margin-bottom: 4px;
  font-size: 24px;
}

.favorites-header p {
  color: #666;
}

.resource-card {
  height: 100%;
  display: flex;
  flex-direction: column;
  transition: all 0.3s;
}

.resource-card:hover {
  transform: translateY(-5px);
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
}

.resource-icon {
  display: flex;
  justify-content: center;
  align-items: center;
  height: 120px;
  background-color: #f5f5f5;
  font-size: 48px;
}

.pdf-icon {
  color: #ff4d4f;
}

.word-icon {
  color: #1890ff;
}

.ppt-icon {
  color: #fa8c16;
}

.excel-icon {
  color: #52c41a;
}

.video-icon {
  color: #722ed1;
}

.image-icon {
  color: #eb2f96;
}

.zip-icon {
  color: #faad14;
}

.file-icon {
  color: #8c8c8c;
}

.resource-meta {
  margin-top: 8px;
  font-size: 12px;
}

.resource-course {
  margin-bottom: 4px;
  color: #333;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.course-name {
  color: #1890ff;
}

.resource-info {
  display: flex;
  justify-content: space-between;
  margin-bottom: 4px;
}

.resource-size {
  color: #666;
}

.resource-type {
  color: #666;
  font-weight: bold;
}

.resource-download {
  color: #666;
}

.empty-resources {
  padding: 40px 0;
  text-align: center;
}

.empty-action {
  margin-top: 16px;
}

.card-actions {
  margin-top: 16px;
  display: flex;
  justify-content: space-between;
}

.card-actions .ant-btn {
  flex: 1;
  margin: 0 4px;
}
</style> 