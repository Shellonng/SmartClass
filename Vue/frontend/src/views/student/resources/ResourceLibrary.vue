<template>
  <div class="resource-library">
    <a-row class="resource-header">
      <a-col :span="12">
        <h2>资源库</h2>
        <p>这里包含了您所选课程的全部学习资源</p>
      </a-col>
    </a-row>

    <a-row class="filter-box">
      <a-col :span="24">
        <a-space>
          <a-input-search
            v-model:value="searchKeyword"
            placeholder="搜索资源"
            @search="handleSearch"
            :loading="searching"
            style="width: 250px"
          />
          
          <a-select
            v-model:value="fileType"
            placeholder="文件类型"
            style="width: 120px"
            @change="filterResources"
          >
            <a-select-option value="">全部类型</a-select-option>
            <a-select-option value="pdf">PDF</a-select-option>
            <a-select-option value="doc">Word文档</a-select-option>
            <a-select-option value="ppt">PPT</a-select-option>
            <a-select-option value="xls">Excel</a-select-option>
            <a-select-option value="mp4">视频</a-select-option>
            <a-select-option value="jpg">图片</a-select-option>
            <a-select-option value="zip">压缩文件</a-select-option>
            <a-select-option value="other">其他</a-select-option>
          </a-select>

          <a-select
            v-model:value="sortBy"
            placeholder="排序方式"
            style="width: 150px"
            @change="filterResources"
          >
            <a-select-option value="createTime">上传时间降序</a-select-option>
            <a-select-option value="createTimeAsc">上传时间升序</a-select-option>
            <a-select-option value="fileSize">文件大小降序</a-select-option>
            <a-select-option value="fileSizeAsc">文件大小升序</a-select-option>
            <a-select-option value="downloadCount">下载次数降序</a-select-option>
            <a-select-option value="name">名称排序</a-select-option>
          </a-select>
          
          <a-select
            v-model:value="courseId"
            placeholder="所属课程"
            style="width: 200px"
            @change="filterResources"
          >
            <a-select-option value="">全部课程</a-select-option>
            <a-select-option v-for="course in courseOptions" :key="course.id" :value="course.id">
              {{ course.title }}
            </a-select-option>
          </a-select>
        </a-space>
      </a-col>
    </a-row>

    <a-spin :spinning="loading">
      <div v-if="filteredResources.length === 0 && !loading" class="empty-resources">
        <a-empty description="暂无资源" />
      </div>
      
      <a-row :gutter="[16, 16]" v-else>
        <a-col :xs="24" :sm="12" :md="8" :lg="6" v-for="resource in filteredResources" :key="resource.id">
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
              <a-button type="default" size="small" @click="handleFavorite(resource)">
                <template #icon><star-outlined /></template>
                收藏
              </a-button>
            </div>
          </a-card>
        </a-col>
      </a-row>
    </a-spin>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted, computed } from 'vue'
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
  StarOutlined
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
const loading = ref(true)
const searching = ref(false)
const resources = ref<Resource[]>([])
const filteredResources = ref<Resource[]>([])
const searchKeyword = ref('')
const fileType = ref('')
const sortBy = ref('createTime')
const courseId = ref('')
const courseOptions = ref<Course[]>([])
const courseMap = ref<Map<number, string>>(new Map())

// 获取所有资源
const fetchResources = async () => {
  loading.value = true
  try {
    const data = await getAllStudentResources()
    resources.value = data
    filterResources()
  } catch (error) {
    console.error('获取资源失败:', error)
    message.error('获取资源列表失败，请稍后再试')
  } finally {
    loading.value = false
  }
}

// 获取学生已选课程
const fetchCourses = async () => {
  try {
    const courses = await getEnrolledCourses()
    courseOptions.value = courses
    
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

// 筛选和排序资源
const filterResources = () => {
  // 先应用搜索和文件类型筛选
  let filtered = resources.value.filter(resource => {
    let matchesKeyword = true
    let matchesFileType = true
    let matchesCourse = true
    
    if (searchKeyword.value) {
      matchesKeyword = resource.name.toLowerCase().includes(searchKeyword.value.toLowerCase())
    }
    
    if (fileType.value) {
      matchesFileType = resource.fileType.toLowerCase() === fileType.value.toLowerCase()
    }
    
    if (courseId.value) {
      matchesCourse = resource.courseId === parseInt(courseId.value)
    }
    
    return matchesKeyword && matchesFileType && matchesCourse
  })
  
  // 再应用排序
  filtered.sort((a, b) => {
    switch (sortBy.value) {
      case 'createTimeAsc':
        return new Date(a.createTime).getTime() - new Date(b.createTime).getTime()
      case 'createTime':
        return new Date(b.createTime).getTime() - new Date(a.createTime).getTime()
      case 'fileSize':
        return b.fileSize - a.fileSize
      case 'fileSizeAsc':
        return a.fileSize - b.fileSize
      case 'downloadCount':
        return (b.downloadCount || 0) - (a.downloadCount || 0)
      case 'name':
        return a.name.localeCompare(b.name)
      default:
        return new Date(b.createTime).getTime() - new Date(a.createTime).getTime()
    }
  })
  
  filteredResources.value = filtered
}

// 搜索处理
const handleSearch = () => {
  searching.value = true
  filterResources()
  setTimeout(() => {
    searching.value = false
  }, 300)
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

// 收藏资源
const handleFavorite = (resource: Resource) => {
  message.success(`已收藏资源：${resource.name}`)
  // 这里可以实现收藏功能的API调用
}

// 页面加载完成后获取资源
onMounted(() => {
  fetchCourses()
  fetchResources()
})
</script> 

<style scoped>
.resource-library {
  padding: 20px;
}

.resource-header {
  margin-bottom: 24px;
}

.resource-header h2 {
  margin-bottom: 4px;
  font-size: 24px;
}

.resource-header p {
  color: #666;
}

.search-box {
  display: flex;
  justify-content: flex-end;
  align-items: center;
}

.filter-box {
  margin-bottom: 24px;
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