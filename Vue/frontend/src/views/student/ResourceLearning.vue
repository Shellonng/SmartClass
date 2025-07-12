<template>
  <div class="resource-learning">
    <a-spin :spinning="loading">
      <div class="resource-header">
        <a-button type="text" @click="goBack" class="back-btn">
          <ArrowLeftOutlined />
          返回课程详情
        </a-button>
        <h1>{{ resource.title }}</h1>
        <div class="resource-meta">
          <span><BookOutlined /> 课程：{{ courseName }}</span>
          <span><FileOutlined /> 类型：{{ getResourceTypeText(resource.fileType) }}</span>
          <span><BarChartOutlined /> 进度：{{ progress }}%</span>
        </div>
      </div>

      <div class="resource-container">
        <div class="resource-viewer">
          <!-- PDF预览 -->
          <div v-if="isPdf" class="pdf-viewer">
            <div class="pdf-controls">
              <a-button-group>
                <a-button @click="prevPage" :disabled="currentPage <= 1">
                  <LeftOutlined />
                </a-button>
                <a-button disabled>{{ currentPage }} / {{ totalPages }}</a-button>
                <a-button @click="nextPage" :disabled="currentPage >= totalPages">
                  <RightOutlined />
                </a-button>
              </a-button-group>
              <a-button-group>
                <a-button @click="zoomOut">
                  <ZoomOutOutlined />
                </a-button>
                <a-button disabled>{{ zoomLevel }}%</a-button>
                <a-button @click="zoomIn">
                  <ZoomInOutlined />
                </a-button>
              </a-button-group>
            </div>
            <div class="pdf-document">
              <img
                v-if="previewUrl"
                :src="previewUrl"
                class="pdf-page"
                :style="{ transform: `scale(${zoomLevel/100})` }"
              />
              <div v-else class="resource-placeholder">
                <FileOutlined />
                <p>PDF加载中或暂无预览</p>
              </div>
            </div>
          </div>
          
          <!-- 图片预览 -->
          <div v-else-if="isImage" class="image-viewer">
            <div class="image-controls">
              <a-button-group>
                <a-button @click="zoomOut">
                  <ZoomOutOutlined />
                </a-button>
                <a-button disabled>{{ zoomLevel }}%</a-button>
                <a-button @click="zoomIn">
                  <ZoomInOutlined />
                </a-button>
              </a-button-group>
            </div>
            <div class="image-container">
              <img
                v-if="previewUrl"
                :src="previewUrl"
                class="image-preview"
                :style="{ transform: `scale(${zoomLevel/100})` }"
              />
              <div v-else class="resource-placeholder">
                <FileImageOutlined />
                <p>图片加载中或暂无预览</p>
              </div>
            </div>
          </div>
          
          <!-- 其他类型资源 -->
          <div v-else class="generic-resource">
            <a-result
              title="资源预览"
              sub-title="此资源类型暂不支持在线预览，请下载后查看"
              :icon="getResourceTypeIcon(resource.fileType)"
            >
              <template #extra>
                <a-button type="primary" @click="downloadResource">
                  <DownloadOutlined />
                  下载资源
                </a-button>
              </template>
            </a-result>
          </div>
        </div>
        
        <div class="progress-bar">
          <a-progress :percent="progress" :stroke-color="progressColor" />
        </div>
        
        <div class="resource-notes">
          <h3>学习笔记</h3>
          <a-textarea
            v-model:value="notes"
            placeholder="在这里记录学习笔记..."
            :rows="4"
            @change="saveNotes"
          />
          <div class="notes-actions">
            <a-button type="primary" @click="saveNotes">
              <SaveOutlined />
              保存笔记
            </a-button>
          </div>
        </div>
      </div>
      
      <div class="resource-info">
        <h3>资源信息</h3>
        <div class="info-grid">
          <div class="info-item">
            <div class="info-label">上传者</div>
            <div class="info-value">{{ resource.uploader || '教师' }}</div>
          </div>
          <div class="info-item">
            <div class="info-label">上传时间</div>
            <div class="info-value">{{ formatDateTime(resource.uploadTime) }}</div>
          </div>
          <div class="info-item">
            <div class="info-label">文件大小</div>
            <div class="info-value">{{ formatFileSize(resource.fileSize) }}</div>
          </div>
          <div class="info-item">
            <div class="info-label">下载次数</div>
            <div class="info-value">{{ resource.downloadCount || 0 }}</div>
          </div>
        </div>
      </div>
      
      <div class="resource-description">
        <h3>资源描述</h3>
        <div class="description-content">
          {{ resource.description || '暂无描述' }}
        </div>
      </div>
    </a-spin>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted, onUnmounted, watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { message } from 'ant-design-vue'
import { 
  ArrowLeftOutlined,
  BookOutlined,
  FileOutlined,
  BarChartOutlined,
  FileImageOutlined,
  FilePdfOutlined,
  FileWordOutlined,
  FileExcelOutlined,
  FilePptOutlined,
  LeftOutlined,
  RightOutlined,
  ZoomInOutlined,
  ZoomOutOutlined,
  DownloadOutlined,
  SaveOutlined
} from '@ant-design/icons-vue'
import dayjs from 'dayjs'
import { startLearningRecord, endLearningRecord } from '@/api/learningRecord'

const route = useRoute()
const router = useRouter()

// 从路由参数获取课程ID和资源ID
const courseId = ref<number>(Number(route.params.courseId) || 0)
const resourceId = ref<number>(Number(route.params.resourceId) || 0)

// 基本数据
const loading = ref<boolean>(true)
const courseName = ref<string>('')
const progress = ref<number>(0)
const notes = ref<string>('')
const resource = ref<any>({})
const previewUrl = ref<string>('')

// 学习记录相关
const learningRecordId = ref<number | null>(null)
const lastProgressUpdate = ref<number>(0)
const progressUpdateInterval = ref<number>(30) // 每30秒更新一次进度
const learningInterval = ref<any>(null)

// PDF浏览器参数
const currentPage = ref<number>(1)
const totalPages = ref<number>(1)
const zoomLevel = ref<number>(100)

// 根据文件类型判断预览方式
const isPdf = computed(() => {
  return resource.value.fileType?.toLowerCase() === 'pdf'
})

const isImage = computed(() => {
  const imageTypes = ['jpg', 'jpeg', 'png', 'gif', 'bmp', 'webp']
  return imageTypes.includes(resource.value.fileType?.toLowerCase())
})

// 加载资源数据
const loadResourceData = async () => {
  try {
    loading.value = true
    
    // 模拟API调用
    await new Promise(resolve => setTimeout(resolve, 1000))
    
    // 模拟数据
    resource.value = {
      id: resourceId.value,
      title: '计算机系统基础知识',
      fileType: 'pdf',
      fileSize: 2458000, // 约2.5MB
      uploadTime: '2025-06-15T10:30:00',
      uploader: '张教授',
      downloadCount: 45,
      description: '本资源包含计算机系统的基础知识，包括硬件、软件、网络等方面的内容。适合初学者阅读学习。'
    }
    
    courseName.value = '计算机组成原理'
    progress.value = 0
    
    // 模拟预览URL (使用一个示例图片作为PDF页面)
    previewUrl.value = 'https://via.placeholder.com/800x1000.png?text=PDF+Preview+Page+1'
    totalPages.value = 10 // 模拟PDF有10页
    
    // 创建学习记录
    await createLearningRecord()
    
    // 设置定时更新进度
    startProgressUpdater()
    
  } catch (error) {
    console.error('加载资源数据失败:', error)
    message.error('加载资源数据失败')
  } finally {
    loading.value = false
  }
}

// 创建学习记录
const createLearningRecord = async () => {
  try {
    const response = await startLearningRecord({
      courseId: courseId.value,
      resourceId: resourceId.value,
      resourceType: resource.value.fileType || 'document'
    })
    
    if (response?.data?.success) {
      learningRecordId.value = response.data.recordId
      console.log('创建学习记录成功, ID:', learningRecordId.value)
    }
  } catch (error) {
    console.error('创建学习记录失败:', error)
  }
}

// 更新学习记录
const updateLearningRecord = async (isCompleted: boolean = false) => {
  if (!learningRecordId.value) return
  
  try {
    await endLearningRecord(
      learningRecordId.value,
      progress.value,
      isCompleted
    )
    console.log('更新学习记录成功')
  } catch (error) {
    console.error('更新学习记录失败:', error)
  }
}

// 启动进度更新器
const startProgressUpdater = () => {
  // 清除现有的定时器
  if (learningInterval.value) {
    clearInterval(learningInterval.value)
  }
  
  // 创建新的定时器
  learningInterval.value = setInterval(() => {
    // 模拟阅读进度
    if (progress.value < 100) {
      progress.value += Math.floor(Math.random() * 5) + 1 // 每次增加1-5%的进度
      progress.value = Math.min(progress.value, 100) // 最大100%
      
      // 当阅读进度超过50%时，自动翻页
      if (isPdf.value && progress.value > 50 && currentPage.value < totalPages.value) {
        currentPage.value += 1
        updatePreviewUrl()
      }
      
      // 当进度变化超过5%，更新学习记录
      if (Math.abs(progress.value - lastProgressUpdate.value) >= 5) {
        lastProgressUpdate.value = progress.value
        updateLearningRecord(progress.value === 100)
        
        // 如果进度达到100%，显示完成提示
        if (progress.value === 100) {
          message.success('恭喜您完成该资源的学习！')
          clearInterval(learningInterval.value)
        }
      }
    }
  }, progressUpdateInterval.value * 1000)
}

// 更新预览URL
const updatePreviewUrl = () => {
  if (isPdf.value) {
    previewUrl.value = `https://via.placeholder.com/800x1000.png?text=PDF+Preview+Page+${currentPage.value}`
  }
}

// 返回课程详情
const goBack = async () => {
  // 在离开页面前更新学习记录
  await updateLearningRecord()
  router.push(`/student/courses/${courseId.value}`)
}

// 上一页
const prevPage = () => {
  if (currentPage.value > 1) {
    currentPage.value--
    updatePreviewUrl()
  }
}

// 下一页
const nextPage = () => {
  if (currentPage.value < totalPages.value) {
    currentPage.value++
    updatePreviewUrl()
  }
}

// 放大
const zoomIn = () => {
  if (zoomLevel.value < 200) {
    zoomLevel.value += 10
  }
}

// 缩小
const zoomOut = () => {
  if (zoomLevel.value > 50) {
    zoomLevel.value -= 10
  }
}

// 下载资源
const downloadResource = () => {
  message.info(`正在下载资源: ${resource.value.title}`)
  // 实际应该调用API下载资源
}

// 保存笔记
const saveNotes = () => {
  message.success('笔记保存成功')
  // 实际应该调用API保存笔记
}

// 获取资源类型文本
const getResourceTypeText = (fileType: string): string => {
  const typeMap: Record<string, string> = {
    'pdf': 'PDF文档',
    'doc': 'Word文档',
    'docx': 'Word文档',
    'ppt': 'PowerPoint演示文稿',
    'pptx': 'PowerPoint演示文稿',
    'xls': 'Excel表格',
    'xlsx': 'Excel表格',
    'jpg': '图片',
    'jpeg': '图片',
    'png': '图片',
    'gif': '图片',
    'txt': '文本文件',
    'zip': '压缩文件',
    'rar': '压缩文件'
  }
  return typeMap[fileType?.toLowerCase()] || '未知类型'
}

// 获取资源类型图标
const getResourceTypeIcon = (fileType: string): any => {
  const iconMap: Record<string, any> = {
    'pdf': FilePdfOutlined,
    'doc': FileWordOutlined,
    'docx': FileWordOutlined,
    'ppt': FilePptOutlined,
    'pptx': FilePptOutlined,
    'xls': FileExcelOutlined,
    'xlsx': FileExcelOutlined,
    'jpg': FileImageOutlined,
    'jpeg': FileImageOutlined,
    'png': FileImageOutlined,
    'gif': FileImageOutlined
  }
  return iconMap[fileType?.toLowerCase()] || FileOutlined
}

// 格式化日期时间
const formatDateTime = (date: string): string => {
  if (!date) return '-'
  return dayjs(date).format('YYYY-MM-DD HH:mm')
}

// 格式化文件大小
const formatFileSize = (bytes: number): string => {
  if (bytes === undefined || bytes === null) return '-'
  
  const units = ['B', 'KB', 'MB', 'GB', 'TB']
  let size = bytes
  let unitIndex = 0
  
  while (size >= 1024 && unitIndex < units.length - 1) {
    size /= 1024
    unitIndex++
  }
  
  return `${size.toFixed(2)} ${units[unitIndex]}`
}

// 进度条颜色
const progressColor = computed(() => {
  if (progress.value >= 80) return '#52c41a'
  if (progress.value >= 60) return '#1890ff'
  return '#faad14'
})

// 清理定时器
onUnmounted(async () => {
  if (learningInterval.value) {
    clearInterval(learningInterval.value)
  }
  await updateLearningRecord()
})

onMounted(() => {
  loadResourceData()
})
</script>

<style scoped>
.resource-learning {
  padding: 24px;
  max-width: 1200px;
  margin: 0 auto;
}

.resource-header {
  margin-bottom: 24px;
}

.back-btn {
  display: block;
  margin-bottom: 16px;
  font-size: 16px;
  padding: 0;
}

.resource-header h1 {
  margin-bottom: 8px;
  font-size: 24px;
}

.resource-meta {
  display: flex;
  gap: 24px;
  color: #666;
}

.resource-container {
  background: white;
  border-radius: 8px;
  padding: 24px;
  margin-bottom: 24px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
}

.resource-viewer {
  background: #f5f5f5;
  border-radius: 8px;
  overflow: hidden;
  margin-bottom: 16px;
}

.pdf-controls, .image-controls {
  display: flex;
  justify-content: center;
  gap: 16px;
  padding: 8px;
  background: #e6e6e6;
}

.pdf-document, .image-container {
  display: flex;
  justify-content: center;
  align-items: center;
  min-height: 500px;
  background: #fff;
  overflow: auto;
}

.pdf-page, .image-preview {
  max-width: 100%;
  transition: transform 0.3s ease;
  transform-origin: center top;
}

.resource-placeholder {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  height: 100%;
  width: 100%;
  color: #999;
  padding: 40px;
}

.resource-placeholder .anticon {
  font-size: 48px;
  margin-bottom: 16px;
}

.progress-bar {
  margin: 16px 0;
}

.resource-notes {
  margin-top: 24px;
}

.resource-notes h3 {
  margin-bottom: 16px;
}

.notes-actions {
  margin-top: 16px;
  display: flex;
  justify-content: flex-end;
}

.resource-info, .resource-description {
  background: white;
  border-radius: 8px;
  padding: 24px;
  margin-bottom: 24px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
}

.resource-info h3, .resource-description h3 {
  margin-bottom: 16px;
  font-size: 18px;
}

.info-grid {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 16px;
}

.info-item {
  display: flex;
  flex-direction: column;
}

.info-label {
  color: #666;
  font-size: 14px;
  margin-bottom: 4px;
}

.info-value {
  font-weight: 500;
}

.generic-resource {
  padding: 24px;
}

@media (max-width: 768px) {
  .resource-meta {
    flex-direction: column;
    gap: 8px;
  }
  
  .info-grid {
    grid-template-columns: 1fr;
  }
}
</style> 