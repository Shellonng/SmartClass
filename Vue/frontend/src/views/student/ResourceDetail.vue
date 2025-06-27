<template>
  <div class="resource-detail">
    <div class="resource-header">
      <h1>资源详情</h1>
      <div class="resource-actions">
        <a-button type="primary" @click="downloadResource">
          <template #icon><DownloadOutlined /></template>
          下载资源
        </a-button>
        <a-button @click="favoriteResource">
          <template #icon><StarOutlined /></template>
          收藏
        </a-button>
      </div>
    </div>
    
    <div class="resource-content">
      <div class="resource-info">
        <h2>{{ resource.title || '资源标题' }}</h2>
        <p class="resource-description">{{ resource.description || '资源描述信息...' }}</p>
        
        <div class="resource-meta">
          <div class="meta-item">
            <span class="label">资源类型:</span>
            <span class="value">{{ resource.type || '文档' }}</span>
          </div>
          <div class="meta-item">
            <span class="label">文件大小:</span>
            <span class="value">{{ resource.size || '1.2MB' }}</span>
          </div>
          <div class="meta-item">
            <span class="label">上传时间:</span>
            <span class="value">{{ resource.uploadTime || '2024-01-01' }}</span>
          </div>
          <div class="meta-item">
            <span class="label">下载次数:</span>
            <span class="value">{{ resource.downloadCount || 0 }}</span>
          </div>
        </div>
      </div>
      
      <div class="resource-preview">
        <h3>资源预览</h3>
        <div class="preview-content">
          <p>资源预览内容将在这里显示...</p>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { message } from 'ant-design-vue'
import { DownloadOutlined, StarOutlined } from '@ant-design/icons-vue'

interface Props {
  id: string
}

const props = defineProps<Props>()
const resourceId = props.id

const resource = ref({
  id: '',
  title: '',
  description: '',
  type: '',
  size: '',
  uploadTime: '',
  downloadCount: 0
})

const loadResourceDetail = async () => {
  try {
    // TODO: 调用API获取资源详情
    console.log('Loading resource detail for ID:', resourceId)
    
    // 模拟数据
    resource.value = {
      id: resourceId,
      title: '示例资源文档',
      description: '这是一个示例资源的详细描述信息，包含了相关的学习内容和使用说明。',
      type: 'PDF文档',
      size: '2.5MB',
      uploadTime: '2024-01-15',
      downloadCount: 25
    }
  } catch (error) {
    console.error('Failed to load resource detail:', error)
    message.error('加载资源详情失败')
  }
}

const downloadResource = () => {
  // TODO: 实现资源下载功能
  message.success('开始下载资源')
}

const favoriteResource = () => {
  // TODO: 实现收藏功能
  message.success('已添加到收藏')
}

onMounted(() => {
  loadResourceDetail()
})
</script>

<style scoped>
.resource-detail {
  padding: 24px;
  max-width: 1200px;
  margin: 0 auto;
}

.resource-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 24px;
  padding-bottom: 16px;
  border-bottom: 1px solid #eee;
}

.resource-header h1 {
  margin: 0;
  color: #333;
}

.resource-actions {
  display: flex;
  gap: 12px;
}

.resource-content {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 24px;
}

.resource-info h2 {
  margin: 0 0 16px 0;
  color: #333;
  font-size: 24px;
}

.resource-description {
  color: #666;
  line-height: 1.6;
  margin-bottom: 24px;
}

.resource-meta {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.meta-item {
  display: flex;
  align-items: center;
}

.meta-item .label {
  font-weight: 500;
  color: #666;
  width: 100px;
  flex-shrink: 0;
}

.meta-item .value {
  color: #333;
}

.resource-preview {
  background: #f8f9fa;
  border-radius: 8px;
  padding: 20px;
}

.resource-preview h3 {
  margin: 0 0 16px 0;
  color: #333;
}

.preview-content {
  background: white;
  border-radius: 4px;
  padding: 16px;
  min-height: 200px;
  border: 1px solid #eee;
}

@media (max-width: 768px) {
  .resource-content {
    grid-template-columns: 1fr;
  }
  
  .resource-header {
    flex-direction: column;
    align-items: flex-start;
    gap: 16px;
  }
}
</style> 