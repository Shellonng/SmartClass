<template>
  <div class="teacher-resources">
    <div class="page-header">
      <h1>资源管理</h1>
      <a-space>
        <a-button @click="uploadResource">
          <UploadOutlined />
          上传资源
        </a-button>
        <a-button @click="batchOperations">
          批量操作
        </a-button>
      </a-space>
    </div>
    
    <div class="resources-content">
      <a-table :dataSource="resources" :columns="columns" />
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue'
import { UploadOutlined } from '@ant-design/icons-vue'

const resources = ref([
  { id: 1, name: '第一章PPT', type: 'ppt', size: '2.5MB' },
  { id: 2, name: '教学视频', type: 'video', size: '125MB' }
])

const columns = [
  { title: '资源名称', dataIndex: 'name', key: 'name' },
  { title: '类型', dataIndex: 'type', key: 'type' },
  { title: '大小', dataIndex: 'size', key: 'size' }
]

const uploadResource = () => {
  console.log('上传资源')
}

const batchOperations = () => {
  console.log('批量操作')
}
</script>

<style scoped>
.teacher-resources {
  padding: 24px;
}

.page-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 24px;
}
</style> 