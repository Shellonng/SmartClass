<template>
  <div class="teacher-tasks">
    <div class="page-header">
      <h1>任务管理</h1>
      <a-button type="primary" @click="publishTask">
        <PlusOutlined />
        发布任务
      </a-button>
    </div>
    
    <div class="tasks-content">
      <a-table :dataSource="tasks" :columns="columns" />
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue'
import { PlusOutlined } from '@ant-design/icons-vue'

const tasks = ref([
  { id: 1, title: '第一章作业', course: '高等数学', deadline: '2024-02-01' },
  { id: 2, title: '期中测验', course: '程序设计', deadline: '2024-02-15' }
])

const columns = [
  { title: '任务标题', dataIndex: 'title', key: 'title' },
  { title: '课程', dataIndex: 'course', key: 'course' },
  { title: '截止时间', dataIndex: 'deadline', key: 'deadline' }
]

const publishTask = () => {
  console.log('发布任务')
}
</script>

<style scoped>
.teacher-tasks {
  padding: 24px;
}

.page-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 24px;
}
</style> 