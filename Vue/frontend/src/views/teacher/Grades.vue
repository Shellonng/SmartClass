<template>
  <div class="teacher-grades">
    <div class="page-header">
      <h1>成绩管理</h1>
      <a-button @click="exportGrades">
        <DownloadOutlined />
        导出报表
      </a-button>
    </div>
    
    <div class="grades-content">
      <a-table :dataSource="grades" :columns="columns" />
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue'
import { DownloadOutlined } from '@ant-design/icons-vue'

const grades = ref([
  { id: 1, student: '张三', course: '高等数学', score: 85 },
  { id: 2, student: '李四', course: '程序设计', score: 92 }
])

const columns = [
  { title: '学生', dataIndex: 'student', key: 'student' },
  { title: '课程', dataIndex: 'course', key: 'course' },
  { title: '成绩', dataIndex: 'score', key: 'score' }
]

const exportGrades = () => {
  console.log('导出报表')
}
</script>

<style scoped>
.teacher-grades {
  padding: 24px;
}

.page-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 24px;
}
</style> 