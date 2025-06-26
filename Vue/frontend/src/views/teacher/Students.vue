<template>
  <div class="teacher-students">
    <div class="page-header">
      <h1>学生管理</h1>
      <a-space>
        <a-button @click="importStudents">
          <UploadOutlined />
          批量导入
        </a-button>
        <a-button @click="exportStudents">
          <DownloadOutlined />
          批量导出
        </a-button>
        <a-button type="primary" @click="addStudent">
          <PlusOutlined />
          新增学生
        </a-button>
      </a-space>
    </div>
    
    <div class="students-content">
      <a-table :dataSource="students" :columns="columns" />
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue'
import { UploadOutlined, DownloadOutlined, PlusOutlined } from '@ant-design/icons-vue'

const students = ref([
  { id: 1, name: '张三', studentId: '2021001', class: '软件1班' },
  { id: 2, name: '李四', studentId: '2021002', class: '软件1班' }
])

const columns = [
  { title: '姓名', dataIndex: 'name', key: 'name' },
  { title: '学号', dataIndex: 'studentId', key: 'studentId' },
  { title: '班级', dataIndex: 'class', key: 'class' }
]

const importStudents = () => {
  console.log('导入学生')
}

const exportStudents = () => {
  console.log('导出学生')
}

const addStudent = () => {
  console.log('新增学生')
}
</script>

<style scoped>
.teacher-students {
  padding: 24px;
}

.page-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 24px;
}
</style> 