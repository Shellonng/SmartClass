<template>
  <div class="teacher-courses">
    <div class="page-header">
      <h1>课程管理</h1>
      <a-button type="primary" @click="createCourse">
        <PlusOutlined />
        新建课程
      </a-button>
    </div>
    
    <div class="courses-content">
      <a-table :dataSource="courses" :columns="columns" />
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue'
import { PlusOutlined } from '@ant-design/icons-vue'

const courses = ref([
  { id: 1, name: '高等数学', code: 'MATH101', credits: 4 },
  { id: 2, name: '程序设计', code: 'CS101', credits: 3 }
])

const columns = [
  { title: '课程名称', dataIndex: 'name', key: 'name' },
  { title: '课程代码', dataIndex: 'code', key: 'code' },
  { title: '学分', dataIndex: 'credits', key: 'credits' }
]

const createCourse = () => {
  console.log('新建课程')
}
</script>

<style scoped>
.teacher-courses {
  padding: 24px;
}

.page-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 24px;
}
</style> 