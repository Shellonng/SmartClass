<template>
  <div class="teacher-classes">
    <div class="page-header">
      <h1>班级管理</h1>
      <a-button type="primary" @click="showCreateModal">
        <PlusOutlined />
        创建班级
      </a-button>
    </div>
    
    <div class="classes-content">
      <a-spin :spinning="loading">
        <div class="classes-grid">
          <div v-for="classItem in classes" :key="classItem.id" class="class-card">
            <h3>{{ classItem.name }}</h3>
            <p>学生人数：{{ classItem.studentCount }}</p>
            <a-button @click="viewClass(classItem.id)">查看详情</a-button>
          </div>
        </div>
      </a-spin>
    </div>

    <a-modal v-model:open="createModalVisible" title="创建班级">
      <a-form :model="form">
        <a-form-item label="班级名称">
          <a-input v-model:value="form.name" />
        </a-form-item>
      </a-form>
    </a-modal>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue'
import { useRouter } from 'vue-router'
import { PlusOutlined } from '@ant-design/icons-vue'

const router = useRouter()
const loading = ref(false)
const createModalVisible = ref(false)
const form = ref({ name: '' })

const classes = ref([
  { id: 1, name: '软件工程2021级1班', studentCount: 45 },
  { id: 2, name: '软件工程2021级2班', studentCount: 43 }
])

const showCreateModal = () => {
  createModalVisible.value = true
}

const viewClass = (id: number) => {
  router.push(`/teacher/classes/${id}`)
}
</script>

<style scoped>
.teacher-classes {
  padding: 24px;
}

.page-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 24px;
}

.classes-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
  gap: 16px;
}

.class-card {
  background: white;
  border-radius: 8px;
  padding: 20px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
}
</style> 