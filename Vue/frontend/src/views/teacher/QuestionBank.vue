<template>
  <div class="teacher-question-bank">
    <div class="page-header">
      <h1>题库管理</h1>
      <a-space>
        <a-button @click="createPaper">
          <FileAddOutlined />
          组卷
        </a-button>
        <a-button type="primary" @click="addQuestion">
          <PlusOutlined />
          录入题目
        </a-button>
      </a-space>
    </div>
    
    <div class="questions-content">
      <a-table :dataSource="questions" :columns="columns" />
    </div>

    <a-modal v-model:open="paperModalVisible" title="组卷设置" width="600px">
      <a-form layout="vertical">
        <a-form-item label="组卷方式">
          <a-radio-group v-model:value="paperType">
            <a-radio value="random">随机组卷</a-radio>
            <a-radio value="knowledge">按知识点</a-radio>
            <a-radio value="difficulty">按难度平衡</a-radio>
          </a-radio-group>
        </a-form-item>
        <a-form-item label="题目数量">
          <a-input-number v-model:value="questionCount" :min="1" :max="100" />
        </a-form-item>
      </a-form>
    </a-modal>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue'
import { FileAddOutlined, PlusOutlined } from '@ant-design/icons-vue'
import { message } from 'ant-design-vue'

const paperModalVisible = ref(false)
const paperType = ref('random')
const questionCount = ref(20)

const questions = ref([
  { id: 1, title: '求函数的导数', type: '计算题', difficulty: '中等', knowledge: '微积分' },
  { id: 2, title: '什么是面向对象', type: '简答题', difficulty: '简单', knowledge: '程序设计' }
])

const columns = [
  { title: '题目', dataIndex: 'title', key: 'title' },
  { title: '类型', dataIndex: 'type', key: 'type' },
  { title: '难度', dataIndex: 'difficulty', key: 'difficulty' },
  { title: '知识点', dataIndex: 'knowledge', key: 'knowledge' }
]

const addQuestion = () => {
  console.log('录入题目')
}

const createPaper = () => {
  paperModalVisible.value = true
}
</script>

<style scoped>
.teacher-question-bank {
  padding: 24px;
}

.page-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 24px;
}
</style> 