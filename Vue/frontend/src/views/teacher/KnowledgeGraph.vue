<template>
  <div class="teacher-knowledge-graph">
    <div class="page-header">
      <h1>知识图谱</h1>
      <a-space>
        <a-button @click="editGraph">
          <EditOutlined />
          编辑图谱
        </a-button>
        <a-button type="primary" @click="generateGraph">
          <ThunderboltOutlined />
          一键生成
        </a-button>
      </a-space>
    </div>
    
    <div class="graph-content">
      <div class="graph-container">
        <p>知识图谱可视化区域</p>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { EditOutlined, ThunderboltOutlined } from '@ant-design/icons-vue'
import { message } from 'ant-design-vue'

const editGraph = () => {
  console.log('编辑知识图谱')
}

const generateGraph = () => {
  message.success('正在生成知识图谱...')
  console.log('一键生成知识图谱')
}
</script>

<style scoped>
.teacher-knowledge-graph {
  padding: 24px;
}

.page-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 24px;
}

.graph-container {
  height: 600px;
  background: #f5f5f5;
  border-radius: 8px;
  display: flex;
  align-items: center;
  justify-content: center;
  border: 2px dashed #d9d9d9;
}
</style> 