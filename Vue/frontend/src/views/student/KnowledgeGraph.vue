<template>
  <div class="student-knowledge-graph">
    <div class="page-header">
      <h1>知识图谱</h1>
      <p class="description">个性化学习路径，掌握知识体系全貌</p>
    </div>

    <div class="graph-container">
      <div class="graph-visualization">
        <p>知识图谱可视化展示</p>
        <p>点击节点查看详情，系统会推荐学习路径</p>
      </div>
    </div>

    <div class="learning-path">
      <h2>推荐学习路径</h2>
      <div class="path-steps">
        <div v-for="(step, index) in learningPath" :key="step.id" class="step-item">
          <div class="step-number">{{ index + 1 }}</div>
          <div class="step-content">
            <h3>{{ step.title }}</h3>
            <p>{{ step.description }}</p>
            <div class="step-status" :class="step.status">
              {{ getStatusText(step.status) }}
            </div>
          </div>
          <div class="step-action">
            <a-button v-if="step.status === 'available'" type="primary" @click="startLearning(step.id)">
              开始学习
            </a-button>
            <a-button v-else-if="step.status === 'completed'" disabled>
              已完成
            </a-button>
            <a-button v-else disabled>
              等待解锁
            </a-button>
          </div>
        </div>
      </div>
    </div>

    <!-- 知识点详情弹窗 -->
    <a-modal v-model:open="detailModalVisible" title="知识点详情" width="600px">
      <div v-if="selectedNode" class="node-detail">
        <h3>{{ selectedNode.title }}</h3>
        <p>{{ selectedNode.description }}</p>
        <div class="learning-resources">
          <h4>学习资源</h4>
          <ul>
            <li v-for="resource in selectedNode.resources" :key="resource.id">
              <a @click="openResource(resource)">{{ resource.title }}</a>
            </li>
          </ul>
        </div>
      </div>
      <template #footer>
        <a-button @click="detailModalVisible = false">关闭</a-button>
        <a-button type="primary" @click="enterLearning">进入学习</a-button>
      </template>
    </a-modal>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue'
import { message } from 'ant-design-vue'

interface LearningStep {
  id: number
  title: string
  description: string
  status: 'completed' | 'available' | 'locked'
}

interface KnowledgeNode {
  id: number
  title: string
  description: string
  resources: Array<{ id: number; title: string; type: string }>
}

const detailModalVisible = ref(false)
const selectedNode = ref<KnowledgeNode | null>(null)

const learningPath = ref<LearningStep[]>([
  {
    id: 1,
    title: '函数基础概念',
    description: '理解函数的定义、定义域、值域等基本概念',
    status: 'completed'
  },
  {
    id: 2,
    title: '函数的极限',
    description: '掌握极限的定义和基本性质，学会计算简单极限',
    status: 'available'
  },
  {
    id: 3,
    title: '导数与微分',
    description: '理解导数的几何意义，掌握求导法则',
    status: 'locked'
  },
  {
    id: 4,
    title: '积分运算',
    description: '学习不定积分和定积分的基本计算方法',
    status: 'locked'
  }
])

const startLearning = (stepId: number) => {
  message.success('开始学习该知识点')
  console.log('开始学习步骤:', stepId)
}

const getStatusText = (status: string) => {
  const statusMap = {
    completed: '已完成',
    available: '可学习',
    locked: '未解锁'
  }
  return statusMap[status as keyof typeof statusMap] || '未知'
}

const openResource = (resource: any) => {
  message.info(`打开资源: ${resource.title}`)
}

const enterLearning = () => {
  if (selectedNode.value) {
    message.success(`进入学习: ${selectedNode.value.title}`)
    detailModalVisible.value = false
  }
}
</script>

<style scoped>
.student-knowledge-graph {
  padding: 24px;
  max-width: 1200px;
  margin: 0 auto;
}

.page-header {
  text-align: center;
  margin-bottom: 32px;
}

.page-header h1 {
  font-size: 28px;
  margin-bottom: 8px;
  color: #333;
}

.description {
  color: #666;
  font-size: 16px;
}

.graph-container {
  background: white;
  border-radius: 12px;
  padding: 32px;
  margin-bottom: 32px;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
}

.graph-visualization {
  height: 400px;
  background: #f8f9fa;
  border-radius: 8px;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  border: 2px dashed #d9d9d9;
  color: #666;
}

.learning-path {
  background: white;
  border-radius: 12px;
  padding: 32px;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
}

.learning-path h2 {
  margin-bottom: 24px;
  color: #333;
}

.path-steps {
  display: flex;
  flex-direction: column;
  gap: 20px;
}

.step-item {
  display: flex;
  align-items: center;
  gap: 20px;
  padding: 20px;
  background: #f8f9fa;
  border-radius: 8px;
  border-left: 4px solid #1890ff;
}

.step-item.completed {
  border-left-color: #52c41a;
  background: #f6ffed;
}

.step-item.locked {
  border-left-color: #d9d9d9;
  background: #fafafa;
  opacity: 0.6;
}

.step-number {
  width: 40px;
  height: 40px;
  background: #1890ff;
  color: white;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-weight: 600;
  flex-shrink: 0;
}

.step-content {
  flex: 1;
}

.step-content h3 {
  margin-bottom: 8px;
  color: #333;
}

.step-content p {
  color: #666;
  margin-bottom: 8px;
}

.step-status {
  font-size: 12px;
  padding: 2px 8px;
  border-radius: 12px;
  display: inline-block;
}

.step-status.completed {
  background: #52c41a;
  color: white;
}

.step-status.available {
  background: #1890ff;
  color: white;
}

.step-status.locked {
  background: #d9d9d9;
  color: #666;
}

.step-action {
  flex-shrink: 0;
}

.node-detail h3 {
  margin-bottom: 16px;
  color: #333;
}

.node-detail p {
  color: #666;
  line-height: 1.6;
  margin-bottom: 20px;
}

.learning-resources h4 {
  margin-bottom: 12px;
  color: #333;
}

.learning-resources ul {
  list-style: none;
  padding: 0;
}

.learning-resources li {
  margin-bottom: 8px;
}

.learning-resources a {
  color: #1890ff;
  cursor: pointer;
}

.learning-resources a:hover {
  text-decoration: underline;
}
</style> 