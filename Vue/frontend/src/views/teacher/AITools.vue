<template>
  <div class="teacher-ai-tools">
    <div class="page-header">
      <h1>AI创新工具</h1>
      <p class="description">利用人工智能技术提升教学效率</p>
    </div>
    
    <div class="tools-grid">
      <div class="tool-card" @click="intelligentGrading">
        <div class="tool-icon">
          <RobotOutlined />
        </div>
        <h3>智能批改</h3>
        <p>自动批改作业和试卷，提供详细反馈</p>
      </div>

      <div class="tool-card" @click="intelligentRecommendation">
        <div class="tool-icon">
          <BulbOutlined />
        </div>
        <h3>智能推荐</h3>
        <p>基于学生学习情况推荐个性化学习内容</p>
      </div>

      <div class="tool-card" @click="abilityAnalysis">
        <div class="tool-icon">
          <BarChartOutlined />
        </div>
        <h3>能力图谱</h3>
        <p>分析学生能力结构，生成能力评估报告</p>
      </div>

      <div class="tool-card" @click="knowledgeGeneration">
        <div class="tool-icon">
          <ThunderboltOutlined />
        </div>
        <h3>知识图谱生成</h3>
        <p>自动生成课程知识图谱和学习路径</p>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { RobotOutlined, BulbOutlined, BarChartOutlined, ThunderboltOutlined } from '@ant-design/icons-vue'
import { message } from 'ant-design-vue'

const intelligentGrading = () => {
  message.info('启动智能批改功能')
  console.log('智能批改')
}

const intelligentRecommendation = () => {
  message.info('启动智能推荐功能')
  console.log('智能推荐')
}

const abilityAnalysis = () => {
  message.info('启动能力分析功能')
  console.log('能力分析')
}

const knowledgeGeneration = () => {
  message.info('启动知识图谱生成功能')
  console.log('知识图谱生成')
}
</script>

<style scoped>
.teacher-ai-tools {
  padding: 24px;
}

.page-header {
  text-align: center;
  margin-bottom: 48px;
}

.page-header h1 {
  font-size: 32px;
  margin-bottom: 8px;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
}

.description {
  font-size: 16px;
  color: #666;
}

.tools-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
  gap: 24px;
  max-width: 1200px;
  margin: 0 auto;
}

.tool-card {
  background: white;
  border-radius: 12px;
  padding: 32px 24px;
  text-align: center;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
  transition: all 0.3s ease;
  cursor: pointer;
  border: 1px solid #f0f0f0;
}

.tool-card:hover {
  transform: translateY(-4px);
  box-shadow: 0 8px 30px rgba(0, 0, 0, 0.12);
  border-color: #1890ff;
}

.tool-icon {
  width: 64px;
  height: 64px;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  margin: 0 auto 20px;
  font-size: 28px;
  color: white;
}

.tool-card h3 {
  font-size: 20px;
  font-weight: 600;
  margin-bottom: 12px;
  color: #333;
}

.tool-card p {
  color: #666;
  line-height: 1.6;
  margin: 0;
}
</style> 