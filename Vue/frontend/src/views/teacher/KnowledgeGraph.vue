<template>
  <div class="knowledge-graph-page">
    <h1>知识图谱管理</h1>
    
    <el-row :gutter="20" class="feature-cards">
      <el-col :span="12">
        <el-card shadow="hover" class="feature-card">
          <template #header>
            <div class="card-header">
              <h2>知识图谱生成器</h2>
            </div>
          </template>
          <div class="card-content">
            <p>使用AI智能生成课程知识图谱，可视化课程内容结构和知识点关联。</p>
            <ul>
              <li>支持多种图谱类型：概念图谱、技能图谱、综合图谱</li>
              <li>自动提取课程章节内容，构建知识点关联</li>
              <li>可视化编辑和管理知识图谱</li>
              <li>支持导出和分享给学生</li>
            </ul>
            <div class="card-actions">
              <el-button type="primary" @click="goToGenerator">进入知识图谱生成器</el-button>
            </div>
          </div>
        </el-card>
      </el-col>
      
      <el-col :span="12">
        <el-card shadow="hover" class="feature-card">
          <template #header>
            <div class="card-header">
              <h2>知识图谱管理</h2>
            </div>
          </template>
          <div class="card-content">
            <p>管理已生成的知识图谱，查看学生学习情况和知识掌握程度。</p>
            <ul>
              <li>查看所有课程的知识图谱</li>
              <li>分析学生对知识点的掌握情况</li>
              <li>发现学习难点和薄弱环节</li>
              <li>优化教学内容和教学策略</li>
            </ul>
            <div class="card-actions">
              <el-button @click="goToManagement">查看我的知识图谱</el-button>
            </div>
    </div>
        </el-card>
      </el-col>
    </el-row>
    
    <div class="knowledge-graph-intro">
      <h2>什么是知识图谱？</h2>
      <p>知识图谱是一种结构化的知识表示方式，通过节点和边来描述知识点之间的关系。在教育领域，知识图谱可以帮助教师和学生更好地理解课程内容的结构和知识点之间的关联。</p>
      
      <h3>知识图谱的优势</h3>
      <el-row :gutter="20" class="benefits">
        <el-col :span="8">
          <div class="benefit-item">
            <el-icon size="large" color="#409EFF"><Connection /></el-icon>
            <h4>可视化知识结构</h4>
            <p>直观展示知识点之间的关联和层次结构，帮助理解复杂概念。</p>
          </div>
        </el-col>
        <el-col :span="8">
          <div class="benefit-item">
            <el-icon size="large" color="#67C23A"><DataLine /></el-icon>
            <h4>个性化学习路径</h4>
            <p>根据学生掌握情况，生成个性化学习路径和推荐内容。</p>
          </div>
        </el-col>
        <el-col :span="8">
          <div class="benefit-item">
            <el-icon size="large" color="#E6A23C"><Search /></el-icon>
            <h4>发现知识盲点</h4>
            <p>识别学生的知识盲点和薄弱环节，有针对性地进行教学调整。</p>
      </div>
        </el-col>
      </el-row>
    </div>
  </div>
</template>

<script setup lang="ts">
import { useRouter } from 'vue-router'
import { Connection, DataLine, Search } from '@element-plus/icons-vue'

const router = useRouter()

const goToGenerator = () => {
  router.push('/teacher/knowledge-graph/generator')
}

const goToManagement = () => {
  router.push('/teacher/knowledge-graph/generator')
}
</script>

<style scoped>
.knowledge-graph-page {
  padding: 20px;
}

.feature-cards {
  margin-top: 20px;
  margin-bottom: 40px;
}

.feature-card {
  height: 100%;
  transition: transform 0.3s;
}

.feature-card:hover {
  transform: translateY(-5px);
}

.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.card-header h2 {
  margin: 0;
  font-size: 20px;
}

.card-content {
  display: flex;
  flex-direction: column;
  height: 100%;
}

.card-content p {
  margin-bottom: 15px;
}

.card-content ul {
  margin-bottom: 20px;
}

.card-actions {
  margin-top: auto;
  text-align: right;
}

.knowledge-graph-intro {
  margin-top: 40px;
}

.benefits {
  margin-top: 30px;
}

.benefit-item {
  text-align: center;
  padding: 20px;
  background-color: #f5f7fa;
  border-radius: 8px;
  height: 100%;
}

.benefit-item h4 {
  margin: 15px 0 10px;
}

.benefit-item p {
  color: #606266;
}
</style> 