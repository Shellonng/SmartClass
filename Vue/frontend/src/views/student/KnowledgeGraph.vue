<template>
  <div class="knowledge-graph-page">
    <h1>课程知识图谱</h1>
    
    <el-row :gutter="20" class="feature-cards">
      <el-col :span="12">
        <el-card shadow="hover" class="feature-card">
          <template #header>
            <div class="card-header">
              <h2>知识图谱浏览器</h2>
    </div>
          </template>
          <div class="card-content">
            <p>浏览课程知识图谱，了解知识点之间的关联和学习路径。</p>
            <ul>
              <li>查看课程的知识结构和知识点关联</li>
              <li>探索学习路径和知识依赖关系</li>
              <li>定位当前学习进度和掌握程度</li>
              <li>获取个性化学习建议</li>
            </ul>
            <div class="card-actions">
              <el-button type="primary" @click="goToViewer">进入知识图谱浏览器</el-button>
      </div>
    </div>
        </el-card>
      </el-col>
      
      <el-col :span="12">
        <el-card shadow="hover" class="feature-card">
          <template #header>
            <div class="card-header">
              <h2>我的学习进度</h2>
            </div>
          </template>
          <div class="card-content">
            <p>查看您在知识图谱中的学习进度和掌握情况。</p>
            <ul>
              <li>跟踪知识点掌握情况</li>
              <li>发现学习中的薄弱环节</li>
              <li>获取针对性的学习建议</li>
              <li>查看学习路径推荐</li>
            </ul>
            <div class="card-actions">
              <el-button @click="goToProgress">查看我的进度</el-button>
            </div>
          </div>
        </el-card>
      </el-col>
    </el-row>
    
    <!-- 推荐知识图谱 -->
    <div class="recommended-graphs">
      <h2>推荐知识图谱</h2>
      <p>根据您的课程和学习情况推荐的知识图谱</p>
      
      <el-row :gutter="20">
        <el-col :span="8" v-for="i in 3" :key="i">
          <el-card shadow="hover" class="graph-card">
            <template #header>
              <div class="card-header">
                <h3>示例知识图谱 {{ i }}</h3>
                <el-tag size="small" type="success">综合图谱</el-tag>
              </div>
            </template>
            <div class="card-content">
              <p class="description">这是一个示例知识图谱，展示了课程的核心概念和知识点关联。</p>
              <div class="graph-preview">
                <img src="https://via.placeholder.com/300x150?text=知识图谱预览" alt="知识图谱预览" />
          </div>
              <div class="card-actions">
                <el-button size="small" @click="viewGraph(i)">查看详情</el-button>
        </div>
      </div>
          </el-card>
        </el-col>
      </el-row>
    </div>

    <div class="knowledge-graph-intro">
      <h2>如何使用知识图谱学习？</h2>
      <p>知识图谱是一种可视化的学习工具，可以帮助您更好地理解课程内容和知识点之间的关联。</p>
      
      <el-steps :active="1" finish-status="success" simple>
        <el-step title="浏览知识图谱" description="了解课程的知识结构和知识点关联" />
        <el-step title="定位当前进度" description="找到您当前的学习位置和掌握程度" />
        <el-step title="探索学习路径" description="根据知识依赖关系规划学习路径" />
        <el-step title="查漏补缺" description="针对薄弱环节进行有针对性的学习" />
      </el-steps>
      </div>
  </div>
</template>

<script setup lang="ts">
import { useRouter } from 'vue-router'

const router = useRouter()

const goToViewer = () => {
  router.push('/student/knowledge-graph/viewer')
}

const goToProgress = () => {
  router.push('/student/knowledge-graph/viewer')
}

const viewGraph = (id: number) => {
  router.push('/student/knowledge-graph/viewer')
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

.card-header h2, .card-header h3 {
  margin: 0;
  font-size: 20px;
}

.card-header h3 {
  font-size: 16px;
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

.recommended-graphs {
  margin-top: 40px;
  margin-bottom: 40px;
}

.graph-card {
  margin-bottom: 20px;
}

.graph-preview {
  margin: 15px 0;
  text-align: center;
}

.graph-preview img {
  width: 100%;
  border-radius: 4px;
}

.description {
  color: #606266;
  overflow: hidden;
  text-overflow: ellipsis;
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
}

.knowledge-graph-intro {
  margin-top: 40px;
}
</style> 