<template>
  <div class="knowledge-graph-view">
    <div class="page-header">
      <h1>知识图谱</h1>
      <p>通过可视化的知识结构，更好地理解和掌握学习内容</p>
    </div>

    <!-- 课程选择和筛选 -->
    <div class="filter-bar">
      <el-select
        v-model="selectedCourse"
        placeholder="选择课程"
        style="width: 200px"
        @change="onCourseChange"
        clearable
      >
        <el-option
          v-for="course in enrolledCourses"
          :key="course.id"
          :label="course.title"
          :value="course.id"
        />
      </el-select>
      
      <el-input
        v-model="searchKeyword"
        placeholder="搜索知识图谱..."
        style="width: 300px; margin-left: 12px"
        clearable
        @input="handleSearch"
      >
        <template #prefix>
          <el-icon><Search /></el-icon>
        </template>
      </el-input>
      
      <el-button-group style="margin-left: 12px">
        <el-button
          :type="viewMode === 'grid' ? 'primary' : 'default'"
          @click="viewMode = 'grid'"
        >
          <el-icon><Grid /></el-icon>
        </el-button>
        <el-button
          :type="viewMode === 'list' ? 'primary' : 'default'"
          @click="viewMode = 'list'"
        >
          <el-icon><List /></el-icon>
        </el-button>
      </el-button-group>
    </div>

    <!-- 推荐图谱 -->
    <div v-if="!selectedCourse && recommendedGraphs.length > 0" class="recommended-section">
      <h3>为您推荐</h3>
      <div class="graph-grid">
        <div
          v-for="graph in recommendedGraphs"
          :key="graph.id"
          class="graph-card recommended"
          @click="viewGraph(graph)"
        >
          <div class="card-header">
            <h4>{{ graph.title }}</h4>
            <el-tag type="warning" size="small">推荐</el-tag>
          </div>
          <div class="card-content">
            <p class="description">{{ graph.description }}</p>
            <div class="card-info">
              <span class="course-name">{{ graph.courseName }}</span>
              <span class="view-count">{{ graph.viewCount }} 次学习</span>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- 课程图谱列表 -->
    <div v-if="selectedCourse" class="course-graphs">
      <div class="section-header">
        <h3>{{ currentCourseName }} - 知识图谱</h3>
        <el-tag>{{ filteredGraphs.length }} 个图谱</el-tag>
      </div>
      
      <div v-if="viewMode === 'grid'" class="graph-grid">
        <div
          v-for="graph in filteredGraphs"
          :key="graph.id"
          class="graph-card"
          @click="viewGraph(graph)"
        >
          <div class="card-header">
            <h4>{{ graph.title }}</h4>
            <div class="progress-badge">
              <el-progress
                type="circle"
                :percentage="getGraphProgress(graph.id)"
                :width="32"
                :show-text="false"
              />
              <span class="progress-text">{{ getGraphProgress(graph.id) }}%</span>
            </div>
          </div>
          
          <div class="card-content">
            <p class="description">{{ graph.description }}</p>
            
            <div class="graph-stats">
              <div class="stat-item">
                <el-icon><Connection /></el-icon>
                <span>{{ getNodeCount(graph) }} 个知识点</span>
              </div>
              <div class="stat-item">
                <el-icon><Clock /></el-icon>
                <span>{{ formatDate(graph.updateTime) }}</span>
              </div>
            </div>
            
            <div class="card-actions">
              <el-button size="small" type="primary" @click.stop="viewGraph(graph)">
                学习
              </el-button>
              <el-button size="small" @click.stop="analyzeGraph(graph)">
                分析
              </el-button>
            </div>
          </div>
        </div>
      </div>
      
      <div v-else class="graph-list">
        <div
          v-for="graph in filteredGraphs"
          :key="graph.id"
          class="list-item"
          @click="viewGraph(graph)"
        >
          <div class="item-content">
            <h4>{{ graph.title }}</h4>
            <p>{{ graph.description }}</p>
            <div class="item-meta">
              <span>{{ getNodeCount(graph) }} 个知识点</span>
              <span>{{ formatDate(graph.updateTime) }}</span>
            </div>
          </div>
          
          <div class="item-progress">
            <el-progress
              :percentage="getGraphProgress(graph.id)"
              :status="getGraphProgress(graph.id) >= 100 ? 'success' : ''"
            />
            <span class="progress-label">{{ getGraphProgress(graph.id) }}% 完成</span>
          </div>
          
          <div class="item-actions">
            <el-button size="small" type="primary">学习</el-button>
          </div>
        </div>
      </div>
    </div>

    <!-- 全部公开图谱 -->
    <div v-if="!selectedCourse" class="public-graphs">
      <h3>探索更多</h3>
      <div class="graph-grid">
        <div
          v-for="graph in publicGraphs"
          :key="graph.id"
          class="graph-card public"
          @click="viewGraph(graph)"
        >
          <div class="card-header">
            <h4>{{ graph.title }}</h4>
            <el-tag type="success" size="small">公开</el-tag>
          </div>
          <div class="card-content">
            <p class="description">{{ graph.description }}</p>
            <div class="card-info">
              <span class="course-name">{{ graph.courseName }}</span>
              <span class="creator-name">by {{ graph.creatorName }}</span>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- 空状态 -->
    <div v-if="showEmptyState" class="empty-state">
      <el-empty
        description="暂无知识图谱"
        :image-size="120"
      >
        <p>选择一个课程开始学习吧</p>
      </el-empty>
    </div>

    <!-- 知识图谱查看对话框 -->
    <el-dialog
      v-model="showGraphDialog"
      :title="currentGraph?.title || '知识图谱'"
      width="90%"
      top="5vh"
      :close-on-click-modal="false"
    >
      <KnowledgeGraph
        v-if="currentGraphData"
        :graph-data="currentGraphData"
        :editable="false"
        container-height="70vh"
        @node-click="handleNodeClick"
        @progress-update="handleProgressUpdate"
      />
      
      <template #footer>
        <div style="display: flex; justify-content: space-between; align-items: center;">
          <div class="dialog-progress">
            <span>学习进度：</span>
            <el-progress
              :percentage="currentGraphProgress"
              :status="currentGraphProgress >= 100 ? 'success' : ''"
              style="width: 200px; margin-left: 8px;"
            />
            <span style="margin-left: 8px;">{{ currentGraphProgress }}%</span>
          </div>
          
          <div>
            <el-button @click="showGraphDialog = false">关闭</el-button>
            <el-button type="primary" @click="analyzeCurrentGraph">
              学习分析
            </el-button>
          </div>
        </div>
      </template>
    </el-dialog>

    <!-- 学习分析对话框 -->
    <el-dialog
      v-model="showAnalysisDialog"
      title="学习分析"
      width="600px"
    >
      <div v-if="analysisData" class="analysis-content">
        <div class="analysis-section">
          <h4>学习路径推荐</h4>
          <ol class="learning-path">
            <li v-for="(step, index) in analysisData.learningPath" :key="index">
              {{ step }}
            </li>
          </ol>
        </div>
        
        <div class="analysis-section">
          <h4>重点知识点</h4>
          <div class="tag-list">
            <el-tag
              v-for="point in analysisData.keyPoints"
              :key="point"
              type="warning"
            >
              {{ point }}
            </el-tag>
          </div>
        </div>
        
        <div class="analysis-section">
          <h4>难点知识点</h4>
          <div class="tag-list">
            <el-tag
              v-for="point in analysisData.difficultPoints"
              :key="point"
              type="danger"
            >
              {{ point }}
            </el-tag>
          </div>
        </div>
      </div>
      
      <template #footer>
        <el-button @click="showAnalysisDialog = false">关闭</el-button>
      </template>
    </el-dialog>
  </div>
</template>

<script setup>
import { ref, reactive, onMounted, computed } from 'vue'
import { ElMessage } from 'element-plus'
import KnowledgeGraph from '@/components/KnowledgeGraph.vue'
import { request } from '@/utils/request'
import { formatDate } from '@/utils/date'
import {
  Search, Grid, List, Connection, Clock
} from '@element-plus/icons-vue'

// 响应式数据
const enrolledCourses = ref([])
const courseGraphs = ref([])
const publicGraphs = ref([])
const recommendedGraphs = ref([])
const selectedCourse = ref('')
const searchKeyword = ref('')
const viewMode = ref('grid')
const loading = ref(false)

// 对话框状态
const showGraphDialog = ref(false)
const showAnalysisDialog = ref(false)
const currentGraph = ref(null)
const currentGraphData = ref(null)
const analysisData = ref(null)

// 学习进度数据
const learningProgress = reactive({})

// 计算属性
const currentCourseName = computed(() => {
  const course = enrolledCourses.value.find(c => c.id === selectedCourse.value)
  return course?.title || ''
})

const filteredGraphs = computed(() => {
  let graphs = courseGraphs.value
  
  if (searchKeyword.value) {
    graphs = graphs.filter(graph =>
      graph.title.includes(searchKeyword.value) ||
      graph.description?.includes(searchKeyword.value)
    )
  }
  
  return graphs
})

const showEmptyState = computed(() => {
  return !selectedCourse.value && 
         recommendedGraphs.value.length === 0 && 
         publicGraphs.value.length === 0
})

const currentGraphProgress = computed(() => {
  if (!currentGraph.value) return 0
  return getGraphProgress(currentGraph.value.id)
})

// 方法
const loadEnrolledCourses = async () => {
  try {
    const response = await request.get('/api/student/course/enrolled')
    if (response.code === 200) {
      enrolledCourses.value = response.data
    }
  } catch (error) {
    console.error('获取已选课程失败:', error)
  }
}

const loadRecommendedGraphs = async () => {
  try {
    const response = await request.get('/api/student/knowledge-graph/recommended')
    if (response.code === 200) {
      recommendedGraphs.value = response.data
    }
  } catch (error) {
    console.error('获取推荐图谱失败:', error)
  }
}

const loadPublicGraphs = async () => {
  try {
    const response = await request.get('/api/student/knowledge-graph/public?limit=6')
    if (response.code === 200) {
      publicGraphs.value = response.data
    }
  } catch (error) {
    console.error('获取公开图谱失败:', error)
  }
}

const onCourseChange = async (courseId) => {
  if (!courseId) {
    courseGraphs.value = []
    return
  }
  
  try {
    const response = await request.get(`/api/student/knowledge-graph/course/${courseId}`)
    if (response.code === 200) {
      courseGraphs.value = response.data
    }
  } catch (error) {
    ElMessage.error('获取课程图谱失败: ' + error.message)
  }
}

const viewGraph = async (graph) => {
  try {
    const response = await request.get(`/api/student/knowledge-graph/${graph.id}`)
    if (response.code === 200) {
      currentGraph.value = graph
      currentGraphData.value = response.data
      showGraphDialog.value = true
    }
  } catch (error) {
    ElMessage.error('获取图谱详情失败: ' + error.message)
  }
}

const analyzeGraph = async (graph) => {
  try {
    const response = await request.post('/api/student/knowledge-graph/analyze', {
      graphId: graph.id,
      analysisType: 'path'
    })
    
    if (response.code === 200) {
      analysisData.value = response.data
      showAnalysisDialog.value = true
    }
  } catch (error) {
    ElMessage.error('分析失败: ' + error.message)
  }
}

const analyzeCurrentGraph = () => {
  if (currentGraph.value) {
    analyzeGraph(currentGraph.value)
  }
}

const handleNodeClick = (node) => {
  console.log('节点点击:', node)
}

const handleProgressUpdate = async (data) => {
  try {
    await request.post(`/api/student/knowledge-graph/${currentGraph.value.id}/progress`, {
      nodeId: data.nodeId,
      completed: data.completed
    })
    
    // 更新本地进度
    if (!learningProgress[currentGraph.value.id]) {
      learningProgress[currentGraph.value.id] = {}
    }
    learningProgress[currentGraph.value.id][data.nodeId] = data.completed ? 100 : 0
    
  } catch (error) {
    ElMessage.error('更新进度失败: ' + error.message)
  }
}

const handleSearch = () => {
  // 搜索在计算属性中处理
}

const getGraphProgress = (graphId) => {
  const graphProgress = learningProgress[graphId]
  if (!graphProgress) return 0
  
  const completedNodes = Object.values(graphProgress).filter(progress => progress >= 100).length
  const totalNodes = Object.keys(graphProgress).length
  
  return totalNodes > 0 ? Math.round((completedNodes / totalNodes) * 100) : 0
}

const getNodeCount = (graph) => {
  // 这里应该解析graph_data获取节点数量，暂时返回默认值
  return Math.floor(Math.random() * 20) + 5
}

// 生命周期
onMounted(() => {
  loadEnrolledCourses()
  loadRecommendedGraphs()
  loadPublicGraphs()
})
</script>

<style scoped>
.knowledge-graph-view {
  padding: 24px;
  background: #f5f7fa;
  min-height: 100vh;
}

.page-header {
  margin-bottom: 24px;
}

.page-header h1 {
  margin: 0 0 8px 0;
  color: #2c3e50;
  font-size: 28px;
  font-weight: 600;
}

.page-header p {
  margin: 0;
  color: #7f8c8d;
  font-size: 16px;
}

.filter-bar {
  display: flex;
  align-items: center;
  margin-bottom: 24px;
  padding: 16px;
  background: #fff;
  border-radius: 8px;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}

.recommended-section,
.course-graphs,
.public-graphs {
  margin-bottom: 32px;
}

.recommended-section h3,
.public-graphs h3 {
  margin: 0 0 16px 0;
  color: #2c3e50;
  font-size: 20px;
  font-weight: 600;
}

.section-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 16px;
}

.section-header h3 {
  margin: 0;
  color: #2c3e50;
  font-size: 20px;
  font-weight: 600;
}

.graph-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
  gap: 20px;
}

.graph-card {
  background: #fff;
  border-radius: 12px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
  transition: all 0.3s ease;
  cursor: pointer;
  overflow: hidden;
}

.graph-card:hover {
  transform: translateY(-4px);
  box-shadow: 0 8px 24px rgba(0, 0, 0, 0.15);
}

.graph-card.recommended {
  border: 2px solid #f39c12;
}

.graph-card.public {
  border: 2px solid #2ecc71;
}

.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 16px 20px;
  border-bottom: 1px solid #f0f0f0;
}

.card-header h4 {
  margin: 0;
  color: #2c3e50;
  font-size: 16px;
  font-weight: 600;
  flex: 1;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.progress-badge {
  display: flex;
  align-items: center;
  gap: 8px;
}

.progress-text {
  font-size: 12px;
  color: #606266;
  font-weight: 600;
}

.card-content {
  padding: 20px;
}

.description {
  margin: 0 0 16px 0;
  color: #606266;
  font-size: 14px;
  line-height: 1.6;
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
  overflow: hidden;
}

.card-info {
  display: flex;
  justify-content: space-between;
  margin-bottom: 16px;
  font-size: 13px;
  color: #909399;
}

.graph-stats {
  display: flex;
  flex-direction: column;
  gap: 8px;
  margin-bottom: 16px;
}

.stat-item {
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 13px;
  color: #909399;
}

.card-actions {
  display: flex;
  gap: 8px;
}

.graph-list {
  background: #fff;
  border-radius: 8px;
  overflow: hidden;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}

.list-item {
  display: flex;
  align-items: center;
  padding: 20px;
  border-bottom: 1px solid #f0f0f0;
  cursor: pointer;
  transition: background-color 0.3s ease;
}

.list-item:hover {
  background-color: #f8f9fa;
}

.list-item:last-child {
  border-bottom: none;
}

.item-content {
  flex: 1;
}

.item-content h4 {
  margin: 0 0 8px 0;
  color: #2c3e50;
  font-size: 16px;
  font-weight: 600;
}

.item-content p {
  margin: 0 0 8px 0;
  color: #606266;
  font-size: 14px;
  line-height: 1.5;
}

.item-meta {
  display: flex;
  gap: 16px;
  font-size: 13px;
  color: #909399;
}

.item-progress {
  width: 200px;
  margin: 0 20px;
}

.progress-label {
  font-size: 12px;
  color: #606266;
  margin-top: 4px;
  display: block;
}

.item-actions {
  width: 80px;
}

.empty-state {
  text-align: center;
  padding: 60px 20px;
}

.dialog-progress {
  display: flex;
  align-items: center;
}

.analysis-content {
  padding: 16px 0;
}

.analysis-section {
  margin-bottom: 24px;
}

.analysis-section h4 {
  margin: 0 0 12px 0;
  color: #2c3e50;
  font-size: 16px;
  font-weight: 600;
}

.learning-path {
  margin: 0;
  padding-left: 20px;
  color: #606266;
  line-height: 1.8;
}

.tag-list {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}

/* 响应式设计 */
@media (max-width: 768px) {
  .filter-bar {
    flex-direction: column;
    gap: 12px;
    align-items: stretch;
  }
  
  .graph-grid {
    grid-template-columns: 1fr;
  }
  
  .list-item {
    flex-direction: column;
    align-items: stretch;
    gap: 16px;
  }
  
  .item-progress,
  .item-actions {
    width: 100%;
  }
}
</style> 