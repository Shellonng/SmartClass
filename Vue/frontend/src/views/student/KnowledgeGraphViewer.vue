<template>
  <div class="knowledge-graph-viewer">
    <h1>课程知识图谱</h1>
    
    <!-- 课程选择 -->
    <div class="course-selector">
      <el-form :inline="true">
        <el-form-item label="选择课程">
          <el-select v-model="selectedCourseId" placeholder="请选择课程" @change="loadCourseGraphs">
            <el-option
              v-for="course in courses"
              :key="course.id"
              :label="course.title"
              :value="course.id"
            />
          </el-select>
        </el-form-item>
      </el-form>
    </div>
    
    <!-- 图谱列表 -->
    <div v-if="selectedCourseId" class="graph-list">
      <h2>可用知识图谱</h2>
      
      <el-empty v-if="courseGraphs.length === 0" description="该课程暂无知识图谱" />
      
      <el-row :gutter="20" v-else>
        <el-col :span="8" v-for="graph in courseGraphs" :key="graph.id">
          <el-card class="graph-card" shadow="hover">
            <template #header>
              <div class="card-header">
                <h3>{{ graph.title }}</h3>
                <el-tag size="small">{{ graphTypeMap[graph.graphType] || graph.graphType }}</el-tag>
              </div>
            </template>
            <div class="card-content">
              <p class="description">{{ graph.description || '暂无描述' }}</p>
              <p class="meta">
                <span>更新时间: {{ formatDate(graph.updateTime) }}</span>
                <span>浏览次数: {{ graph.viewCount }}</span>
              </p>
              <div class="actions">
                <el-button type="primary" @click="viewGraph(graph)">查看图谱</el-button>
              </div>
            </div>
          </el-card>
        </el-col>
      </el-row>
    </div>
    
    <!-- 推荐图谱 -->
    <div class="recommended-graphs" v-if="!selectedCourseId && publicGraphs.length > 0">
      <h2>推荐知识图谱</h2>
      
      <el-row :gutter="20">
        <el-col :span="8" v-for="graph in publicGraphs" :key="graph.id">
          <el-card class="graph-card" shadow="hover">
            <template #header>
              <div class="card-header">
                <h3>{{ graph.title }}</h3>
                <div>
                  <el-tag size="small">{{ graph.courseName }}</el-tag>
                  <el-tag size="small" type="success" style="margin-left: 5px">
                    {{ graphTypeMap[graph.graphType] || graph.graphType }}
                  </el-tag>
                </div>
              </div>
            </template>
            <div class="card-content">
              <p class="description">{{ graph.description || '暂无描述' }}</p>
              <p class="meta">
                <span>更新时间: {{ formatDate(graph.updateTime) }}</span>
                <span>浏览次数: {{ graph.viewCount }}</span>
              </p>
              <div class="actions">
                <el-button type="primary" @click="viewGraph(graph)">查看图谱</el-button>
              </div>
            </div>
          </el-card>
        </el-col>
      </el-row>
    </div>
    
    <!-- 图谱详情对话框 -->
    <el-dialog
      v-model="graphDialogVisible"
      :title="currentGraph?.title || '知识图谱'"
      width="80%"
      destroy-on-close
    >
      <div v-if="graphData">
        <p v-if="graphData.description" class="graph-description">
          {{ graphData.description }}
        </p>
        
        <!-- 图谱可视化区域 -->
        <div class="graph-container" ref="graphContainer"></div>
        
        <!-- 节点详情 -->
        <div v-if="selectedNode" class="node-details">
          <h3>{{ selectedNode.name }}</h3>
          <el-descriptions :column="1" border>
            <el-descriptions-item label="类型">
              {{ nodeTypeMap[selectedNode.type] || selectedNode.type }}
            </el-descriptions-item>
            <el-descriptions-item label="重要性">
              <el-rate
                v-model="selectedNode.level"
                disabled
                :max="5"
                :colors="['#99A9BF', '#F7BA2A', '#FF9900']"
              />
            </el-descriptions-item>
            <el-descriptions-item label="描述">
              {{ selectedNode.description || '暂无描述' }}
            </el-descriptions-item>
          </el-descriptions>
          
          <!-- 学习建议 -->
          <div class="learning-suggestions" v-if="selectedNode.properties?.learningMaterials">
            <h4>学习资料</h4>
            <ul>
              <li v-for="(material, index) in selectedNode.properties.learningMaterials" :key="index">
                {{ material }}
              </li>
            </ul>
          </div>
        </div>
      </div>
      
      <template #footer>
        <span class="dialog-footer">
          <el-button @click="graphDialogVisible = false">关闭</el-button>
          <el-button type="primary" @click="exportGraph">导出图片</el-button>
        </span>
      </template>
    </el-dialog>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted, nextTick } from 'vue'
import * as echarts from 'echarts'
import { ElMessage } from 'element-plus'
import { studentKnowledgeGraphAPI } from '@/api/knowledgeGraph'
import { courseAPI } from '@/api/course'
import type { 
  KnowledgeGraphData, 
  KnowledgeGraph, 
  KnowledgeGraphNode 
} from '@/api/knowledgeGraph'
import type { ApiResponse } from '@/api/course'

// 状态变量
const courses = ref<any[]>([])
const selectedCourseId = ref<number | null>(null)
const courseGraphs = ref<KnowledgeGraph[]>([])
const publicGraphs = ref<KnowledgeGraph[]>([])
const graphDialogVisible = ref(false)
const currentGraph = ref<KnowledgeGraph | null>(null)
const graphData = ref<KnowledgeGraphData | null>(null)
const selectedNode = ref<KnowledgeGraphNode | null>(null)
const graphChart = ref<echarts.ECharts | null>(null)
const graphContainer = ref<HTMLElement | null>(null)

// 映射表
const graphTypeMap = {
  'concept': '概念图谱',
  'skill': '技能图谱',
  'comprehensive': '综合图谱'
}

const nodeTypeMap = {
  'concept': '概念',
  'skill': '技能',
  'topic': '主题',
  'chapter': '章节'
}

// 生命周期钩子
onMounted(async () => {
  await loadCourses()
  await loadPublicGraphs()
})

// 加载课程列表
const loadCourses = async () => {
  try {
    const response = await courseAPI.getStudentCourses()
    courses.value = response?.data?.content || []
  } catch (error) {
    console.error('加载课程失败:', error)
    ElMessage.error('加载课程列表失败')
  }
}

// 加载公开图谱
const loadPublicGraphs = async () => {
  try {
    const response: ApiResponse<KnowledgeGraph[]> = await studentKnowledgeGraphAPI.getPublicGraphs(6) // 限制加载6个
    publicGraphs.value = response?.data || []
  } catch (error) {
    console.error('加载公开图谱失败:', error)
    ElMessage.error('加载推荐图谱失败')
  }
}

// 加载课程图谱
const loadCourseGraphs = async () => {
  if (!selectedCourseId.value) {
    courseGraphs.value = []
    return
  }
  
  try {
    const response: ApiResponse<KnowledgeGraph[]> = await studentKnowledgeGraphAPI.getCourseGraphs(selectedCourseId.value)
    courseGraphs.value = response?.data || []
  } catch (error) {
    console.error('加载课程图谱失败:', error)
    ElMessage.error('加载课程图谱失败')
  }
}

// 查看图谱
const viewGraph = async (graph: KnowledgeGraph) => {
  try {
    currentGraph.value = graph
    graphDialogVisible.value = true
    
    const response: ApiResponse<KnowledgeGraphData> = await studentKnowledgeGraphAPI.getGraphDetail(graph.id)
    if (response?.data) {
      graphData.value = response.data
      
      // 渲染图谱
      nextTick(() => {
        renderGraph(graphData.value!)
      })
    }
  } catch (error) {
    console.error('获取图谱详情失败:', error)
    ElMessage.error('获取图谱详情失败')
  }
}

// 渲染图谱
const renderGraph = (data: KnowledgeGraphData | null = null) => {
  if (!graphContainer.value) {
    console.error('图谱容器不存在')
    return
  }

  try {
    const finalData = data || graphData.value
    if (!finalData || !finalData.nodes || !finalData.edges) {
      console.error('图谱数据不完整或不存在')
      return
    }
  
  // 销毁旧图表
    const element = graphContainer.value as HTMLElement
    let chart = echarts.getInstanceByDom(element)
    if (chart) {
      chart.dispose()
  }
  
    // 创建新图表实例
    chart = echarts.init(element)
    graphChart.value = chart
    
    // 节点分类
    const categories = [
      { name: '章节' },
      { name: '概念' },
      { name: '技能' },
      { name: '主题' }
    ]
    
    // 处理节点数据
    const nodes = finalData.nodes.map(node => ({
    id: node.id,
    name: node.name,
      symbolSize: node.style?.size || (node.type === 'chapter' ? 50 : 40),
      category: node.type === 'chapter' ? 0 : node.type === 'concept' ? 1 : node.type === 'skill' ? 2 : 3,
    value: node.level || 1,
    itemStyle: {
      color: node.style?.color || getNodeColor(node.type)
    },
      x: node.position?.x,
      y: node.position?.y,
      fixed: node.position?.fixed || false,
      draggable: true,
    label: {
        show: true
      }
    }))
    
    // 处理边数据
    const links = finalData.edges.map(edge => ({
    source: edge.source,
    target: edge.target,
      value: edge.weight || 1,
      name: edge.description || '',
    lineStyle: {
        width: edge.weight ? Math.max(1, Math.min(5, edge.weight)) : 2,
      curveness: 0.2
      }
  }))
  
    // 图表配置
  const option = {
    title: {
        text: finalData.title || '知识图谱',
        subtext: finalData.description || '',
      top: 'top',
      left: 'center'
    },
    tooltip: {
      trigger: 'item',
        formatter: function(params: any) {
        if (params.dataType === 'node') {
            return `<div style="font-weight:bold">${params.name}</div>` +
                   `<div>类型: ${nodeTypeMap[params.data.category] || '未知'}</div>` +
                   `<div>重要性: ${params.value}</div>`
        } else {
            return `${params.data.source} → ${params.data.target}`
        }
      }
    },
    legend: {
        data: categories.map(category => category.name),
        top: 'bottom',
        left: 'center',
        selectedMode: 'multiple'
    },
    animationDuration: 1500,
    animationEasingUpdate: 'quinticInOut' as const,
    series: [
      {
          name: '知识图谱',
        type: 'graph',
        layout: 'force',
        data: nodes,
          links: links,
          categories: categories,
        roam: true,
        label: {
            show: true,
            position: 'right',
            formatter: '{b}'
          },
          edgeSymbol: ['none', 'arrow'],
          edgeLabel: {
            fontSize: 10,
            formatter: '{c}'
          },
          force: {
            repulsion: 200,
            gravity: 0.1,
            edgeLength: 80,
            layoutAnimation: true
        },
        lineStyle: {
          color: 'source',
            curveness: 0.2
        },
        emphasis: {
          focus: 'adjacency',
          lineStyle: {
              width: 4
          }
        }
      }
    ]
  }
  
    chart.setOption(option)
  
    // 注册点击事件
    chart.on('click', (params: any) => {
    if (params.dataType === 'node') {
        // 从原始数据中找到对应节点
        const node = finalData.nodes.find(n => n.id === params.data.id)
        if (node) {
          selectedNode.value = node
        }
    }
  })
  
    // 响应窗口大小变化
  window.addEventListener('resize', () => {
      chart.resize()
  })
  } catch (error) {
    console.error('渲染图谱失败:', error)
    ElMessage.error('渲染知识图谱失败，请稍后重试')
  
    // 尝试使用默认数据
    createDefaultGraph()
  }
}

// 获取节点颜色
const getNodeColor = (type: string) => {
  const colorMap: Record<string, string> = {
    'chapter': '#3498db', // 章节 - 蓝色
    'concept': '#2ecc71', // 概念 - 绿色
    'skill': '#e74c3c',   // 技能 - 红色
    'topic': '#f39c12'    // 主题 - 橙色
  }
  return colorMap[type] || '#95a5a6' // 默认灰色
}

// 创建默认图谱（演示用）
const createDefaultGraph = () => {
  if (courses.value.length === 0) return
  
  // 创建一个简单的演示图谱
  const courseTitle = currentGraph.value?.title || 
    (selectedCourseId.value && courses.value.find(c => c.id === selectedCourseId.value)?.title) || 
    '示例课程'
  
  const demoData: KnowledgeGraphData = {
    title: `${courseTitle}知识图谱`,
    description: '知识结构可视化',
    nodes: [
      {
        id: 'root',
        name: courseTitle,
        type: 'topic',
        level: 3,
        description: '课程主题',
        position: { x: 300, y: 300 }
      },
      {
        id: 'node1',
        name: '第一章：基础知识',
        type: 'chapter',
        level: 2,
        position: { x: 150, y: 200 }
      },
      {
        id: 'node2',
        name: '第二章：核心概念',
        type: 'chapter',
        level: 2,
        position: { x: 450, y: 200 }
      },
      {
        id: 'node3',
        name: '实践应用',
        type: 'skill',
        level: 2,
        position: { x: 300, y: 450 }
}
    ],
    edges: [
      { id: 'e1', source: 'root', target: 'node1', type: 'includes', description: '包含' },
      { id: 'e2', source: 'root', target: 'node2', type: 'includes', description: '包含' },
      { id: 'e3', source: 'root', target: 'node3', type: 'includes', description: '包含' },
      { id: 'e4', source: 'node1', target: 'node2', type: 'prerequisite', description: '先导' }
    ]
  }
  
  graphData.value = demoData
  renderGraph(demoData)
}

// 导出图谱
const exportGraph = () => {
  if (!graphChart.value) return
  
  try {
    // 获取图表的数据URL
    const dataURL = graphChart.value.getDataURL({
      pixelRatio: 2,
      backgroundColor: '#fff'
    })
    
    // 创建下载链接
    const link = document.createElement('a')
    link.download = `${currentGraph.value?.title || '知识图谱'}.png`
    link.href = dataURL
    link.click()
  } catch (error) {
    console.error('导出图谱失败:', error)
    ElMessage.error('导出图谱失败')
  }
}

// 格式化日期
const formatDate = (dateStr: string) => {
  if (!dateStr) return ''
  const date = new Date(dateStr)
  return date.toLocaleDateString()
}
</script>

<style scoped>
.knowledge-graph-viewer {
  padding: 20px;
}

.course-selector {
  margin-bottom: 30px;
}

.graph-list {
  margin-bottom: 40px;
}

.recommended-graphs {
  margin-top: 40px;
}

.graph-card {
  margin-bottom: 20px;
  height: 100%;
}

.card-header {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
}

.card-header h3 {
  margin: 0;
  font-size: 16px;
}

.card-content {
  display: flex;
  flex-direction: column;
  height: 150px;
}

.description {
  flex-grow: 1;
  overflow: hidden;
  text-overflow: ellipsis;
  display: -webkit-box;
  -webkit-line-clamp: 3;
  -webkit-box-orient: vertical;
  color: #606266;
}

.meta {
  font-size: 12px;
  color: #909399;
  margin-bottom: 10px;
  display: flex;
  justify-content: space-between;
}

.actions {
  text-align: right;
}

.graph-container {
  width: 100%;
  height: 500px;
  border: 1px solid #dcdfe6;
  border-radius: 4px;
  margin-bottom: 20px;
}

.node-details {
  padding: 15px;
  background-color: #f5f7fa;
  border-radius: 4px;
}

.graph-description {
  margin-bottom: 20px;
  color: #606266;
}

.learning-suggestions {
  margin-top: 15px;
}

.learning-suggestions h4 {
  margin-bottom: 10px;
}
</style> 