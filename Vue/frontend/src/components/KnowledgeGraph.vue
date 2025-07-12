<template>
  <div class="knowledge-graph-container">
    <!-- 工具栏 -->
    <div class="graph-toolbar">
      <div class="toolbar-left">
        <el-input
          v-model="searchKeyword"
          placeholder="搜索知识点..."
          size="small"
          style="width: 200px"
          clearable
          @input="onSearch"
        >
          <template #prefix>
            <el-icon><Search /></el-icon>
          </template>
        </el-input>
        
        <el-select
          v-model="filterType"
          placeholder="筛选类型"
          size="small"
          style="width: 120px; margin-left: 10px"
          @change="onFilterChange"
        >
          <el-option label="全部" value="all" />
          <el-option label="概念" value="concept" />
          <el-option label="技能" value="skill" />
          <el-option label="主题" value="topic" />
          <el-option label="章节" value="chapter" />
        </el-select>
      </div>
      
      <div class="toolbar-right">
        <el-button-group size="small">
          <el-button @click="zoomIn">
            <el-icon><ZoomIn /></el-icon>
          </el-button>
          <el-button @click="zoomOut">
            <el-icon><ZoomOut /></el-icon>
          </el-button>
          <el-button @click="resetZoom">
            <el-icon><Refresh /></el-icon>
          </el-button>
        </el-button-group>
        
        <el-button
          v-if="editable"
          type="primary"
          size="small"
          style="margin-left: 10px"
          @click="saveGraph"
          :loading="saving"
        >
          保存图谱
        </el-button>
      </div>
    </div>
    
    <!-- 图谱主体 -->
    <div 
      ref="graphContainer" 
      class="graph-canvas"
      :style="{ height: containerHeight }"
    ></div>
    
    <!-- 侧边面板 -->
    <div v-if="selectedNode" class="node-panel">
      <div class="panel-header">
        <h4>{{ selectedNode.name }}</h4>
        <el-button
          type="text"
          size="small"
          @click="selectedNode = null"
        >
          <el-icon><Close /></el-icon>
        </el-button>
      </div>
      
      <div class="panel-content">
        <div class="node-info">
          <p><strong>类型:</strong> {{ getNodeTypeLabel(selectedNode.type) }}</p>
          <p><strong>级别:</strong> {{ selectedNode.level }}</p>
          <p v-if="selectedNode.description">
            <strong>描述:</strong> {{ selectedNode.description }}
          </p>
        </div>
        
        <!-- 学习进度（学生端） -->
        <div v-if="!editable && userRole === 'student'" class="learning-progress">
          <h5>学习进度</h5>
          <el-progress 
            :percentage="getNodeProgress(selectedNode.id)"
            :status="getNodeProgress(selectedNode.id) >= 100 ? 'success' : ''"
          />
          <div class="progress-actions">
            <el-button
              size="small"
              @click="markAsCompleted(selectedNode.id)"
              :disabled="getNodeProgress(selectedNode.id) >= 100"
            >
              标记完成
            </el-button>
          </div>
        </div>
        
        <!-- 相关章节链接 -->
        <div v-if="selectedNode.chapterId" class="related-links">
          <h5>相关内容</h5>
          <el-button
            type="text"
            size="small"
            @click="goToChapter(selectedNode.chapterId, selectedNode.sectionId)"
          >
            查看章节内容
          </el-button>
        </div>
      </div>
    </div>
    
    <!-- 图例 -->
    <div class="graph-legend">
      <h5>图例</h5>
      <div class="legend-items">
        <div class="legend-item">
          <div class="legend-node concept"></div>
          <span>概念</span>
        </div>
        <div class="legend-item">
          <div class="legend-node skill"></div>
          <span>技能</span>
        </div>
        <div class="legend-item">
          <div class="legend-node topic"></div>
          <span>主题</span>
        </div>
        <div class="legend-item">
          <div class="legend-node chapter"></div>
          <span>章节</span>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, reactive, onMounted, nextTick, watch, computed } from 'vue'
import { useRouter } from 'vue-router'
import { useAuthStore } from '@/stores/auth'
import { ElMessage } from 'element-plus'
import * as echarts from 'echarts'
import { Search, ZoomIn, ZoomOut, Refresh, Close } from '@element-plus/icons-vue'

// Props
const props = defineProps({
  graphData: {
    type: Object,
    required: true
  },
  editable: {
    type: Boolean,
    default: false
  },
  containerHeight: {
    type: String,
    default: '600px'
  }
})

// Emits
const emit = defineEmits(['save', 'nodeClick', 'progressUpdate'])

// 状态管理
const router = useRouter()
const authStore = useAuthStore()
const userRole = computed(() => authStore.user?.role?.toLowerCase())

// 响应式数据
const graphContainer = ref(null)
const chartInstance = ref(null)
const searchKeyword = ref('')
const filterType = ref('all')
const selectedNode = ref(null)
const saving = ref(false)

// 学习进度数据（学生端）
const learningProgress = reactive({})

// 图谱配置
const graphOption = reactive({
  tooltip: {
    show: true,
    formatter: (params) => {
      if (params.dataType === 'node') {
        const node = params.data
        return `
          <div style="padding: 8px;">
            <h4 style="margin: 0 0 8px 0;">${node.name}</h4>
            <p style="margin: 0;"><strong>类型:</strong> ${getNodeTypeLabel(node.type)}</p>
            <p style="margin: 0;"><strong>级别:</strong> ${node.level}</p>
            ${node.description ? `<p style="margin: 0;"><strong>描述:</strong> ${node.description}</p>` : ''}
          </div>
        `
      } else if (params.dataType === 'edge') {
        const edge = params.data
        return `
          <div style="padding: 8px;">
            <h4 style="margin: 0 0 8px 0;">关系: ${edge.description || edge.type}</h4>
            <p style="margin: 0;">${edge.source} → ${edge.target}</p>
          </div>
        `
      }
    }
  },
  series: [{
    type: 'graph',
    layout: 'force',
    symbolSize: (value, params) => {
      return params.data.style?.size || 20
    },
    roam: true,
    label: {
      show: true,
      position: 'bottom',
      fontSize: (params) => {
        return params.data.style?.fontSize || 12
      }
    },
    edgeSymbol: ['none', 'arrow'],
    edgeSymbolSize: [0, 10],
    edgeLabel: {
      fontSize: 10
    },
    // 力导向图配置
    force: {
      repulsion: 1000,  // 节点之间的斥力，值越大距离越远
      gravity: 0.05,    // 节点受到的向中心的引力，值越小越分散
      edgeLength: 200,  // 边的长度，值越大节点间距越大
      layoutAnimation: true
    },
    data: [],
    links: [],
    categories: [
      { name: 'concept', itemStyle: { color: '#3498db' } },
      { name: 'skill', itemStyle: { color: '#e74c3c' } },
      { name: 'topic', itemStyle: { color: '#2ecc71' } },
      { name: 'chapter', itemStyle: { color: '#f39c12' } }
    ],
    emphasis: {
      focus: 'adjacency'
    },
    lineStyle: {
      opacity: 0.8,
      width: 2,
      curveness: 0.1
    }
  }]
})

// 计算属性
const filteredNodes = computed(() => {
  if (!props.graphData?.nodes) return []
  
  let nodes = props.graphData.nodes
  
  // 类型筛选
  if (filterType.value !== 'all') {
    nodes = nodes.filter(node => node.type === filterType.value)
  }
  
  // 搜索筛选
  if (searchKeyword.value) {
    const keyword = searchKeyword.value.toLowerCase()
    nodes = nodes.filter(node => 
      node.name.toLowerCase().includes(keyword) ||
      (node.description && node.description.toLowerCase().includes(keyword))
    )
  }
  
  return nodes
})

// 方法
const initChart = () => {
  if (!graphContainer.value) return
  
  chartInstance.value = echarts.init(graphContainer.value)
  updateChartData()
  
  // 绑定事件
  chartInstance.value.on('click', onNodeClick)
  chartInstance.value.on('dataZoom', onDataZoom)
  
  // 响应式调整
  window.addEventListener('resize', () => {
    chartInstance.value?.resize()
  })
}

const updateChartData = () => {
  if (!chartInstance.value || !props.graphData) return
  
  const nodes = filteredNodes.value.map(node => ({
    ...node,
    x: node.position?.x,
    y: node.position?.y,
    // 在力导向布局中，只有fixed为true的节点会保持固定位置
    fixed: node.position?.fixed === true,
    category: node.type,
    itemStyle: {
      color: node.style?.color || getDefaultColor(node.type),
      borderWidth: selectedNode.value?.id === node.id ? 3 : 0,
      borderColor: '#409eff'
    },
    label: {
      show: true,
      fontSize: node.style?.fontSize || 12,
      // 增加标签与节点的距离
      distance: 5
    }
  }))
  
  const links = props.graphData.edges?.filter(edge => {
    const sourceExists = filteredNodes.value.some(node => node.id === edge.source)
    const targetExists = filteredNodes.value.some(node => node.id === edge.target)
    return sourceExists && targetExists
  }).map(edge => ({
    ...edge,
    // 使用边的权重来调整边的长度
    value: edge.weight || 1,
    lineStyle: {
      color: edge.style?.color || '#7f8c8d',
      width: edge.style?.width || 2,
      type: edge.style?.lineType || 'solid'
    }
  })) || []
  
  graphOption.series[0].data = nodes
  graphOption.series[0].links = links
  
  chartInstance.value.setOption(graphOption, true)
}

const onNodeClick = (params) => {
  if (params.dataType === 'node') {
    selectedNode.value = params.data
    emit('nodeClick', params.data)
  }
}

const onDataZoom = (params) => {
  // 处理缩放事件
}

const onSearch = () => {
  updateChartData()
}

const onFilterChange = () => {
  updateChartData()
}

const zoomIn = () => {
  chartInstance.value?.dispatchAction({
    type: 'dataZoom',
    zoom: 1.2
  })
}

const zoomOut = () => {
  chartInstance.value?.dispatchAction({
    type: 'dataZoom',
    zoom: 0.8
  })
}

const resetZoom = () => {
  chartInstance.value?.dispatchAction({
    type: 'restore'
  })
}

const saveGraph = async () => {
  if (!props.editable) return
  
  saving.value = true
  try {
    // 获取当前节点位置信息
    const currentOption = chartInstance.value.getOption()
    const updatedData = {
      ...props.graphData,
      nodes: props.graphData.nodes.map((node, index) => {
        const currentNode = currentOption.series[0].data[index]
        return {
          ...node,
          position: {
            x: currentNode?.x || node.position?.x || 0,
            y: currentNode?.y || node.position?.y || 0,
            fixed: currentNode?.fixed || false
          }
        }
      })
    }
    
    emit('save', updatedData)
    ElMessage.success('图谱保存成功')
  } catch (error) {
    ElMessage.error('保存失败: ' + error.message)
  } finally {
    saving.value = false
  }
}

const getNodeTypeLabel = (type) => {
  const labels = {
    concept: '概念',
    skill: '技能',
    topic: '主题',
    chapter: '章节'
  }
  return labels[type] || type
}

const getDefaultColor = (type) => {
  const colors = {
    concept: '#3498db',
    skill: '#e74c3c',
    topic: '#2ecc71',
    chapter: '#f39c12'
  }
  return colors[type] || '#95a5a6'
}

const getNodeProgress = (nodeId) => {
  return learningProgress[nodeId] || 0
}

const markAsCompleted = async (nodeId) => {
  try {
    learningProgress[nodeId] = 100
    emit('progressUpdate', { nodeId, completed: true })
    ElMessage.success('已标记为完成')
  } catch (error) {
    ElMessage.error('更新进度失败')
  }
}

const goToChapter = (chapterId, sectionId) => {
  if (sectionId) {
    router.push(`/student/courses/${props.graphData.courseId}/sections/${sectionId}`)
  } else {
    router.push(`/student/courses/${props.graphData.courseId}`)
  }
}

// 监听数据变化
watch(() => props.graphData, () => {
  nextTick(() => {
    updateChartData()
  })
}, { deep: true })

watch(filteredNodes, () => {
  updateChartData()
})

// 组件挂载
onMounted(() => {
  nextTick(() => {
    initChart()
  })
})
</script>

<style scoped>
.knowledge-graph-container {
  position: relative;
  background: #fff;
  border-radius: 8px;
  box-shadow: 0 2px 12px rgba(0, 0, 0, 0.1);
  overflow: hidden;
}

.graph-toolbar {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 16px;
  border-bottom: 1px solid #e4e7ed;
  background: #f8f9fa;
}

.toolbar-left {
  display: flex;
  align-items: center;
}

.toolbar-right {
  display: flex;
  align-items: center;
}

.graph-canvas {
  width: 100%;
  background: #fafafa;
}

.node-panel {
  position: absolute;
  top: 60px;
  right: 16px;
  width: 280px;
  background: #fff;
  border-radius: 8px;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
  z-index: 1000;
}

.panel-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 16px;
  border-bottom: 1px solid #e4e7ed;
  background: #409eff;
  color: #fff;
  border-radius: 8px 8px 0 0;
}

.panel-header h4 {
  margin: 0;
  font-size: 16px;
}

.panel-content {
  padding: 16px;
}

.node-info p {
  margin: 8px 0;
  font-size: 14px;
  line-height: 1.4;
}

.learning-progress {
  margin-top: 16px;
  padding-top: 16px;
  border-top: 1px solid #e4e7ed;
}

.learning-progress h5 {
  margin: 0 0 12px 0;
  font-size: 14px;
  color: #303133;
}

.progress-actions {
  margin-top: 12px;
}

.related-links {
  margin-top: 16px;
  padding-top: 16px;
  border-top: 1px solid #e4e7ed;
}

.related-links h5 {
  margin: 0 0 8px 0;
  font-size: 14px;
  color: #303133;
}

.graph-legend {
  position: absolute;
  bottom: 16px;
  left: 16px;
  background: rgba(255, 255, 255, 0.9);
  border-radius: 6px;
  padding: 12px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
}

.graph-legend h5 {
  margin: 0 0 8px 0;
  font-size: 12px;
  color: #606266;
}

.legend-items {
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.legend-item {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 12px;
  color: #606266;
}

.legend-node {
  width: 12px;
  height: 12px;
  border-radius: 50%;
}

.legend-node.concept {
  background-color: #3498db;
}

.legend-node.skill {
  background-color: #e74c3c;
}

.legend-node.topic {
  background-color: #2ecc71;
}

.legend-node.chapter {
  background-color: #f39c12;
}

/* 响应式设计 */
@media (max-width: 768px) {
  .node-panel {
    position: fixed;
    top: 0;
    right: 0;
    width: 100%;
    height: 100%;
    z-index: 2000;
  }
  
  .graph-legend {
    display: none;
  }
  
  .graph-toolbar {
    flex-direction: column;
    gap: 12px;
  }
  
  .toolbar-left,
  .toolbar-right {
    width: 100%;
    justify-content: center;
  }
}
</style> 