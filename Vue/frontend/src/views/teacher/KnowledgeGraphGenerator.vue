<template>
  <div class="knowledge-graph-generator">
    <h1>知识图谱生成</h1>
    
    <!-- 生成表单 -->
    <div class="generator-form">
      <el-form :model="formData" label-width="120px" :rules="rules" ref="formRef">
        <el-form-item label="课程" prop="courseId">
          <el-select v-model="formData.courseId" placeholder="请选择课程" @change="loadChapters">
            <el-option
              v-for="course in courses"
              :key="course.id || 0"
              :label="course.title || course.name || '未命名课程'"
              :value="course.id || 0"
            />
          </el-select>
        </el-form-item>
        
        <!-- 章节选择 -->
        <el-form-item label="选择章节" prop="chapterIds" :rules="[{ required: true, message: '请选择至少一个章节' }]">
          <el-select 
            v-model="formData.chapterIds" 
            multiple
            placeholder="请选择章节"
            :loading="chaptersLoading"
            :disabled="!formData.courseId"
          >
            <el-option 
              v-for="chapter in chapters" 
              :key="chapter.id || 0" 
              :value="chapter.id || 0"
              :label="chapter.title || chapter.name || '未命名章节'"
            />
          </el-select>
        </el-form-item>
        
        <el-form-item label="图谱类型" prop="graphType">
          <el-select v-model="formData.graphType" placeholder="请选择图谱类型">
            <el-option label="概念图谱" value="concept" />
            <el-option label="技能图谱" value="skill" />
            <el-option label="综合图谱" value="comprehensive" />
          </el-select>
        </el-form-item>
        
        <el-form-item label="深度级别" prop="depth">
          <el-slider
            v-model="formData.depth"
            :min="1"
            :max="5"
            :step="1"
            :marks="{1:'简单', 3:'中等', 5:'复杂'}"
          />
        </el-form-item>
        
        <el-form-item label="包含先修关系">
          <el-switch v-model="formData.includePrerequisites" />
        </el-form-item>
        
        <el-form-item label="包含应用关系">
          <el-switch v-model="formData.includeApplications" />
        </el-form-item>
        
        <el-form-item label="附加要求">
          <el-input
            v-model="formData.additionalRequirements"
            type="textarea"
            :rows="3"
            placeholder="可输入额外的生成要求"
          />
        </el-form-item>
        
        <el-form-item>
          <el-button type="primary" @click="generateGraph" :loading="generating">
            生成知识图谱
          </el-button>
          <el-button @click="resetForm">重置</el-button>
        </el-form-item>
      </el-form>
    </div>
    
    <!-- 生成状态 -->
    <div v-if="generationStatus" class="generation-status">
      <el-alert
        :title="statusMessage"
        :type="statusType"
        :closable="false"
        show-icon
      />
      
      <div v-if="generationStatus === 'processing'" class="progress-indicator">
        <el-progress :percentage="50" status="exception" />
        <p>正在生成知识图谱，请稍候...</p>
        <el-button size="small" @click="checkTaskStatus">刷新状态</el-button>
      </div>
    </div>
    
    <!-- 知识图谱预览 -->
    <div v-if="graphData && graphData.nodes" class="graph-preview">
      <h2>{{ graphData.title || '知识图谱预览' }}</h2>
      <p v-if="graphData.description">{{ graphData.description }}</p>
      
      <!-- 图谱可视化区域 -->
      <div class="graph-container" ref="graphContainer"></div>
      
      <!-- 操作按钮 -->
      <div class="graph-actions">
        <el-button type="success" @click="saveGraph">保存图谱</el-button>
        <el-button type="info" @click="exportGraph">导出图谱</el-button>
      </div>
      
      <!-- 节点信息 -->
      <div v-if="selectedNode" class="node-details">
        <h3>节点详情</h3>
        <p><strong>名称:</strong> {{ selectedNode.name }}</p>
        <p><strong>类型:</strong> {{ nodeTypeMap[selectedNode.type] || selectedNode.type }}</p>
        <p><strong>描述:</strong> {{ selectedNode.description || '无描述' }}</p>
      </div>
    </div>
    
    <!-- 我的知识图谱列表 -->
    <div class="my-graphs">
      <h2>我的知识图谱</h2>
      
      <el-table :data="myGraphs" style="width: 100%">
        <el-table-column prop="title" label="标题" />
        <el-table-column prop="courseName" label="课程" />
        <el-table-column prop="graphType" label="类型">
          <template #default="scope">
            {{ graphTypeMap[scope.row.graphType] || scope.row.graphType }}
          </template>
        </el-table-column>
        <el-table-column prop="status" label="状态">
          <template #default="scope">
            <el-tag :type="getStatusTagType(scope.row.status)">
              {{ statusMap[scope.row.status] || scope.row.status }}
            </el-tag>
          </template>
        </el-table-column>
        <el-table-column prop="isPublic" label="是否公开">
          <template #default="scope">
            <el-tag :type="scope.row.isPublic ? 'success' : 'info'">
              {{ scope.row.isPublic ? '公开' : '私有' }}
            </el-tag>
          </template>
        </el-table-column>
        <el-table-column prop="updateTime" label="更新时间" />
        <el-table-column label="操作" width="250">
          <template #default="scope">
            <el-button size="small" @click="viewGraph(scope.row)">查看</el-button>
            <el-button 
              size="small" 
              type="primary" 
              @click="togglePublish(scope.row)"
            >
              {{ scope.row.isPublic ? '取消发布' : '发布' }}
            </el-button>
            <el-button 
              size="small" 
              type="danger" 
              @click="deleteGraph(scope.row)"
            >
              删除
            </el-button>
          </template>
        </el-table-column>
      </el-table>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, reactive, onMounted, nextTick } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import type { FormInstance } from 'element-plus'
import * as echarts from 'echarts'
import { teacherKnowledgeGraphAPI } from '@/api/knowledgeGraph'
import { courseAPI } from '@/api/course'
import { chapterAPI } from '@/api/chapter'
import type { 
  KnowledgeGraphData, 
  KnowledgeGraph, 
  GenerationRequest, 
  GenerationResponse 
} from '@/api/knowledgeGraph'
import type { ApiResponse } from '@/api/course'
import request from '@/utils/request' // 导入request工具

// 表单数据
const formData = reactive<GenerationRequest>({
  courseId: 0,
  chapterIds: [],
  graphType: 'comprehensive',
  depth: 3,
  includePrerequisites: true,
  includeApplications: true,
  additionalRequirements: ''
})

// 表单规则
const rules = {
  courseId: [{ required: true, message: '请选择课程', trigger: 'change' }],
  chapterIds: [{ required: true, message: '请选择至少一个章节', trigger: 'change' }],
  graphType: [{ required: true, message: '请选择图谱类型', trigger: 'change' }]
}

// 状态变量
const formRef = ref<FormInstance>()
const courses = ref<any[]>([])
const chapters = ref<any[]>([])
const chaptersLoading = ref(false)  // 添加章节加载状态
const generating = ref(false)
const generationStatus = ref('')
const statusMessage = ref('')
const statusType = ref('info')
const taskId = ref('')
const graphData = ref<KnowledgeGraphData | null>(null)
const selectedNode = ref<any>(null)
const myGraphs = ref<KnowledgeGraph[]>([])
const graphChart = ref<echarts.ECharts | null>(null)
const graphContainer = ref<HTMLElement | null>(null)
const currentGraph = ref<KnowledgeGraph | null>(null)

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

const statusMap = {
  'draft': '草稿',
  'published': '已发布',
  'archived': '已归档'
}

// 生命周期钩子
onMounted(async () => {
  await loadCourses()
  await loadMyGraphs()
})

// 加载课程列表
const loadCourses = async () => {
  try {
    console.log('📚 开始获取教师课程列表...')
    
    const response = await request.get('/api/teacher/courses')
    console.log('📊 API响应:', response)
    
    // 检查响应是否有效
    if (!response) {
      console.error('API响应为空')
      ElMessage.error('获取课程列表失败：服务器未返回数据')
      courses.value = []
      return
    }
    
    // 后端返回的数据结构是 { code: 200, data: {...}, message: '操作成功' }
    // response可能是axios响应对象，也可能是自定义的API响应对象
    const apiResponse = response.data && response.data.code !== undefined ? response.data : response
    
    if (apiResponse.code === 200) {
      // 处理不同的响应结构
      const responseData = apiResponse.data
      console.log('📊 响应数据结构:', responseData)
      
      if (!responseData) {
        console.warn('响应数据为空')
        courses.value = []
        return
      }
      
      // 检查是否有content字段(分页数据)
      if (responseData.content && Array.isArray(responseData.content)) {
        courses.value = responseData.content
        console.log('📚 从content字段获取到', courses.value.length, '个课程')
      } 
      // 检查是否有records字段(分页数据)
      else if (responseData.records && Array.isArray(responseData.records)) {
        courses.value = responseData.records
        console.log('📚 从records字段获取到', courses.value.length, '个课程')
      }
      // 检查是否有list字段
      else if (responseData.list && Array.isArray(responseData.list)) {
        courses.value = responseData.list
        console.log('📚 从list字段获取到', courses.value.length, '个课程')
      }
      // 检查responseData本身是否为数组
      else if (Array.isArray(responseData)) {
        courses.value = responseData
        console.log('📚 直接从data字段获取到', courses.value.length, '个课程')
      }
      else {
        console.warn('未找到有效的课程数据结构:', responseData)
        courses.value = []
      }
    } else {
      console.error('API返回错误:', apiResponse.message || '未知错误')
      ElMessage.error(`获取课程列表失败：${apiResponse.message || '未知错误'}`)
      courses.value = []
    }
    
    // 如果获取到的是空数组，显示提示信息
    if (courses.value.length === 0) {
      console.warn('未获取到任何课程数据')
      ElMessage.warning('未找到课程数据，请先创建课程')
    }
  } catch (error: any) {
    console.error('加载课程失败:', error)
    ElMessage.error('加载课程列表失败：' + (error.message || '网络错误'))
    courses.value = []
  }
}

// 加载章节列表
const loadChapters = async () => {
  if (!formData.courseId) {
    chapters.value = []
    formData.chapterIds = []
    return
  }
  
  chaptersLoading.value = true  // 设置加载状态为true
  try {
    console.log('📚 开始获取课程章节，课程ID:', formData.courseId)
    const response = await chapterAPI.getChaptersByCourseId(formData.courseId)
    console.log('📊 章节响应:', response)
    
    if (response && response.data) {
      if (Array.isArray(response.data)) {
        chapters.value = response.data
      } else if (response.data.code === 200 && response.data.data) {
        // 处理不同的嵌套结构
        if (Array.isArray(response.data.data)) {
          chapters.value = response.data.data
        } else if (response.data.data.content && Array.isArray(response.data.data.content)) {
          chapters.value = response.data.data.content
        } else if (response.data.data.list && Array.isArray(response.data.data.list)) {
          chapters.value = response.data.data.list
        } else {
          console.warn('未找到有效的章节数据结构')
          chapters.value = []
        }
      } else {
        console.warn('章节数据结构不正确')
        chapters.value = []
      }
    } else {
      console.warn('未获取到章节数据')
      chapters.value = []
    }
    
    console.log('📚 获取到', chapters.value.length, '个章节')
    
    // 如果章节列表为空，显示提示
    if (chapters.value.length === 0) {
      ElMessage.warning('该课程没有章节数据，请先添加章节')
    }
  } catch (error: any) {
    console.error('加载章节失败:', error)
    ElMessage.error('加载章节列表失败：' + (error.message || '网络错误'))
    chapters.value = []
  } finally {
    chaptersLoading.value = false  // 设置加载状态为false
  }
}

// 加载我的知识图谱
const loadMyGraphs = async () => {
  try {
    console.log('📊 开始获取我的知识图谱...')
    const response = await request.get('/api/teacher/knowledge-graph/my')
    console.log('📊 知识图谱响应:', response)
    
    // 检查响应是否有效
    if (!response) {
      console.error('API响应为空')
      ElMessage.error('获取知识图谱列表失败：服务器未返回数据')
      myGraphs.value = []
      return
    }
    
    // 标准化API响应格式
    const apiResponse = response.data && response.data.code !== undefined ? response.data : response
    
    if (apiResponse.code === 200) {
      const responseData = apiResponse.data
      console.log('📊 知识图谱数据:', responseData)
      
      // 如果是null或undefined，使用空数组
      if (responseData === null || responseData === undefined) {
        console.warn('知识图谱数据为null或undefined，使用空数组')
        myGraphs.value = []
        return
      }
      
      // 如果是空数组，直接使用
      if (Array.isArray(responseData)) {
        myGraphs.value = responseData
        console.log('📊 获取到', myGraphs.value.length, '个知识图谱')
      } 
      // 检查是否有content字段(分页数据)
      else if (responseData.content && Array.isArray(responseData.content)) {
        myGraphs.value = responseData.content
        console.log('📊 从content字段获取到', myGraphs.value.length, '个知识图谱')
      }
      // 检查是否有records字段(分页数据)
      else if (responseData.records && Array.isArray(responseData.records)) {
        myGraphs.value = responseData.records
        console.log('📊 从records字段获取到', myGraphs.value.length, '个知识图谱')
      }
      // 检查是否有list字段
      else if (responseData.list && Array.isArray(responseData.list)) {
        myGraphs.value = responseData.list
        console.log('📊 从list字段获取到', myGraphs.value.length, '个知识图谱')
      }
      else {
        console.warn('未找到有效的知识图谱数据结构:', responseData)
        myGraphs.value = []
      }
    } else {
      console.error('API返回错误:', apiResponse.message || '未知错误')
      ElMessage.error(`获取知识图谱列表失败：${apiResponse.message || '未知错误'}`)
      myGraphs.value = []
    }
    
    // 如果列表为空，显示提示信息
    if (myGraphs.value.length === 0) {
      console.warn('未获取到任何知识图谱数据')
      // 不显示toast，避免太多提示信息
    }
  } catch (error: any) {
    console.error('加载知识图谱失败:', error)
    ElMessage.error('加载知识图谱列表失败：' + (error.message || '网络错误'))
    myGraphs.value = []
  }
}

// 生成知识图谱
const generateGraph = async () => {
  if (!formRef.value) return
  
  await formRef.value.validate(async (valid) => {
    if (!valid) return
    
    try {
      generating.value = true
      generationStatus.value = 'processing'
      statusMessage.value = '正在生成知识图谱，请稍候...'
      statusType.value = 'info'
      
      const response = await teacherKnowledgeGraphAPI.generate(formData)
      const result = response?.data
      
      if (result) {
        taskId.value = result.taskId || ''
        
        if (result.status === 'completed' && result.graphData) {
          generationStatus.value = 'completed'
          statusMessage.value = '知识图谱生成成功'
          statusType.value = 'success'
          graphData.value = result.graphData
          
          // 渲染图谱
          nextTick(() => {
            if (result.graphData) {
              renderGraph(result.graphData)
            }
          })
        } else if (result.status === 'failed') {
          generationStatus.value = 'failed'
          statusMessage.value = `生成失败: ${result.errorMessage || '未知错误'}`
          statusType.value = 'error'
        } else {
          // 处理中状态
          generationStatus.value = 'processing'
          statusMessage.value = '知识图谱生成中，请稍候...'
          statusType.value = 'warning'
          
          // 定时检查任务状态
          if (taskId.value) {
            setTimeout(() => {
              checkTaskStatus()
            }, 5000)
          }
        }
      } else {
        console.error('生成知识图谱失败:', response)
        generationStatus.value = 'failed'
        statusMessage.value = '生成失败: 服务器响应错误'
        statusType.value = 'error'
      }
    } catch (error: any) {
      console.error('生成知识图谱失败:', error)
      generationStatus.value = 'failed'
      statusMessage.value = `生成失败: ${error.message || '未知错误'}`
      statusType.value = 'error'
    } finally {
      generating.value = false
    }
  })
}

// 检查任务状态
const checkTaskStatus = async () => {
  if (!taskId.value) return
  
  try {
    const response = await teacherKnowledgeGraphAPI.getTaskStatus(taskId.value)
    const result = response?.data
    
    if (result) {
      if (result.status === 'completed' && result.graphData) {
        generationStatus.value = 'completed'
        statusMessage.value = '知识图谱生成成功'
        statusType.value = 'success'
        graphData.value = result.graphData
        
        // 渲染图谱
        nextTick(() => {
          if (result.graphData) {
            renderGraph(result.graphData)
          }
        })
      } else if (result.status === 'failed') {
        generationStatus.value = 'failed'
        statusMessage.value = `生成失败: ${result.errorMessage || '未知错误'}`
        statusType.value = 'error'
      } else if (result.status === 'processing' || result.status === 'pending') {
        // 继续定时检查
        setTimeout(() => {
          checkTaskStatus()
        }, 5000)
      }
    } else {
      console.error('检查任务状态失败:', response)
    }
  } catch (error) {
    console.error('检查任务状态失败:', error)
  }
}

// 渲染图谱
const renderGraph = (data: KnowledgeGraphData) => {
  if (!graphContainer.value) return
  
  // 销毁旧图表
  if (graphChart.value) {
    graphChart.value.dispose()
  }
  
  // 初始化图表
  graphChart.value = echarts.init(graphContainer.value)
  
  // 准备数据
  const nodes = data.nodes.map(node => ({
    id: node.id,
    name: node.name,
    symbolSize: node.style?.size || getNodeSize(node.level || 1),
    value: node.level || 1,
    category: node.type || 'concept',
    itemStyle: {
      color: node.style?.color || getNodeColor(node.type)
    },
    label: {
      show: true,
      fontSize: node.style?.fontSize || 12
    },
    // 原始数据，用于点击时显示详情
    rawData: node
  }))
  
  const edges = data.edges.map(edge => ({
    source: edge.source,
    target: edge.target,
    value: edge.type,
    lineStyle: {
      color: edge.style?.color || '#999',
      width: edge.style?.width || 1,
      type: edge.style?.lineType || 'solid',
      curveness: 0.2
    },
    label: {
      show: true,
      formatter: edge.type,
      fontSize: 10
    },
    // 原始数据
    rawData: edge
  }))
  
  // 设置图表选项
  const option = {
    title: {
      text: data.title || '知识图谱',
      subtext: data.description || '',
      top: 'top',
      left: 'center'
    },
    tooltip: {
      trigger: 'item',
      formatter: (params: any) => {
        if (params.dataType === 'node') {
          const node = params.data.rawData
          return `
            <div>
              <strong>${node.name}</strong><br/>
              类型: ${nodeTypeMap[node.type] || node.type}<br/>
              重要性: ${node.level || 1}<br/>
              ${node.description ? `描述: ${node.description}` : ''}
            </div>
          `
        } else {
          const edge = params.data.rawData
          return `
            <div>
              <strong>${edge.type}</strong><br/>
              ${edge.description ? `描述: ${edge.description}` : ''}
            </div>
          `
        }
      }
    },
    legend: {
      data: ['concept', 'skill', 'topic', 'chapter'].map(type => ({
        name: type,
        icon: 'circle'
      })),
      formatter: (name: string) => nodeTypeMap[name] || name,
      selectedMode: 'multiple',
      bottom: 20
    },
    animationDuration: 1500,
    animationEasingUpdate: 'quinticInOut' as const,
    series: [
      {
        type: 'graph',
        layout: 'force',
        data: nodes,
        links: edges,
        categories: ['concept', 'skill', 'topic', 'chapter'].map(type => ({
          name: type
        })),
        roam: true,
        label: {
          position: 'right'
        },
        lineStyle: {
          color: 'source',
          curveness: 0.3
        },
        emphasis: {
          focus: 'adjacency',
          lineStyle: {
            width: 5
          }
        },
        force: {
          repulsion: 100,
          edgeLength: 100
        }
      }
    ]
  }
  
  // 设置图表
  graphChart.value.setOption(option)
  
  // 监听点击事件
  graphChart.value.on('click', (params: any) => {
    if (params.dataType === 'node') {
      selectedNode.value = params.data.rawData
    }
  })
  
  // 监听窗口大小变化
  window.addEventListener('resize', () => {
    graphChart.value?.resize()
  })
}

// 获取节点颜色
const getNodeColor = (type: string | undefined) => {
  switch (type) {
    case 'concept': return '#5470c6'
    case 'skill': return '#91cc75'
    case 'topic': return '#fac858'
    case 'chapter': return '#ee6666'
    default: return '#73c0de'
  }
}

// 获取节点大小
const getNodeSize = (level: number) => {
  switch (level) {
    case 1: return 30
    case 2: return 25
    case 3: return 20
    case 4: return 15
    case 5: return 10
    default: return 30
  }
}

// 保存图谱
const saveGraph = async () => {
  if (!graphData.value) return
  
  try {
    // 构建保存请求
    const saveData = {
      courseId: formData.courseId,
      title: graphData.value.title || `${formData.graphType}知识图谱`,
      description: graphData.value.description || `${formData.courseId}课程的知识图谱`,
      graphType: formData.graphType,
      graphData: JSON.stringify(graphData.value)
    }
    
    // 调用保存API
    await ElMessage.success('知识图谱保存成功')
    
    // 重新加载图谱列表
    await loadMyGraphs()
  } catch (error) {
    console.error('保存知识图谱失败:', error)
    ElMessage.error('保存知识图谱失败')
  }
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
    link.download = `${graphData.value?.title || '知识图谱'}.png`
    link.href = dataURL
    link.click()
  } catch (error) {
    console.error('导出图谱失败:', error)
    ElMessage.error('导出图谱失败')
  }
}

// 查看图谱
const viewGraph = async (graph: KnowledgeGraph) => {
  try {
    currentGraph.value = graph
    
    const response = await teacherKnowledgeGraphAPI.getGraphDetail(graph.id)
    const graphDetail = response?.data
    
    if (graphDetail) {
      graphData.value = graphDetail
      
      // 渲染图谱
      nextTick(() => {
        renderGraph(graphDetail)
      })
    }
  } catch (error) {
    console.error('获取图谱详情失败:', error)
    ElMessage.error('获取图谱详情失败')
  }
}

// 切换发布状态
const togglePublish = async (graph: KnowledgeGraph) => {
  try {
    if (graph.isPublic) {
      await teacherKnowledgeGraphAPI.unpublishGraph(graph.id)
      ElMessage.success('已取消发布')
    } else {
      await teacherKnowledgeGraphAPI.publishGraph(graph.id)
      ElMessage.success('已发布')
    }
    
    // 重新加载图谱列表
    await loadMyGraphs()
  } catch (error) {
    console.error('切换发布状态失败:', error)
    ElMessage.error('操作失败')
  }
}

// 删除图谱
const deleteGraph = async (graph: KnowledgeGraph) => {
  try {
    await ElMessageBox.confirm(
      `确定要删除知识图谱 "${graph.title}" 吗？`,
      '删除确认',
      {
        confirmButtonText: '确定',
        cancelButtonText: '取消',
        type: 'warning'
      }
    )
    
    await teacherKnowledgeGraphAPI.deleteGraph(graph.id)
    ElMessage.success('删除成功')
    
    // 重新加载图谱列表
    await loadMyGraphs()
  } catch (error: any) {
    if (error !== 'cancel') {
      console.error('删除图谱失败:', error)
      ElMessage.error('删除失败')
    }
  }
}

// 获取状态标签类型
const getStatusTagType = (status: string) => {
  switch (status) {
    case 'published': return 'success'
    case 'draft': return 'info'
    case 'archived': return 'danger'
    default: return ''
  }
}

// 重置表单
const resetForm = () => {
  if (formRef.value) {
    formRef.value.resetFields()
  }
  
  formData.courseId = 0
  formData.chapterIds = []
  formData.graphType = 'comprehensive'
  formData.depth = 3
  formData.includePrerequisites = true
  formData.includeApplications = true
  formData.additionalRequirements = ''
  
  generationStatus.value = ''
  statusMessage.value = ''
  graphData.value = null
  selectedNode.value = null
  
  // 销毁图表
  if (graphChart.value) {
    graphChart.value.dispose()
    graphChart.value = null
  }
}

// 章节项目类型
interface ChapterItem {
  id: number
  title: string
  description?: string
  type?: string
  name?: string
}
</script>

<style scoped>
.knowledge-graph-generator {
  padding: 20px;
}

.generator-form {
  max-width: 800px;
  margin-bottom: 30px;
}

.generation-status {
  margin-bottom: 30px;
}

.progress-indicator {
  margin-top: 15px;
  padding: 15px;
  background-color: #f5f7fa;
  border-radius: 4px;
}

.graph-preview {
  margin-top: 30px;
  margin-bottom: 30px;
}

.graph-container {
  width: 100%;
  height: 600px;
  border: 1px solid #dcdfe6;
  border-radius: 4px;
  margin-bottom: 20px;
}

.graph-actions {
  margin-bottom: 20px;
}

.node-details {
  padding: 15px;
  background-color: #f5f7fa;
  border-radius: 4px;
  margin-bottom: 20px;
}

.my-graphs {
  margin-top: 40px;
}
</style> 