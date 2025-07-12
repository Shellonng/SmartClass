import request from '@/utils/request'
import type { ApiResponse } from '@/api/course'

// 知识图谱相关接口类型定义
export interface KnowledgeGraphNode {
  id: string
  name: string
  type: 'concept' | 'skill' | 'topic' | 'chapter'
  level: number
  description?: string
  chapterId?: number
  sectionId?: number
  style?: {
    color?: string
    size?: number
    shape?: string
    fontSize?: number
    highlighted?: boolean
  }
  position?: {
    x?: number
    y?: number
    fixed?: boolean
  }
  properties?: Record<string, any>
}

export interface KnowledgeGraphEdge {
  id: string
  source: string
  target: string
  type: string
  description?: string
  weight?: number
  style?: {
    color?: string
    width?: number
    lineType?: string
    showArrow?: boolean
  }
  properties?: Record<string, any>
}

export interface KnowledgeGraphData {
  id?: number
  title: string
  description?: string
  nodes: KnowledgeGraphNode[]
  edges: KnowledgeGraphEdge[]
  metadata?: Record<string, any>
}

export interface GenerationRequest {
  courseId: number
  chapterIds: number[]
  graphType?: 'concept' | 'skill' | 'comprehensive'
  depth?: number
  includePrerequisites?: boolean
  includeApplications?: boolean
  additionalRequirements?: string
}

export interface GenerationResponse {
  status: 'pending' | 'processing' | 'completed' | 'failed' | 'error'
  taskId?: string
  graphData?: KnowledgeGraphData
  errorMessage?: string
  suggestions?: string
}

export interface KnowledgeGraph {
  id: number
  courseId: number
  title: string
  description?: string
  graphType: string
  creatorId: number
  status: string
  version: number
  isPublic: boolean
  viewCount: number
  createTime: string
  updateTime: string
  courseName?: string
  creatorName?: string
}

export interface QueryRequest {
  courseId?: number
  graphType?: string
  keyword?: string
  includeStyle?: boolean
}

export interface AnalysisRequest {
  graphId: number
  studentId?: number
  analysisType: 'mastery' | 'difficulty' | 'path'
}

export interface AnalysisResponse {
  analysis: Record<string, any>
  learningPath: string[]
  keyPoints: string[]
  difficultPoints: string[]
}

// 教师端API
export const teacherKnowledgeGraphAPI = {
  /**
   * 生成知识图谱
   */
  generate(data: GenerationRequest): Promise<ApiResponse<GenerationResponse>> {
    return request.post('/api/teacher/knowledge-graph/create', data)
  },

  /**
   * 获取课程的知识图谱列表
   */
  getCourseGraphs(courseId: number): Promise<ApiResponse<KnowledgeGraph[]>> {
    return request.get(`/api/teacher/knowledge-graph/course/${courseId}/graphs`)
  },

  /**
   * 获取知识图谱详情
   */
  getGraphDetail(graphId: number): Promise<ApiResponse<KnowledgeGraphData>> {
    return request.get(`/api/teacher/knowledge-graph/detail/${graphId}`)
  },

  /**
   * 更新知识图谱
   */
  updateGraph(graphId: number, data: KnowledgeGraphData): Promise<ApiResponse<void>> {
    return request.put(`/api/teacher/knowledge-graph/update/${graphId}`, data)
  },

  /**
   * 删除知识图谱
   */
  deleteGraph(graphId: number): Promise<ApiResponse<void>> {
    return request.delete(`/api/teacher/knowledge-graph/delete/${graphId}`)
  },

  /**
   * 发布知识图谱
   */
  publishGraph(graphId: number): Promise<ApiResponse<void>> {
    return request.put(`/api/teacher/knowledge-graph/${graphId}/publish`)
  },

  /**
   * 取消发布知识图谱
   */
  unpublishGraph(graphId: number): Promise<ApiResponse<void>> {
    return request.put(`/api/teacher/knowledge-graph/${graphId}/unpublish`)
  },

  /**
   * 获取任务状态
   */
  getTaskStatus(taskId: string): Promise<ApiResponse<GenerationResponse>> {
    return request.get(`/api/teacher/knowledge-graph/task-status/${taskId}`)
  },

  /**
   * 分页查询知识图谱
   */
  getGraphsPage(page: number, size: number, query: QueryRequest): Promise<ApiResponse<any>> {
    return request.post(`/api/teacher/knowledge-graph/page?page=${page}&size=${size}`, query)
  },

  /**
   * 搜索知识图谱
   */
  searchGraphs(keyword: string): Promise<ApiResponse<KnowledgeGraph[]>> {
    return request.get(`/api/teacher/knowledge-graph/search?keyword=${encodeURIComponent(keyword)}`)
  },

  /**
   * 获取我创建的知识图谱
   */
  getMyGraphs(): Promise<ApiResponse<KnowledgeGraph[]>> {
    return request.get('/api/teacher/knowledge-graph/my')
  },

  /**
   * 知识点分析
   */
  analyzeGraph(data: AnalysisRequest): Promise<ApiResponse<AnalysisResponse>> {
    return request.post('/api/teacher/knowledge-graph/analyze', data)
  }
}

// 学生端API
export const studentKnowledgeGraphAPI = {
  /**
   * 获取课程的知识图谱列表
   */
  getCourseGraphs(courseId: number): Promise<ApiResponse<KnowledgeGraph[]>> {
    return request.get(`/api/student/knowledge-graph/course/${courseId}`)
  },

  /**
   * 获取知识图谱详情
   */
  getGraphDetail(graphId: number): Promise<ApiResponse<KnowledgeGraphData>> {
    return request.get(`/api/student/knowledge-graph/${graphId}`)
  },

  /**
   * 获取公开的知识图谱
   */
  getPublicGraphs(limit?: number): Promise<ApiResponse<KnowledgeGraph[]>> {
    return request.get(`/api/student/knowledge-graph/public${limit ? `?limit=${limit}` : ''}`)
  },

  /**
   * 分页查询公开的知识图谱
   */
  getPublicGraphsPage(page: number, size: number, query: QueryRequest): Promise<ApiResponse<any>> {
    return request.post(`/api/student/knowledge-graph/page?page=${page}&size=${size}`, query)
  },

  /**
   * 知识点学习路径分析
   */
  getLearningPath(graphId: number): Promise<ApiResponse<string[]>> {
    return request.get(`/api/student/knowledge-graph/${graphId}/learning-path`)
  }
}

// 通用工具函数
export const knowledgeGraphUtils = {
  /**
   * 获取节点类型标签
   */
  getNodeTypeLabel(type: string): string {
    const labels: Record<string, string> = {
      concept: '概念',
      skill: '技能',
      topic: '主题',
      chapter: '章节'
    }
    return labels[type] || type
  },

  /**
   * 获取图谱类型标签
   */
  getGraphTypeLabel(type: string): string {
    const labels: Record<string, string> = {
      concept: '概念图谱',
      skill: '技能图谱',
      comprehensive: '综合图谱'
    }
    return labels[type] || type
  },

  /**
   * 获取节点默认颜色
   */
  getNodeDefaultColor(type: string): string {
    const colors: Record<string, string> = {
      concept: '#3498db',
      skill: '#e74c3c',
      topic: '#2ecc71',
      chapter: '#f39c12'
    }
    return colors[type] || '#95a5a6'
  },

  /**
   * 验证图谱数据
   */
  validateGraphData(data: KnowledgeGraphData): boolean {
    if (!data.title || !data.nodes || !Array.isArray(data.nodes)) {
      return false
    }
    
    // 验证节点
    for (const node of data.nodes) {
      if (!node.id || !node.name || !node.type) {
        return false
      }
    }
    
    // 验证边
    if (data.edges) {
      for (const edge of data.edges) {
        if (!edge.id || !edge.source || !edge.target) {
          return false
        }
        
        // 验证边的节点存在
        const sourceExists = data.nodes.some(node => node.id === edge.source)
        const targetExists = data.nodes.some(node => node.id === edge.target)
        if (!sourceExists || !targetExists) {
          return false
        }
      }
    }
    
    return true
  },

  /**
   * 计算图谱统计信息
   */
  calculateGraphStats(data: KnowledgeGraphData): {
    nodeCount: number
    edgeCount: number
    typeDistribution: Record<string, number>
    complexity: 'low' | 'medium' | 'high'
  } {
    const nodeCount = data.nodes.length
    const edgeCount = data.edges?.length || 0
    
    // 统计节点类型分布
    const typeDistribution: Record<string, number> = {}
    data.nodes.forEach(node => {
      typeDistribution[node.type] = (typeDistribution[node.type] || 0) + 1
    })
    
    // 计算复杂度
    let complexity: 'low' | 'medium' | 'high' = 'low'
    if (nodeCount > 50 || edgeCount > 100) {
      complexity = 'high'
    } else if (nodeCount > 20 || edgeCount > 40) {
      complexity = 'medium'
    }
    
    return {
      nodeCount,
      edgeCount,
      typeDistribution,
      complexity
    }
  },

  /**
   * 生成图谱预览数据
   */
  generatePreviewData(data: KnowledgeGraphData, maxNodes: number = 10): KnowledgeGraphData {
    if (data.nodes.length <= maxNodes) {
      return data
    }
    
    // 选择重要节点（根据级别和连接数）
    const nodeImportance = data.nodes.map(node => {
      const connectionCount = (data.edges || []).filter(edge => 
        edge.source === node.id || edge.target === node.id
      ).length
      return {
        node,
        importance: (node.level || 1) * 10 + connectionCount
      }
    })
    
    nodeImportance.sort((a, b) => b.importance - a.importance)
    const selectedNodes = nodeImportance.slice(0, maxNodes).map(item => item.node)
    const selectedNodeIds = new Set(selectedNodes.map(node => node.id))
    
    // 筛选相关的边
    const filteredEdges = (data.edges || []).filter(edge =>
      selectedNodeIds.has(edge.source) && selectedNodeIds.has(edge.target)
    )
    
    return {
      ...data,
      nodes: selectedNodes,
      edges: filteredEdges
    }
  }
} 