<template>
  <div class="learning-pathway">
    <div class="page-header">
      <h1>🛣️ 个性化学习路径</h1>
      <p class="description">智能分析您的学习情况，为您规划最佳学习路径</p>
    </div>

    <div class="pathway-container">
      <!-- 知识掌握度分析 -->
      <a-row :gutter="24" class="analysis-row">
        <a-col :span="16">
          <a-card title="知识掌握度分析" class="analysis-card">
            <template #extra>
              <a-button type="primary" @click="refreshAnalysis" :loading="refreshingAnalysis">
                <ReloadOutlined />
                更新分析
              </a-button>
            </template>
            
            <div class="chart-container">
              <div ref="radarChartRef" class="radar-chart"></div>
              <div class="legend">
                <div class="legend-item">
                  <span class="legend-color current"></span>
                  <span>当前水平</span>
                </div>
                <div class="legend-item">
                  <span class="legend-color target"></span>
                  <span>目标水平</span>
                </div>
              </div>
            </div>
          </a-card>
        </a-col>
        
        <a-col :span="8">
          <a-card title="学习状态概览" class="status-card">
            <div class="status-summary">
              <div class="summary-item">
                <div class="summary-label">学习进度</div>
                <div class="summary-value">{{ studyProgress }}%</div>
                <a-progress :percent="studyProgress" size="small" status="active" />
              </div>
              
              <div class="summary-item">
                <div class="summary-label">薄弱知识点</div>
                <div class="summary-value">{{ weakKnowledgePoints.length }}</div>
                <div class="knowledge-tags">
                  <a-tag 
                    v-for="(point, index) in weakKnowledgePoints.slice(0, 2)" 
                    :key="index" 
                    color="orange"
                  >
                    {{ point.name }}
                  </a-tag>
                  <a-tag v-if="weakKnowledgePoints.length > 2" color="orange">
                    +{{ weakKnowledgePoints.length - 2 }}
                  </a-tag>
                </div>
              </div>
              
              <div class="summary-item">
                <div class="summary-label">擅长知识点</div>
                <div class="summary-value">{{ strongKnowledgePoints.length }}</div>
                <div class="knowledge-tags">
                  <a-tag 
                    v-for="(point, index) in strongKnowledgePoints.slice(0, 2)" 
                    :key="index" 
                    color="green"
                  >
                    {{ point.name }}
                  </a-tag>
                  <a-tag v-if="strongKnowledgePoints.length > 2" color="green">
                    +{{ strongKnowledgePoints.length - 2 }}
                  </a-tag>
                </div>
              </div>
            </div>
          </a-card>
        </a-col>
      </a-row>

      <!-- 个性化学习路径 -->
      <a-card title="个性化学习路径" class="pathway-card">
        <template #extra>
          <a-space>
            <a-select 
              v-model:value="selectedSubject" 
              style="width: 120px" 
              @change="handleSubjectChange"
              placeholder="选择课程"
            >
              <a-select-option v-for="subject in subjects" :key="subject.id" :value="subject.id">
                {{ subject.name }}
              </a-select-option>
            </a-select>
            
            <a-button @click="generatePathway" type="primary" :loading="generatingPathway">
              生成路径
            </a-button>
          </a-space>
        </template>
        
        <div class="pathway-wrapper">
          <div class="pathway-steps">
            <!-- 学习路径步骤 -->
            <div v-for="(step, index) in learningPathway" :key="index" class="pathway-step" :class="{ 'active': index === currentStep }">
              <div class="step-header" @click="toggleStep(index)">
                <div class="step-number">{{ index + 1 }}</div>
                <div class="step-info">
                  <h3>{{ step.title }}</h3>
                  <div class="step-meta">
                    <a-tag :color="getDifficultyColor(step.difficulty)">{{ step.difficulty }}</a-tag>
                    <span class="step-duration">预计学习时间: {{ step.duration }}</span>
                  </div>
                </div>
                <div class="step-status">
                  <a-tag :color="getStatusColor(step.status)">{{ getStatusText(step.status) }}</a-tag>
                </div>
                <div class="step-expand">
                  <DownOutlined v-if="expandedSteps[index]" />
                  <RightOutlined v-else />
                </div>
              </div>
              
              <div v-show="expandedSteps[index]" class="step-content">
                <div class="step-description">
                  <p>{{ step.description }}</p>
                </div>
                
                <div class="knowledge-points">
                  <h4>知识点:</h4>
                  <div class="knowledge-list">
                    <a-tag v-for="(point, pidx) in step.knowledgePoints" :key="pidx">
                      {{ point }}
                    </a-tag>
                  </div>
                </div>
                
                <div class="resources-list">
                  <h4>推荐学习资源:</h4>
                  <a-list size="small" :data-source="step.resources" :bordered="false">
                    <template #renderItem="{ item }">
                      <a-list-item>
                        <a-list-item-meta>
                          <template #title>
                            <a :href="item.url" target="_blank">{{ item.title }}</a>
                          </template>
                          <template #description>
                            <span>{{ item.type }} · {{ item.duration }}</span>
                          </template>
                          <template #avatar>
                            <a-avatar :style="{ backgroundColor: getResourceColor(item.type) }">
                              {{ getResourceIcon(item.type) }}
                            </a-avatar>
                          </template>
                        </a-list-item-meta>
                        <template #actions>
                          <a-button size="small" type="link" @click="startLearning(item)">
                            开始学习
                          </a-button>
                        </template>
                      </a-list-item>
                    </template>
                  </a-list>
                </div>
                
                <div class="practice-section">
                  <h4>巩固练习:</h4>
                  <a-button type="primary" @click="startPractice(step)">
                    开始练习
                  </a-button>
                </div>
                
                <div class="step-actions">
                  <a-space>
                    <a-button @click="markStepCompleted(index)" :disabled="step.status === 'completed'">
                      标记为已完成
                    </a-button>
                    <a-button type="primary" @click="goToNextStep(index)" :disabled="index === learningPathway.length - 1">
                      下一步
                    </a-button>
                  </a-space>
                </div>
              </div>
            </div>
          </div>
          
          <div class="pathway-visualization">
            <div ref="pathwayChartRef" class="pathway-chart"></div>
          </div>
        </div>
      </a-card>
      
      <!-- 学习建议 -->
      <a-card title="学习建议" class="suggestions-card">
        <div class="suggestions-list">
          <div v-for="(suggestion, index) in learningRecommendations" :key="index" class="suggestion-item">
            <div class="suggestion-icon">{{ suggestion.icon }}</div>
            <div class="suggestion-content">
              <h4>{{ suggestion.title }}</h4>
              <p>{{ suggestion.description }}</p>
            </div>
          </div>
        </div>
      </a-card>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, reactive, onMounted, nextTick } from 'vue'
import { message } from 'ant-design-vue'
import * as echarts from 'echarts'
import { 
  ReloadOutlined,
  DownOutlined,
  RightOutlined
} from '@ant-design/icons-vue'
import type { EChartsOption } from 'echarts'

// 响应式数据
const radarChartRef = ref<HTMLElement | null>(null)
const pathwayChartRef = ref<HTMLElement | null>(null)
const radarChart = ref<echarts.ECharts | null>(null)
const pathwayChart = ref<echarts.ECharts | null>(null)

const refreshingAnalysis = ref(false)
const generatingPathway = ref(false)
const selectedSubject = ref<number | null>(null)
const currentStep = ref(0)
const studyProgress = ref(65)
const expandedSteps = ref<Record<number, boolean>>({})

// 模拟数据 - 学科列表
const subjects = ref([
  { id: 1, name: '高等数学' },
  { id: 2, name: '线性代数' },
  { id: 3, name: '概率论' },
  { id: 4, name: '数据结构' }
])

// 模拟数据 - 知识点掌握情况
const knowledgePointsData = ref([
  { name: '函数极限', score: 85, target: 90 },
  { name: '导数计算', score: 65, target: 90 },
  { name: '微分方程', score: 40, target: 85 },
  { name: '多元积分', score: 75, target: 85 },
  { name: '级数收敛', score: 55, target: 80 }
])

// 薄弱知识点
const weakKnowledgePoints = ref([
  { name: '微分方程', score: 40 },
  { name: '级数收敛', score: 55 },
  { name: '导数计算', score: 65 }
])

// 擅长知识点
const strongKnowledgePoints = ref([
  { name: '函数极限', score: 85 },
  { name: '多元积分', score: 75 }
])

// 学习路径
const learningPathway = ref([
  {
    id: 1,
    title: '微分方程基础概念',
    difficulty: '基础',
    duration: '2小时',
    status: 'in_progress',
    description: '学习微分方程的基本概念、分类及解法思路，掌握一阶常微分方程的求解方法。',
    knowledgePoints: ['微分方程定义', '一阶微分方程', '变量分离法'],
    resources: [
      { 
        title: '微分方程入门', 
        type: '视频',
        url: '/student/courses/1/videos/101',
        duration: '45分钟' 
      },
      { 
        title: '一阶微分方程求解指南', 
        type: '文档',
        url: '/student/resources/202',
        duration: '30分钟' 
      }
    ]
  },
  {
    id: 2,
    title: '二阶线性微分方程',
    difficulty: '中等',
    duration: '3小时',
    status: 'pending',
    description: '掌握二阶线性微分方程的结构特点和求解方法，学会求解常系数二阶线性微分方程。',
    knowledgePoints: ['二阶线性微分方程', '常系数方程', '特征方程法'],
    resources: [
      { 
        title: '二阶线性微分方程详解', 
        type: '视频',
        url: '/student/courses/1/videos/102',
        duration: '60分钟' 
      },
      { 
        title: '常系数微分方程习题集', 
        type: '习题',
        url: '/student/resources/203',
        duration: '45分钟' 
      }
    ]
  },
  {
    id: 3,
    title: '微分方程应用',
    difficulty: '高级',
    duration: '4小时',
    status: 'pending',
    description: '学习微分方程在物理、工程等领域的应用，掌握建立微分方程模型的方法。',
    knowledgePoints: ['微分方程建模', '物理应用', '工程应用'],
    resources: [
      { 
        title: '微分方程在物理中的应用', 
        type: '视频',
        url: '/student/courses/1/videos/103',
        duration: '50分钟' 
      },
      { 
        title: '微分方程应用案例分析', 
        type: '案例',
        url: '/student/resources/204',
        duration: '40分钟' 
      }
    ]
  }
])

// 学习建议
const learningRecommendations = ref([
  {
    icon: '📚',
    title: '强化微分方程基础',
    description: '根据您的学习分析，微分方程是您需要重点提升的知识点。建议先巩固基础概念再进行进阶学习。'
  },
  {
    icon: '⏱️',
    title: '制定合理学习计划',
    description: '每天建议学习1-2个知识点，学习时间控制在2小时以内，注重质量而非数量。'
  },
  {
    icon: '✏️',
    title: '多做练习题',
    description: '针对薄弱环节，建议多做相关练习题，特别是变量分离法和二阶常系数微分方程的计算。'
  }
])

// 方法
const initRadarChart = () => {
  if (radarChartRef.value) {
    radarChart.value = echarts.init(radarChartRef.value)
    
    const indicator = knowledgePointsData.value.map(item => ({
      name: item.name,
      max: 100
    }))
    
    const currentData = knowledgePointsData.value.map(item => item.score)
    const targetData = knowledgePointsData.value.map(item => item.target)
    
    const option: EChartsOption = {
      radar: {
        indicator,
        radius: '65%',
        splitNumber: 5,
        axisName: {
          color: '#333',
          fontSize: 12
        }
      },
      series: [
        {
          type: 'radar',
          data: [
            {
              value: currentData,
              name: '当前水平',
              symbol: 'circle',
              symbolSize: 6,
              lineStyle: {
                color: '#1890ff',
                width: 2
              },
              areaStyle: {
                color: 'rgba(24, 144, 255, 0.2)'
              }
            },
            {
              value: targetData,
              name: '目标水平',
              symbol: 'circle',
              symbolSize: 6,
              lineStyle: {
                color: '#52c41a',
                width: 2,
                type: 'dashed'
              },
              areaStyle: {
                color: 'rgba(82, 196, 26, 0.1)'
              }
            }
          ]
        }
      ]
    }
    
    radarChart.value.setOption(option)
  }
}

const initPathwayChart = () => {
  if (pathwayChartRef.value) {
    pathwayChart.value = echarts.init(pathwayChartRef.value)
    
    const steps = learningPathway.value.map(step => step.title)
    const data = learningPathway.value.map((step, index) => {
      const statusMap: Record<string, number> = {
        'completed': 100,
        'in_progress': 50,
        'pending': 0
      }
      return {
        name: step.title,
        value: statusMap[step.status] || 0
      }
    })
    
    const option: EChartsOption = {
      tooltip: {
        trigger: 'item',
        formatter: '{b}: {c}%'
      },
      series: [
        {
          type: 'funnel',
          left: '10%',
          width: '80%',
          minSize: '0%',
          maxSize: '100%',
          sort: 'none',
          gap: 2,
          label: {
            show: true,
            position: 'inside'
          },
          itemStyle: {
            borderColor: '#fff',
            borderWidth: 1
          },
          emphasis: {
            label: {
              fontSize: 16
            }
          },
          data: data
        }
      ]
    }
    
    pathwayChart.value.setOption(option)
  }
}

const refreshAnalysis = async () => {
  try {
    refreshingAnalysis.value = true
    await new Promise(resolve => setTimeout(resolve, 1000))
    
    // 模拟更新数据
    knowledgePointsData.value = knowledgePointsData.value.map(point => ({
      ...point,
      score: Math.floor(Math.random() * 40) + 60 // 随机生成60-100的分数
    }))
    
    // 重新初始化雷达图
    nextTick(() => {
      initRadarChart()
    })
    
    message.success('分析数据已更新')
  } catch (error) {
    message.error('更新失败')
  } finally {
    refreshingAnalysis.value = false
  }
}

const handleSubjectChange = (value: number) => {
  console.log('选择学科:', value)
  
  // 仅记录所选科目，不立即生成路径
  // 清除当前路径数据，用户需点击"生成路径"按钮才会显示新的路径
  
  // 清空当前路径数据
  learningPathway.value = []
  
  // 清空学习建议
  learningRecommendations.value = []
  
  // 更新雷达图 (保留能力分析功能)
  updateKnowledgeAnalysis(value)
}

// 更新知识点分析数据，但不生成路径
const updateKnowledgeAnalysis = (subjectId: number) => {
  if (subjectId && subjectPathways[subjectId]) {
    // 更新知识点数据
    knowledgePointsData.value = subjectPathways[subjectId].knowledgePoints
    
    // 更新薄弱知识点和擅长知识点
    weakKnowledgePoints.value = knowledgePointsData.value
      .filter(point => point.score < 60)
      .sort((a, b) => a.score - b.score)
      .map(point => ({ name: point.name, score: point.score }))

    strongKnowledgePoints.value = knowledgePointsData.value
      .filter(point => point.score >= 75)
      .sort((a, b) => b.score - a.score)
      .map(point => ({ name: point.name, score: point.score }))
      
    // 更新雷达图
    nextTick(() => {
      initRadarChart()
    })
  }
}

// 科目路径数据
const subjectPathways: Record<number, any> = {
  1: { // 高等数学
    knowledgePoints: [
      { name: '函数极限', score: 85, target: 90 },
      { name: '导数计算', score: 65, target: 90 },
      { name: '微分方程', score: 40, target: 85 },
      { name: '多元积分', score: 75, target: 85 },
      { name: '级数收敛', score: 55, target: 80 }
    ],
    pathway: [
      {
        id: 1,
        title: '微分方程基础概念',
        difficulty: '基础',
        duration: '2小时',
        status: 'in_progress',
        description: '学习微分方程的基本概念、分类及解法思路，掌握一阶常微分方程的求解方法。',
        knowledgePoints: ['微分方程定义', '一阶微分方程', '变量分离法'],
        resources: [
          { title: '微分方程入门', type: '视频', url: '/student/courses/1/videos/101', duration: '45分钟' },
          { title: '一阶微分方程求解指南', type: '文档', url: '/student/resources/202', duration: '30分钟' }
        ]
      },
      {
        id: 2,
        title: '二阶线性微分方程',
        difficulty: '中等',
        duration: '3小时',
        status: 'pending',
        description: '掌握二阶线性微分方程的结构特点和求解方法，学会求解常系数二阶线性微分方程。',
        knowledgePoints: ['二阶线性微分方程', '常系数方程', '特征方程法'],
        resources: [
          { title: '二阶线性微分方程详解', type: '视频', url: '/student/courses/1/videos/102', duration: '60分钟' },
          { title: '常系数微分方程习题集', type: '习题', url: '/student/resources/203', duration: '45分钟' }
        ]
      },
      {
        id: 3,
        title: '微分方程应用',
        difficulty: '高级',
        duration: '4小时',
        status: 'pending',
        description: '学习微分方程在物理、工程等领域的应用，掌握建立微分方程模型的方法。',
        knowledgePoints: ['微分方程建模', '物理应用', '工程应用'],
        resources: [
          { title: '微分方程在物理中的应用', type: '视频', url: '/student/courses/1/videos/103', duration: '50分钟' },
          { title: '微分方程应用案例分析', type: '案例', url: '/student/resources/204', duration: '40分钟' }
        ]
      }
    ],
    recommendations: [
      {
        icon: '📚',
        title: '强化微分方程基础',
        description: '根据您的学习分析，微分方程是您需要重点提升的知识点。建议先巩固基础概念再进行进阶学习。'
      },
      {
        icon: '⏱️',
        title: '制定合理学习计划',
        description: '每天建议学习1-2个知识点，学习时间控制在2小时以内，注重质量而非数量。'
      },
      {
        icon: '✏️',
        title: '多做练习题',
        description: '针对薄弱环节，建议多做相关练习题，特别是变量分离法和二阶常系数微分方程的计算。'
      }
    ]
  },
  2: { // 线性代数
    knowledgePoints: [
      { name: '矩阵运算', score: 75, target: 90 },
      { name: '向量空间', score: 55, target: 85 },
      { name: '特征值', score: 45, target: 80 },
      { name: '线性变换', score: 60, target: 85 },
      { name: '正交分解', score: 70, target: 85 }
    ],
    pathway: [
      {
        id: 1,
        title: '矩阵基本运算',
        difficulty: '基础',
        duration: '2.5小时',
        status: 'in_progress',
        description: '掌握矩阵的加减法、数乘、矩阵乘法等基本运算法则和性质。',
        knowledgePoints: ['矩阵定义', '矩阵运算', '初等变换'],
        resources: [
          { title: '矩阵运算基础', type: '视频', url: '/student/courses/2/videos/201', duration: '40分钟' },
          { title: '矩阵运算习题精讲', type: '习题', url: '/student/resources/222', duration: '35分钟' }
        ]
      },
      {
        id: 2,
        title: '向量空间与线性相关性',
        difficulty: '中等',
        duration: '3小时',
        status: 'pending',
        description: '学习向量空间的概念，理解向量组线性相关与线性无关的判定方法。',
        knowledgePoints: ['向量空间', '线性相关性', '基与维数'],
        resources: [
          { title: '向量空间与线性相关', type: '视频', url: '/student/courses/2/videos/202', duration: '55分钟' },
          { title: '向量空间习题解析', type: '文档', url: '/student/resources/223', duration: '40分钟' }
        ]
      },
      {
        id: 3,
        title: '特征值与特征向量',
        difficulty: '高级',
        duration: '3.5小时',
        status: 'pending',
        description: '掌握特征值和特征向量的概念与计算方法，学习矩阵对角化的条件与步骤。',
        knowledgePoints: ['特征值', '特征向量', '矩阵对角化'],
        resources: [
          { title: '特征值与特征向量详解', type: '视频', url: '/student/courses/2/videos/203', duration: '60分钟' },
          { title: '矩阵对角化应用案例', type: '案例', url: '/student/resources/224', duration: '45分钟' }
        ]
      }
    ],
    recommendations: [
      {
        icon: '🧮',
        title: '加强向量空间概念理解',
        description: '您在向量空间概念方面需要加强，建议重点学习基与维数相关内容，多做习题巩固。'
      },
      {
        icon: '🔢',
        title: '注重特征值计算',
        description: '特征值计算是您的薄弱环节，建议多练习特征多项式求解和特征向量计算。'
      },
      {
        icon: '📐',
        title: '建立应用意识',
        description: '线性代数的实际应用很广泛，建议结合计算机图形学、机器学习等领域学习，加深理解。'
      }
    ]
  },
  3: { // 概率论
    knowledgePoints: [
      { name: '随机事件', score: 80, target: 90 },
      { name: '条件概率', score: 65, target: 85 },
      { name: '随机变量', score: 50, target: 85 },
      { name: '大数定律', score: 40, target: 80 },
      { name: '中心极限定理', score: 35, target: 75 }
    ],
    pathway: [
      {
        id: 1,
        title: '概率论基础',
        difficulty: '基础',
        duration: '2小时',
        status: 'in_progress',
        description: '学习随机事件、概率公理及概率计算的基本方法。',
        knowledgePoints: ['样本空间', '随机事件', '概率计算'],
        resources: [
          { title: '概率论入门', type: '视频', url: '/student/courses/3/videos/301', duration: '45分钟' },
          { title: '概率计算基础题集', type: '习题', url: '/student/resources/302', duration: '30分钟' }
        ]
      },
      {
        id: 2,
        title: '条件概率与全概率公式',
        difficulty: '中等',
        duration: '2.5小时',
        status: 'pending',
        description: '理解条件概率的概念，掌握全概率公式和贝叶斯公式的应用。',
        knowledgePoints: ['条件概率', '全概率公式', '贝叶斯公式'],
        resources: [
          { title: '条件概率详解', type: '视频', url: '/student/courses/3/videos/302', duration: '50分钟' },
          { title: '贝叶斯公式应用案例', type: '案例', url: '/student/resources/303', duration: '35分钟' }
        ]
      },
      {
        id: 3,
        title: '随机变量与分布函数',
        difficulty: '中等',
        duration: '3小时',
        status: 'pending',
        description: '学习离散型和连续型随机变量的概念、分布函数及其性质。',
        knowledgePoints: ['随机变量', '分布函数', '概率密度'],
        resources: [
          { title: '随机变量与分布函数', type: '视频', url: '/student/courses/3/videos/303', duration: '55分钟' },
          { title: '常见分布详解', type: '文档', url: '/student/resources/304', duration: '40分钟' }
        ]
      },
      {
        id: 4,
        title: '大数定律与中心极限定理',
        difficulty: '高级',
        duration: '4小时',
        status: 'pending',
        description: '理解大数定律和中心极限定理的内涵及应用。',
        knowledgePoints: ['大数定律', '中心极限定理', '统计推断'],
        resources: [
          { title: '大数定律讲解', type: '视频', url: '/student/courses/3/videos/304', duration: '60分钟' },
          { title: '中心极限定理及应用', type: '案例', url: '/student/resources/305', duration: '50分钟' }
        ]
      }
    ],
    recommendations: [
      {
        icon: '🎲',
        title: '加强随机变量概念理解',
        description: '您在随机变量及其分布方面需要加强，建议重点学习常见分布的性质及应用场景。'
      },
      {
        icon: '📊',
        title: '多做概率计算练习',
        description: '建议多做条件概率和全概率公式的应用题，提高解题能力。'
      },
      {
        icon: '📈',
        title: '注重统计学应用',
        description: '大数定律和中心极限定理是您的薄弱环节，这些在数据分析中非常重要，建议结合实际案例学习。'
      }
    ]
  },
  4: { // 数据结构
    knowledgePoints: [
      { name: '线性表', score: 75, target: 90 },
      { name: '树结构', score: 60, target: 85 },
      { name: '图算法', score: 45, target: 80 },
      { name: '排序算法', score: 70, target: 90 },
      { name: '查找算法', score: 55, target: 85 }
    ],
    pathway: [
      {
        id: 1,
        title: '线性表及其实现',
        difficulty: '基础',
        duration: '2.5小时',
        status: 'in_progress',
        description: '学习线性表的基本概念、顺序存储和链式存储的实现方法。',
        knowledgePoints: ['线性表', '顺序表', '链表'],
        resources: [
          { title: '线性表基础', type: '视频', url: '/student/courses/4/videos/401', duration: '45分钟' },
          { title: '链表操作实现', type: '代码', url: '/student/resources/402', duration: '40分钟' }
        ]
      },
      {
        id: 2,
        title: '树与二叉树',
        difficulty: '中等',
        duration: '3小时',
        status: 'pending',
        description: '掌握树和二叉树的基本概念、存储结构和遍历方法。',
        knowledgePoints: ['树的基本概念', '二叉树', '树的遍历'],
        resources: [
          { title: '二叉树详解', type: '视频', url: '/student/courses/4/videos/402', duration: '55分钟' },
          { title: '二叉树遍历算法实现', type: '代码', url: '/student/resources/403', duration: '35分钟' }
        ]
      },
      {
        id: 3,
        title: '图及其算法',
        difficulty: '高级',
        duration: '4小时',
        status: 'pending',
        description: '学习图的基本概念、存储结构及常用算法。',
        knowledgePoints: ['图的基本概念', '图的遍历', '最短路径算法'],
        resources: [
          { title: '图论基础', type: '视频', url: '/student/courses/4/videos/403', duration: '60分钟' },
          { title: '图算法实现', type: '代码', url: '/student/resources/404', duration: '50分钟' }
        ]
      },
      {
        id: 4,
        title: '排序与查找',
        difficulty: '中等',
        duration: '3.5小时',
        status: 'pending',
        description: '掌握各种排序和查找算法的原理与实现。',
        knowledgePoints: ['内部排序', '外部排序', '查找算法'],
        resources: [
          { title: '排序算法详解', type: '视频', url: '/student/courses/4/videos/404', duration: '65分钟' },
          { title: '查找算法实现与分析', type: '代码', url: '/student/resources/405', duration: '45分钟' }
        ]
      }
    ],
    recommendations: [
      {
        icon: '🌳',
        title: '加强树结构理解',
        description: '您在树结构方面需要加强，特别是平衡树和B树等高级树结构。'
      },
      {
        icon: '🔍',
        title: '图算法需要突破',
        description: '图算法是您的薄弱环节，建议重点学习图的遍历、最短路径和最小生成树等算法。'
      },
      {
        icon: '💻',
        title: '多动手实践',
        description: '数据结构需要通过编程实践加深理解，建议实现各种数据结构和算法，提高编程能力。'
      }
    ]
  }
};

const generatePathway = async () => {
  if (!selectedSubject.value) {
    message.warning('请先选择课程')
    return
  }
  
  try {
    // 显示生成中状态
    generatingPathway.value = true
    
    // 模拟生成过程延迟
    await new Promise(resolve => setTimeout(resolve, 1800))
    
    // 获取当前选择的科目ID
    const subjectId = selectedSubject.value
    
    if (subjectId && subjectPathways[subjectId]) {
      // 从数据中获取当前科目的学习路径
      learningPathway.value = JSON.parse(JSON.stringify(subjectPathways[subjectId].pathway))
      
      // 获取当前科目的学习建议
      learningRecommendations.value = JSON.parse(JSON.stringify(subjectPathways[subjectId].recommendations))
      
      // 获取薄弱知识点
      const weakPoints = knowledgePointsData.value
        .filter(point => point.score < 70)
        .sort((a, b) => a.score - b.score) // 按分数升序排序，优先安排最薄弱的知识点
      
      // 增强学习路径的个性化程度
      if (weakPoints.length > 0) {
        // 为薄弱知识点相关的学习步骤添加额外资源
        learningPathway.value = learningPathway.value.map(step => {
          // 检查该步骤是否涉及薄弱知识点
          const isWeakPointStep = step.knowledgePoints.some(kp => 
            weakPoints.some(wp => wp.name.includes(kp) || kp.includes(wp.name))
          )
          
          if (isWeakPointStep) {
            // 为薄弱知识点添加额外的学习资源
            const extraResource = {
              title: `${step.title}强化训练`,
              type: '习题',
              url: `/student/resources/extra-${step.id}`,
              duration: '30分钟'
            }
            
            // 避免重复添加资源
            if (!step.resources.some(r => r.title === extraResource.title)) {
              step.resources = [...step.resources, extraResource]
            }
          }
          
          return step
        })
        
        // 重新排序学习路径，将薄弱知识点相关步骤提前
        const weakSteps = learningPathway.value.filter(step => 
          step.knowledgePoints.some(kp => 
            weakPoints.some(wp => wp.name.includes(kp) || kp.includes(wp.name))
          )
        )
        
        const otherSteps = learningPathway.value.filter(step => 
          !step.knowledgePoints.some(kp => 
            weakPoints.some(wp => wp.name.includes(kp) || kp.includes(wp.name))
          )
        )
        
        // 如果找到薄弱知识点相关步骤，调整学习顺序
        if (weakSteps.length > 0 && otherSteps.length > 0) {
          // 将薄弱知识点步骤放在前面，但保持内部顺序
          learningPathway.value = [...weakSteps, ...otherSteps]
          
          // 更新步骤状态
          learningPathway.value[0].status = 'in_progress'
          for (let i = 1; i < learningPathway.value.length; i++) {
            learningPathway.value[i].status = 'pending'
          }
        }
      }
      
      // 重置展开状态，默认展开第一个步骤
      expandedSteps.value = { 0: true }
      currentStep.value = 0
      
      // 计算学习进度
      const completedCount = learningPathway.value.filter(step => step.status === 'completed').length
      studyProgress.value = Math.floor(completedCount / learningPathway.value.length * 100)
      
      // 更新路径可视化
      nextTick(() => {
        initPathwayChart()
      })
      
      message.success(`已为您生成 ${subjects.value.find(s => s.id === subjectId)?.name} 的个性化学习路径`)
    } else {
      message.error('生成路径失败，无法获取课程数据')
    }
  } catch (error) {
    message.error('生成失败：' + (error as Error).message)
  } finally {
    generatingPathway.value = false
  }
}

// 学习路径生成现在集成在handleSubjectChange和generatePathway函数中

const toggleStep = (index: number) => {
  expandedSteps.value = {
    ...expandedSteps.value,
    [index]: !expandedSteps.value[index]
  }
}

const markStepCompleted = (index: number) => {
  learningPathway.value[index].status = 'completed'
  
  // 更新进度条
  const completedCount = learningPathway.value.filter(step => step.status === 'completed').length
  studyProgress.value = Math.floor(completedCount / learningPathway.value.length * 100)
  
  // 更新可视化
  nextTick(() => {
    initPathwayChart()
  })
  
  message.success('已标记为完成')
}

const goToNextStep = (index: number) => {
  if (index < learningPathway.value.length - 1) {
    // 关闭当前步骤
    expandedSteps.value = {
      ...expandedSteps.value,
      [index]: false,
      [index + 1]: true
    }
    
    // 更新当前步骤
    currentStep.value = index + 1
    
    // 更新下一个步骤的状态
    if (learningPathway.value[index + 1].status === 'pending') {
      learningPathway.value[index + 1].status = 'in_progress'
      
      // 更新可视化
      nextTick(() => {
        initPathwayChart()
      })
    }
  }
}

const startLearning = (resource: any) => {
  message.info(`开始学习: ${resource.title}`)
  // 实际应用中应该跳转到相应的学习资源页面
}

const startPractice = (step: any) => {
  message.info(`开始练习: ${step.title}`)
  // 实际应用中应该跳转到练习页面
}

const getDifficultyColor = (difficulty: string) => {
  const colorMap: Record<string, string> = {
    '基础': 'green',
    '中等': 'blue',
    '高级': 'red'
  }
  return colorMap[difficulty] || 'default'
}

const getStatusColor = (status: string) => {
  const colorMap: Record<string, string> = {
    'completed': 'green',
    'in_progress': 'blue',
    'pending': 'orange'
  }
  return colorMap[status] || 'default'
}

const getStatusText = (status: string) => {
  const textMap: Record<string, string> = {
    'completed': '已完成',
    'in_progress': '进行中',
    'pending': '待学习'
  }
  return textMap[status] || status
}

const getResourceColor = (type: string) => {
  const colorMap: Record<string, string> = {
    '视频': '#1890ff',
    '文档': '#52c41a',
    '习题': '#fa8c16',
    '案例': '#722ed1'
  }
  return colorMap[type] || '#d9d9d9'
}

const getResourceIcon = (type: string) => {
  const iconMap: Record<string, string> = {
    '视频': '📹',
    '文档': '📄',
    '习题': '📝',
    '案例': '📊'
  }
  return iconMap[type] || '📑'
}

onMounted(() => {
  // 初始化展开第一个步骤
  expandedSteps.value = { 0: true }
  
  // 初始化雷达图
  nextTick(() => {
    initRadarChart()
    initPathwayChart()
    
    // 默认选择第一个科目但不自动生成路径
    if (subjects.value.length > 0) {
      selectedSubject.value = subjects.value[0].id
    }
  })
})
</script>

<style scoped>
.learning-pathway {
  padding: 24px;
  background-color: #f5f5f5;
  min-height: 100vh;
}

.page-header {
  text-align: center;
  margin-bottom: 32px;
}

.page-header h1 {
  font-size: 28px;
  margin-bottom: 8px;
  color: #1890ff;
}

.description {
  color: #666;
  font-size: 16px;
}

.pathway-container {
  max-width: 1400px;
  margin: 0 auto;
  display: flex;
  flex-direction: column;
  gap: 24px;
}

.analysis-row {
  margin-bottom: 24px;
}

.analysis-card, .status-card, .pathway-card, .suggestions-card {
  border-radius: 12px;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
}

.chart-container {
  position: relative;
  height: 400px;
  display: flex;
  flex-direction: column;
}

.radar-chart {
  height: 350px;
  width: 100%;
}

.legend {
  display: flex;
  justify-content: center;
  gap: 24px;
  margin-top: 16px;
}

.legend-item {
  display: flex;
  align-items: center;
  gap: 8px;
}

.legend-color {
  display: inline-block;
  width: 16px;
  height: 8px;
  border-radius: 2px;
}

.legend-color.current {
  background-color: #1890ff;
}

.legend-color.target {
  background-color: #52c41a;
}

.status-summary {
  display: flex;
  flex-direction: column;
  gap: 24px;
}

.summary-item {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.summary-label {
  font-size: 14px;
  color: #666;
}

.summary-value {
  font-size: 24px;
  font-weight: 600;
  color: #1890ff;
}

.knowledge-tags {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-top: 8px;
}

.pathway-wrapper {
  display: flex;
  gap: 24px;
  margin-top: 16px;
}

.pathway-steps {
  flex: 1;
}

.pathway-step {
  margin-bottom: 16px;
  border: 1px solid #e8e8e8;
  border-radius: 8px;
  overflow: hidden;
  background-color: #fff;
  transition: all 0.3s;
}

.pathway-step.active {
  border-color: #1890ff;
  box-shadow: 0 2px 8px rgba(24, 144, 255, 0.2);
}

.step-header {
  display: flex;
  align-items: center;
  padding: 16px;
  cursor: pointer;
  transition: background-color 0.3s;
}

.step-header:hover {
  background-color: #f5f5f5;
}

.step-number {
  width: 32px;
  height: 32px;
  border-radius: 50%;
  background-color: #1890ff;
  color: white;
  display: flex;
  align-items: center;
  justify-content: center;
  font-weight: 600;
  margin-right: 16px;
}

.step-info {
  flex: 1;
}

.step-info h3 {
  margin: 0 0 8px 0;
  font-size: 16px;
}

.step-meta {
  display: flex;
  align-items: center;
  gap: 8px;
}

.step-duration {
  color: #666;
  font-size: 12px;
}

.step-status {
  margin-right: 16px;
}

.step-content {
  padding: 0 16px 16px 64px;
  border-top: 1px solid #f0f0f0;
}

.step-description {
  margin-bottom: 16px;
}

.knowledge-points, .resources-list, .practice-section {
  margin-bottom: 16px;
}

.knowledge-list {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-top: 8px;
}

.step-actions {
  margin-top: 24px;
  display: flex;
  justify-content: flex-end;
}

.pathway-visualization {
  width: 300px;
}

.pathway-chart {
  height: 400px;
  width: 100%;
}

.suggestions-list {
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.suggestion-item {
  display: flex;
  gap: 16px;
  padding: 16px;
  background-color: #f9f9f9;
  border-radius: 8px;
}

.suggestion-icon {
  font-size: 24px;
}

.suggestion-content {
  flex: 1;
}

.suggestion-content h4 {
  margin: 0 0 8px 0;
  color: #333;
}

.suggestion-content p {
  margin: 0;
  color: #666;
}
</style> 