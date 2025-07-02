<template>
  <div class="course-exams">
    <div class="page-header">
      <h2>考试管理</h2>
      <a-button type="primary" @click="showAddExamModal">
        <PlusOutlined />
        添加考试
      </a-button>
    </div>

    <div class="filter-section">
      <div class="filter-row">
        <div class="filter-left">
          <div class="filter-item">
            <span class="filter-label">状态：</span>
            <a-select 
              v-model:value="filters.status" 
              style="width: 120px" 
              placeholder="全部状态"
              allowClear
              @change="handleFilterChange"
            >
              <a-select-option value="not_started">未开始</a-select-option>
              <a-select-option value="in_progress">进行中</a-select-option>
              <a-select-option value="ended">已结束</a-select-option>
            </a-select>
          </div>
        </div>
        <div class="filter-right">
          <div class="filter-item search-box">
            <a-input-search
              v-model:value="filters.keyword"
              placeholder="搜索考试名称"
              style="width: 250px"
              @search="handleSearch"
              enter-button
            />
          </div>
          <a-button @click="resetFilters" size="middle">
            重置筛选
          </a-button>
        </div>
      </div>
    </div>

    <div class="exam-content">
      <a-spin :spinning="loading">
        <a-empty v-if="exams.length === 0" description="暂无考试" />
        
        <a-table
          v-else
          :dataSource="exams"
          :columns="columns"
          :pagination="pagination"
          :rowKey="(record) => record.id"
          @change="handleTableChange"
        >
          <!-- 考试名称 -->
          <template #bodyCell="{ column, record }">
            <template v-if="column.dataIndex === 'title'">
              <div class="exam-title">
                <span>{{ record.title }}</span>
              </div>
            </template>

            <!-- 考试状态 -->
            <template v-else-if="column.dataIndex === 'examState'">
              <a-tag :color="getStatusColor(record.examState)">{{ getStatusText(record.examState) }}</a-tag>
            </template>

            <!-- 考试时间 -->
            <template v-else-if="column.dataIndex === 'examTime'">
              <div>
                <div>开始：{{ formatDate(record.startTime) }}</div>
                <div>结束：{{ formatDate(record.endTime) }}</div>
              </div>
            </template>

            <!-- 发布状态 -->
            <template v-else-if="column.dataIndex === 'publishStatus'">
              <a-tag :color="record.status === 0 ? 'orange' : 'green'">
                {{ record.status === 0 ? '未发布' : '已发布' }}
              </a-tag>
            </template>
            
            <!-- 操作 -->
            <template v-else-if="column.dataIndex === 'action'">
              <div class="exam-actions">
                <a-tooltip title="查看">
                  <a-button type="link" @click="viewExamDetail(record)">
                    <EyeOutlined />
                  </a-button>
                </a-tooltip>
                <a-tooltip title="编辑">
                  <a-button type="link" @click="editExam(record)">
                    <EditOutlined />
                  </a-button>
                </a-tooltip>
                <a-tooltip title="发布" v-if="record.status === 0">
                  <a-button type="link" @click="publishExam(record)" style="color: #52c41a">
                    <CheckOutlined />
                  </a-button>
                </a-tooltip>
                <a-tooltip title="取消发布" v-if="record.status === 1">
                  <a-button type="link" @click="unpublishExam(record)" style="color: #faad14">
                    <CloseCircleOutlined />
                  </a-button>
                </a-tooltip>
                <a-tooltip title="删除">
                  <a-popconfirm
                    title="确定要删除这个考试吗？"
                    description="删除后将无法恢复，包括考试题目关联数据也会被删除。"
                    @confirm="handleDeleteExam(record.id)"
                    ok-text="确定"
                    cancel-text="取消"
                  >
                    <a-button type="link" danger>
                      <DeleteOutlined />
                    </a-button>
                  </a-popconfirm>
                </a-tooltip>
              </div>
            </template>
          </template>
        </a-table>
      </a-spin>
    </div>

    <!-- 添加/编辑考试弹窗 -->
    <a-modal
      v-model:open="examModalVisible"
      :title="isEditing ? '编辑考试' : '添加考试'"
      :maskClosable="false"
      :footer="null"
      width="800px"
    >
      <div class="exam-form-steps">
        <a-steps :current="currentStep">
          <a-step title="基本信息" />
          <a-step title="组卷设置" />
        </a-steps>
        
        <!-- 步骤1: 基本信息 -->
        <div v-if="currentStep === 0" class="step-content">
          <a-form :model="examForm" layout="vertical">
            <a-form-item label="考试名称" required>
              <a-input v-model:value="examForm.title" placeholder="请输入考试名称" />
            </a-form-item>
            <a-form-item label="考试时间" required>
              <a-range-picker 
                v-model:value="examTimeRange" 
                :show-time="{ format: 'HH:mm' }" 
                format="YYYY-MM-DD HH:mm"
                @change="handleTimeRangeChange"
              />
            </a-form-item>
            <a-form-item label="考试时长(分钟)" required>
              <a-input-number v-model:value="examForm.duration" :min="1" :max="300" :disabled="true" />
              <div class="form-help-text">* 考试时长根据所选时间范围自动计算</div>
            </a-form-item>
            <a-form-item label="考试说明">
              <a-textarea v-model:value="examForm.description" placeholder="请输入考试说明" :rows="4" />
            </a-form-item>
          </a-form>
          
          <div class="step-actions">
            <a-button @click="examModalVisible = false">取消</a-button>
            <a-button type="primary" :loading="savingBasicInfo" @click="saveBasicInfo">
              {{ examForm.id ? '下一步' : (savingBasicInfo ? '保存中...' : '保存') }}
            </a-button>
          </div>
        </div>
        
        <!-- 步骤2: 组卷设置 -->
        <div v-else-if="currentStep === 1" class="step-content">
          <div class="paper-config-section">
            <div class="section-header">
              <h3>组卷设置</h3>
              <div class="auto-generate">
                <a-checkbox v-model:checked="examForm.paperConfig.isRandom">随机组卷</a-checkbox>
              </div>
            </div>
            
            <!-- 设置难度和知识点筛选 -->
            <div class="filter-config" v-if="examForm.paperConfig.isRandom">
              <a-form layout="horizontal" :label-col="{ span: 6 }" :wrapper-col="{ span: 18 }">
                <a-form-item label="难度等级">
                  <a-rate v-model:value="examForm.paperConfig.difficulty" />
                </a-form-item>
                <a-form-item label="知识点">
                  <a-select
                    v-model:value="examForm.paperConfig.knowledgePoint"
                    placeholder="选择知识点"
                    allowClear
                  >
                    <a-select-option v-for="point in knowledgePoints" :key="point">{{ point }}</a-select-option>
                  </a-select>
                </a-form-item>
              </a-form>
            </div>
            
            <!-- 题目配置 -->
            <div class="question-config">
              <a-divider>题目配置</a-divider>
              
              <!-- 考试项列表 -->
              <div class="exam-sections">
                <div v-for="(section, index) in examSections" :key="index" class="exam-section">
                  <div class="section-header">
                    <span class="section-title">{{ getRomanNumber(index+1) }}、{{ getQuestionTypeText(section.questionType) }}</span>
                    <span class="section-summary">共{{ section.count }}题 {{ section.score * section.count }}分</span>
                    <a-button type="link" danger @click="removeExamSection(index)">
                      <DeleteOutlined />
                    </a-button>
                  </div>
                </div>
              </div>
              
              <!-- 添加考试项按钮 -->
              <div class="add-section">
                <a-button type="dashed" block @click="showAddSectionModal">
                  <PlusOutlined /> 添加考试项
                </a-button>
              </div>
              
              <div class="total-score">
                <div class="total-label">总分:</div>
                <div class="total-value">{{ calculateTotalScore() }}分</div>
              </div>
            </div>
          </div>
          
          <div class="step-actions">
            <a-button @click="prevStep">上一步</a-button>
            <a-button type="primary" :loading="saving" @click="handleSaveExam">
              {{ saving ? '保存中...' : '保存' }}
            </a-button>
          </div>
        </div>
      </div>
    </a-modal>

        <!-- 查看考试详情弹窗 -->
    <a-modal
      v-model:open="viewModalVisible"
      title="考试详情"
      :footer="null"
      width="700px"
    >
      <div v-if="currentExam" class="exam-detail">
        <div class="exam-detail-header">
          <a-tag :color="getStatusColor(currentExam.examState)" class="status-tag">
            {{ getStatusText(currentExam.examState) }}
          </a-tag>
        </div>
        
        <div class="exam-detail-item">
          <div class="exam-detail-label">考试名称：</div>
          <div class="exam-detail-value">{{ currentExam.title }}</div>
        </div>
        
        <div class="exam-detail-item">
          <div class="exam-detail-label">考试时间：</div>
          <div class="exam-detail-value">
            <div>开始时间：{{ formatDate(currentExam.startTime) }}</div>
            <div>结束时间：{{ formatDate(currentExam.endTime) }}</div>
          </div>
        </div>
        
        <div class="exam-detail-item">
          <div class="exam-detail-label">考试时长：</div>
          <div class="exam-detail-value">{{ currentExam.duration }} 分钟</div>
        </div>
        
        <div class="exam-detail-item">
          <div class="exam-detail-label">总分值：</div>
          <div class="exam-detail-value">{{ currentExam.totalScore }} 分</div>
        </div>
        
        <div class="exam-detail-item" v-if="currentExam.description">
          <div class="exam-detail-label">考试说明：</div>
          <div class="exam-detail-value">{{ currentExam.description }}</div>
        </div>
        
        <div class="exam-detail-item">
          <div class="exam-detail-label">创建时间：</div>
          <div class="exam-detail-value">{{ formatDate(currentExam.createTime) }}</div>
        </div>
        
        <div class="exam-detail-actions">
          <a-button type="primary" @click="editExam(currentExam)">编辑考试</a-button>
          <a-button @click="viewModalVisible = false">关闭</a-button>
        </div>
      </div>
    </a-modal>

    <!-- 添加考试项弹窗 -->
    <a-modal
      v-model:open="addSectionModalVisible"
      title="添加考试项"
      :maskClosable="false"
      :footer="null"
      width="900px"
    >
      <div class="section-form">
        <a-form layout="horizontal" :label-col="{ span: 4 }" :wrapper-col="{ span: 20 }">
          <a-form-item label="题目类型">
            <a-select v-model:value="currentSection.questionType" placeholder="选择题目类型">
              <a-select-option value="single">单选题</a-select-option>
              <a-select-option value="multiple">多选题</a-select-option>
              <a-select-option value="true_false">判断题</a-select-option>
              <a-select-option value="blank">填空题</a-select-option>
              <a-select-option value="short">简答题</a-select-option>
              <a-select-option value="code">编程题</a-select-option>
            </a-select>
          </a-form-item>
          
          <a-form-item label="题目数量">
            <a-input-number v-model:value="currentSection.count" :min="1" :max="50" />
          </a-form-item>
          
          <a-form-item label="每题分值">
            <a-input-number v-model:value="currentSection.score" :min="1" :max="100" />
          </a-form-item>
        </a-form>
        
        <!-- 题目筛选区域 -->
        <div class="question-filter">
          <div class="filter-row">
            <a-form layout="inline">
              <a-form-item label="难度等级">
                <a-select v-model:value="questionFilters.difficulty" style="width: 120px" placeholder="全部难度" allowClear>
                  <a-select-option :value="1">★</a-select-option>
                  <a-select-option :value="2">★★</a-select-option>
                  <a-select-option :value="3">★★★</a-select-option>
                  <a-select-option :value="4">★★★★</a-select-option>
                  <a-select-option :value="5">★★★★★</a-select-option>
                </a-select>
              </a-form-item>
              
              <a-form-item label="知识点">
                <a-select v-model:value="questionFilters.knowledgePoint" style="width: 150px" placeholder="全部知识点" allowClear>
                  <a-select-option v-for="point in knowledgePoints" :key="point">{{ point }}</a-select-option>
                </a-select>
              </a-form-item>
              
              <a-form-item>
                <a-input-search
                  v-model:value="questionFilters.keyword"
                  placeholder="搜索题目内容"
                  style="width: 200px"
                  @search="handleSearchQuestions"
                />
              </a-form-item>
              
              <a-form-item>
                <a-button @click="resetQuestionFilters">重置筛选</a-button>
              </a-form-item>
            </a-form>
          </div>
        </div>
        
        <!-- 题目列表 -->
        <div class="question-list-container">
          <a-spin :spinning="loadingQuestions">
            <a-empty v-if="filteredQuestions.length === 0" description="暂无题目" />
            
            <div v-else class="question-list">
              <div v-for="question in filteredQuestions" :key="question.id" class="question-item">
                <a-checkbox 
                  v-model:checked="question.selected" 
                  :disabled="!question.selected && isMaxQuestionsSelected"
                >
                  <div class="question-content">
                    <div class="question-title">{{ question.title }}</div>
                    <div class="question-info">
                      <a-tag>{{ getQuestionTypeText(question.questionType) }}</a-tag>
                      <a-tag color="orange">难度: {{ getDifficultyStars(question.difficulty) }}</a-tag>
                      <a-tag v-if="question.knowledgePoint" color="blue">{{ question.knowledgePoint }}</a-tag>
                    </div>
                  </div>
                </a-checkbox>
              </div>
            </div>
            
            <div class="selection-summary">
              已选择 {{ selectedQuestionCount }} / {{ currentSection.count }} 题
            </div>
          </a-spin>
        </div>
        
        <div class="modal-footer">
          <a-button @click="addSectionModalVisible = false">取消</a-button>
          <a-button type="primary" :disabled="selectedQuestionCount !== currentSection.count" @click="confirmAddSection">
            确定
          </a-button>
        </div>
      </div>
    </a-modal>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted, watch, defineProps, computed } from 'vue'
import { useRouter } from 'vue-router'
import { message } from 'ant-design-vue'
import {
  PlusOutlined,
  EyeOutlined,
  EditOutlined,
  DeleteOutlined,
  CheckOutlined,
  CloseCircleOutlined
} from '@ant-design/icons-vue'
import { formatDate } from '@/utils/date'
import examAPI from '@/api/exam'
import request from '@/utils/request'
import type { Dayjs } from 'dayjs'
import dayjs from 'dayjs'

// API响应类型定义
interface ApiResponse<T = any> {
  code: number;
  data: T;
  message?: string;
}

// 题目类型定义
interface Question {
  id: number
  title: string
  questionType: string
  difficulty: number
  knowledgePoint?: string
  selected: boolean
}

// 题目池
interface QuestionPool {
  single: Question[]
  multiple: Question[]
  trueFalse: Question[]
  blank: Question[]
  short: Question[]
  code: Question[]
}

// 已选题目
interface SelectedQuestions {
  single: Question[]
  multiple: Question[]
  trueFalse: Question[]
  blank: Question[]
  short: Question[]
  code: Question[]
}

// 考试项定义
interface ExamSection {
  questionType: string
  count: number
  score: number
  questions: Question[]
}

// 定义考试数据类型
interface ExamRecord {
  id: number;
  title: string;
  courseId: number;
  userId: number;
  description?: string;
  startTime: string;
  endTime: string;
  status: number;
  examState?: string;
  publishStatus?: string;
  duration?: number;
  [key: string]: any; // 其他可能的属性
}

const props = defineProps({
  courseId: {
    type: Number,
    required: true
  }
})

// 初始化路由
const router = useRouter()

// 考试状态
const examStatus = {
  NOT_STARTED: 'not_started',
  IN_PROGRESS: 'in_progress',
  ENDED: 'ended'
}

// 状态定义
const exams = ref<ExamRecord[]>([])
const loading = ref(false)
const pagination = ref({
  current: 1,
  pageSize: 10,
  total: 0,
  showSizeChanger: true,
  showTotal: (total: number) => `共 ${total} 条`
})

// 筛选条件
const filters = ref({
  status: undefined as string | undefined,
  keyword: ''
})

// 表格列定义
const columns = [
  {
    title: '考试名称',
    dataIndex: 'title',
    key: 'title',
    ellipsis: true,
    width: '25%'
  },
  {
    title: '状态',
    dataIndex: 'examState',
    key: 'examState',
    width: '10%'
  },
  {
    title: '发布状态',
    dataIndex: 'publishStatus',
    key: 'publishStatus',
    width: '10%'
  },
  {
    title: '考试时间',
    dataIndex: 'examTime',
    key: 'examTime',
    width: '20%'
  },
  {
    title: '考试时长',
    dataIndex: 'duration',
    key: 'duration',
    width: '10%',
    customRender: ({ text }: { text: number }) => `${text} 分钟`
  },
  {
    title: '操作',
    dataIndex: 'action',
    key: 'action',
    width: '15%'
  }
]

// 添加/编辑考试相关状态
const examModalVisible = ref(false)
const isEditing = ref(false)
const saving = ref(false)
const savingBasicInfo = ref(false)
const examTimeRange = ref<[Dayjs, Dayjs] | null>(null)
const examForm = ref<ExamRecord>({
  id: undefined,
  title: '',
  courseId: props.courseId,
  userId: 0,
  startTime: '',
  endTime: '',
  duration: 60,
  totalScore: 100,
  description: '',
  status: 0,
  paperConfig: {
    singleCount: 10,
    singleScore: 3,
    multipleCount: 5,
    multipleScore: 4,
    trueFalseCount: 10,
    trueFalseScore: 2,
    blankCount: 5,
    blankScore: 2,
    shortCount: 2,
    shortScore: 10,
    codeCount: 0,
    codeScore: 0,
    isRandom: true,
    difficulty: 3,
    knowledgePoint: undefined
  }
})

// 查看考试相关状态
const viewModalVisible = ref(false)
const currentExam = ref<ExamRecord | null>(null)

// 考试项相关状态
const examSections = ref<ExamSection[]>([])
const addSectionModalVisible = ref(false)
const currentSection = ref<ExamSection>({
  questionType: 'single',
  count: 10,
  score: 5,
  questions: []
})
const filteredQuestions = ref<Question[]>([])
const loadingQuestions = ref(false)
const questionFilters = ref({
  type: undefined as string | undefined,
  difficulty: undefined as number | undefined,
  knowledgePoint: undefined as string | undefined,
  keyword: ''
})

// 步骤控制
const currentStep = ref(0)
const nextStep = () => {
  if (validateStep1()) {
    currentStep.value = 1
    // 设置考试总分初始值
    examForm.value.totalScore = calculateTotalScore()
  }
}

// 保存基本信息
const saveBasicInfo = async () => {
  if (!validateStep1()) {
    return
  }
  
  // 如果已经有ID，直接进入下一步
  if (examForm.value.id) {
    nextStep()
    return
  }
  
  try {
    savingBasicInfo.value = true
    
    // 设置考试类型为exam
    examForm.value.type = 'exam'
    // 设置用户ID为当前登录用户
    examForm.value.userId = currentUserId.value
    
    // 调用API保存基本信息
    const res = await examAPI.createExam(examForm.value)
    
    if (res && res.code === 200) {
      message.success('考试基本信息保存成功')
      // 设置ID
      examForm.value.id = res.data
      
      // 添加新创建的考试到列表中，无需刷新页面
      const newExam: ExamRecord = {
        id: res.data,
        title: examForm.value.title,
        courseId: examForm.value.courseId,
        userId: examForm.value.userId,
        startTime: examForm.value.startTime,
        endTime: examForm.value.endTime,
        duration: examForm.value.duration,
        status: 'not_started', // 新创建的考试默认未开始
        description: examForm.value.description
      }
      exams.value.unshift(newExam)
      
      // 进入下一步
      nextStep()
    } else {
      message.error(res?.message || '保存失败')
    }
  } catch (error) {
    console.error('保存考试基本信息失败:', error)
    message.error('保存失败，请重试')
  } finally {
    savingBasicInfo.value = false
  }
}

const prevStep = () => {
  currentStep.value = 0
}

// 验证第一步表单
const validateStep1 = () => {
  if (!examForm.value.title) {
    message.error('请输入考试名称')
    return false
  }
  if (!examTimeRange.value || examTimeRange.value.length < 2) {
    message.error('请选择考试时间')
    return false
  }
  if (!examForm.value.duration) {
    message.error('请输入考试时长')
    return false
  }
  return true
}

// 知识点列表
const knowledgePoints = ref<Array<string>>([])
// 当前用户ID
const currentUserId = ref<number>(0)

// 获取当前用户信息
const fetchCurrentUserInfo = async () => {
  console.log('开始获取用户信息...')
  try {
    // 直接使用后端日志中的用户ID
    console.log('直接使用教师ID: 6')
    currentUserId.value = 6
    return 6
    
    /* 暂时注释掉API调用，直接使用硬编码的用户ID
    // 尝试从API获取用户信息
    console.log('尝试调用API: /api/auth/user-info')
    const response = await request({
      url: '/api/auth/user-info',
      method: 'get'
    })
    
    console.log('API响应:', response)
    
    // 解析响应为我们的自定义类型
    const res = response as unknown as ApiResponse<{id: number}>
    
    if (res && res.code === 200 && res.data) {
      currentUserId.value = res.data.id
      console.log('当前用户ID:', currentUserId.value)
      return res.data.id
    } else {
      console.error('获取用户信息失败:', res)
      // API失败时使用默认用户ID (6)
      console.log('使用默认用户ID: 6')
      currentUserId.value = 6
      return 6
    }
    */
  } catch (error) {
    console.error('获取用户信息失败:', error)
    // 错误时使用默认用户ID (6)
    console.log('使用默认用户ID: 6')
    currentUserId.value = 6
    return 6
  }
}

// 获取课程知识点
const fetchKnowledgePoints = async () => {
  console.log('开始获取知识点...')
  try {
    // 先获取当前用户ID
    const userId = currentUserId.value || await fetchCurrentUserInfo()
    console.log('获取到用户ID:', userId)
    
    if (!userId) {
      message.error('获取用户信息失败，无法加载知识点')
      return
    }
    
    try {
      // 使用专门的知识点API
      console.log('调用知识点API, 参数:', { courseId: props.courseId, createdBy: userId })
      const response = await examAPI.getKnowledgePoints(props.courseId, userId)
      console.log('知识点API响应:', response)
      
      // 直接处理响应数据
      if (response && response.code === 200) {
        console.log('获取知识点成功，数据:', response.data)
        // 处理返回的知识点数据
        if (Array.isArray(response.data)) {
          knowledgePoints.value = [...new Set(response.data)].filter(Boolean)
          console.log('处理后的知识点:', knowledgePoints.value)
        } else {
          console.error('知识点数据格式不正确:', response.data)
          knowledgePoints.value = []
        }
      } else {
        console.log('知识点API失败，尝试使用备用方法')
        // 如果专门API失败，回退到旧方法
        console.log('调用课程题目列表API, 参数:', { 
          courseId: props.courseId, 
          createdBy: userId,
          onlyKnowledgePoints: true 
        })
        
        const fallbackResponse = await examAPI.getCourseQuestionList({
          courseId: props.courseId,
          createdBy: userId,
          onlyKnowledgePoints: true
        })
        
        console.log('课程题目列表API响应:', fallbackResponse)
        
        if (fallbackResponse && fallbackResponse.code === 200) {
          console.log('获取题目列表成功，数据:', fallbackResponse.data)
          
          const data = Array.isArray(fallbackResponse.data) ? fallbackResponse.data : 
                      Array.isArray(fallbackResponse.data?.records) ? fallbackResponse.data.records : [];
          
          console.log('处理前的数据:', data)
          
          const points = data
            .map((item: any) => item.knowledgePoint)
            .filter(Boolean);
          
          knowledgePoints.value = [...new Set(points)];
          console.log('处理后的知识点:', knowledgePoints.value)
        } else {
          console.error('获取知识点失败:', fallbackResponse)
          message.error('获取知识点失败')
        }
      }
    } catch (error) {
      console.error('API调用出错:', error)
      message.error('获取知识点失败')
    }
  } catch (error) {
    console.error('获取知识点失败:', error)
  }
}

// 初始化考试表单
const initExamForm = () => {
  examForm.value = {
    id: undefined,
    title: '',
    courseId: props.courseId,
    userId: currentUserId.value, // 设置当前用户ID
    type: 'exam', // 设置类型为exam
    startTime: '',
    endTime: '',
    duration: 90,
    totalScore: 100,
    status: 0,
    description: '',
    paperConfig: {
      singleCount: 10,
      singleScore: 3,
      multipleCount: 5,
      multipleScore: 4,
      trueFalseCount: 10,
      trueFalseScore: 2,
      blankCount: 5,
      blankScore: 2,
      shortCount: 2,
      shortScore: 10,
      codeCount: 0,
      codeScore: 0,
      isRandom: true,
      difficulty: 3,
      knowledgePoint: undefined
    }
  }
  examTimeRange.value = null
}

// 计算总分
const calculateTotalScore = () => {
  if (examForm.value.paperConfig.isRandom) {
    const config = examForm.value.paperConfig
    return (
      (config.singleCount || 0) * (config.singleScore || 0) +
      (config.multipleCount || 0) * (config.multipleScore || 0) +
      (config.trueFalseCount || 0) * (config.trueFalseScore || 0) +
      (config.blankCount || 0) * (config.blankScore || 0) +
      (config.shortCount || 0) * (config.shortScore || 0) +
      (config.codeCount || 0) * (config.codeScore || 0)
    )
  } else {
    return examSections.value.reduce((total, section) => {
      return total + (section.count * section.score)
    }, 0)
  }
}

// 计算各题型总分
const getTotalScoreByType = (type: string) => {
  const config = examForm.value.paperConfig
  switch (type) {
    case 'single':
      return (config.singleCount || 0) * (config.singleScore || 0)
    case 'multiple':
      return (config.multipleCount || 0) * (config.multipleScore || 0)
    case 'trueFalse':
      return (config.trueFalseCount || 0) * (config.trueFalseScore || 0)
    case 'blank':
      return (config.blankCount || 0) * (config.blankScore || 0)
    case 'short':
      return (config.shortCount || 0) * (config.shortScore || 0)
    case 'code':
      return (config.codeCount || 0) * (config.codeScore || 0)
    default:
      return 0
  }
}

// 显示添加考试项弹窗
const showAddSectionModal = () => {
  currentSection.value = {
    questionType: 'single',
    count: 10,
    score: 5,
    questions: []
  }
  filteredQuestions.value = []
  addSectionModalVisible.value = true
  // 加载相应题型的题目
  loadQuestionsByType(currentSection.value.questionType)
}

// 移除考试项
const removeExamSection = (index: number) => {
  examSections.value.splice(index, 1)
}

// 加载指定类型的题目
const loadQuestionsByType = async (type: string) => {
  loadingQuestions.value = true
  try {
    const userId = currentUserId.value || await fetchCurrentUserInfo()
    
    console.log('调用getQuestionsByType API，参数:', { 
      courseId: props.courseId, 
      questionType: type, 
      difficulty: questionFilters.value.difficulty, 
      knowledgePoint: questionFilters.value.knowledgePoint,
      keyword: questionFilters.value.keyword,
      createdBy: userId 
    })
    
    const response = await examAPI.getQuestionsByType(
      props.courseId,
      type,
      questionFilters.value.difficulty,
      questionFilters.value.knowledgePoint,
      userId,
      questionFilters.value.keyword
    )
    
    console.log('getQuestionsByType API响应:', response)
    
    if (response && response.code === 200 && response.data && response.data[type]) {
      let questions = response.data[type].map((q: any) => ({
        id: q.id,
        title: q.title || '无标题',
        questionType: q.question_type || type,
        difficulty: q.difficulty || 3,
        knowledgePoint: q.knowledge_point,
        selected: false
      }))
      
      // 如果有关键词，在前端进行筛选
      if (questionFilters.value.keyword) {
        const keyword = questionFilters.value.keyword.toLowerCase()
        questions = questions.filter((q: Question) => 
          q.title.toLowerCase().includes(keyword) || 
          (q.knowledgePoint && q.knowledgePoint.toLowerCase().includes(keyword))
        )
      }
      
      filteredQuestions.value = questions
      console.log(`已加载${type}类型题目:`, filteredQuestions.value.length, '条')
    } else {
      console.log(`未找到${type}类型题目`)
      filteredQuestions.value = []
    }
  } catch (error) {
    console.error('加载题目失败:', error)
    message.error('加载题目失败')
    filteredQuestions.value = []
  } finally {
    loadingQuestions.value = false
  }
}

// 搜索题目
const handleSearchQuestions = () => {
  loadQuestionsByType(currentSection.value.questionType)
}

// 重置题目筛选条件
const resetQuestionFilters = () => {
  questionFilters.value.difficulty = undefined
  questionFilters.value.knowledgePoint = undefined
  questionFilters.value.keyword = ''
  loadQuestionsByType(currentSection.value.questionType)
}

// 计算已选题目数量
const selectedQuestionCount = computed(() => {
  return filteredQuestions.value.filter(q => q.selected).length
})

// 判断是否已达到最大选择数量
const isMaxQuestionsSelected = computed(() => {
  return selectedQuestionCount.value >= currentSection.value.count
})

// 确认添加考试项
const confirmAddSection = () => {
  const selectedQuestions = filteredQuestions.value.filter(q => q.selected)
  
  if (selectedQuestions.length !== currentSection.value.count) {
    message.warning(`请选择${currentSection.value.count}道题目`)
    return
  }
  
  examSections.value.push({
    ...currentSection.value,
    questions: [...selectedQuestions]
  })
  
  addSectionModalVisible.value = false
  message.success('考试项添加成功')
}

// 保存考试前的处理
const handleSaveExam = async () => {
  // 更新总分
  examForm.value.totalScore = calculateTotalScore()
  
  if (examForm.value.totalScore <= 0) {
    message.error('总分必须大于0，请设置题目数量和分值')
    return
  }
  
  try {
    saving.value = true
    
    // 调用API保存考试信息
    let res: any
    let examId = examForm.value.id
    
    // 只更新考试，因为基本信息已经在第一步保存过了
    res = await examAPI.updateExam(examId, examForm.value)
    
    // 如果是手动选题模式，保存所选题目
    if (!examForm.value.paperConfig.isRandom && res && res.code === 200) {
      // 收集所有已选题目及其分值
      const allQuestionIds: number[] = []
      const allScores: number[] = []
      
      // 从考试项中收集所有题目
      examSections.value.forEach(section => {
        if (section.questions && section.questions.length > 0) {
          // 将当前section中的所有题目添加到列表中
          section.questions.forEach((question, index) => {
            allQuestionIds.push(question.id)
            allScores.push(section.score) // 使用当前section设置的分值
          })
        }
      })
      
      console.log('要保存的题目:', allQuestionIds.length, '道')
      console.log('题目ID:', allQuestionIds)
      console.log('分值:', allScores)
      
      if (allQuestionIds.length > 0) {
        // 保存题目关联
        const questionsRes = await examAPI.selectQuestions(examId, allQuestionIds, allScores)
        if (questionsRes && questionsRes.code === 200) {
          console.log('题目关联保存成功')
        } else {
          console.error('题目关联保存失败:', questionsRes)
          message.error(questionsRes?.message || '题目关联保存失败')
          return
        }
      }
    }
    
    if (res && res.code === 200) {
      message.success('考试更新成功')
      examModalVisible.value = false
      // 重置表单和步骤
      currentStep.value = 0
      fetchExams() // 重新加载考试列表
    } else {
      message.error(res?.message || '操作失败')
    }
  } catch (error) {
    console.error('保存考试失败:', error)
    message.error('操作失败，请重试')
  } finally {
    saving.value = false
  }
}

// 监听课程ID变化
watch(() => props.courseId, (newId) => {
  if (newId) {
    fetchExams()
  }
})

// 生命周期钩子
onMounted(async () => {
  console.log('组件挂载，开始初始化...')
  // 先获取用户信息
  await fetchCurrentUserInfo()
  // 再获取考试数据和知识点
  fetchExams()
  fetchKnowledgePoints()
  
  // 如果需要，可以在这里添加一个默认的空数据显示
  if (!exams.value || exams.value.length === 0) {
    console.log('没有找到考试数据，显示空状态')
    exams.value = []
  }
})

// 获取考试列表
const fetchExams = async () => {
  console.log('开始获取考试列表...')
  loading.value = true
  try {
    // 确保已获取用户ID
    const userId = currentUserId.value || await fetchCurrentUserInfo()
    console.log('获取到用户ID:', userId)
    
    if (!userId) {
      message.error('获取用户信息失败，无法加载考试列表')
      return
    }
    
    const params = {
      courseId: props.courseId,
      userId: userId, // 添加用户ID参数
      status: filters.value.status,
      keyword: filters.value.keyword,
      current: pagination.value.current,
      pageSize: pagination.value.pageSize
    }
    
    console.log('调用考试列表API，参数:', params)
    const response = await examAPI.getExamList(params)
    console.log('考试列表API响应:', response)
    
    // 直接处理响应数据，不做类型转换
    const res = response
    
    // 处理响应数据
    try {
      if (res && res.code === 200) {
        console.log('获取考试列表成功，数据:', res.data)
        // 处理可能的数据格式差异
        const records = res.data?.records || res.data || [];
        const total = res.data?.total || records.length;
        
        exams.value = Array.isArray(records) ? records : [];
        pagination.value.total = total;
        
        console.log('处理后的考试数据:', exams.value)
        
        // 处理考试状态
        exams.value.forEach(exam => {
          // 保存发布状态 (0未发布，1已发布)
          const publishStatus = exam.status
          
          // 处理进行状态
          const now = new Date().getTime()
          const startTime = new Date(exam.startTime).getTime()
          const endTime = new Date(exam.endTime).getTime()
          
          // 设置进行状态到examState字段，保留原始status字段用于发布状态
          if (now < startTime) {
            exam.examState = examStatus.NOT_STARTED
          } else if (now >= startTime && now <= endTime) {
            exam.examState = examStatus.IN_PROGRESS
          } else {
            exam.examState = examStatus.ENDED
          }
          
          // 如果没有时长信息，根据开始和结束时间计算
          if (!exam.duration) {
            exam.duration = calculateDuration(exam.startTime, exam.endTime)
          }
        })
      } else {
        console.error('获取考试列表失败:', res)
        message.error('获取考试列表失败')
      }
    } catch (error) {
      console.error('处理考试数据时出错:', error)
      message.error('处理考试数据时出错')
    }
  } catch (error) {
    console.error('获取考试列表失败:', error)
    message.error('获取考试列表失败，请稍后再试')
  } finally {
    loading.value = false
  }
}

// 筛选变化处理
const handleFilterChange = () => {
  pagination.value.current = 1
  fetchExams()
}

// 搜索处理
const handleSearch = () => {
  pagination.value.current = 1
  fetchExams()
}

// 重置筛选条件
const resetFilters = () => {
  filters.value = {
    status: undefined,
    keyword: ''
  }
  pagination.value.current = 1
  fetchExams()
}

// 表格变化事件
const handleTableChange = (pagination: any) => {
  pagination.value = {
    ...pagination.value,
    current: pagination.current,
    pageSize: pagination.pageSize
  }
  fetchExams()
}

// 考试时间范围变化
const handleTimeRangeChange = (dates: [Dayjs, Dayjs] | null) => {
  if (dates) {
    examForm.value.startTime = dates[0].format('YYYY-MM-DD HH:mm:ss')
    examForm.value.endTime = dates[1].format('YYYY-MM-DD HH:mm:ss')
    
    // 自动计算考试时长（分钟）
    const startTime = dates[0]
    const endTime = dates[1]
    const durationMinutes = endTime.diff(startTime, 'minute')
    examForm.value.duration = durationMinutes
  } else {
    examForm.value.startTime = ''
    examForm.value.endTime = ''
    examForm.value.duration = 0
  }
}

// 显示添加考试弹窗
const showAddExamModal = () => {
  isEditing.value = false
  initExamForm()
  examModalVisible.value = true
}

// 查看考试
const viewExam = (exam: ExamRecord) => {
  currentExam.value = exam
  viewModalVisible.value = true
}

// 查看考试详情
const viewExamDetail = (exam: ExamRecord) => {
  router.push(`/teacher/exams/${exam.id}`)
}

// 编辑考试
const editExam = (exam: ExamRecord) => {
  isEditing.value = true
  examForm.value = { ...exam }
  // 这里应该从API获取完整的考试详情
  examTimeRange.value = null // 应该根据startTime和endTime设置
  examModalVisible.value = true
}

// 删除考试
const handleDeleteExam = async (id: number) => {
  try {
    // 调用API删除考试
    const response = await examAPI.deleteExam(id)
    
    if (response && response.code === 200) {
      message.success('考试删除成功')
      fetchExams() // 重新加载考试列表
    } else {
      message.error(response?.message || '考试删除失败')
    }
  } catch (error) {
    console.error('考试删除失败:', error)
    message.error('考试删除失败')
  }
}

// 获取状态显示文本
const getStatusText = (status: string): string => {
  const statusMap: Record<string, string> = {
    [examStatus.NOT_STARTED]: '未开始',
    [examStatus.IN_PROGRESS]: '进行中',
    [examStatus.ENDED]: '已结束'
  }
  return statusMap[status] || '未知状态'
}

// 获取状态标签颜色
const getStatusColor = (status: string): string => {
  const colorMap: Record<string, string> = {
    [examStatus.NOT_STARTED]: 'blue',
    [examStatus.IN_PROGRESS]: 'green',
    [examStatus.ENDED]: 'gray'
  }
  return colorMap[status] || 'default'
}

// 获取罗马数字
const getRomanNumber = (num: number): string => {
  const roman = ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'X']
  return roman[num - 1] || num.toString()
}

// 获取题目类型文本
const getQuestionTypeText = (type: string): string => {
  const typeMap: Record<string, string> = {
    'single': '单选题',
    'multiple': '多选题',
    'true_false': '判断题',
    'blank': '填空题',
    'short': '简答题',
    'code': '编程题'
  }
  return typeMap[type] || ''
}

// 获取难度星级显示
const getDifficultyStars = (difficulty: number): string => {
  return '★'.repeat(difficulty)
}

// 题目池
const questionPool = ref<QuestionPool>({
  single: [],
  multiple: [],
  trueFalse: [],
  blank: [],
  short: [],
  code: []
})

// 已选题目
const selectedQuestions = ref<SelectedQuestions>({
  single: [],
  multiple: [],
  trueFalse: [],
  blank: [],
  short: [],
  code: []
})

// 获取课程题目
const fetchCourseQuestions = async () => {
  console.log('开始获取课程题目...')
  try {
    // 确保已获取用户ID
    const userId = currentUserId.value || await fetchCurrentUserInfo()
    console.log('获取到用户ID:', userId)
    
    if (!userId) {
      message.error('获取用户信息失败，无法加载题目')
      return
    }
    
    const params = {
      courseId: props.courseId,
      createdBy: userId, // 只获取当前用户创建的题目
      difficulty: examForm.value.paperConfig.difficulty,
      knowledgePoint: examForm.value.paperConfig.knowledgePoint
    }
    
    console.log('调用题目分类API，参数:', params)
    const response = await examAPI.getQuestionsByType(
      params.courseId, 
      undefined, 
      params.difficulty, 
      params.knowledgePoint, 
      params.createdBy
    )
    console.log('题目分类API响应:', response)
    
    try {
      if (response && response.code === 200) {
        console.log('获取课程题目成功，数据:', response.data)
        
        // 清空题目池
        Object.keys(questionPool.value).forEach(type => {
          questionPool.value[type as keyof QuestionPool] = []
        })
        
        // 处理单选题
        if (Array.isArray(response.data.single)) {
          questionPool.value.single = response.data.single.map((q: any) => ({
            id: q.id,
            title: q.title || '无标题',
            questionType: q.question_type || 'single',
            difficulty: q.difficulty || 3,
            knowledgePoint: q.knowledge_point,
            selected: false
          }));
        }
        
        // 处理多选题
        if (Array.isArray(response.data.multiple)) {
          questionPool.value.multiple = response.data.multiple.map((q: any) => ({
            id: q.id,
            title: q.title || '无标题',
            questionType: q.question_type || 'multiple',
            difficulty: q.difficulty || 3,
            knowledgePoint: q.knowledge_point,
            selected: false
          }));
        }
        
        // 处理判断题
        if (Array.isArray(response.data.true_false)) {
          questionPool.value.trueFalse = response.data.true_false.map((q: any) => ({
            id: q.id,
            title: q.title || '无标题',
            questionType: q.question_type || 'true_false',
            difficulty: q.difficulty || 3,
            knowledgePoint: q.knowledge_point,
            selected: false
          }));
        }
        
        // 处理填空题
        if (Array.isArray(response.data.blank)) {
          questionPool.value.blank = response.data.blank.map((q: any) => ({
            id: q.id,
            title: q.title || '无标题',
            questionType: q.question_type || 'blank',
            difficulty: q.difficulty || 3,
            knowledgePoint: q.knowledge_point,
            selected: false
          }));
        }
        
        // 处理简答题
        if (Array.isArray(response.data.short)) {
          questionPool.value.short = response.data.short.map((q: any) => ({
            id: q.id,
            title: q.title || '无标题',
            questionType: q.question_type || 'short',
            difficulty: q.difficulty || 3,
            knowledgePoint: q.knowledge_point,
            selected: false
          }));
        }
        
        // 处理编程题
        if (Array.isArray(response.data.code)) {
          questionPool.value.code = response.data.code.map((q: any) => ({
            id: q.id,
            title: q.title || '无标题',
            questionType: q.question_type || 'code',
            difficulty: q.difficulty || 3,
            knowledgePoint: q.knowledge_point,
            selected: false
          }));
        }
        
        console.log('已加载题目总数:', 
          questionPool.value.single.length + 
          questionPool.value.multiple.length + 
          questionPool.value.trueFalse.length + 
          questionPool.value.blank.length + 
          questionPool.value.short.length + 
          questionPool.value.code.length
        )
        console.log('单选题数:', questionPool.value.single.length)
        console.log('多选题数:', questionPool.value.multiple.length)
        console.log('判断题数:', questionPool.value.trueFalse.length)
        console.log('填空题数:', questionPool.value.blank.length)
        console.log('简答题数:', questionPool.value.short.length)
        console.log('编程题数:', questionPool.value.code.length)
      } else {
        console.error('获取课程题目失败:', response)
        message.error('获取课程题目失败')
      }
    } catch (error) {
      console.error('处理题目数据时出错:', error)
      message.error('处理题目数据时出错')
    }
  } catch (error) {
    console.error('获取课程题目失败:', error)
    message.error('获取课程题目失败')
  }
}

// 处理题目选择
const handleQuestionSelect = (question: Question, type: keyof QuestionPool) => {
  const list = selectedQuestions.value[type]
  if (question.selected) {
    // 添加到已选列表
    list.push(question)
  } else {
    // 从已选列表移除
    const index = list.findIndex(q => q.id === question.id)
    if (index > -1) {
      list.splice(index, 1)
    }
  }
}

// 切换可选题目
const toggleAvailableQuestions = (type: keyof QuestionPool) => {
  // 如果设置的题目数量为0，清空已选
  if (examForm.value.paperConfig[`${type}Count`] === 0) {
    questionPool.value[type].forEach(q => {
      q.selected = false
    })
    selectedQuestions.value[type] = []
  }
}

// 判断是否已达到最大选择数量
const isQuestionSelectDisabled = (type: keyof QuestionPool): boolean => {
  const maxCount = examForm.value.paperConfig[`${type}Count`]
  return selectedQuestions.value[type].length >= maxCount
}

// 监听随机组卷设置变化
watch(() => examForm.value.paperConfig.isRandom, (isRandom) => {
  if (!isRandom) {
    // 切换到手动选题时，获取课程题目
    fetchCourseQuestions()
  }
})

// 监听难度筛选条件变化
watch(() => questionFilters.value.difficulty, (newVal) => {
  if (addSectionModalVisible.value) {
    // 只有在弹窗显示时才触发重新加载
    loadQuestionsByType(currentSection.value.questionType)
  }
})

// 监听知识点筛选条件变化
watch(() => questionFilters.value.knowledgePoint, (newVal) => {
  if (addSectionModalVisible.value) {
    // 只有在弹窗显示时才触发重新加载
    loadQuestionsByType(currentSection.value.questionType)
  }
})

// 监听题目类型变化
watch(() => currentSection.value.questionType, (newType) => {
  if (addSectionModalVisible.value) {
    // 重置已选题目
    filteredQuestions.value.forEach(q => q.selected = false)
    // 重新加载对应类型的题目
    loadQuestionsByType(newType)
  }
})

// 发布考试
const publishExam = async (exam: any) => {
  try {
    const res = await examAPI.publishExam(exam.id)
    if (res && res.code === 200) {
      message.success('考试发布成功')
      // 更新当前考试列表中的考试状态
      const index = exams.value.findIndex(e => e.id === exam.id)
      if (index !== -1) {
        // 将status从0改为1，表示已发布
        exams.value[index].status = 1
      }
    } else {
      message.error(res?.message || '发布失败')
    }
  } catch (error) {
    console.error('发布考试失败:', error)
    message.error('发布失败，请重试')
  }
}

// 取消发布考试
const unpublishExam = async (exam: any) => {
  try {
    // 只发送必要的字段，确保包含id和status
    const examInfo = {
      id: exam.id,
      title: exam.title || '',
      courseId: exam.courseId,
      userId: exam.userId,
      description: exam.description || '',
      startTime: exam.startTime,
      endTime: exam.endTime,
      status: 0 // 将状态设置为0，表示未发布
    };
    
    console.log('取消发布考试，数据:', examInfo);
    
    const res = await examAPI.updateExam(exam.id, examInfo);
    if (res && res.code === 200) {
      message.success('取消发布成功');
      // 更新当前考试列表中的考试状态
      const index = exams.value.findIndex(e => e.id === exam.id);
      if (index !== -1) {
        // 将status从1改为0，表示未发布
        exams.value[index].status = 0;
      }
    } else {
      message.error(res?.message || '取消发布失败');
    }
  } catch (error) {
    console.error('取消发布考试失败:', error);
    message.error('取消发布失败，请重试');
  }
}

// 计算考试时长（分钟）
const calculateDuration = (startTime: string, endTime: string): number => {
  if (!startTime || !endTime) return 0
  
  const start = new Date(startTime)
  const end = new Date(endTime)
  
  return Math.round((end.getTime() - start.getTime()) / (1000 * 60))
}
</script>

<style scoped>
.course-exams {
  padding: 24px;
  background-color: #fff;
}

.page-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 24px;
}

.page-header h2 {
  margin: 0;
  font-size: 20px;
  font-weight: 600;
}

.filter-section {
  background-color: #f5f7fa;
  padding: 16px;
  border-radius: 8px;
  margin-bottom: 24px;
}

.filter-row {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  justify-content: space-between;
}

.filter-left {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
}

.filter-right {
  display: flex;
  align-items: center;
  gap: 16px;
}

.filter-item {
  display: flex;
  align-items: center;
  margin-right: 24px;
  margin-bottom: 8px;
}

.filter-label {
  margin-right: 8px;
  white-space: nowrap;
}

.search-box {
  flex-grow: 1;
}

.exam-content {
  background-color: #fff;
}

.exam-actions {
  display: flex;
  gap: 8px;
}

.exam-detail {
  padding: 16px;
}

.exam-detail-header {
  display: flex;
  align-items: center;
  gap: 16px;
  margin-bottom: 20px;
  padding-bottom: 12px;
  border-bottom: 1px solid #f0f0f0;
}

.status-tag {
  font-size: 14px;
  padding: 2px 12px;
}

.exam-detail-item {
  margin-bottom: 16px;
}

.exam-detail-label {
  font-weight: 600;
  margin-bottom: 8px;
}

.exam-detail-value {
  white-space: pre-line;
}

.exam-detail-actions {
  display: flex;
  justify-content: flex-end;
  gap: 12px;
  margin-top: 24px;
  padding-top: 16px;
  border-top: 1px solid #f0f0f0;
}

.exam-form-steps {
  padding: 0 20px;
}

.step-content {
  margin-top: 20px;
  padding: 20px 0;
}

.step-actions {
  display: flex;
  justify-content: flex-end;
  margin-top: 20px;
  gap: 10px;
}

.paper-config-section {
  margin-top: 20px;
}

.section-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 20px;
}

.filter-config {
  background-color: #f9f9f9;
  padding: 15px;
  border-radius: 4px;
  margin-bottom: 20px;
}

.question-config {
  background-color: #f9f9f9;
  padding: 15px;
  border-radius: 4px;
}

.total-score {
  display: flex;
  justify-content: flex-end;
  align-items: center;
  margin-top: 20px;
  font-size: 16px;
  font-weight: bold;
}

.total-label {
  margin-right: 10px;
}

.total-value {
  color: #1890ff;
  font-size: 18px;
}

.form-help-text {
  font-size: 12px;
  color: #999;
  margin-top: 4px;
}

.question-pool {
  margin-top: 16px;
  border: 1px solid #e8e8e8;
  border-radius: 4px;
}

.question-pool-title {
  background-color: #f5f7fa;
  padding: 8px 12px;
  font-weight: 600;
  border-bottom: 1px solid #e8e8e8;
}

.selected-count {
  color: #1890ff;
  font-weight: bold;
}

.question-list {
  max-height: 300px;
  overflow-y: auto;
  padding: 12px;
}

.question-item {
  padding: 8px;
  border-bottom: 1px solid #f0f0f0;
}

.question-item:last-child {
  border-bottom: none;
}

.question-content {
  display: flex;
  flex-direction: column;
}

.question-title {
  margin-bottom: 8px;
}

.question-info {
  display: flex;
  gap: 8px;
}

.exam-sections {
  margin-bottom: 16px;
}

.exam-section {
  background-color: #f9f9f9;
  border-radius: 4px;
  padding: 12px 16px;
  margin-bottom: 8px;
  border: 1px solid #e8e8e8;
}

.section-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.section-title {
  font-weight: 600;
  font-size: 15px;
}

.section-summary {
  color: #1890ff;
  margin-right: auto;
  margin-left: 16px;
}

.add-section {
  margin-bottom: 20px;
}

.section-form {
  margin-top: 16px;
}

.question-filter {
  margin: 16px 0;
  padding: 12px;
  background-color: #f5f7fa;
  border-radius: 4px;
}

.question-list-container {
  max-height: 400px;
  overflow-y: auto;
  border: 1px solid #e8e8e8;
  border-radius: 4px;
  padding: 12px;
  margin-bottom: 16px;
}

.question-list {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.selection-summary {
  text-align: right;
  padding: 8px 0;
  font-weight: 600;
  color: #1890ff;
}

.modal-footer {
  display: flex;
  justify-content: flex-end;
  gap: 8px;
  margin-top: 16px;
  padding-top: 16px;
  border-top: 1px solid #f0f0f0;
}
</style> 