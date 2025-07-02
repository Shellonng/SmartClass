<template>
  <div class="question-bank-management">
    <div class="question-bank-header">
      <h2>{{ currentCourseId > 0 ? '课程题库' : '题库管理' }}</h2>
      <a-button type="primary" @click="showAddQuestionModal">
        <PlusOutlined />
        添加题目
      </a-button>
    </div>

    <!-- 分类筛选栏 -->
    <div class="filter-section">
      <div class="filter-row">
        <div class="filter-left">
          <div class="filter-item">
            <span class="filter-label">题目类型：</span>
            <a-select 
              v-model:value="filters.questionType" 
              style="width: 120px" 
              placeholder="全部类型"
              allowClear
              @change="handleFilterChange"
            >
              <a-select-option v-for="(desc, type) in QuestionTypeDesc" :key="type" :value="type">{{ desc }}</a-select-option>
            </a-select>
          </div>
          <div class="filter-item">
            <span class="filter-label">难度等级：</span>
            <a-select 
              v-model:value="filters.difficulty" 
              style="width: 120px" 
              placeholder="全部难度"
              allowClear
              @change="handleFilterChange"
            >
              <a-select-option v-for="level in DifficultyLevels" :key="level.value" :value="level.value">{{ level.label }}</a-select-option>
            </a-select>
          </div>
          <div class="filter-item">
            <span class="filter-label">知识点：</span>
            <a-auto-complete
              v-model:value="filters.knowledgePoint" 
              style="width: 180px" 
              placeholder="输入知识点"
              :options="filteredKnowledgePoints.map(point => ({ value: point }))"
              @search="handleKnowledgePointSearch"
              @select="handleKnowledgePointSelect"
              allowClear
            >
              <template #option="{ value: point }">
                <div class="auto-complete-option">
                  {{ point }}
                </div>
              </template>
            </a-auto-complete>
          </div>
        </div>
        <div class="filter-right">
          <div class="filter-item search-box">
            <a-auto-complete
              v-model:value="filters.keyword"
              style="width: 250px"
              :options="filteredTitleSuggestions.map(title => ({ value: title }))"
              @search="handleTitleSearch"
              @select="handleTitleSelect"
            >
              <template #default>
            <a-input-search
              v-model:value="filters.keyword"
              placeholder="搜索题目内容"
              @search="handleSearch"
              enter-button
            />
              </template>
              <template #option="{ value: title }">
                <div class="auto-complete-option">
                  {{ title }}
                </div>
              </template>
            </a-auto-complete>
          </div>
          <a-button @click="resetFilters" size="middle">
            重置筛选
          </a-button>
        </div>
      </div>
    </div>

    <div class="question-bank-content">
      <a-spin :spinning="loading">
        <a-empty v-if="questions.length === 0" description="暂无题目" />
        
        <a-table
          v-else
          :dataSource="questions"
          :columns="columns"
          :pagination="pagination"
          :rowKey="(record) => record.id"
          @change="handleTableChange"
        >
          <!-- 题目内容 -->
          <template #bodyCell="{ column, record }: { column: any, record: Question }">
            <template v-if="column.dataIndex === 'title'">
              <div class="question-content">
                <span>{{ truncateText(record.title, 50) }}</span>
              </div>
            </template>

            <!-- 题目类型 -->
            <template v-else-if="column.dataIndex === 'questionType'">
              <a-tag :color="getQuestionTypeColor(record.questionType)">{{ record.questionTypeDesc }}</a-tag>
            </template>

            <!-- 难度 -->
            <template v-else-if="column.dataIndex === 'difficulty'">
              <a-rate 
                :value="record.difficulty" 
                disabled 
                :count="5"
                style="font-size: 12px"
              />
            </template>

            <!-- 知识点 -->
            <template v-else-if="column.dataIndex === 'knowledgePoint'">
              <a-tag color="cyan">{{ record.knowledgePoint || '未分类' }}</a-tag>
            </template>

            <!-- 创建时间 -->
            <template v-else-if="column.dataIndex === 'createTime'">
              {{ formatDate(record.createTime) }}
            </template>

            <!-- 操作 -->
            <template v-else-if="column.dataIndex === 'action'">
              <div class="question-actions">
                <a-tooltip title="查看">
                  <a-button type="link" @click="viewQuestion(record)">
                    <EyeOutlined />
                  </a-button>
                </a-tooltip>
                <a-tooltip title="编辑">
                  <a-button type="link" @click="editQuestion(record)">
                    <EditOutlined />
                  </a-button>
                </a-tooltip>
                <a-tooltip title="删除">
                  <a-popconfirm
                    title="确定要删除这个题目吗？"
                    @confirm="handleDeleteQuestion(record.id)"
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

    <!-- 添加/编辑题目弹窗 -->
    <a-modal
      v-model:open="questionModalVisible"
      :title="isEditing ? '编辑题目' : '添加题目'"
      :maskClosable="false"
      @ok="handleSaveQuestion"
      :okButtonProps="{ loading: saving }"
      :okText="saving ? '保存中...' : '保存'"
      width="700px"
    >
      <a-form :model="questionForm" layout="vertical">
        <a-form-item label="题目内容" required>
          <a-textarea v-model:value="questionForm.title" placeholder="请输入题目内容" :rows="4" />
        </a-form-item>
        <a-form-item label="题目类型" required>
          <a-select v-model:value="questionForm.questionType" placeholder="请选择题目类型" @change="handleQuestionTypeChange">
            <a-select-option v-for="(desc, type) in QuestionTypeDesc" :key="type" :value="type">{{ desc }}</a-select-option>
          </a-select>
        </a-form-item>
        <a-form-item label="难度" required>
          <a-rate v-model:value="questionForm.difficulty" :count="5" />
        </a-form-item>
        <a-form-item label="知识点">
          <a-input v-model:value="questionForm.knowledgePoint" placeholder="请输入知识点" allowClear />
        </a-form-item>
        
        <!-- 选择题选项 -->
        <template v-if="['single', 'multiple', 'true_false'].includes(questionForm.questionType)">
          <a-divider>选项</a-divider>
          <a-form-item v-for="(option, index) in questionForm.options" :key="index">
            <div style="display: flex; align-items: center; gap: 8px;">
              <a-input v-model:value="option.optionLabel" style="width: 60px;" placeholder="A/B/C" />
              <a-input v-model:value="option.optionText" placeholder="选项内容" />
              <a-button type="text" danger @click="removeOption(index)">
                <DeleteOutlined />
              </a-button>
            </div>
          </a-form-item>
          <a-button type="dashed" block @click="addOption">
            <PlusOutlined /> 添加选项
          </a-button>
        </template>
        
        <a-form-item label="标准答案" required>
          <!-- 单选题答案 -->
          <div v-if="questionForm.questionType === 'single' && questionForm.options && questionForm.options.length > 0">
            <a-radio-group v-model:value="questionForm.correctAnswer">
              <a-radio v-for="option in questionForm.options" :key="option.optionLabel" :value="option.optionLabel">
                {{ option.optionLabel }}. {{ option.optionText }}
              </a-radio>
            </a-radio-group>
            <div v-if="questionForm.options.length === 0" class="answer-tip">
              请先添加选项
            </div>
          </div>
          
          <!-- 多选题答案 -->
          <div v-else-if="questionForm.questionType === 'multiple' && questionForm.options && questionForm.options.length > 0">
            <a-checkbox-group v-model:value="multipleAnswers" @change="handleMultipleAnswerChange">
              <a-checkbox v-for="option in questionForm.options" :key="option.optionLabel" :value="option.optionLabel">
                {{ option.optionLabel }}. {{ option.optionText }}
              </a-checkbox>
            </a-checkbox-group>
            <div v-if="questionForm.options.length === 0" class="answer-tip">
              请先添加选项
            </div>
          </div>
          
          <!-- 判断题答案 -->
          <div v-else-if="questionForm.questionType === 'true_false'">
            <a-radio-group v-model:value="questionForm.correctAnswer">
              <a-radio value="T">正确</a-radio>
              <a-radio value="F">错误</a-radio>
            </a-radio-group>
          </div>
          
          <!-- 其他题型答案 -->
          <a-textarea v-else v-model:value="questionForm.correctAnswer" placeholder="请输入标准答案" :rows="3" />
        </a-form-item>
        <a-form-item label="答案解析">
          <a-textarea v-model:value="questionForm.explanation" placeholder="请输入答案解析" :rows="3" />
        </a-form-item>
        
        <a-form-item v-if="currentCourseId > 0" label="章节">
          <a-select v-model:value="questionForm.chapterId" placeholder="请选择章节">
            <a-select-option v-for="chapter in chapters" :key="chapter.id" :value="chapter.id">{{ chapter.title }}</a-select-option>
          </a-select>
        </a-form-item>
      </a-form>
    </a-modal>

    <!-- 查看题目弹窗 -->
    <a-modal
      v-model:open="viewModalVisible"
      title="题目详情"
      :footer="null"
      width="700px"
    >
      <div v-if="currentQuestion" class="question-detail">
        <!-- 题目类型和难度水平排列 -->
        <div class="question-detail-header">
          <a-tag :color="getQuestionTypeColor(currentQuestion.questionType)" class="question-type-tag">
            {{ (currentQuestion.questionTypeDesc || QuestionTypeDesc[currentQuestion.questionType] || '') }}
          </a-tag>
          <a-rate :value="currentQuestion.difficulty || 1" disabled :count="5" class="difficulty-stars" />
        </div>
        
        <div class="question-detail-item">
          <div class="question-detail-label">题目内容：</div>
          <div class="question-detail-value">{{ currentQuestion.title || '未设置' }}</div>
        </div>
        
        <!-- 选项 -->
        <div class="question-detail-item" v-if="currentQuestion.options && currentQuestion.options.length > 0">
          <div class="question-detail-label">选项：</div>
          <div class="question-detail-value">
            <div v-for="option in currentQuestion.options" :key="option.id || option.optionLabel" class="question-option">
              {{ option.optionLabel }}. {{ option.optionText }}
          </div>
        </div>
          </div>
        
        <div class="question-detail-item" v-if="currentQuestion.knowledgePoint">
          <div class="question-detail-label">知识点：</div>
          <div class="question-detail-value">
            <a-tag color="cyan">{{ currentQuestion.knowledgePoint }}</a-tag>
          </div>
        </div>
        
        <div class="question-detail-item">
          <div class="question-detail-label">标准答案：</div>
          <div class="question-detail-value">{{ currentQuestion.correctAnswer || '未设置' }}</div>
        </div>
        <div class="question-detail-item" v-if="currentQuestion.explanation">
          <div class="question-detail-label">答案解析：</div>
          <div class="question-detail-value">{{ currentQuestion.explanation }}</div>
        </div>
        <div class="question-detail-item">
          <div class="question-detail-label">创建时间：</div>
          <div class="question-detail-value">{{ formatDate(currentQuestion.createTime) }}</div>
        </div>
      </div>
    </a-modal>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted, computed, watch } from 'vue'
import { useRoute } from 'vue-router'
import { message } from 'ant-design-vue'
import {
  PlusOutlined,
  EyeOutlined,
  EditOutlined,
  DeleteOutlined
} from '@ant-design/icons-vue'
import { formatDate } from '@/utils/date'
import {
  QuestionType,
  QuestionTypeDesc,
  DifficultyLevels,
  Question,
  QuestionOption,
  PageResponse,
  addQuestion,
  updateQuestion,
  deleteQuestion,
  getQuestionDetail,
  getQuestionPage,
  getQuestionsByCourse
} from '@/api/question'
import axios from 'axios'

// 定义组件属性
const props = defineProps<{
  courseId?: number
}>()

// 状态定义
const questions = ref<Question[]>([])
const loading = ref(false)
const pagination = ref({
  current: 1,
  pageSize: 10,
  total: 0,
  showSizeChanger: true,
  showTotal: (total: number) => `共 ${total} 条`
})

// 多选题答案处理
const multipleAnswers = ref<string[]>([])

// 多选题答案变化处理
const handleMultipleAnswerChange = (checkedValues: string[]) => {
  // 将选中的选项标签转换为逗号分隔的字符串作为答案
  questionForm.value.correctAnswer = checkedValues.join(',')
}

// 获取当前课程ID
const currentCourseId = computed(() => {
  return props.courseId || -1
})

// 章节列表
const chapters = ref<{ id: number, title: string }[]>([])

// 自动补全数据列表
const knowledgePoints = ref<string[]>([]) // 知识点列表
const titleSuggestions = ref<string[]>([]) // 题目内容建议列表
const filteredKnowledgePoints = ref<string[]>([]) // 过滤后的知识点列表
const filteredTitleSuggestions = ref<string[]>([]) // 过滤后的题目内容建议列表

// 从题目列表中提取知识点和题目内容
const extractSuggestionsFromQuestions = (questionsList: Question[]) => {
  // 提取知识点
  const pointsSet = new Set<string>()
  // 提取题目内容（取前50个字符作为建议）
  const titlesSet = new Set<string>()

  questionsList.forEach(question => {
    // 提取知识点
    if (question.knowledgePoint) {
      pointsSet.add(question.knowledgePoint)
    }
    // 提取题目内容（对长内容做截断）
    if (question.title) {
      const titleSuggestion = question.title.length > 50 
        ? question.title.substring(0, 50) + '...'
        : question.title
      titlesSet.add(titleSuggestion)
    }
  })

  // 更新建议列表
  knowledgePoints.value = Array.from(pointsSet)
  titleSuggestions.value = Array.from(titlesSet)
}

// 筛选条件
const filters = ref({
  questionType: undefined as string | undefined,
  difficulty: undefined as number | undefined,
  knowledgePoint: undefined as string | undefined,
  keyword: ''
})

// 表格列定义
const columns = [
  {
    title: '题目内容',
    dataIndex: 'title',
    key: 'title',
    ellipsis: true,
    width: '30%'
  },
  {
    title: '题目类型',
    dataIndex: 'questionType',
    key: 'questionType',
    width: '12%'
  },
  {
    title: '难度',
    dataIndex: 'difficulty',
    key: 'difficulty',
    width: '12%'
  },
  {
    title: '知识点',
    dataIndex: 'knowledgePoint',
    key: 'knowledgePoint',
    width: '12%'
  },
  {
    title: '创建时间',
    dataIndex: 'createTime',
    key: 'createTime',
    width: '15%'
  },
  {
    title: '操作',
    dataIndex: 'action',
    key: 'action',
    width: '15%'
  }
]

// 添加/编辑题目相关状态
const questionModalVisible = ref(false)
const isEditing = ref(false)
const saving = ref(false)
const questionForm = ref<Question & { options: QuestionOption[] }>({
  id: undefined,
  title: '',
  questionType: QuestionType.SINGLE,
  difficulty: 3,
  knowledgePoint: '',
  correctAnswer: '',
  explanation: '',
  courseId: currentCourseId.value,
  chapterId: undefined as unknown as number,
  options: []
})

// 查看题目相关状态
const viewModalVisible = ref(false)
const currentQuestion = ref<Question | null>(null)

// 生命周期钩子
onMounted(() => {
  console.log('QuestionBank组件已加载，课程ID:', currentCourseId.value)
  fetchQuestions()
  if (currentCourseId.value > 0) {
    fetchChapters()
  }
})

// 监听课程ID变化
watch(() => props.courseId, (newVal) => {
  if (newVal) {
    questionForm.value.courseId = newVal
    fetchQuestions()
    fetchChapters()
  }
})

// 获取题目列表
const fetchQuestions = async () => {
  loading.value = true
  try {
    const params = {
      pageNum: pagination.value.current,
      pageSize: pagination.value.pageSize,
      courseId: currentCourseId.value > 0 ? currentCourseId.value : undefined,
      questionType: filters.value.questionType,
      difficulty: filters.value.difficulty,
      knowledgePoint: filters.value.knowledgePoint,
      keyword: filters.value.keyword || undefined
    }

    if (currentCourseId.value > 0) {
      // 如果有课程ID，使用课程相关API
      const res = await getQuestionsByCourse(currentCourseId.value)
      
      // 提取所有题目数据用于自动补全建议
      extractSuggestionsFromQuestions(res.data || [])
      
      // 在前端进行筛选，确保同时满足所有筛选条件
      let filteredQuestions = res.data || []
      
      // 按题目类型筛选
      if (filters.value.questionType) {
        filteredQuestions = filteredQuestions.filter(q => 
          q.questionType === filters.value.questionType
        )
      }
      
      // 按难度筛选
      if (filters.value.difficulty) {
        filteredQuestions = filteredQuestions.filter(q => 
          q.difficulty === filters.value.difficulty
        )
      }
      
      // 按知识点筛选
      if (filters.value.knowledgePoint) {
        filteredQuestions = filteredQuestions.filter(q => 
          q.knowledgePoint && q.knowledgePoint.toLowerCase().includes(filters.value.knowledgePoint!.toLowerCase())
        )
      }
      
      // 按关键词筛选题目内容
      if (filters.value.keyword) {
        filteredQuestions = filteredQuestions.filter(q => 
          q.title && q.title.toLowerCase().includes(filters.value.keyword.toLowerCase())
        )
      }
      
      // 保存完整的筛选结果，用于前端分页
      allFilteredQuestions.value = filteredQuestions
      
      // 计算当前页的数据
      const start = (pagination.value.current - 1) * pagination.value.pageSize
      const end = start + pagination.value.pageSize
      questions.value = filteredQuestions.slice(start, end)
      
      pagination.value.total = filteredQuestions.length
    } else {
      // 否则使用分页查询
      const res = await getQuestionPage(params)
      if (res.data && res.data.records) {
        questions.value = res.data.records
        pagination.value.total = res.data.total
        
        // 提取建议列表用于自动补全
        extractSuggestionsFromQuestions(res.data.records)
      } else {
        questions.value = []
        pagination.value.total = 0
      }
    }
  } catch (error) {
    console.error('获取题目列表失败:', error)
    // 设置为空数组，防止界面卡死
    questions.value = []
    pagination.value.total = 0
    message.error('获取题目列表失败，请稍后再试')
  } finally {
    loading.value = false
  }
}

// 获取章节列表
const fetchChapters = async () => {
  try {
    if (currentCourseId.value > 0) {
      // 从API获取课程下的章节
      const res = await axios.get(`/teacher/chapters/course/${currentCourseId.value}`);
      if (res.data && res.data.code === 200 && Array.isArray(res.data.data)) {
        chapters.value = res.data.data;
        console.log('获取到章节列表:', chapters.value);
        
        // 如果有章节，默认选择第一个
        if (chapters.value.length > 0 && questionForm.value) {
          questionForm.value.chapterId = chapters.value[0].id;
        } else {
          // 如果没有章节，显示提示
          message.warning('当前课程没有章节，请先添加章节再添加题目');
        }
      } else {
        chapters.value = [];
        message.warning('获取章节列表失败，请确保课程已添加章节');
      }
    }
  } catch (error) {
    console.error('获取章节列表失败:', error);
    chapters.value = [];
    message.error('获取章节列表失败，请稍后再试');
  }
}

// 筛选变化处理
const handleFilterChange = () => {
  pagination.value.current = 1
  fetchQuestions()
}

// 搜索处理
const handleSearch = () => {
  pagination.value.current = 1
  fetchQuestions()
}

// 重置筛选条件
const resetFilters = () => {
  filters.value = {
    questionType: undefined,
    difficulty: undefined,
    knowledgePoint: undefined,
    keyword: ''
  }
  // 重置过滤列表
  filteredKnowledgePoints.value = []
  filteredTitleSuggestions.value = []
  allFilteredQuestions.value = []
  pagination.value.current = 1
  fetchQuestions()
}

// 处理知识点输入变化
const handleKnowledgePointSearch = (value: string) => {
  if (!value) {
    filteredKnowledgePoints.value = []
    return
  }
  // 模糊匹配
  filteredKnowledgePoints.value = knowledgePoints.value
    .filter(point => point.toLowerCase().includes(value.toLowerCase()))
}

// 处理题目内容输入变化
const handleTitleSearch = (value: string) => {
  if (!value) {
    filteredTitleSuggestions.value = []
    return
  }
  // 模糊匹配
  filteredTitleSuggestions.value = titleSuggestions.value
    .filter(title => title.toLowerCase().includes(value.toLowerCase()))
}

// 处理知识点选择
const handleKnowledgePointSelect = (value: string) => {
  filters.value.knowledgePoint = value
  handleFilterChange()
}

// 处理题目内容选择
const handleTitleSelect = (value: string) => {
  // 处理末尾的"..."
  filters.value.keyword = value.endsWith('...') ? value.substring(0, value.length - 3) : value
  handleFilterChange()
}

// 显示添加题目弹窗
const showAddQuestionModal = () => {
  isEditing.value = false
  questionForm.value = {
    id: undefined,
    title: '',
    questionType: QuestionType.SINGLE,
    difficulty: 3,
    knowledgePoint: '',
    correctAnswer: '',
    explanation: '',
    courseId: currentCourseId.value,
    chapterId: chapters.value.length > 0 ? chapters.value[0].id : undefined as unknown as number,
    options: []
  }
  // 如果是判断题，自动添加两个选项
  if (questionForm.value.questionType === QuestionType.TRUE_FALSE) {
    questionForm.value.options = [
      { optionLabel: 'T', optionText: '正确' },
      { optionLabel: 'F', optionText: '错误' }
    ]
    // 默认选择"正确"
    questionForm.value.correctAnswer = 'T'
  }
  questionModalVisible.value = true
}

// 查看题目
const viewQuestion = async (question: Question) => {
  try {
    console.log('查看题目ID:', question.id)
    const res = await getQuestionDetail(question.id!)
    if (res.data && res.data.code === 200 && res.data.data) {
      currentQuestion.value = res.data.data
      console.log('获取到题目详情:', currentQuestion.value)
    } else {
      // 如果API没有返回详细数据，先使用当前行数据
      currentQuestion.value = question
      console.log('使用当前行数据作为详情:', currentQuestion.value)
    }
    viewModalVisible.value = true
  } catch (error) {
    console.error('获取题目详情失败:', error)
    message.error('获取题目详情失败')
    // 发生错误时也使用当前行数据
    currentQuestion.value = question
    viewModalVisible.value = true
  }
}

// 编辑题目
const editQuestion = async (question: Question) => {
  try {
    console.log('编辑题目ID:', question.id)
    const res = await getQuestionDetail(question.id!)
    
    // 设置编辑状态
    isEditing.value = true
    
    if (res.data && res.data.code === 200 && res.data.data) {
      // 如果API返回了详细数据
      const detailData = res.data.data
      console.log('获取到题目详情用于编辑:', detailData)
      
    questionForm.value = {
      ...detailData,
        options: detailData.options || [],
        images: detailData.images || []
    }
      
      // 处理多选题答案
      if (detailData.questionType === QuestionType.MULTIPLE && detailData.correctAnswer) {
        multipleAnswers.value = detailData.correctAnswer.split(',')
      }
    } else {
      // 如果API没有返回详细数据，使用当前行数据
      console.log('使用当前行数据用于编辑:', question)
      questionForm.value = {
        ...question,
        options: question.options || [],
        images: question.images || []
      }
      
      // 处理多选题答案
      if (question.questionType === QuestionType.MULTIPLE && question.correctAnswer) {
        multipleAnswers.value = question.correctAnswer.split(',')
      }
    }
    
    // 显示编辑弹窗
    questionModalVisible.value = true
  } catch (error) {
    console.error('获取题目详情失败:', error)
    message.error('获取题目详情失败')
    
    // 即使出错也尝试使用当前行数据进行编辑
    questionForm.value = {
      ...question,
      options: question.options || [],
      images: question.images || []
    }
    questionModalVisible.value = true
  }
}

// 删除题目
const handleDeleteQuestion = async (id: number) => {
  try {
    await deleteQuestion(id)
    message.success('题目删除成功')
    fetchQuestions()
  } catch (error) {
    console.error('题目删除失败:', error)
    message.error('题目删除失败')
  }
}

// 处理题目类型变化
const handleQuestionTypeChange = (type: string) => {
  console.log('题目类型变化:', type)
  
  // 重置答案
  questionForm.value.correctAnswer = ''
  multipleAnswers.value = []
  
  // 判断题特殊处理
  if (type === QuestionType.TRUE_FALSE) {
    // 判断题固定选项：正确和错误
    questionForm.value.options = [
      { optionLabel: 'T', optionText: '正确' },
      { optionLabel: 'F', optionText: '错误' }
    ]
    // 默认选择"正确"
    questionForm.value.correctAnswer = 'T'
  } else if (type === QuestionType.SINGLE || type === QuestionType.MULTIPLE) {
    // 其他有选项的题型，如果没有选项则默认添加一个空选项
    if (!questionForm.value.options?.length) {
      questionForm.value.options = []
      addOption()
    }
  }
}

// 添加选项
const addOption = () => {
  // 判断题不允许添加选项
  if (questionForm.value.questionType === QuestionType.TRUE_FALSE) {
    return
  }
  
  const label = String.fromCharCode(65 + questionForm.value.options.length) // A, B, C...
  questionForm.value.options.push({
    optionLabel: label,
    optionText: ''
  })
}

// 删除选项
const removeOption = (index: number) => {
  questionForm.value.options.splice(index, 1)
  // 重新排序选项标签
  questionForm.value.options.forEach((option, idx) => {
    option.optionLabel = String.fromCharCode(65 + idx)
  })
}

// 保存题目
const handleSaveQuestion = async () => {
  // 表单验证
  if (!questionForm.value.title) {
    message.error('请输入题目内容')
    return
  }
  if (!questionForm.value.correctAnswer) {
    // 判断题型
    if (questionForm.value.questionType === QuestionType.MULTIPLE && multipleAnswers.value.length === 0) {
      message.error('请选择正确选项')
      return
    } else if (!questionForm.value.correctAnswer) {
    message.error('请输入标准答案')
    return
    }
  }
  
  // 验证章节ID是否有效
  if (currentCourseId.value > 0) {
    if (!questionForm.value.chapterId) {
      message.error('请选择章节')
      return
    }
    
    // 检查章节是否存在于章节列表中
    const chapterExists = chapters.value.some(chapter => chapter.id === questionForm.value.chapterId)
    if (!chapterExists) {
      message.error('所选章节不存在，请重新选择')
      return
    }
  }
  
  if (['single', 'multiple', 'true_false'].includes(questionForm.value.questionType) && 
      questionForm.value.options.length === 0) {
    message.error('请添加选项')
    return
  }

  saving.value = true
  try {
    const formData = { ...questionForm.value }
    
    if (isEditing.value) {
      // 编辑现有题目
      const res = await updateQuestion(formData)
      if (res.data && res.data.code === 200) {
      message.success('题目更新成功')
        // 更新本地列表中的题目
        const updatedIndex = questions.value.findIndex(q => q.id === formData.id)
        if (updatedIndex !== -1) {
          questions.value[updatedIndex] = res.data.data || formData
        }
      } else {
        message.error('题目更新失败: ' + (res.data?.message || '未知错误'))
        saving.value = false
        return
      }
    } else {
      // 添加新题目
      const res = await addQuestion(formData)
      if (res.data && res.data.code === 200 && res.data.data && res.data.data.id) {
        message.success('题目添加成功')
        // 将新添加的题目添加到当前列表中
        const newQuestion = res.data.data;
        questions.value = [newQuestion, ...questions.value]
      } else {
        message.error('题目添加失败，请检查表单数据')
        saving.value = false
        return
      }
    }
    questionModalVisible.value = false
    fetchQuestions()
  } catch (error: any) { // 使用any类型避免TypeScript错误
    console.error('保存题目失败:', error)
    if (error?.response?.data?.message) {
      message.error(`保存题目失败: ${error.response.data.message}`)
    } else {
      message.error('保存题目失败，请检查网络连接')
    }
  } finally {
    saving.value = false
  }
}

// 用于存储未经分页的完整筛选结果
const allFilteredQuestions = ref<Question[]>([])

// 表格变化事件
const handleTableChange = (pag: any) => {
  pagination.value.current = pag.current
  pagination.value.pageSize = pag.pageSize
  
  if (currentCourseId.value > 0 && allFilteredQuestions.value.length > 0) {
    // 如果是课程ID查询模式，前端处理分页
    const start = (pagination.value.current - 1) * pagination.value.pageSize
    const end = start + pagination.value.pageSize
    questions.value = allFilteredQuestions.value.slice(start, end)
  } else {
    // 否则调用后端分页
  fetchQuestions()
  }
}

// 工具函数
const truncateText = (text: string, maxLength: number) => {
  if (!text) return ''
  if (text.length <= maxLength) return text
  return text.substring(0, maxLength) + '...'
}

const getQuestionTypeColor = (type: string) => {
  const colorMap: Record<string, string> = {
    [QuestionType.SINGLE]: 'blue',
    [QuestionType.MULTIPLE]: 'purple',
    [QuestionType.TRUE_FALSE]: 'green',
    [QuestionType.BLANK]: 'orange',
    [QuestionType.SHORT]: 'red',
    [QuestionType.CODE]: 'geekblue'
  }
  return colorMap[type] || 'default'
}
</script>

<style scoped>
.question-bank-management {
  padding: 24px;
  background-color: #fff;
}

.question-bank-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 24px;
}

.question-bank-header h2 {
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
  margin-bottom: 16px;
}

.filter-row:last-child {
  margin-bottom: 0;
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

.filter-actions {
  display: flex;
  align-items: center;
  gap: 8px;
}

.search-box {
  flex-grow: 1;
}

.question-bank-content {
  background-color: #fff;
}

.question-content {
  white-space: pre-line;
}

.question-actions {
  display: flex;
  gap: 8px;
}

.question-detail {
  padding: 16px;
}

.question-detail-header {
  display: flex;
  align-items: center;
  gap: 16px;
  margin-bottom: 20px;
  padding-bottom: 12px;
  border-bottom: 1px solid #f0f0f0;
}

.question-type-tag {
  font-size: 14px;
  padding: 2px 12px;
}

.difficulty-stars {
  margin-left: auto;
  font-size: 16px;
}

.question-detail-item {
  margin-bottom: 16px;
}

.question-detail-label {
  font-weight: 600;
  margin-bottom: 8px;
}

.question-detail-value {
  white-space: pre-line;
}

.question-option {
  margin-bottom: 8px;
}

.answer-tip {
  color: #ff4d4f;
  font-size: 12px;
  margin-top: 8px;
}

.ant-radio-group, .ant-checkbox-group {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

/* 自动补全下拉项样式 */
.auto-complete-option {
  padding: 6px;
  cursor: pointer;
  transition: background-color 0.3s;
}

.auto-complete-option:hover {
  background-color: #f5f5f5;
}

.highlight-text {
  color: #1890ff;
  font-weight: bold;
}

/* 提供足够的下拉高度 */
.ant-select-dropdown {
  max-height: 300px;
}
</style> 