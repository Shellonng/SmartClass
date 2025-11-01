<template>
  <div class="personalized-practice">
    <div class="page-header">
      <h1>🎯 个性化练习</h1>
      <p class="description">AI为您定制专属练习题，针对性提升学习效果</p>
    </div>

    <div class="practice-container">
      <!-- 练习配置区域 -->
      <div class="config-section">
        <a-row :gutter="24">
          <!-- 练习类型选择 -->
          <a-col :span="8">
            <a-card title="📚 练习类型" class="type-card">
              <div class="practice-types">
                <div 
                  v-for="type in practiceTypes" 
                  :key="type.key"
                  :class="['type-item', { active: selectedType === type.key }]"
                  @click="selectPracticeType(type.key)"
                >
                  <div class="type-icon">{{ type.icon }}</div>
                  <div class="type-info">
                    <h4>{{ type.title }}</h4>
                    <p>{{ type.description }}</p>
                  </div>
                </div>
              </div>
            </a-card>
          </a-col>

          <!-- 配置参数 -->
          <a-col :span="16">
            <a-card title="⚙️ 练习配置" class="config-card">
              <a-form :model="practiceConfig" :label-col="{ span: 6 }" :wrapper-col="{ span: 18 }">
                <!-- 课程选择 -->
                <a-form-item label="选择课程" name="courseId" :rules="[{ required: true, message: '请选择课程' }]">
                  <a-select v-model:value="practiceConfig.courseId" placeholder="请选择课程" @change="handleCourseChange">
                    <a-select-option v-for="course in courses" :key="course.id" :value="course.id">
                      {{ course.name }}
                    </a-select-option>
                  </a-select>
                </a-form-item>

                <!-- 薄弱知识点选择 -->
                <a-form-item v-if="selectedType === 'personalized'" label="薄弱知识点">
                  <a-select 
                    v-model:value="practiceConfig.weakKnowledgePoints" 
                    mode="multiple"
                    placeholder="AI已为您分析出薄弱知识点"
                    :options="weakKnowledgeOptions"
                  />
                </a-form-item>

                <!-- 错题知识点选择 -->
                <a-form-item v-if="selectedType === 'retry'" label="错题知识点">
                  <a-select 
                    v-model:value="practiceConfig.errorKnowledgePoints" 
                    mode="multiple"
                    placeholder="基于历史错题分析"
                    :options="errorKnowledgeOptions"
                  />
                </a-form-item>

                <!-- 能力水平 -->
                <a-form-item v-if="selectedType === 'personalized'" label="当前水平">
                  <a-radio-group v-model:value="practiceConfig.abilityLevel">
                    <a-radio value="LOW">基础水平</a-radio>
                    <a-radio value="MEDIUM">中等水平</a-radio>
                    <a-radio value="HIGH">高级水平</a-radio>
                  </a-radio-group>
                </a-form-item>

                <!-- 题目数量 -->
                <a-form-item label="题目数量">
                  <a-slider 
                    v-model:value="practiceConfig.questionCount" 
                    :min="5" 
                    :max="30" 
                    :marks="{ 5: '5题', 15: '15题', 30: '30题' }"
                  />
                </a-form-item>

                <!-- 偏好题型 -->
                <a-form-item label="偏好题型">
                  <div class="question-types">
                    <a-checkbox-group v-model:value="selectedQuestionTypes">
                      <a-checkbox value="SINGLE_CHOICE">单选题</a-checkbox>
                      <a-checkbox value="MULTIPLE_CHOICE">多选题</a-checkbox>
                      <a-checkbox value="TRUE_FALSE">判断题</a-checkbox>
                      <a-checkbox value="FILL_BLANK">填空题</a-checkbox>
                    </a-checkbox-group>
                  </div>
                </a-form-item>

                <!-- 操作按钮 -->
                <a-form-item :wrapper-col="{ offset: 6, span: 18 }">
                  <a-space>
                    <a-button type="primary" @click="handleGeneratePractice" :loading="generating">
                      <ThunderboltOutlined />
                      生成练习
                    </a-button>
                    <a-button @click="handleQuickRecommend" :loading="recommending">
                      <StarOutlined />
                      智能推荐
                    </a-button>
                  </a-space>
                </a-form-item>
              </a-form>
            </a-card>
          </a-col>
        </a-row>
      </div>

      <!-- 练习题展示区域 -->
      <div v-if="practiceQuestions.length > 0" class="practice-section">
        <a-card title="📝 练习题目" class="practice-card">
          <template #extra>
            <a-space>
              <span class="practice-info">
                共{{ practiceQuestions.length }}题 | 预计用时{{ estimatedTime }}分钟
              </span>
              <a-button @click="handleStartPractice" type="primary">
                开始练习
              </a-button>
            </a-space>
          </template>

          <div class="questions-preview">
            <div v-for="(question, index) in practiceQuestions" :key="index" class="question-preview">
              <div class="question-header">
                <span class="question-number">{{ index + 1 }}.</span>
                <a-tag :color="getDifficultyColor(question.difficulty)">{{ question.difficulty }}</a-tag>
                <a-tag color="blue">{{ getQuestionTypeLabel(question.questionType) }}</a-tag>
                <span class="question-score">{{ question.score }}分</span>
              </div>
              <div class="question-content">
                <p class="question-text">{{ question.questionText }}</p>
                <div class="question-meta">
                  <span class="knowledge-point">知识点：{{ question.knowledgePoint }}</span>
                </div>
              </div>
            </div>
          </div>
        </a-card>
      </div>

      <!-- 练习历史 -->
      <div class="history-section">
        <a-card title="📊 练习历史" class="history-card">
          <template #extra>
            <a-button @click="refreshHistory" :loading="loadingHistory">
              <ReloadOutlined />
              刷新
            </a-button>
          </template>

          <a-table 
            :columns="historyColumns" 
            :data-source="practiceHistory"
            :loading="loadingHistory"
            row-key="id"
            :pagination="{ pageSize: 5 }"
            size="small"
          >
            <template #bodyCell="{ column, record }">
              <template v-if="column.key === 'title'">
                <div class="history-title">
                  <span>{{ record.title }}</span>
                  <a-tag size="small" color="cyan">{{ record.course_name }}</a-tag>
                </div>
              </template>

              <template v-if="column.key === 'score'">
                <div class="score-display">
                  <span class="score">{{ record.score }}</span>
                  <span class="total">/{{ record.total_score }}</span>
                  <a-progress 
                    :percent="(record.score / record.total_score * 100)" 
                    size="small" 
                    :show-info="false"
                    style="margin-left: 8px; width: 60px;"
                  />
                </div>
              </template>

              <template v-if="column.key === 'status'">
                <a-tag :color="getHistoryStatusColor(record.status)">
                  {{ getHistoryStatusText(record.status) }}
                </a-tag>
              </template>

              <template v-if="column.key === 'action'">
                <a-space>
                  <a-button size="small" @click="handleViewHistory(record)">查看</a-button>
                  <a-button size="small" @click="handleRetryPractice(record)">重做</a-button>
                </a-space>
              </template>
            </template>
          </a-table>
        </a-card>
      </div>
    </div>

    <!-- 开始练习弹窗 -->
    <a-modal 
      v-model:open="practiceModalVisible"
      title="开始练习"
      width="1000px"
      :footer="null"
      :closable="false"
      :mask-closable="false"
    >
      <div class="practice-modal">
        <div class="practice-header">
          <div class="timer">
            <ClockCircleOutlined />
            剩余时间：{{ formatTime(remainingTime) }}
          </div>
          <div class="progress">
            题目进度：{{ currentQuestionIndex + 1 }} / {{ practiceQuestions.length }}
          </div>
        </div>

        <div v-if="currentQuestion" class="current-question">
          <div class="question-info">
            <h3>第{{ currentQuestionIndex + 1 }}题 ({{ currentQuestion.score }}分)</h3>
            <div class="question-tags">
              <a-tag :color="getDifficultyColor(currentQuestion.difficulty)">
                {{ currentQuestion.difficulty }}
              </a-tag>
              <a-tag color="blue">{{ getQuestionTypeLabel(currentQuestion.questionType) }}</a-tag>
              <a-tag color="green">{{ currentQuestion.knowledgePoint }}</a-tag>
            </div>
          </div>

          <div class="question-content">
            <p class="question-text">{{ currentQuestion.questionText }}</p>

            <!-- 选择题选项 -->
            <div v-if="currentQuestion.options" class="question-options">
              <a-radio-group 
                v-if="currentQuestion.questionType === 'SINGLE_CHOICE'"
                v-model:value="currentAnswer"
              >
                <div v-for="(option, index) in currentQuestion.options" :key="index" class="option-item">
                  <a-radio :value="String.fromCharCode(65 + index)">
                    {{ String.fromCharCode(65 + index) }}. {{ option }}
                  </a-radio>
                </div>
              </a-radio-group>

              <a-checkbox-group 
                v-else-if="currentQuestion.questionType === 'MULTIPLE_CHOICE'"
                v-model:value="currentAnswer"
              >
                <div v-for="(option, index) in currentQuestion.options" :key="index" class="option-item">
                  <a-checkbox :value="String.fromCharCode(65 + index)">
                    {{ String.fromCharCode(65 + index) }}. {{ option }}
                  </a-checkbox>
                </div>
              </a-checkbox-group>

              <a-radio-group 
                v-else-if="currentQuestion.questionType === 'TRUE_FALSE'"
                v-model:value="currentAnswer"
              >
                <a-radio value="T">正确</a-radio>
                <a-radio value="F">错误</a-radio>
              </a-radio-group>
            </div>

            <!-- 填空题 -->
            <div v-else-if="currentQuestion.questionType === 'FILL_BLANK'" class="fill-blank">
              <a-input v-model:value="currentAnswer" placeholder="请输入答案" />
            </div>

            <!-- 简答题 -->
            <div v-else-if="currentQuestion.questionType === 'ESSAY'" class="essay">
              <a-textarea v-model:value="currentAnswer" :rows="4" placeholder="请输入您的答案" />
            </div>
          </div>

          <div class="question-actions">
            <a-space>
              <a-button @click="handlePrevQuestion" :disabled="currentQuestionIndex === 0">
                上一题
              </a-button>
              <a-button 
                type="primary" 
                @click="handleNextQuestion"
                :disabled="!currentAnswer"
              >
                {{ currentQuestionIndex === practiceQuestions.length - 1 ? '提交答案' : '下一题' }}
              </a-button>
            </a-space>
          </div>
        </div>
      </div>
    </a-modal>

    <!-- 练习结果弹窗 -->
    <a-modal 
      v-model:open="resultModalVisible"
      title="练习结果"
      width="800px"
      :footer="null"
    >
      <div v-if="practiceResult" class="practice-result">
        <div class="result-summary">
          <div class="summary-header">
            <h2>练习完成！</h2>
            <div class="score-circle">
              <div class="score-value">{{ practiceResult.score }}</div>
              <div class="score-total">/{{ practiceResult.totalScore }}</div>
            </div>
          </div>
          
          <div class="summary-stats">
            <div class="stat-item">
              <div class="stat-label">正确率</div>
              <div class="stat-value">{{ practiceResult.accuracy }}%</div>
            </div>
            <div class="stat-item">
              <div class="stat-label">用时</div>
              <div class="stat-value">{{ practiceResult.timeUsed }}分钟</div>
            </div>
            <div class="stat-item">
              <div class="stat-label">击败用户</div>
              <div class="stat-value">{{ practiceResult.ranking }}%</div>
            </div>
          </div>
        </div>

        <div class="result-analysis">
          <h3>AI分析报告</h3>
          <div class="analysis-content">
            <div class="strength-analysis">
              <h4>💪 优势分析</h4>
              <ul>
                <li v-for="strength in practiceResult.strengths" :key="strength">{{ strength }}</li>
              </ul>
            </div>
            
            <div class="weakness-analysis">
              <h4>📈 提升建议</h4>
              <ul>
                <li v-for="weakness in practiceResult.weaknesses" :key="weakness">{{ weakness }}</li>
              </ul>
            </div>
          </div>
        </div>

        <div class="result-actions">
          <a-space>
            <a-button @click="resultModalVisible = false">关闭</a-button>
            <a-button @click="handleRetryCurrentPractice">重新练习</a-button>
            <a-button type="primary" @click="handleGenerateRelated">生成相关练习</a-button>
          </a-space>
        </div>
      </div>
    </a-modal>
  </div>
</template>

<script setup lang="ts">
import { ref, reactive, computed, onMounted, onBeforeUnmount } from 'vue'
import { message } from 'ant-design-vue'
import { 
  ThunderboltOutlined,
  StarOutlined,
  ReloadOutlined,
  ClockCircleOutlined
} from '@ant-design/icons-vue'
import { studentPaperApi, type PaperGenerationResponse } from '@/api/dify'

// 响应式数据
const generating = ref(false)
const recommending = ref(false)
const loadingHistory = ref(false)
const practiceModalVisible = ref(false)
const resultModalVisible = ref(false)

const selectedType = ref('personalized')
const selectedQuestionTypes = ref(['SINGLE_CHOICE', 'MULTIPLE_CHOICE'])
const practiceQuestions = ref<any[]>([])
const currentQuestionIndex = ref(0)
const currentAnswer = ref<any>('')
const studentAnswers = ref<Record<number, any>>({})
const remainingTime = ref(1800) // 30分钟
const timer = ref<ReturnType<typeof setInterval> | null>(null)

const practiceConfig = reactive({
  courseId: 0,
  weakKnowledgePoints: [],
  errorKnowledgePoints: [],
  abilityLevel: 'MEDIUM' as 'LOW' | 'MEDIUM' | 'HIGH',
  questionCount: 10
})

// 练习类型配置
const practiceTypes = ref([
  {
    key: 'personalized',
    icon: '🎯',
    title: '个性化练习',
    description: '基于AI分析为您量身定制'
  },
  {
    key: 'retry',
    icon: '🔄',
    title: '错题重练',
    description: '针对历史错题生成相似题目'
  },
  {
    key: 'recommend',
    icon: '⭐',
    title: '智能推荐',
    description: '系统推荐最适合的练习'
  }
])

// 模拟数据
const courses = ref([
  { id: 1, name: '高等数学' },
  { id: 2, name: '线性代数' },
  { id: 3, name: '概率论与数理统计' }
])

const weakKnowledgeOptions = ref([
  { label: '函数极限', value: '函数极限' },
  { label: '导数计算', value: '导数计算' },
  { label: '积分应用', value: '积分应用' }
])

const errorKnowledgeOptions = ref([
  { label: '微分方程', value: '微分方程' },
  { label: '无穷级数', value: '无穷级数' },
  { label: '多元函数', value: '多元函数' }
])

const practiceHistory = ref([
  {
    id: 1,
    title: '个性化练习 1',
    course_name: '高等数学',
    question_count: 10,
    score: 85,
    total_score: 100,
    created_time: '2024-01-15 14:30:00',
    status: 'completed'
  },
  {
    id: 2,
    title: '错题重练 1',
    course_name: '线性代数',
    question_count: 8,
    score: 70,
    total_score: 80,
    created_time: '2024-01-14 16:20:00',
    status: 'completed'
  }
])

const practiceResult = ref<any>(null)

// 表格列定义
const historyColumns = [
  {
    title: '练习标题',
    key: 'title',
    width: 200
  },
  {
    title: '题目数',
    dataIndex: 'question_count',
    key: 'question_count',
    width: 80
  },
  {
    title: '得分',
    key: 'score',
    width: 150
  },
  {
    title: '状态',
    key: 'status',
    width: 80
  },
  {
    title: '创建时间',
    dataIndex: 'created_time',
    key: 'created_time',
    width: 150
  },
  {
    title: '操作',
    key: 'action',
    width: 120
  }
]

// 计算属性
const currentQuestion = computed(() => {
  return practiceQuestions.value[currentQuestionIndex.value]
})

const estimatedTime = computed(() => {
  return Math.ceil(practiceQuestions.value.length * 2) // 每题预计2分钟
})

// 方法
const selectPracticeType = (type: string) => {
  selectedType.value = type
}

const handleCourseChange = (courseId: number) => {
  console.log('课程变更:', courseId)
  // TODO: 根据课程加载对应的知识点和历史数据
}

const handleGeneratePractice = async () => {
  if (!practiceConfig.courseId) {
    message.warning('请先选择课程')
    return
  }

  try {
    generating.value = true

    const questionTypes: Record<string, number> = {}
    const totalCount = practiceConfig.questionCount
    const typeCount = Math.floor(totalCount / selectedQuestionTypes.value.length)

    selectedQuestionTypes.value.forEach((type, index) => {
      if (index === selectedQuestionTypes.value.length - 1) {
        questionTypes[type] = totalCount - typeCount * index
      } else {
        questionTypes[type] = typeCount
      }
    })

    let response: any

    if (selectedType.value === 'personalized') {
      response = await studentPaperApi.generatePractice({
        courseId: practiceConfig.courseId,
        weakKnowledgePoints: practiceConfig.weakKnowledgePoints,
        abilityLevel: practiceConfig.abilityLevel,
        questionCount: practiceConfig.questionCount,
        preferredQuestionTypes: questionTypes
      })
    } else if (selectedType.value === 'retry') {
      response = await studentPaperApi.generateRetry({
        courseId: practiceConfig.courseId,
        errorKnowledgePoints: practiceConfig.errorKnowledgePoints,
        errorTypes: selectedQuestionTypes.value,
        retryCount: practiceConfig.questionCount
      })
    }

    if (response && response.data && response.data.status === 'completed') {
      practiceQuestions.value = response.data.questions || []
      message.success('练习题生成成功！')
    } else {
      message.error('练习题生成失败')
    }
  } catch (error) {
    message.error('生成失败: ' + (error as any).message)
  } finally {
    generating.value = false
  }
}

const handleQuickRecommend = async () => {
  if (!practiceConfig.courseId) {
    message.warning('请先选择课程')
    return
  }

  try {
    recommending.value = true
    const response = await studentPaperApi.recommendPractice(practiceConfig.courseId, practiceConfig.questionCount)
    
    if (response.data && response.data.status === 'completed') {
      practiceQuestions.value = response.data.questions || []
      message.success('智能推荐成功！')
    }
  } catch (error) {
    message.error('推荐失败: ' + (error as any).message)
  } finally {
    recommending.value = false
  }
}

const handleStartPractice = () => {
  practiceModalVisible.value = true
  currentQuestionIndex.value = 0
  currentAnswer.value = ''
  studentAnswers.value = {}
  remainingTime.value = estimatedTime.value * 60 // 转换为秒
  startTimer()
}

const startTimer = () => {
  timer.value = setInterval(() => {
    remainingTime.value--
    if (remainingTime.value <= 0) {
      handleTimeUp()
    }
  }, 1000)
}

const stopTimer = () => {
  if (timer.value) {
    clearInterval(timer.value)
    timer.value = null
  }
}

const handleTimeUp = () => {
  stopTimer()
  message.warning('时间到！自动提交答案')
  handleSubmitPractice()
}

const handlePrevQuestion = () => {
  saveCurrentAnswer()
  currentQuestionIndex.value--
  loadQuestionAnswer()
}

const handleNextQuestion = () => {
  saveCurrentAnswer()
  
  if (currentQuestionIndex.value === practiceQuestions.value.length - 1) {
    handleSubmitPractice()
  } else {
    currentQuestionIndex.value++
    loadQuestionAnswer()
  }
}

const saveCurrentAnswer = () => {
  if (currentQuestion.value) {
    studentAnswers.value[currentQuestion.value.id || currentQuestionIndex.value] = currentAnswer.value
  }
}

const loadQuestionAnswer = () => {
  const questionId = currentQuestion.value?.id || currentQuestionIndex.value
  currentAnswer.value = studentAnswers.value[questionId] || ''
}

const handleSubmitPractice = () => {
  saveCurrentAnswer()
  stopTimer()
  
  // 计算练习结果
  const usedTime = Math.ceil((estimatedTime.value * 60 - remainingTime.value) / 60)
  const totalScore = practiceQuestions.value.reduce((sum, q) => sum + q.score, 0)
  
  // 模拟计算得分（实际应该调用后端API）
  const correctCount = Math.floor(Math.random() * practiceQuestions.value.length * 0.8)
  const score = Math.floor(correctCount / practiceQuestions.value.length * totalScore)
  
  practiceResult.value = {
    score,
    totalScore,
    accuracy: Math.floor(correctCount / practiceQuestions.value.length * 100),
    timeUsed: usedTime,
    ranking: Math.floor(Math.random() * 50 + 50), // 模拟排名
    strengths: [
      '基础概念掌握较好',
      '计算能力较强',
      '解题思路清晰'
    ],
    weaknesses: [
      '应用题分析有待加强',
      '复杂计算易出错',
      '建议多练习相关题型'
    ]
  }
  
  practiceModalVisible.value = false
  resultModalVisible.value = true
  
  // 添加到练习历史
  practiceHistory.value.unshift({
    id: Date.now(),
    title: `${selectedType.value === 'personalized' ? '个性化练习' : '错题重练'} ${practiceHistory.value.length + 1}`,
    course_name: courses.value.find(c => c.id === practiceConfig.courseId)?.name || '未知课程',
    question_count: practiceQuestions.value.length,
    score,
    total_score: totalScore,
    created_time: new Date().toLocaleString(),
    status: 'completed'
  })
}

const refreshHistory = async () => {
  try {
    loadingHistory.value = true
    const response = await studentPaperApi.getPracticeHistory(1, 10)
    // practiceHistory.value = response.data.records
    message.success('历史记录已刷新')
  } catch (error) {
    message.error('刷新失败')
  } finally {
    loadingHistory.value = false
  }
}

const handleViewHistory = (record: any) => {
  message.info('查看历史详情功能开发中...')
}

const handleRetryPractice = (record: any) => {
  message.info('重做练习功能开发中...')
}

const handleRetryCurrentPractice = () => {
  resultModalVisible.value = false
  handleStartPractice()
}

const handleGenerateRelated = () => {
  resultModalVisible.value = false
  handleGeneratePractice()
}

const getQuestionTypeLabel = (type: string) => {
  const typeMap: Record<string, string> = {
    'SINGLE_CHOICE': '单选题',
    'MULTIPLE_CHOICE': '多选题',
    'TRUE_FALSE': '判断题',
    'FILL_BLANK': '填空题',
    'ESSAY': '简答题'
  }
  return typeMap[type] || type
}

const getDifficultyColor = (difficulty: string) => {
  const colorMap: Record<string, string> = {
    'EASY': 'green',
    'MEDIUM': 'orange',
    'HARD': 'red'
  }
  return colorMap[difficulty] || 'blue'
}

const getHistoryStatusColor = (status: string) => {
  const colorMap: Record<string, string> = {
    'completed': 'green',
    'in_progress': 'blue',
    'abandoned': 'orange'
  }
  return colorMap[status] || 'default'
}

const getHistoryStatusText = (status: string) => {
  const textMap: Record<string, string> = {
    'completed': '已完成',
    'in_progress': '进行中',
    'abandoned': '已放弃'
  }
  return textMap[status] || status
}

const formatTime = (seconds: number) => {
  const mins = Math.floor(seconds / 60)
  const secs = seconds % 60
  return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`
}

onMounted(() => {
  // 初始化数据
})

onBeforeUnmount(() => {
  stopTimer()
})
</script>

<style scoped>
.personalized-practice {
  padding: 24px;
  background: #f5f5f5;
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

.practice-container {
  max-width: 1400px;
  margin: 0 auto;
  display: flex;
  flex-direction: column;
  gap: 24px;
}

.type-card, .config-card, .practice-card, .history-card {
  border-radius: 12px;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
}

.practice-types {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.type-item {
  display: flex;
  align-items: center;
  padding: 16px;
  border: 2px solid #f0f0f0;
  border-radius: 8px;
  cursor: pointer;
  transition: all 0.3s;
}

.type-item:hover {
  border-color: #1890ff;
  background: #f6ffed;
}

.type-item.active {
  border-color: #1890ff;
  background: #e6f7ff;
}

.type-icon {
  font-size: 24px;
  margin-right: 16px;
}

.type-info h4 {
  margin: 0 0 4px 0;
  color: #333;
}

.type-info p {
  margin: 0;
  color: #666;
  font-size: 14px;
}

.question-types {
  display: flex;
  flex-wrap: wrap;
  gap: 16px;
}

.practice-info {
  color: #666;
  font-size: 14px;
}

.questions-preview {
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.question-preview {
  background: white;
  border: 1px solid #e8e8e8;
  border-radius: 8px;
  padding: 16px;
}

.question-header {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 12px;
}

.question-number {
  font-weight: 600;
  color: #1890ff;
}

.question-score {
  margin-left: auto;
  font-weight: 600;
  color: #f5222d;
}

.question-text {
  font-size: 16px;
  line-height: 1.6;
  margin-bottom: 8px;
  color: #333;
}

.question-meta {
  font-size: 14px;
  color: #666;
}

.history-title {
  display: flex;
  align-items: center;
  gap: 8px;
}

.score-display {
  display: flex;
  align-items: center;
}

.score {
  font-weight: 600;
  color: #1890ff;
}

.total {
  color: #666;
}

.practice-modal {
  padding: 16px 0;
}

.practice-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 24px;
  padding: 16px;
  background: #f8f9fa;
  border-radius: 8px;
}

.timer {
  display: flex;
  align-items: center;
  gap: 8px;
  font-weight: 600;
  color: #f5222d;
}

.progress {
  color: #666;
}

.current-question {
  background: white;
  border-radius: 8px;
  padding: 24px;
}

.question-info {
  margin-bottom: 24px;
}

.question-info h3 {
  margin: 0 0 12px 0;
  color: #333;
}

.question-tags {
  display: flex;
  gap: 8px;
}

.question-content {
  margin-bottom: 24px;
}

.question-options {
  margin: 16px 0;
}

.option-item {
  margin: 12px 0;
  padding: 8px 0;
}

.fill-blank, .essay {
  margin: 16px 0;
}

.question-actions {
  text-align: right;
}

.practice-result {
  padding: 16px 0;
}

.result-summary {
  text-align: center;
  margin-bottom: 32px;
}

.summary-header {
  display: flex;
  flex-direction: column;
  align-items: center;
  margin-bottom: 24px;
}

.summary-header h2 {
  margin: 0 0 16px 0;
  color: #333;
}

.score-circle {
  display: flex;
  align-items: baseline;
  justify-content: center;
  width: 120px;
  height: 120px;
  border: 4px solid #1890ff;
  border-radius: 50%;
  background: #f6ffed;
}

.score-value {
  font-size: 36px;
  font-weight: 600;
  color: #1890ff;
}

.score-total {
  font-size: 18px;
  color: #666;
}

.summary-stats {
  display: flex;
  justify-content: center;
  gap: 40px;
}

.stat-item {
  text-align: center;
}

.stat-label {
  color: #666;
  font-size: 14px;
  margin-bottom: 4px;
}

.stat-value {
  font-size: 20px;
  font-weight: 600;
  color: #1890ff;
}

.result-analysis {
  margin-bottom: 24px;
}

.result-analysis h3 {
  margin: 0 0 16px 0;
  color: #333;
}

.analysis-content {
  display: flex;
  gap: 24px;
}

.strength-analysis, .weakness-analysis {
  flex: 1;
  padding: 16px;
  border-radius: 8px;
}

.strength-analysis {
  background: #f6ffed;
  border-left: 4px solid #52c41a;
}

.weakness-analysis {
  background: #fff7e6;
  border-left: 4px solid #fa8c16;
}

.strength-analysis h4, .weakness-analysis h4 {
  margin: 0 0 12px 0;
}

.strength-analysis ul, .weakness-analysis ul {
  margin: 0;
  padding-left: 20px;
}

.strength-analysis li, .weakness-analysis li {
  margin: 6px 0;
  line-height: 1.5;
}

.result-actions {
  text-align: right;
}
</style> 