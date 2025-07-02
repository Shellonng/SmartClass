<template>
  <div class="assignment-page">
    <a-spin :spinning="loading">
      <div class="assignment-header">
        <div class="header-left">
          <h1 class="assignment-title">{{ assignment?.title || '作业' }}</h1>
          <a-tag color="blue">作业进行中</a-tag>
        </div>
        <div class="header-right">
          <div class="deadline">
            <calendar-outlined /> 截止时间：{{ formatDateTime(assignment?.endTime) }}
          </div>
          <a-button type="primary" @click="submitAssignment">提交作业</a-button>
        </div>
      </div>

      <div class="assignment-content">
        <div class="question-navigation">
          <div class="navigation-header">
            <div class="navigation-title">题目导航</div>
            <div class="question-stats">
              已答: {{ answeredCount }}/{{ totalQuestions }}
            </div>
          </div>
          <div class="question-buttons">
            <a-button 
              v-for="(q, index) in questions" 
              :key="q.id"
              :type="isQuestionAnswered(q.id) ? 'primary' : 'default'"
              :class="{ 'current-question': currentQuestionIndex === index }"
              size="small"
              @click="goToQuestion(index)"
            >
              {{ index + 1 }}
            </a-button>
          </div>
        </div>

        <div class="question-container">
          <div v-if="currentQuestion" class="question">
            <div class="question-header">
              <div class="question-index">
                第 {{ currentQuestionIndex + 1 }} 题 
                <a-tag :color="getQuestionTypeColor(currentQuestion.questionType)">
                  {{ getQuestionTypeText(currentQuestion.questionType) }}
                </a-tag>
                <span class="question-score">{{ currentQuestion.score }}分</span>
              </div>
            </div>
            
            <div class="question-content">{{ currentQuestion.title }}</div>
            
            <!-- 选择题选项 -->
            <div v-if="['single', 'multiple'].includes(currentQuestion.questionType)" class="question-options">
              <a-radio-group 
                v-if="currentQuestion.questionType === 'single'" 
                v-model:value="answers[currentQuestion.id]"
                @change="saveCurrentAnswer"
              >
                <div v-for="option in currentQuestion.options" :key="option.id" class="option-item">
                  <a-radio :value="option.optionKey">
                    {{ option.optionKey }}. {{ option.content }}
                  </a-radio>
                </div>
              </a-radio-group>
              
              <a-checkbox-group 
                v-else-if="currentQuestion.questionType === 'multiple'" 
                v-model:value="answers[currentQuestion.id]"
                @change="saveCurrentAnswer"
              >
                <div v-for="option in currentQuestion.options" :key="option.id" class="option-item">
                  <a-checkbox :value="option.optionKey">
                    {{ option.optionKey }}. {{ option.content }}
                  </a-checkbox>
                </div>
              </a-checkbox-group>
            </div>
            
            <!-- 判断题选项 -->
            <div v-else-if="currentQuestion.questionType === 'judge'" class="question-options">
              <a-radio-group 
                v-model:value="answers[currentQuestion.id]"
                @change="saveCurrentAnswer"
              >
                <a-radio value="true">正确</a-radio>
                <a-radio value="false">错误</a-radio>
              </a-radio-group>
            </div>
            
            <!-- 填空题 -->
            <div v-else-if="currentQuestion.questionType === 'fill'" class="question-blank">
              <a-input 
                v-model:value="answers[currentQuestion.id]" 
                placeholder="请输入答案"
                @blur="saveCurrentAnswer"
              />
            </div>
            
            <!-- 简答题 -->
            <div v-else-if="currentQuestion.questionType === 'short'" class="question-short">
              <a-textarea 
                v-model:value="answers[currentQuestion.id]" 
                placeholder="请输入答案" 
                :rows="6"
                @blur="saveCurrentAnswer"
              />
            </div>
            
            <!-- 其他题型 -->
            <div v-else class="question-other">
              <a-textarea 
                v-model:value="answers[currentQuestion.id]" 
                placeholder="请输入答案" 
                :rows="6"
                @blur="saveCurrentAnswer"
              />
            </div>
          </div>
          
          <div v-else class="empty-question">
            <a-empty description="题目加载中..." />
          </div>
          
          <div class="question-actions">
            <a-button 
              v-if="currentQuestionIndex > 0" 
              @click="prevQuestion"
            >
              上一题
            </a-button>
            <a-button 
              type="primary" 
              v-if="currentQuestionIndex < questions.length - 1" 
              @click="nextQuestion"
            >
              下一题
            </a-button>
            <a-button 
              type="primary" 
              v-if="currentQuestionIndex === questions.length - 1" 
              @click="submitAssignment"
            >
              提交作业
            </a-button>
          </div>
        </div>
      </div>
    </a-spin>
    
    <a-modal
      v-model:visible="submitModalVisible"
      title="确认提交"
      @ok="confirmSubmit"
      okText="确认提交"
      cancelText="继续作答"
    >
      <p>您还有 {{ totalQuestions - answeredCount }} 道题未作答，确定要提交吗？</p>
      <p>提交后将无法再次修改答案。</p>
    </a-modal>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted, onBeforeUnmount } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { message } from 'ant-design-vue'
import { CalendarOutlined } from '@ant-design/icons-vue'
import assignmentApi from '@/api/assignment'
import dayjs from 'dayjs'

const route = useRoute()
const router = useRouter()
const assignmentId = ref<number>(Number(route.params.id) || 0)
const loading = ref<boolean>(true)
const assignment = ref<any>(null)
const questions = ref<any[]>([])
const currentQuestionIndex = ref<number>(0)
const answers = ref<Record<number, any>>({})
const submitModalVisible = ref<boolean>(false)

// 当前题目
const currentQuestion = computed(() => {
  if (!questions.value || questions.value.length === 0) return null
  return questions.value[currentQuestionIndex.value]
})

// 已答题数量
const answeredCount = computed(() => {
  return Object.keys(answers.value).filter(id => {
    const answer = answers.value[Number(id)]
    return answer !== undefined && answer !== null && 
           !(Array.isArray(answer) && answer.length === 0) && 
           !(typeof answer === 'string' && answer.trim() === '')
  }).length
})

// 总题数
const totalQuestions = computed(() => questions.value.length)

// 加载作业详情和题目
const loadAssignmentData = async () => {
  try {
    loading.value = true
    
    // 获取作业详情
    const assignmentResponse = await assignmentApi.getStudentAssignmentDetail(assignmentId.value)
    assignment.value = assignmentResponse
    
    // 获取作业题目
    const questionsResponse = await assignmentApi.getAssignmentQuestions(assignmentId.value)
    questions.value = questionsResponse || []
    
    // 初始化答案对象
    questions.value.forEach(q => {
      // 根据题型初始化不同类型的默认值
      if (q.questionType === 'multiple') {
        answers.value[q.id] = []  // 多选题初始化为空数组
      } else {
        answers.value[q.id] = ''  // 其他题型初始化为空字符串
      }
    })
    
    // 加载本地保存的答案（如果有）
    loadSavedAnswers()
    
  } catch (error) {
    console.error('加载作业数据失败:', error)
    message.error('加载作业数据失败')
  } finally {
    loading.value = false
  }
}

// 保存当前答案
const saveCurrentAnswer = () => {
  if (!currentQuestion.value) return
  saveAnswersToLocal()
}

// 保存答案到本地存储
const saveAnswersToLocal = () => {
  try {
    localStorage.setItem(`assignment_${assignmentId.value}_answers`, JSON.stringify(answers.value))
  } catch (error) {
    console.error('保存答案到本地失败:', error)
  }
}

// 加载本地保存的答案
const loadSavedAnswers = () => {
  try {
    const savedAnswers = localStorage.getItem(`assignment_${assignmentId.value}_answers`)
    if (savedAnswers) {
      const parsed = JSON.parse(savedAnswers)
      // 合并已保存的答案
      Object.keys(parsed).forEach(key => {
        answers.value[Number(key)] = parsed[key]
      })
    }
  } catch (error) {
    console.error('加载本地答案失败:', error)
  }
}

// 清除本地保存的答案
const clearSavedAnswers = () => {
  try {
    localStorage.removeItem(`assignment_${assignmentId.value}_answers`)
  } catch (error) {
    console.error('清除本地答案失败:', error)
  }
}

// 判断题目是否已答
const isQuestionAnswered = (questionId: number) => {
  const answer = answers.value[questionId]
  return answer !== undefined && answer !== null && 
         !(Array.isArray(answer) && answer.length === 0) && 
         !(typeof answer === 'string' && answer.trim() === '')
}

// 上一题
const prevQuestion = () => {
  if (currentQuestionIndex.value > 0) {
    currentQuestionIndex.value--
  }
}

// 下一题
const nextQuestion = () => {
  if (currentQuestionIndex.value < questions.value.length - 1) {
    currentQuestionIndex.value++
  }
}

// 跳转到指定题目
const goToQuestion = (index: number) => {
  currentQuestionIndex.value = index
}

// 提交作业
const submitAssignment = () => {
  // 如果已经全部作答，直接提交
  if (answeredCount.value === totalQuestions.value) {
    confirmSubmit()
  } else {
    // 否则显示确认对话框
    submitModalVisible.value = true
  }
}

// 确认提交
const confirmSubmit = async () => {
  try {
    loading.value = true
    
    // 调用API提交答案
    await assignmentApi.submitAssignmentAnswers(assignmentId.value, answers.value)
    
    // 清除本地存储
    clearSavedAnswers()
    
    message.success('作业提交成功')
    
    // 跳转到作业列表页面
    router.push('/student/assignments')
  } catch (error) {
    console.error('提交作业失败:', error)
    message.error('提交作业失败，请重试')
  } finally {
    loading.value = false
    submitModalVisible.value = false
  }
}

// 格式化日期时间
const formatDateTime = (date: string | Date) => {
  if (!date) return '未设置'
  return dayjs(date).format('YYYY-MM-DD HH:mm')
}

// 获取题目类型文本
const getQuestionTypeText = (type: string) => {
  const typeMap: Record<string, string> = {
    'single': '单选题',
    'multiple': '多选题',
    'judge': '判断题',
    'fill': '填空题',
    'short': '简答题'
  }
  return typeMap[type] || '其他题型'
}

// 获取题目类型颜色
const getQuestionTypeColor = (type: string) => {
  const colorMap: Record<string, string> = {
    'single': 'blue',
    'multiple': 'purple',
    'judge': 'green',
    'fill': 'orange',
    'short': 'cyan'
  }
  return colorMap[type] || 'default'
}

// 页面离开前确认
const handleBeforeUnload = (e: BeforeUnloadEvent) => {
  e.preventDefault()
  e.returnValue = '离开页面将丢失未保存的答案，确定要离开吗？'
}

onMounted(() => {
  loadAssignmentData()
  // 添加页面离开确认
  window.addEventListener('beforeunload', handleBeforeUnload)
})

onBeforeUnmount(() => {
  // 移除页面离开确认
  window.removeEventListener('beforeunload', handleBeforeUnload)
})
</script>

<style scoped>
.assignment-page {
  background-color: #f5f7fa;
  min-height: 100vh;
  padding: 24px;
}

.assignment-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  background-color: #fff;
  padding: 16px 24px;
  border-radius: 8px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
  margin-bottom: 24px;
}

.header-left {
  display: flex;
  align-items: center;
  gap: 12px;
}

.assignment-title {
  font-size: 20px;
  margin: 0;
}

.header-right {
  display: flex;
  align-items: center;
  gap: 16px;
}

.deadline {
  font-size: 14px;
  color: #ff4d4f;
}

.assignment-content {
  display: flex;
  gap: 24px;
}

.question-navigation {
  width: 200px;
  background-color: #fff;
  border-radius: 8px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
  padding: 16px;
  flex-shrink: 0;
}

.navigation-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 16px;
}

.navigation-title {
  font-weight: 500;
}

.question-stats {
  font-size: 12px;
  color: #666;
}

.question-buttons {
  display: grid;
  grid-template-columns: repeat(5, 1fr);
  gap: 8px;
}

.current-question {
  border-color: #1890ff;
  background-color: #e6f7ff;
}

.question-container {
  flex: 1;
  background-color: #fff;
  border-radius: 8px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
  padding: 24px;
}

.question {
  margin-bottom: 24px;
}

.question-header {
  margin-bottom: 16px;
  padding-bottom: 12px;
  border-bottom: 1px solid #f0f0f0;
}

.question-index {
  font-size: 16px;
  font-weight: 500;
  display: flex;
  align-items: center;
  gap: 8px;
}

.question-score {
  color: #ff4d4f;
  margin-left: auto;
}

.question-content {
  font-size: 16px;
  line-height: 1.6;
  margin-bottom: 24px;
}

.question-options {
  margin-left: 16px;
}

.option-item {
  margin-bottom: 12px;
}

.question-blank,
.question-short,
.question-other {
  margin-top: 16px;
}

.question-actions {
  display: flex;
  justify-content: space-between;
  margin-top: 32px;
}

.empty-question {
  display: flex;
  justify-content: center;
  align-items: center;
  height: 300px;
}
</style> 