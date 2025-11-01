<template>
  <div class="exam-container">
    <div class="exam-header">
      <h1>{{ examTitle }}</h1>
      <div class="exam-status">
        <span class="status-tag" :class="{ 'in-progress': submissionStatus === 0, 'submitted': submissionStatus > 0 }">{{ examStatus }}</span>
        <div class="timer" v-if="submissionStatus === 0">
          <el-icon><Timer /></el-icon>
          <span>剩余时间: {{ formatTime(remainingTime) }}</span>
        </div>
        <el-button type="danger" @click="handleSubmit" :disabled="submissionStatus > 0">提交考试</el-button>
      </div>
    </div>

    <div class="exam-content">
      <div class="question-nav">
        <h3>题目导航</h3>
        <div class="question-nav-list">
          <div v-for="(group, groupIndex) in questionGroups" :key="'group-' + groupIndex">
            <div class="question-group-title">{{ getQuestionTypeName(group.type) }}</div>
            <div class="question-buttons">
              <el-button 
                v-for="(question, qIndex) in group.questions" 
                :key="'q-' + question.id"
                size="small"
                :type="isAnswered(question.id) ? 'primary' : 'default'"
                @click="scrollToQuestion(question.id)"
              >
                {{ qIndex + 1 }}
              </el-button>
            </div>
          </div>
        </div>
      </div>

      <div class="question-content">
        <div v-if="loading" class="loading-container">
          <div class="loading-icon">
            <el-icon class="is-loading"><Loading /></el-icon>
          </div>
          <div class="loading-text">题目加载中...</div>
        </div>
        
        <div v-else-if="error" class="error-message">
          <el-alert
            title="加载失败"
            type="error"
            :description="error"
            show-icon
          />
        </div>

        <div v-else>
          <!-- 按题型分组显示题目 -->
          <div v-for="(group, groupIndex) in questionGroups" :key="'group-' + groupIndex" class="question-section">
            <div class="section-header">
              <h2>{{ getRomanNumeral(groupIndex + 1) }}、{{ getQuestionTypeName(group.type) }} <span class="section-info">（共{{ group.questions.length }}题，共{{ group.totalScore }}分）</span></h2>
            </div>
            
            <!-- 题目列表 -->
            <div v-for="(question, qIndex) in group.questions" :key="question.id" class="question-item" :id="`question-${question.id}`">
              <div class="question-header">
                <span class="question-number">{{ qIndex + 1 }}.</span>
                <span class="question-title" v-html="question.title"></span>
                <span class="question-score">（{{ question.score }}分）</span>
              </div>
              
              <!-- 单选题 -->
              <div v-if="group.type === 'single'" class="question-options">
                <div class="options-container">
                  <div v-for="option in question.options" :key="option.id" class="option-row">
                    <input 
                      type="radio" 
                      :id="`q${question.id}_${option.optionKey}`" 
                      :name="`question_${question.id}`"
                      :value="option.optionKey" 
                      v-model="answers[question.id]"
                      class="custom-radio"
                      :disabled="submissionStatus > 0"
                    >
                    <label :for="`q${question.id}_${option.optionKey}`" class="option-label">
                      {{ option.optionKey }}. {{ option.content }}
                    </label>
                  </div>
                </div>
              </div>
              
              <!-- 多选题 -->
              <div v-else-if="group.type === 'multiple'" class="question-options">
                <div class="options-container">
                  <div v-for="option in question.options" :key="option.id" class="option-row">
                    <input 
                      type="checkbox" 
                      :id="`q${question.id}_${option.optionKey}`" 
                      :value="option.optionKey" 
                      :checked="isOptionChecked(question.id, option.optionKey)"
                      class="custom-checkbox"
                      :disabled="submissionStatus > 0"
                      @change="toggleMultipleChoice(question.id, option.optionKey, $event)"
                    >
                    <label :for="`q${question.id}_${option.optionKey}`" class="option-label">
                      {{ option.optionKey }}. {{ option.content }}
                    </label>
                  </div>
                </div>
              </div>
              
              <!-- 判断题 -->
              <div v-else-if="group.type === 'true_false'" class="question-options">
                <div class="options-container">
                  <div v-for="option in [...(question.options || [])].sort((a, b) => {
                    if (a.optionKey === 'T') return -1;
                    if (a.optionKey === 'F') return 1;
                    return 0;
                  })" :key="option.id" class="option-row">
                    <input 
                      type="radio" 
                      :id="`q${question.id}_${option.optionKey}`" 
                      :name="`question_${question.id}`"
                      :value="option.optionKey" 
                      v-model="answers[question.id]"
                      class="custom-radio"
                      :disabled="submissionStatus > 0"
                    >
                    <label :for="`q${question.id}_${option.optionKey}`" class="option-label">
                      {{ option.optionKey }}. {{ option.content }}
                    </label>
                  </div>
                </div>
              </div>
              
              <!-- 填空题 -->
              <div v-else-if="group.type === 'blank'" class="question-blank">
                <el-input
                  v-model="answers[question.id]"
                  placeholder="输入你的答案"
                  type="text"
                  :disabled="submissionStatus > 0"
                />
              </div>
              
              <!-- 简答题 -->
              <div v-else-if="group.type === 'short'" class="question-short">
                <el-input
                  v-model="answers[question.id]"
                  placeholder="输入你的答案"
                  type="textarea"
                  :rows="4"
                  :disabled="submissionStatus > 0"
                />
              </div>
              
              <!-- 代码题 -->
              <div v-else-if="group.type === 'code'" class="question-code">
                <el-input
                  v-model="answers[question.id]"
                  placeholder="请在此处编写代码"
                  type="textarea"
                  :rows="8"
                  :disabled="submissionStatus > 0"
                />
              </div>
              
              <!-- 保存按钮 -->
              <div class="question-actions">
                <el-button size="small" type="primary" @click="saveAnswer(question.id)" :disabled="submissionStatus > 0">保存答案</el-button>
              </div>
            </div>
          </div>
          
          <!-- 提交按钮 -->
          <div class="submit-section">
            <!-- 移除此处的提交按钮，只保留右上角的按钮 -->
          </div>
        </div>
      </div>
    </div>

    <!-- 提交确认对话框 -->
    <el-dialog
      v-model="submitDialogVisible"
      title="确认提交"
      width="30%"
    >
      <div class="submit-dialog-content">
        <p>您确定要提交本次考试吗？</p>
        <p class="warning">提交后将无法再修改答案！</p>
        <div class="submit-stats">
          <div class="stat-item">
            <span class="stat-label">总题数：</span>
            <span class="stat-value">{{ totalQuestions }}</span>
          </div>
          <div class="stat-item">
            <span class="stat-label">已答题数：</span>
            <span class="stat-value">{{ answeredCount }}</span>
          </div>
          <div class="stat-item">
            <span class="stat-label">未答题数：</span>
            <span class="stat-value">{{ totalQuestions - answeredCount }}</span>
          </div>
        </div>
      </div>
      <template #footer>
        <span class="dialog-footer">
          <el-button @click="submitDialogVisible = false">取消</el-button>
          <el-button type="danger" @click="submitExam">确认提交</el-button>
        </span>
      </template>
    </el-dialog>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted, onBeforeUnmount } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { ElMessage } from 'element-plus'
import { Timer, Loading } from '@element-plus/icons-vue'
import assignmentApi from '@/api/assignment'

const route = useRoute()
const router = useRouter()
const examId = ref<number>(Number(route.params.id) || 0)

// 考试基本信息
const examTitle = ref('')
const examStatus = ref('考试进行中')
const remainingTime = ref(7200) // 默认2小时
const timer = ref<number | null>(null)
const submissionStatus = ref(0) // 提交状态：0-未提交，1-已提交未批改，2-已批改

// 题目数据
interface QuestionOption {
  id: number
  optionKey: string
  content: string
}

interface Question {
  id: number
  title: string
  questionType: string
  difficulty: number
  score: number
  sequence: number
  knowledgePoint: string
  options?: QuestionOption[]
}

interface QuestionGroup {
  type: string
  questions: Question[]
  totalScore: number
}

const questionGroups = ref<QuestionGroup[]>([])
const answers = ref<Record<number, any>>({})
const loading = ref(true)
const error = ref('')

// 提交对话框
const submitDialogVisible = ref(false)

// 计算总题数
const totalQuestions = computed(() => {
  let count = 0
  questionGroups.value.forEach(group => {
    count += group.questions.length
  })
  return count
})

// 计算已答题数
const answeredCount = computed(() => {
  return Object.keys(answers.value).filter(key => {
    const answer = answers.value[Number(key)]
    if (Array.isArray(answer)) {
      return answer.length > 0
    }
    return answer !== undefined && answer !== null && answer !== ''
  }).length
})

// 检查题目是否已答
const isAnswered = (questionId: number) => {
  const answer = answers.value[questionId]
  if (Array.isArray(answer)) {
    return answer.length > 0
  }
  return answer !== undefined && answer !== null && answer !== ''
}

// 格式化剩余时间
const formatTime = (seconds: number) => {
  const hours = Math.floor(seconds / 3600)
  const minutes = Math.floor((seconds % 3600) / 60)
  const secs = seconds % 60
  
  return `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`
}

// 获取题型名称
const getQuestionTypeName = (type: string) => {
  const typeMap: Record<string, string> = {
    'single': '单选题',
    'multiple': '多选题',
    'true_false': '判断题',
    'blank': '填空题',
    'short': '简答题',
    'code': '代码题'
  }
  return typeMap[type] || type
}

// 获取罗马数字
const getRomanNumeral = (num: number) => {
  const roman = ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'X']
  return roman[num - 1] || num.toString()
}

// 滚动到指定题目
const scrollToQuestion = (questionId: number) => {
  const element = document.getElementById(`question-${questionId}`)
  if (element) {
    element.scrollIntoView({ behavior: 'smooth', block: 'start' })
  }
}

// 修改保存答案函数
const saveAnswer = async (questionId: number) => {
  try {
    // 检查是否已提交
    if (submissionStatus.value > 0) {
      ElMessage.warning('考试已提交，无法修改答案');
      return;
    }
    
    const answer = answers.value[questionId];
    
    // 保存到本地存储
    const storageKey = `exam_${examId.value}_answers`;
    localStorage.setItem(storageKey, JSON.stringify(answers.value));
    
    // 保存到服务器
    const response = await assignmentApi.saveQuestionAnswer(examId.value, questionId, {
      answer: answer
    });
    
    if (response.code === 200) {
      ElMessage.success('答案已保存');
    } else {
      ElMessage.error(response.message || '保存失败');
    }
  } catch (err: any) {
    console.error('保存答案失败', err);
    ElMessage.error('保存失败: ' + (err.message || '未知错误'));
  }
};

// 检查多选题选项是否已选中
const isOptionChecked = (questionId: number, optionKey: string) => {
  if (!answers.value[questionId]) {
    answers.value[questionId] = [];
  }
  return Array.isArray(answers.value[questionId]) && answers.value[questionId].includes(optionKey);
};

// 切换多选题选项
const toggleMultipleChoice = (questionId: number, optionKey: string, event: Event) => {
  const isChecked = (event.target as HTMLInputElement).checked;
  
  // 确保是数组
  if (!Array.isArray(answers.value[questionId])) {
    answers.value[questionId] = [];
  }
  
  if (isChecked) {
    // 添加选项
    if (!answers.value[questionId].includes(optionKey)) {
      answers.value[questionId].push(optionKey);
      // 对选项进行排序，确保按ABCD顺序
      answers.value[questionId].sort();
    }
  } else {
    // 移除选项
    const index = answers.value[questionId].indexOf(optionKey);
    if (index !== -1) {
      answers.value[questionId].splice(index, 1);
    }
  }
};

// 自动保存答案
const autoSaveAnswers = () => {
  // 如果已提交，不再自动保存
  if (submissionStatus.value > 0) return;
  
  const storageKey = `exam_${examId.value}_answers`
  localStorage.setItem(storageKey, JSON.stringify(answers.value))
}

// 加载本地存储的答案
const loadSavedAnswers = () => {
  const storageKey = `exam_${examId.value}_answers`
  const savedAnswers = localStorage.getItem(storageKey)
  if (savedAnswers) {
    try {
      const parsedAnswers = JSON.parse(savedAnswers);
      // 确保多选题的答案类型为数组
      questionGroups.value.forEach(group => {
        if (group.type === 'multiple') {
          group.questions.forEach(question => {
            const savedAnswer = parsedAnswers[question.id];
            if (savedAnswer && !Array.isArray(savedAnswer)) {
              // 如果保存的不是数组，则转换为数组
              if (typeof savedAnswer === 'string') {
                parsedAnswers[question.id] = [savedAnswer];
              } else {
                // 如果是其他类型，则初始化为空数组
                parsedAnswers[question.id] = [];
              }
            } else if (!savedAnswer) {
              // 如果没有保存的答案，初始化为空数组
              parsedAnswers[question.id] = [];
            }
          });
        }
      });
      answers.value = parsedAnswers;
    } catch (e) {
      console.error('Failed to parse saved answers', e)
    }
  }
}

// 倒计时
const startTimer = () => {
  timer.value = setInterval(() => {
    if (remainingTime.value > 0) {
      remainingTime.value--
      // 每分钟自动保存一次
      if (remainingTime.value % 60 === 0 && submissionStatus.value === 0) {
        autoSaveAnswers()
      }
    } else {
      // 时间到，自动提交
      if (timer.value) clearInterval(timer.value)
      // 检查是否已提交
      if (submissionStatus.value === 0) {
      ElMessage.warning('考试时间已到，系统将自动提交您的答案')
      submitExam()
      } else {
        ElMessage.info('考试已结束')
      }
    }
  }, 1000) as unknown as number
}

// 提交前确认
const handleSubmit = () => {
  // 检查是否已提交
  if (submissionStatus.value > 0) {
    ElMessage.warning('考试已提交，无法重复提交');
    return;
  }
  submitDialogVisible.value = true
}

// 修改提交考试函数
const submitExam = async () => {
  try {
    // 检查是否已提交
    if (submissionStatus.value > 0) {
      ElMessage.warning('考试已提交，无法重复提交');
      submitDialogVisible.value = false;
      return;
    }
    
    ElMessage.info('正在保存所有答案并提交，请稍候...');
    
    // 自动保存所有题目的答案，即使用户没有手动点击保存
    const savePromises: Promise<any>[] = [];
    
    // 遍历所有题目组
    questionGroups.value.forEach(group => {
      group.questions.forEach(question => {
        // 获取题目ID
        const questionId = question.id;
        
        // 检查是否有该题的答案（即使是空数组也保存，因为多选题可能故意不选）
        if (questionId in answers.value) {
          const savePromise = assignmentApi.saveQuestionAnswer(examId.value, questionId, {
            answer: answers.value[questionId]
          }).catch(e => {
            console.error(`保存题目 ${questionId} 答案失败`, e);
            // 即使保存失败也继续处理其他题目
            return null;
          });
          
          savePromises.push(savePromise);
        }
      });
    });
    
    // 等待所有保存操作完成
    await Promise.all(savePromises);
    
    // 提交到服务器
    const response = await assignmentApi.submitAssignment(examId.value, {});
    
    if (response.code === 200) {
      // 更新提交状态
      submissionStatus.value = 1;
      examStatus.value = '已提交';
      
      ElMessage.success('提交成功');
      // 清除本地存储
      localStorage.removeItem(`exam_${examId.value}_answers`);
      // 跳转到考试详情页
      router.push(`/student/exams/${examId.value}`);
    } else {
      ElMessage.error(response.message || '提交失败');
      submitDialogVisible.value = false;
    }
  } catch (err: any) {
    console.error('提交考试失败', err);
    ElMessage.error('提交失败: ' + (err.message || '未知错误'));
    submitDialogVisible.value = false;
  }
};

// 加载考试题目
const loadExamQuestions = async () => {
  try {
    loading.value = true
    error.value = ''
    
    // 先获取考试信息和提交状态
    const detailResponse = await assignmentApi.getAssignmentDetail(examId.value);
    
    if (detailResponse.code === 200 && detailResponse.data.submission) {
      submissionStatus.value = detailResponse.data.submission.status || 0;
      
      // 如果已提交，显示提示信息
      if (submissionStatus.value > 0) {
        examStatus.value = '已提交';
        ElMessage.info('本次考试已提交，您可以查看题目但无法修改答案');
      }
    }
    
    const response = await assignmentApi.getAssignmentQuestionsStudent(examId.value)
    
    if (response.code === 200) {
      questionGroups.value = response.data
      
      // 设置考试信息
      examTitle.value = route.query.title?.toString() || '考试'
      
      // 设置考试时间
      if (route.query.timeLimit) {
        remainingTime.value = parseInt(route.query.timeLimit.toString()) * 60 // 转换为秒
      }
      
      // 初始化多选题的答案为空数组
      questionGroups.value.forEach(group => {
        if (group.type === 'multiple') {
          group.questions.forEach(question => {
            if (!answers.value[question.id] || !Array.isArray(answers.value[question.id])) {
              answers.value[question.id] = [];
            }
          });
        }
      });
      
      // 加载保存的答案
      loadSavedAnswers()
      
      // 只有未提交状态才开始倒计时
      if (submissionStatus.value === 0) {
      startTimer()
      }
    } else {
      error.value = response.message || '加载题目失败'
    }
  } catch (err: any) {
    console.error('加载考试题目失败', err)
    error.value = err.message || '加载题目失败'
  } finally {
    loading.value = false
  }
}

// 页面离开确认
const setupBeforeUnloadListener = () => {
  window.addEventListener('beforeunload', (e) => {
    // 自动保存
    autoSaveAnswers()
    
    // 显示确认提示
    e.preventDefault()
    e.returnValue = '考试尚未完成，确定要离开吗？'
    return '考试尚未完成，确定要离开吗？'
  })
}

// 生命周期钩子
onMounted(() => {
  loadExamQuestions()
  setupBeforeUnloadListener()
})

onBeforeUnmount(() => {
  // 清除定时器
  if (timer.value) {
    clearInterval(timer.value)
  }
  
  // 自动保存
  autoSaveAnswers()
  
  // 移除事件监听
  window.removeEventListener('beforeunload', () => {})
})
</script>

<style scoped>
.exam-container {
  display: flex;
  flex-direction: column;
  height: 100%;
  background-color: #f5f7fa;
}

.exam-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 15px 20px;
  background-color: #fff;
  box-shadow: 0 2px 12px 0 rgba(0, 0, 0, 0.1);
}

.exam-header h1 {
  margin: 0;
  font-size: 20px;
}

.exam-status {
  display: flex;
  align-items: center;
  gap: 20px;
}

.status-tag {
  padding: 4px 8px;
  border-radius: 4px;
  font-size: 14px;
}

.status-tag.in-progress {
  background-color: #e6f7ff;
  color: #1890ff;
}

.status-tag.submitted {
  background-color: #f6ffed;
  color: #52c41a;
}

.timer {
  display: flex;
  align-items: center;
  gap: 5px;
  font-size: 16px;
  font-weight: bold;
  color: #f56c6c;
}

.exam-content {
  display: flex;
  flex: 1;
  padding: 20px;
  gap: 20px;
  overflow: hidden;
}

.question-nav {
  width: 200px;
  background-color: #fff;
  border-radius: 4px;
  padding: 15px;
  box-shadow: 0 2px 12px 0 rgba(0, 0, 0, 0.1);
  overflow-y: auto;
}

.question-nav h3 {
  margin-top: 0;
  margin-bottom: 15px;
  font-size: 16px;
  border-bottom: 1px solid #eee;
  padding-bottom: 10px;
}

.question-group-title {
  font-size: 14px;
  font-weight: bold;
  margin: 10px 0 5px;
}

.question-buttons {
  display: flex;
  flex-wrap: wrap;
  gap: 5px;
  margin-bottom: 15px;
}

.question-content {
  flex: 1;
  background-color: #fff;
  border-radius: 4px;
  padding: 20px;
  box-shadow: 0 2px 12px 0 rgba(0, 0, 0, 0.1);
  overflow-y: auto;
}

.loading-container {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  height: 300px;
}

.loading-icon {
  margin-bottom: 20px;
}

.loading-text {
  color: #909399;
}

.question-section {
  margin-bottom: 30px;
}

.section-header {
  margin-bottom: 20px;
  border-bottom: 2px solid #409eff;
  padding-bottom: 10px;
}

.section-header h2 {
  font-size: 18px;
  margin: 0;
}

.section-info {
  font-size: 14px;
  font-weight: normal;
  color: #606266;
}

.question-item {
  margin-bottom: 30px;
  padding: 20px;
  border: 1px solid #ebeef5;
  border-radius: 4px;
}

/* 添加已提交状态下的样式 */
.question-item input:disabled,
.question-item .el-input.is-disabled .el-input__inner,
.question-item .el-textarea.is-disabled .el-textarea__inner {
  cursor: not-allowed;
  background-color: #f5f7fa;
  color: #909399;
  border-color: #e4e7ed;
}

.question-header {
  margin-bottom: 15px;
}

.question-number {
  font-weight: bold;
  margin-right: 5px;
}

.question-title {
  font-weight: bold;
}

.question-score {
  color: #f56c6c;
  margin-left: 10px;
}

.question-options {
  margin: 20px 0;
}

.options-container {
  display: flex;
  flex-direction: column;
  align-items: flex-start;
  width: 100%;
}

.option-row {
  margin: 15px 0;
  display: flex;
  align-items: flex-start;
  width: 100%;
}

.custom-radio,
.custom-checkbox {
  margin-top: 3px;
  margin-right: 10px;
}

.option-label {
  cursor: pointer;
}

.question-blank,
.question-short,
.question-code {
  margin-bottom: 15px;
}

.question-actions {
  display: flex;
  justify-content: flex-end;
}

.submit-section {
  display: flex;
  justify-content: center;
  margin-top: 30px;
  padding-top: 20px;
  border-top: 1px solid #ebeef5;
}

.submit-dialog-content {
  text-align: center;
}

.submit-dialog-content .warning {
  color: #f56c6c;
  font-weight: bold;
}

.submit-stats {
  margin-top: 20px;
  text-align: left;
}

.stat-item {
  margin-bottom: 10px;
}

.stat-label {
  font-weight: bold;
}
</style> 