<template>
  <div class="exam-detail-page">
    <!-- 考试头部信息 -->
    <div class="exam-header">
      <h1 class="exam-title">{{ exam.title }}</h1>
      <div class="exam-info">
        <div class="info-item">
          <span class="info-label">题量：</span>
          <span class="info-value">{{ exam.questionCount || '0' }}</span>
        </div>
        <div class="info-item">
          <span class="info-label">满分：</span>
          <span class="info-value">{{ exam.totalScore || '100' }}分</span>
        </div>
        <div class="info-item">
          <span class="info-label">考试时间：</span>
          <span class="info-value">
            {{ formatDate(exam.startTime) }} ~ {{ formatDate(exam.endTime) }}
          </span>
        </div>
      </div>
      <div v-if="exam.description" class="exam-description">
        <div class="description-label">考试说明：</div>
        <div class="description-content">{{ exam.description }}</div>
      </div>
    </div>

    <!-- 题目区域 -->
    <div class="exam-content">
      <a-spin :spinning="loading">
        <div v-if="!questions || questions.length === 0" class="empty-content">
          <a-empty description="暂无题目" />
        </div>
        <div v-else class="question-list">
          <!-- 按题型分组显示题目 -->
          <div v-for="(group, groupIndex) in questionGroups" :key="group.type" class="question-group">
            <div class="group-header">
              <h2 class="group-title">{{ getRomanNumber(groupIndex + 1) }}、{{ getQuestionTypeText(group.type) }} （共{{ group.questions.length }}题，{{ group.totalScore }}分）</h2>
            </div>
            
            <div class="questions">
              <div v-for="(question, questionIndex) in group.questions" :key="question.id" class="question-item">
                <div class="question-header">
                  <div class="question-index">{{ questionIndex + 1 }}.（{{ getQuestionTypeText(question.questionType) }}，{{ question.score }}分）</div>
                </div>
                
                <div class="question-content">{{ question.title }}</div>
                
                <!-- 选择题选项 -->
                <div v-if="['single', 'multiple'].includes(question.questionType)" class="question-options">
                  <div v-for="option in question.options" :key="option.id" class="option-item">
                    <a-radio-group v-if="question.questionType === 'single'" v-model:value="answers[question.id]">
                      <a-radio :value="option.optionKey">{{ option.optionKey }}. {{ option.content }}</a-radio>
                    </a-radio-group>
                    
                    <a-checkbox-group v-else-if="question.questionType === 'multiple'" v-model:value="answers[question.id]">
                      <a-checkbox :value="option.optionKey">{{ option.optionKey }}. {{ option.content }}</a-checkbox>
                    </a-checkbox-group>
                  </div>
                </div>
                
                <!-- 判断题选项 -->
                <div v-else-if="question.questionType === 'true_false'" class="question-options">
                  <a-radio-group v-model:value="answers[question.id]">
                    <a-radio value="true">正确</a-radio>
                    <a-radio value="false">错误</a-radio>
                  </a-radio-group>
                </div>
                
                <!-- 填空题 -->
                <div v-else-if="question.questionType === 'blank'" class="question-blank">
                  <a-input v-model:value="answers[question.id]" placeholder="输入你的答案" />
                </div>
                
                <!-- 简答题 -->
                <div v-else-if="question.questionType === 'short'" class="question-short">
                  <a-textarea v-model:value="answers[question.id]" placeholder="输入你的答案" :rows="4" />
                </div>
                
                <!-- 其他题型 -->
                <div v-else class="question-other">
                  <a-textarea v-model:value="answers[question.id]" placeholder="输入你的答案" :rows="6" />
                </div>
                
                <!-- 保存按钮 -->
                <div class="question-actions">
                  <a-button type="primary" size="small" @click="saveAnswer(question.id)">保存</a-button>
                </div>
              </div>
            </div>
          </div>
        </div>
      </a-spin>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted, computed, watch } from 'vue'
import { useRoute } from 'vue-router'
import { message } from 'ant-design-vue'
import { formatDate } from '@/utils/date'
import examAPI from '@/api/exam'

// 路由参数
const route = useRoute()
const examId = computed(() => Number(route.params.id))

// 状态定义
const loading = ref(true)
const exam = ref<any>({})
const questions = ref<any[]>([])
const answers = ref<Record<number, any>>({})

// 计算按题型分组的题目
const questionGroups = computed(() => {
  if (!questions.value || questions.value.length === 0) return []
  
  const groups: Record<string, any> = {}
  
  // 按题型分组
  questions.value.forEach(q => {
    const type = q.questionType || 'other'
    if (!groups[type]) {
      groups[type] = {
        type,
        questions: [],
        totalScore: 0
      }
    }
    groups[type].questions.push(q)
    groups[type].totalScore += (q.score || 0)
  })
  
  // 转换为数组并排序
  const order = ['single', 'multiple', 'true_false', 'blank', 'short', 'code', 'other']
  return Object.values(groups).sort((a, b) => {
    return order.indexOf(a.type) - order.indexOf(b.type)
  })
})

// 获取考试详情
const fetchExamDetail = async () => {
  loading.value = true
  try {
    // 调用API获取考试详情
    const response = await examAPI.getExamDetail(examId.value)
    
    if (response && response.code === 200) {
      exam.value = response.data
      // 初始化问题
      fetchExamQuestions()
    } else {
      message.error(response?.message || '获取考试信息失败')
    }
  } catch (error) {
    console.error('获取考试详情失败:', error)
    message.error('获取考试详情失败')
  } finally {
    loading.value = false
  }
}

// 获取考试题目
const fetchExamQuestions = async () => {
  loading.value = true
  try {
    // 调用API获取考试题目
    // 注：这里需要后端提供获取考试题目的API
    const response = await examAPI.getExamQuestions(examId.value)
    
    if (response && response.code === 200) {
      questions.value = response.data || []
      
      // 初始化答案对象
      questions.value.forEach(q => {
        // 根据题型初始化不同类型的默认值
        if (q.questionType === 'multiple') {
          answers.value[q.id] = []  // 多选题初始化为空数组
        } else {
          answers.value[q.id] = ''  // 其他题型初始化为空字符串
        }
      })
    } else {
      message.error(response?.message || '获取考试题目失败')
    }
  } catch (error) {
    console.error('获取考试题目失败:', error)
    message.error('获取考试题目失败')
  } finally {
    loading.value = false
  }
}

// 保存答案
const saveAnswer = async (questionId: number) => {
  try {
    const answer = answers.value[questionId]
    
    // 检查答案是否为空
    if (answer === undefined || answer === null || 
        (Array.isArray(answer) && answer.length === 0) || 
        (typeof answer === 'string' && answer.trim() === '')) {
      message.warning('请先填写答案')
      return
    }
    
    // 调用API保存答案
    const response = await examAPI.saveExamAnswer(examId.value, questionId, answer)
    
    if (response && response.code === 200) {
      message.success('答案保存成功')
    } else {
      message.error(response?.message || '答案保存失败')
    }
  } catch (error) {
    console.error('保存答案失败:', error)
    message.error('保存答案失败')
  }
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
    'code': '编程题',
    'other': '其他题型'
  }
  return typeMap[type] || ''
}

// 生命周期钩子
onMounted(() => {
  fetchExamDetail()
})
</script>

<style scoped>
.exam-detail-page {
  padding: 24px;
  background-color: #fff;
}

.exam-header {
  margin-bottom: 24px;
  padding-bottom: 20px;
  border-bottom: 1px solid #e8e8e8;
}

.exam-title {
  font-size: 24px;
  font-weight: 600;
  margin-bottom: 16px;
}

.exam-info {
  display: flex;
  flex-wrap: wrap;
  gap: 24px;
  margin-bottom: 16px;
}

.info-item {
  display: flex;
  align-items: center;
}

.info-label {
  font-weight: 600;
  margin-right: 8px;
}

.exam-description {
  background-color: #f5f7fa;
  padding: 12px 16px;
  border-radius: 4px;
  margin-top: 16px;
}

.description-label {
  font-weight: 600;
  margin-bottom: 8px;
}

.description-content {
  white-space: pre-line;
}

.exam-content {
  margin-top: 24px;
}

.question-group {
  margin-bottom: 32px;
}

.group-header {
  margin-bottom: 16px;
}

.group-title {
  font-size: 18px;
  font-weight: 600;
}

.questions {
  margin-left: 12px;
}

.question-item {
  margin-bottom: 24px;
  padding: 16px;
  border: 1px solid #e8e8e8;
  border-radius: 8px;
  background-color: #fff;
}

.question-header {
  display: flex;
  align-items: center;
  margin-bottom: 12px;
}

.question-index {
  font-weight: 600;
}

.question-content {
  margin-bottom: 16px;
  line-height: 1.6;
}

.question-options {
  margin-left: 8px;
}

.option-item {
  margin-bottom: 8px;
}

.question-blank,
.question-short,
.question-other {
  margin-top: 16px;
}

.question-actions {
  display: flex;
  justify-content: flex-end;
  margin-top: 16px;
}
</style> 