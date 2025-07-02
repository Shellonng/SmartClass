<template>
  <div class="wrong-questions">
    <div class="page-header">
      <h1 class="page-title">错题集</h1>
      <p class="page-description">记录学习中的错误，加强薄弱环节</p>
    </div>

    <div class="content-wrapper">
      <div class="filter-section">
        <a-select
          v-model:value="subjectFilter"
          placeholder="选择科目"
          style="width: 150px"
          @change="handleFilter"
        >
          <a-select-option value="">全部科目</a-select-option>
          <a-select-option v-for="subject in subjects" :key="subject" :value="subject">
            {{ subject }}
          </a-select-option>
        </a-select>

        <a-select
          v-model:value="typeFilter"
          placeholder="题目类型"
          style="width: 150px"
          @change="handleFilter"
        >
          <a-select-option value="">全部类型</a-select-option>
          <a-select-option value="single">单选题</a-select-option>
          <a-select-option value="multiple">多选题</a-select-option>
          <a-select-option value="fill">填空题</a-select-option>
          <a-select-option value="essay">问答题</a-select-option>
        </a-select>

        <a-input-search
          v-model:value="searchKeyword"
          placeholder="搜索题目内容..."
          style="width: 250px"
          @search="handleSearch"
        />
      </div>

      <div class="questions-list">
        <a-spin :spinning="loading">
          <a-empty v-if="questions.length === 0" description="暂无错题记录" />
          
          <div v-else class="question-cards">
            <a-card v-for="question in questions" :key="question.id" class="question-card">
              <template #title>
                <div class="question-header">
                  <a-tag :color="getQuestionTypeColor(question.type)">{{ getQuestionTypeText(question.type) }}</a-tag>
                  <span class="question-source">{{ question.source }}</span>
                </div>
              </template>
              
              <div class="question-content">
                <div class="question-text" v-html="question.content"></div>
                
                <div v-if="question.type === 'single' || question.type === 'multiple'" class="question-options">
                  <div v-for="option in question.options" :key="option.id" class="option-item">
                    <a-tag 
                      :color="getOptionColor(option, question)"
                      style="min-width: 32px; text-align: center; margin-right: 8px;"
                    >
                      {{ option.label }}
                    </a-tag>
                    <span>{{ option.content }}</span>
                  </div>
                </div>
                
                <div v-else-if="question.type === 'fill'" class="question-fill">
                  <div class="correct-answer">
                    <span class="label">正确答案：</span>
                    <span class="answer">{{ question.answer }}</span>
                  </div>
                  <div class="your-answer">
                    <span class="label">你的答案：</span>
                    <span class="answer wrong">{{ question.yourAnswer }}</span>
                  </div>
                </div>
                
                <div v-else-if="question.type === 'essay'" class="question-essay">
                  <div class="correct-answer">
                    <div class="label">参考答案：</div>
                    <div class="answer" v-html="question.answer"></div>
                  </div>
                  <div class="your-answer">
                    <div class="label">你的答案：</div>
                    <div class="answer wrong" v-html="question.yourAnswer"></div>
                  </div>
                </div>
              </div>
              
              <div class="question-analysis">
                <div class="analysis-title">解析：</div>
                <div class="analysis-content" v-html="question.analysis"></div>
              </div>
              
              <div class="question-footer">
                <span class="wrong-count">错误次数：{{ question.wrongCount }}</span>
                <span class="last-wrong-time">最近错误：{{ formatDate(question.lastWrongTime) }}</span>
                <div class="question-actions">
                  <a-button type="primary" size="small" @click="practiceAgain(question)">
                    再次练习
                  </a-button>
                  <a-button size="small" @click="markAsMastered(question)">
                    标记为已掌握
                  </a-button>
                </div>
              </div>
            </a-card>
          </div>
          
          <div class="pagination">
            <a-pagination
              v-model:current="pagination.current"
              :total="pagination.total"
              :pageSize="pagination.pageSize"
              @change="handlePageChange"
              show-size-changer
              show-quick-jumper
            />
          </div>
        </a-spin>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { message } from 'ant-design-vue'
import dayjs from 'dayjs'

// 筛选条件
const subjectFilter = ref<string>('')
const typeFilter = ref<string>('')
const searchKeyword = ref<string>('')
const subjects = ref<string[]>(['数学', '英语', '物理', '化学', '计算机'])

// 分页
const pagination = ref({
  current: 1,
  pageSize: 10,
  total: 0
})

// 数据
const loading = ref<boolean>(false)
const questions = ref<any[]>([])

// 加载错题数据
const loadQuestions = async () => {
  try {
    loading.value = true
    
    // 模拟API调用
    await new Promise(resolve => setTimeout(resolve, 1000))
    
    // 模拟数据
    questions.value = [
      {
        id: 1,
        type: 'single',
        content: '在计算机网络中，以下哪个协议工作在应用层？',
        options: [
          { id: 1, label: 'A', content: 'TCP' },
          { id: 2, label: 'B', content: 'IP' },
          { id: 3, label: 'C', content: 'HTTP' },
          { id: 4, label: 'D', content: 'ARP' }
        ],
        answer: 'C',
        yourAnswer: 'A',
        analysis: 'HTTP（超文本传输协议）工作在应用层，TCP工作在传输层，IP工作在网络层，ARP工作在网络接口层。',
        wrongCount: 2,
        lastWrongTime: '2025-06-30 14:30:22',
        source: '计算机网络 - 第3章测验'
      },
      {
        id: 2,
        type: 'multiple',
        content: '以下哪些是Java的基本数据类型？',
        options: [
          { id: 1, label: 'A', content: 'int' },
          { id: 2, label: 'B', content: 'String' },
          { id: 3, label: 'C', content: 'boolean' },
          { id: 4, label: 'D', content: 'Float' }
        ],
        answer: ['A', 'C'],
        yourAnswer: ['A', 'B', 'C'],
        analysis: 'Java的基本数据类型包括byte、short、int、long、float、double、boolean和char。String是引用类型，Float是包装类。',
        wrongCount: 1,
        lastWrongTime: '2025-07-01 09:15:30',
        source: 'Java程序设计 - 期中考试'
      },
      {
        id: 3,
        type: 'fill',
        content: '微观经济学中，需求量与价格呈________关系。',
        answer: '反比',
        yourAnswer: '正比',
        analysis: '根据需求定律，在其他条件不变的情况下，商品的价格与需求量呈反比关系，即价格上升，需求量下降；价格下降，需求量上升。',
        wrongCount: 3,
        lastWrongTime: '2025-07-02 16:45:12',
        source: '微观经济学原理 - 第2章作业'
      }
    ]
    
    pagination.value.total = 23 // 模拟总数据量
    
  } catch (error) {
    console.error('加载错题数据失败:', error)
    message.error('加载错题数据失败')
  } finally {
    loading.value = false
  }
}

// 处理筛选
const handleFilter = () => {
  pagination.value.current = 1
  loadQuestions()
}

// 处理搜索
const handleSearch = () => {
  pagination.value.current = 1
  loadQuestions()
}

// 处理分页
const handlePageChange = (page: number, pageSize?: number) => {
  pagination.value.current = page
  if (pageSize) {
    pagination.value.pageSize = pageSize
  }
  loadQuestions()
}

// 获取题目类型文本
const getQuestionTypeText = (type: string): string => {
  const typeMap: Record<string, string> = {
    'single': '单选题',
    'multiple': '多选题',
    'fill': '填空题',
    'essay': '问答题'
  }
  return typeMap[type] || '未知类型'
}

// 获取题目类型颜色
const getQuestionTypeColor = (type: string): string => {
  const colorMap: Record<string, string> = {
    'single': 'blue',
    'multiple': 'purple',
    'fill': 'green',
    'essay': 'orange'
  }
  return colorMap[type] || 'default'
}

// 获取选项颜色
const getOptionColor = (option: any, question: any): string => {
  if (Array.isArray(question.answer)) {
    // 多选题
    const isCorrect = question.answer.includes(option.label)
    const isSelected = question.yourAnswer.includes(option.label)
    
    if (isCorrect && isSelected) return 'success'
    if (isCorrect) return 'success'
    if (isSelected) return 'error'
    return 'default'
  } else {
    // 单选题
    if (option.label === question.answer) return 'success'
    if (option.label === question.yourAnswer) return 'error'
    return 'default'
  }
}

// 格式化日期
const formatDate = (date: string): string => {
  return dayjs(date).format('YYYY-MM-DD HH:mm')
}

// 再次练习
const practiceAgain = (question: any) => {
  message.info(`开始练习题目：${question.id}`)
}

// 标记为已掌握
const markAsMastered = (question: any) => {
  message.success(`已将题目 ${question.id} 标记为已掌握`)
  // 实际应调用API
  questions.value = questions.value.filter(q => q.id !== question.id)
}

onMounted(() => {
  loadQuestions()
})
</script>

<style scoped>
.wrong-questions {
  padding: 24px;
}

.page-header {
  margin-bottom: 24px;
}

.page-title {
  font-size: 24px;
  font-weight: 600;
  margin: 0 0 8px 0;
}

.page-description {
  color: #666;
  margin: 0;
}

.content-wrapper {
  background: white;
  border-radius: 8px;
  padding: 24px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);
}

.filter-section {
  display: flex;
  gap: 16px;
  margin-bottom: 24px;
  flex-wrap: wrap;
}

.questions-list {
  min-height: 400px;
}

.question-cards {
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.question-card {
  margin-bottom: 16px;
}

.question-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.question-source {
  color: #999;
  font-size: 14px;
}

.question-content {
  margin-bottom: 16px;
}

.question-text {
  margin-bottom: 16px;
  font-weight: 500;
}

.question-options {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.option-item {
  display: flex;
  align-items: center;
}

.question-fill, .question-essay {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.correct-answer, .your-answer {
  display: flex;
  gap: 8px;
}

.label {
  font-weight: 500;
  min-width: 80px;
}

.answer.wrong {
  color: #ff4d4f;
}

.question-analysis {
  margin-top: 16px;
  padding-top: 16px;
  border-top: 1px dashed #eee;
}

.analysis-title {
  font-weight: 500;
  margin-bottom: 8px;
}

.analysis-content {
  color: #666;
}

.question-footer {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-top: 16px;
  flex-wrap: wrap;
  gap: 8px;
}

.wrong-count, .last-wrong-time {
  color: #999;
  font-size: 14px;
}

.question-actions {
  display: flex;
  gap: 8px;
}

.pagination {
  margin-top: 24px;
  text-align: center;
}
</style> 