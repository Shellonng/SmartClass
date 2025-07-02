<template>
  <div class="question-detail-page">
    <div class="page-header">
      <a-button @click="goBack">
        <ArrowLeftOutlined /> 返回题库
      </a-button>
      <h2>题目详情</h2>
    </div>

    <a-spin :spinning="loading">
      <a-card v-if="question" class="question-card">
        <div class="question-header">
          <div class="question-meta">
            <a-tag :color="getQuestionTypeColor(question.questionType)">{{ question.questionTypeDesc }}</a-tag>
            <a-rate :value="question.difficulty" disabled :count="5" />
            <span class="question-date">创建时间: {{ formatDate(question.createTime) }}</span>
          </div>
          <div class="question-actions">
            <a-button type="primary" @click="editQuestion">
              <EditOutlined /> 编辑
            </a-button>
            <a-popconfirm
              title="确定要删除这个题目吗？"
              @confirm="handleDeleteQuestion"
              ok-text="确定"
              cancel-text="取消"
            >
              <a-button danger>
                <DeleteOutlined /> 删除
              </a-button>
            </a-popconfirm>
          </div>
        </div>

        <div class="question-content">
          <h3>题目内容</h3>
          <div class="content-text">{{ question.title }}</div>
        </div>

        <!-- 选项 -->
        <div v-if="question.options && question.options.length > 0" class="question-options">
          <h3>选项</h3>
          <div class="options-list">
            <div v-for="option in question.options" :key="option.id" class="option-item">
              <strong>{{ option.optionLabel }}.</strong> {{ option.optionText }}
            </div>
          </div>
        </div>

        <div class="question-answer">
          <h3>标准答案</h3>
          <div class="answer-text">{{ question.correctAnswer }}</div>
        </div>

        <div v-if="question.explanation" class="question-explanation">
          <h3>解析</h3>
          <div class="explanation-text">{{ question.explanation }}</div>
        </div>

        <div v-if="question.knowledgePoint" class="question-knowledge">
          <h3>知识点</h3>
          <a-tag color="cyan">{{ question.knowledgePoint }}</a-tag>
        </div>

        <div class="question-info">
          <div v-if="question.courseName" class="info-item">
            <span class="info-label">所属课程:</span>
            <span class="info-value">{{ question.courseName }}</span>
          </div>
          <div v-if="question.chapterName" class="info-item">
            <span class="info-label">所属章节:</span>
            <span class="info-value">{{ question.chapterName }}</span>
          </div>
          <div v-if="question.teacherName" class="info-item">
            <span class="info-label">出题教师:</span>
            <span class="info-value">{{ question.teacherName }}</span>
          </div>
        </div>
      </a-card>

      <a-empty v-else description="题目不存在或已被删除" />
    </a-spin>

    <!-- 编辑题目弹窗 -->
    <a-modal
      v-model:open="modalVisible"
      title="编辑题目"
      :maskClosable="false"
      @ok="handleSave"
      :okButtonProps="{ loading: saving }"
      :okText="saving ? '保存中...' : '保存'"
      width="700px"
    >
      <a-form :model="form" layout="vertical">
        <a-form-item label="题目内容" required>
          <a-textarea v-model:value="form.title" placeholder="请输入题目内容" :rows="4" />
        </a-form-item>
        <a-form-item label="题目类型" required>
          <a-select v-model:value="form.questionType" placeholder="请选择题目类型">
            <a-select-option v-for="(desc, type) in QuestionTypeDesc" :key="type" :value="type">{{ desc }}</a-select-option>
          </a-select>
        </a-form-item>
        <a-form-item label="难度" required>
          <a-rate v-model:value="form.difficulty" :count="5" />
        </a-form-item>
        <a-form-item label="知识点">
          <a-select v-model:value="form.knowledgePoint" placeholder="请选择知识点" allowClear>
            <a-select-option v-for="point in knowledgePoints" :key="point" :value="point">{{ point }}</a-select-option>
          </a-select>
        </a-form-item>
        
        <!-- 选择题选项 -->
        <template v-if="['single', 'multiple', 'true_false'].includes(form.questionType)">
          <a-divider>选项</a-divider>
          <a-form-item v-for="(option, index) in form.options" :key="index">
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
          <a-textarea v-model:value="form.correctAnswer" placeholder="请输入标准答案" :rows="3" />
        </a-form-item>
        <a-form-item label="答案解析">
          <a-textarea v-model:value="form.explanation" placeholder="请输入答案解析" :rows="3" />
        </a-form-item>
        
        <a-form-item v-if="form.courseId" label="章节">
          <a-select v-model:value="form.chapterId" placeholder="请选择章节">
            <a-select-option v-for="chapter in chapters" :key="chapter.id" :value="chapter.id">{{ chapter.title }}</a-select-option>
          </a-select>
        </a-form-item>
      </a-form>
    </a-modal>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { message } from 'ant-design-vue'
import {
  ArrowLeftOutlined,
  EditOutlined,
  DeleteOutlined,
  PlusOutlined
} from '@ant-design/icons-vue'
import { formatDate } from '@/utils/date'
import {
  QuestionType,
  QuestionTypeDesc,
  getQuestionDetail,
  updateQuestion,
  deleteQuestion
} from '@/api/question'
import type { Question, QuestionOption } from '@/api/question'

const route = useRoute()
const router = useRouter()
const questionId = ref(Number(route.params.id))
const courseId = ref(Number(route.query.courseId || 0))

// 状态定义
const question = ref<Question | null>(null)
const loading = ref(false)
const modalVisible = ref(false)
const saving = ref(false)
const form = ref<Question & { options: QuestionOption[] }>({
  id: undefined,
  title: '',
  questionType: QuestionType.SINGLE,
  difficulty: 3,
  knowledgePoint: '',
  correctAnswer: '',
  explanation: '',
  courseId: 0,
  chapterId: 0,
  options: []
})

// 章节列表
const chapters = ref<{ id: number, title: string }[]>([])

// 知识点列表 (示例数据，实际应该从后端获取)
const knowledgePoints = ref([
  'JavaScript', 'HTML', 'CSS', 'React', 'Vue', 'Node.js', 
  '数据库', '网络', '算法', '设计模式', '操作系统'
])

// 获取题目详情
const fetchQuestionDetail = async () => {
  loading.value = true
  try {
    const res = await getQuestionDetail(questionId.value)
    question.value = res.data
    loading.value = false
  } catch (error) {
    console.error('获取题目详情失败:', error)
    message.error('获取题目详情失败')
    loading.value = false
  }
}

// 获取章节列表
const fetchChapters = async () => {
  try {
    // 这里应该调用获取章节列表的API
    // 暂时使用模拟数据
    chapters.value = [
      { id: 1, title: '第一章：基础知识' },
      { id: 2, title: '第二章：进阶内容' },
      { id: 3, title: '第三章：高级特性' }
    ]
  } catch (error) {
    console.error('获取章节列表失败:', error)
  }
}

// 返回题库列表
const goBack = () => {
  if (courseId.value) {
    router.push(`/teacher/course/${courseId.value}/question-bank`)
  } else {
    router.push('/teacher/question-bank')
  }
}

// 编辑题目
const editQuestion = () => {
  if (question.value) {
    form.value = {
      ...question.value,
      options: question.value.options || []
    }
    modalVisible.value = true
  }
}

// 添加选项
const addOption = () => {
  const label = String.fromCharCode(65 + form.value.options.length) // A, B, C...
  form.value.options.push({
    optionLabel: label,
    optionText: ''
  })
}

// 删除选项
const removeOption = (index: number) => {
  form.value.options.splice(index, 1)
  // 重新排序选项标签
  form.value.options.forEach((option, idx) => {
    option.optionLabel = String.fromCharCode(65 + idx)
  })
}

// 保存题目
const handleSave = async () => {
  // 表单验证
  if (!form.value.title) {
    message.error('请输入题目内容')
    return
  }
  if (!form.value.correctAnswer) {
    message.error('请输入标准答案')
    return
  }
  if (['single', 'multiple', 'true_false'].includes(form.value.questionType) && 
      form.value.options.length === 0) {
    message.error('请添加选项')
    return
  }

  saving.value = true
  try {
    await updateQuestion(form.value)
    message.success('题目更新成功')
    modalVisible.value = false
    fetchQuestionDetail()
  } catch (error) {
    console.error('保存题目失败:', error)
    message.error('保存题目失败')
  } finally {
    saving.value = false
  }
}

// 删除题目
const handleDeleteQuestion = async () => {
  try {
    await deleteQuestion(questionId.value)
    message.success('题目删除成功')
    goBack()
  } catch (error) {
    console.error('删除题目失败:', error)
    message.error('删除题目失败')
  }
}

// 工具函数
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

// 生命周期钩子
onMounted(() => {
  fetchQuestionDetail()
  fetchChapters()
})
</script>

<style scoped>
.question-detail-page {
  padding: 24px;
  background-color: #fff;
}

.page-header {
  display: flex;
  align-items: center;
  margin-bottom: 24px;
}

.page-header h2 {
  margin: 0 0 0 16px;
  font-size: 20px;
  font-weight: 600;
}

.question-card {
  margin-bottom: 24px;
}

.question-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 24px;
}

.question-meta {
  display: flex;
  align-items: center;
  gap: 16px;
}

.question-date {
  color: #999;
  font-size: 14px;
}

.question-actions {
  display: flex;
  gap: 8px;
}

.question-content,
.question-options,
.question-answer,
.question-explanation,
.question-knowledge {
  margin-bottom: 24px;
}

.question-content h3,
.question-options h3,
.question-answer h3,
.question-explanation h3,
.question-knowledge h3 {
  font-size: 16px;
  font-weight: 600;
  margin-bottom: 12px;
}

.content-text,
.answer-text,
.explanation-text {
  white-space: pre-line;
  line-height: 1.6;
}

.options-list {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.option-item {
  padding: 8px 12px;
  background-color: #f5f7fa;
  border-radius: 4px;
}

.question-info {
  border-top: 1px solid #eee;
  padding-top: 16px;
  margin-top: 24px;
}

.info-item {
  margin-bottom: 8px;
}

.info-label {
  font-weight: 600;
  margin-right: 8px;
}
</style> 