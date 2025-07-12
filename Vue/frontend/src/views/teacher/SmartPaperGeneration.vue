<template>
  <div class="smart-paper-generation">
    <div class="page-header">
      <h1>🧠 智能组卷</h1>
      <p class="description">基于AI技术，智能生成高质量试卷</p>
    </div>

    <div class="generation-container">
      <!-- 左侧：参数配置 -->
      <div class="config-panel">
        <a-card title="📝 组卷配置" class="config-card">
          <a-form 
            :model="formData" 
            :label-col="{ span: 6 }" 
            :wrapper-col="{ span: 18 }"
            @finish="handleGenerate"
          >
            <!-- 课程选择 -->
            <a-form-item label="选择课程" name="courseId" :rules="[{ required: true, message: '请选择课程' }]">
              <a-select 
                v-model:value="formData.courseId" 
                placeholder="请选择课程"
                @change="handleCourseChange"
                :loading="coursesLoading"
              >
                <a-select-option v-for="course in courses" :key="course.id" :value="course.id">
                  {{ course.title || course.name }}
                </a-select-option>
              </a-select>
            </a-form-item>

            <!-- 知识点选择 -->
            <a-form-item label="知识点范围" name="knowledgePoints" :rules="[{ required: true, message: '请选择知识点' }]">
              <a-select 
                v-model:value="formData.knowledgePoints" 
                mode="multiple"
                placeholder="请选择要考查的知识点"
                :loading="chaptersLoading"
              >
                <a-select-option v-for="point in knowledgePointOptions" :key="point.id" :value="point.id">
                  {{ point.name || point.title || point.pointName }}
                </a-select-option>
              </a-select>
            </a-form-item>

            <!-- 难度级别 -->
            <a-form-item label="难度级别" name="difficulty" :rules="[{ required: true, message: '请选择难度级别' }]">
              <a-radio-group v-model:value="formData.difficulty">
                <a-radio value="EASY">简单</a-radio>
                <a-radio value="MEDIUM">中等</a-radio>
                <a-radio value="HARD">困难</a-radio>
              </a-radio-group>
            </a-form-item>

            <!-- 题目数量 -->
            <a-form-item label="题目数量" name="questionCount" :rules="[{ required: true, message: '请输入题目数量' }]">
              <a-input-number 
                v-model:value="formData.questionCount" 
                :min="5" 
                :max="50" 
                placeholder="请输入题目数量"
                style="width: 100%"
              />
            </a-form-item>

            <!-- 题型分布 -->
            <a-form-item label="题型分布">
              <div class="question-types">
                <div v-for="(count, type) in formData.questionTypes" :key="type" class="type-item">
                  <span class="type-label">{{ getQuestionTypeLabel(type) }}</span>
                  <a-input-number 
                    v-model:value="formData.questionTypes[type]" 
                    :min="0" 
                    :max="formData.questionCount"
                    size="small"
                  />
                </div>
              </div>
            </a-form-item>

            <!-- 考试时长 -->
            <a-form-item label="考试时长" name="duration">
              <a-input-number 
                v-model:value="formData.duration" 
                :min="10" 
                :max="180" 
                addon-after="分钟"
                placeholder="考试时长"
                style="width: 100%"
              />
            </a-form-item>

            <!-- 总分 -->
            <a-form-item label="总分" name="totalScore">
              <a-input-number 
                v-model:value="formData.totalScore" 
                :min="50" 
                :max="200" 
                addon-after="分"
                placeholder="试卷总分"
                style="width: 100%"
              />
            </a-form-item>

            <!-- 额外要求 -->
            <a-form-item label="额外要求">
              <a-textarea 
                v-model:value="formData.additionalRequirements" 
                placeholder="可输入特殊要求，如：注重实际应用、包含计算题等"
                :rows="3"
              />
            </a-form-item>

            <!-- 操作按钮 -->
            <a-form-item :wrapper-col="{ offset: 6, span: 18 }">
              <a-space>
                <a-button @click="handlePreview" :loading="previewLoading">
                  <EyeOutlined />
                  预览参数
                </a-button>
                <a-button type="primary" html-type="submit" :loading="generateLoading">
                  <ThunderboltOutlined />
                  智能生成
                </a-button>
                <a-button @click="handleAsyncGenerate" :loading="asyncLoading">
                  <ClockCircleOutlined />
                  异步生成
                </a-button>
              </a-space>
            </a-form-item>
          </a-form>
        </a-card>
      </div>

      <!-- 右侧：结果展示 -->
      <div class="result-panel">
        <!-- 参数预览 -->
        <a-card v-if="previewData" title="📊 参数预览" class="preview-card" style="margin-bottom: 16px;">
          <div class="preview-content">
            <div class="preview-item">
              <span class="label">预计生成题目：</span>
              <span class="value">{{ previewData.estimated_questions }} 道</span>
            </div>
            <div class="preview-item">
              <span class="label">预计生成时间：</span>
              <span class="value">{{ previewData.estimated_time }}</span>
            </div>
            <div class="preview-item">
              <span class="label">可用题型：</span>
              <a-tag v-for="type in previewData.available_types" :key="type" color="blue">
                {{ type }}
              </a-tag>
            </div>
          </div>
        </a-card>

        <!-- 生成结果 -->
        <a-card title="📋 生成结果" class="result-card">
          <!-- 加载状态 -->
          <div v-if="generateLoading || asyncLoading" class="loading-content">
            <a-spin size="large">
              <div class="loading-text">
                <p>🤖 AI正在智能分析课程内容...</p>
                <p>📊 正在匹配最适合的题目...</p>
                <p>⚡ 即将完成试卷生成...</p>
              </div>
            </a-spin>
          </div>

          <!-- 异步任务状态 -->
          <div v-else-if="asyncTaskId && !paperResult" class="async-status">
            <a-alert 
              message="异步任务进行中" 
              :description="`任务ID: ${asyncTaskId}，请稍后查看结果`"
              type="info" 
              show-icon 
            />
            <a-button @click="checkTaskStatus" :loading="checkingStatus" style="margin-top: 16px;">
              <ReloadOutlined />
              检查状态
            </a-button>
          </div>

          <!-- 生成成功 -->
          <div v-else-if="paperResult && paperResult.status === 'completed'" class="success-content">
            <div class="result-header">
              <h2>{{ paperResult.title || '智能生成试卷' }}</h2>
              <a-space>
                <a-dropdown>
                  <template #overlay>
                    <a-menu>
                      <a-menu-item key="word" @click="exportAsWord">
                        <FileWordOutlined /> Word文档
                      </a-menu-item>
                      <a-menu-item key="text" @click="exportAsText">
                        <FileTextOutlined /> 文本文件
                      </a-menu-item>
                    </a-menu>
                  </template>
                  <a-button>
                    <DownloadOutlined />
                    下载试卷 <DownOutlined />
                  </a-button>
                </a-dropdown>
                <a-button @click="handleSave">
                  <SaveOutlined />
                  保存到题库
                </a-button>
                <a-button @click="handlePreviewPaper">
                  <EyeOutlined />
                  预览试卷
                </a-button>
              </a-space>
            </div>

            <div class="questions-list">
              <div v-for="(question, index) in paperResult.questions" :key="index" class="question-item">
                <div class="question-header">
                  <span class="question-number">{{ index + 1 }}.</span>
                  <a-tag :color="getDifficultyColor(question.difficulty)">
                    {{ question.difficulty || '未知难度' }}
                  </a-tag>
                  <a-tag color="blue">{{ getQuestionTypeLabel(question.questionType) }}</a-tag>
                  <span v-if="question.score" class="score">{{ question.score }}分</span>
                </div>
                
                <div class="question-content">
                  <!-- AI直接输出类型的特殊处理 -->
                  <div v-if="question.questionType === 'AI_OUTPUT' || question.questionType === 'ERROR'" class="ai-output">
                    <pre style="white-space: pre-wrap; word-break: break-word;">{{ question.questionText }}</pre>
                  </div>
                  <!-- 普通题目类型的处理 -->
                  <p v-else class="question-text">{{ question.questionText }}</p>
                  
                  <!-- 选择题选项 -->
                  <div v-if="question.options" class="options">
                    <div v-for="(option, optIndex) in question.options" :key="optIndex" class="option">
                      {{ String.fromCharCode(65 + optIndex) }}. {{ option }}
                    </div>
                  </div>
                  
                  <div v-if="question.questionType !== 'AI_OUTPUT' && question.questionType !== 'ERROR'" class="question-meta">
                    <span class="knowledge-point">知识点：{{ question.knowledgePoint }}</span>
                    <span class="correct-answer">正确答案：{{ question.correctAnswer }}</span>
                  </div>
                  
                  <div v-if="question.explanation" class="explanation">
                    <strong>解析：</strong>{{ question.explanation }}
                  </div>
                </div>
              </div>
            </div>
          </div>

          <!-- 生成失败 -->
          <div v-else-if="paperResult && paperResult.status === 'failed'" class="error-content">
            <a-result
              status="error"
              title="生成失败"
              :sub-title="paperResult.errorMessage || '智能组卷失败，请重试'"
            >
              <template #extra>
                <a-button type="primary" @click="handleRetry">
                  <ReloadOutlined />
                  重新生成
                </a-button>
              </template>
            </a-result>
          </div>

          <!-- 初始状态 -->
          <div v-else class="empty-content">
            <a-empty description="请配置参数并生成试卷" />
          </div>
        </a-card>
      </div>
    </div>

    <!-- 试卷预览弹窗 -->
    <a-modal 
      v-model:open="previewModalVisible" 
      title="试卷预览" 
      width="800px"
      :footer="null"
    >
      <div class="paper-preview">
        <!-- 试卷预览内容 -->
        <div v-if="paperResult" class="preview-paper">
          <div class="paper-header">
            <h1>{{ paperResult.title || '智能生成试卷' }}</h1>
            <div class="paper-info">
              <span>总分：{{ formData.totalScore }}分</span>
              <span>时长：{{ formData.duration }}分钟</span>
              <span>题数：{{ paperResult.questions.length }}道</span>
            </div>
          </div>
          
          <div class="paper-questions">
            <div v-for="(question, index) in paperResult.questions" :key="index" class="preview-question">
              <div class="question-title">
                {{ index + 1 }}. {{ question.score ? `(${question.score}分)` : '' }} {{ question.questionText }}
              </div>
              
              <div v-if="question.options" class="question-options">
                <div v-for="(option, optIndex) in question.options" :key="optIndex">
                  {{ String.fromCharCode(65 + optIndex) }}. {{ option }}
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </a-modal>
  </div>
</template>

<script setup lang="ts">
import { ref, reactive, computed, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { message } from 'ant-design-vue'
import { 
  EyeOutlined, 
  ThunderboltOutlined, 
  ClockCircleOutlined,
  DownloadOutlined,
  SaveOutlined,
  ReloadOutlined,
  FileWordOutlined,
  FileTextOutlined,
  DownOutlined
} from '@ant-design/icons-vue'
import { teacherPaperApi, type PaperGenerationRequest, type PaperGenerationResponse } from '@/api/dify'
import axios from 'axios'
import request from '@/utils/request' // 导入已配置的request实例
// 恢复导入，启用Word文档导出
import { saveAs } from 'file-saver'
import { Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType } from 'docx'
import { useAuthStore } from '@/stores/auth'

// 定义题目类型接口
interface Question {
  questionText: string
  questionType: 'SINGLE_CHOICE' | 'MULTIPLE_CHOICE' | 'TRUE_FALSE' | 'FILL_BLANK' | 'ESSAY' | 'AI_OUTPUT' | 'ERROR' | string
  options?: string[]
  correctAnswer?: string
  score?: number
  knowledgePoint?: string
  difficulty?: string
  explanation?: string
}

// 定义试卷结果接口
interface PaperResult {
  title: string
  questions: Question[]
  status: string
  taskId?: string
  errorMessage?: string
}

// 响应式数据
const generateLoading = ref(false)
const asyncLoading = ref(false)
const previewLoading = ref(false)
const checkingStatus = ref(false)
const previewModalVisible = ref(false)
const coursesLoading = ref(false)
const chaptersLoading = ref(false)

const formData = reactive<PaperGenerationRequest>({
  courseId: 0,
  knowledgePoints: [],
  difficulty: 'MEDIUM',
  questionCount: 10,
  questionTypes: {
    'SINGLE_CHOICE': 5,
    'MULTIPLE_CHOICE': 3,
    'TRUE_FALSE': 2
  },
  duration: 90,
  totalScore: 100,
  additionalRequirements: ''
})

// 真实课程数据
const courses = ref<any[]>([])
const knowledgePointOptions = ref<any[]>([])
const chapters = ref<any[]>([])

const previewData = ref<any>(null)
const paperResult = ref<PaperResult | null>(null)
const asyncTaskId = ref<string>('')

// 加载教师课程列表
const loadTeacherCourses = async () => {
  try {
    console.log('📚 开始获取教师课程列表...')
    coursesLoading.value = true
    
    // 从authStore获取token
    const authStore = useAuthStore()
    let token = authStore.token
    
    // 如果authStore中没有token，尝试从localStorage获取
    if (!token) {
      token = localStorage.getItem('token') || localStorage.getItem('user-token')
      if (token && authStore.setToken) {
        authStore.setToken(token)
      }
    }
    
    // 使用全局配置的request实例替代axios
    const response = await request.get('/api/teacher/courses', {
      headers: {
        'Authorization': token ? `Bearer ${token}` : ''
      }
    })
    
    console.log('📊 API原始响应:', response)
    
    if (response && response.data) {
      // 处理不同的响应结构
      const responseData = response.data
      console.log('📊 响应数据类型:', typeof responseData, '是否为数组:', Array.isArray(responseData))
      
      if (Array.isArray(responseData)) {
        // 直接是数组
        courses.value = responseData
        console.log('📚 获取到', courses.value.length, '个课程 (直接数组)')
      } else if (responseData.records || responseData.content || responseData.list) {
        // 分页响应
        courses.value = responseData.records || responseData.content || responseData.list || []
        console.log('📚 获取到', courses.value.length, '个课程 (分页对象)')
      } else if (responseData.code === 200 && responseData.data) {
        // Result包装的数据
        console.log('📊 Result包装的数据:', responseData.data)
        if (Array.isArray(responseData.data)) {
          courses.value = responseData.data
          console.log('📚 获取到', courses.value.length, '个课程 (Result包装数组)')
        } else if (responseData.data.records || responseData.data.content || responseData.data.list) {
          courses.value = responseData.data.records || responseData.data.content || responseData.data.list || []
          console.log('📚 获取到', courses.value.length, '个课程 (Result包装分页对象)')
        } else {
          // 尝试查找更多可能的字段
          if (responseData.data && typeof responseData.data === 'object') {
            // 查找任何可能包含课程数组的字段
            for (const key in responseData.data) {
              if (Array.isArray(responseData.data[key]) && responseData.data[key].length > 0) {
                // 检查数组的第一个元素是否有课程的典型字段
                const firstItem = responseData.data[key][0]
                if (firstItem && (firstItem.id !== undefined || firstItem.title !== undefined)) {
                  courses.value = responseData.data[key]
                  console.log('📚 找到课程数组字段:', key, courses.value.length, '个课程')
                  return
                }
              }
            }
          }
          console.warn('未找到有效的课程数据结构:', responseData.data)
          courses.value = []
        }
      } else {
        // 其他情况
        console.warn('未能识别的课程数据结构:', responseData)
        courses.value = []
      }
    } else {
      console.warn('未获取到课程数据')
      courses.value = []
    }
    
    // 如果没有获取到课程数据，使用模拟数据
    if (courses.value.length === 0) {
      courses.value = [
        { id: 19, title: 'Java程序设计', code: 'CS101' },
        { id: 20, title: '数据结构与算法', code: 'CS201' },
        { id: 21, title: 'Python程序设计', code: 'CS102' }
      ]
      console.log('📚 使用模拟数据 (API处理失败):', courses.value.length, '个课程')
    }
    
  } catch (error: any) {
    console.error('获取课程列表失败:', error)
    // 使用模拟数据
    courses.value = [
      { id: 19, title: 'Java程序设计', code: 'CS101' },
      { id: 20, title: '数据结构与算法', code: 'CS201' },
      { id: 21, title: 'Python程序设计', code: 'CS102' }
    ]
    console.log('📚 使用模拟数据 (异常):', courses.value.length, '个课程')
  } finally {
    coursesLoading.value = false
  }
}

// 加载课程知识点
const loadCourseKnowledgePoints = async (courseId: number) => {
  try {
    console.log('🧠 开始获取课程知识点，课程ID:', courseId)
    chaptersLoading.value = true
    
    // 从authStore获取token
    const authStore = useAuthStore()
    let token = authStore.token
    
    // 如果authStore中没有token，尝试从localStorage获取
    if (!token) {
      token = localStorage.getItem('token') || localStorage.getItem('user-token')
    }
    
    // 使用正确的API路径获取章节
    const response = await request.get(`/api/teacher/chapters/course/${courseId}`, {
      headers: {
        'Authorization': token ? `Bearer ${token}` : ''
      }
    })
    
    console.log('📖 章节列表响应:', response)
    
    if (response && response.data) {
      // 处理可能的嵌套数据结构
      let chapterData = response.data.data || response.data
      
      // 将章节转换为知识点选项
      if (Array.isArray(chapterData) && chapterData.length > 0) {
        knowledgePointOptions.value = chapterData.flatMap((chapter: any) => {
          // 如果有小节，使用小节作为知识点
          if (chapter.sections && chapter.sections.length > 0) {
            return chapter.sections.map((section: any) => ({
              id: `section-${section.id}`,
              name: `${chapter.title} - ${section.title}`,
              title: `${chapter.title} - ${section.title}`
            }))
          }
          
          // 否则使用章节作为知识点
          return {
            id: `chapter-${chapter.id}`,
            name: chapter.title,
            title: chapter.title
          }
        })
        console.log('🧠 获取到', knowledgePointOptions.value.length, '个知识点 (章节和小节)')
        return // 成功获取章节，直接返回
      } else {
        console.warn('章节数据为空或格式不正确:', chapterData)
        // 继续尝试其他方式获取知识点
      }
    }
    
    // 如果没有获取到章节数据或数据为空，尝试使用作业API获取知识点
    console.log('尝试使用作业API获取知识点...')
    try {
      const assignmentResponse = await request.get('/api/teacher/assignments/questions/knowledge-points', {
        params: {
          courseId: courseId
        },
        headers: {
          'Authorization': token ? `Bearer ${token}` : ''
        }
      })
      
      console.log('📚 作业知识点响应:', assignmentResponse)
      
      if (assignmentResponse && assignmentResponse.data && assignmentResponse.data.code === 200 && 
          Array.isArray(assignmentResponse.data.data) && assignmentResponse.data.data.length > 0) {
        // 转换为知识点选项格式
        knowledgePointOptions.value = assignmentResponse.data.data.map((point: string) => ({
          id: point,
          name: point,
          title: point
        }))
        console.log('🧠 获取到', knowledgePointOptions.value.length, '个知识点 (作业API)')
        return
      }
    } catch (assignmentError) {
      console.error('获取作业知识点失败:', assignmentError)
    }
    
    // 如果以上方法都失败，使用模拟数据
    console.log('无法从API获取知识点，使用模拟数据')
    useDefaultKnowledgePoints(courseId)
  } catch (error: any) {
    console.error('获取知识点失败:', error)
    // 使用默认知识点
    useDefaultKnowledgePoints(courseId)
  } finally {
    chaptersLoading.value = false
  }
}

// 计算属性
const totalQuestionCount = computed(() => {
  return Object.values(formData.questionTypes).reduce((sum, count) => sum + count, 0)
})

// 方法
// 处理课程选择变化
const handleCourseChange = (courseId: number) => {
  // 清空知识点选择
  formData.knowledgePoints = []
  
  // 如果没有选择课程，直接返回
  if (!courseId) {
    knowledgePointOptions.value = []
    return
  }
  
  // 根据课程加载对应的知识点
  console.log('课程变更:', courseId)
  loadCourseKnowledgePoints(courseId)
}

const handlePreview = async () => {
  try {
    previewLoading.value = true
    const response = await teacherPaperApi.previewPaper(formData)
    previewData.value = response.data
    message.success('参数预览成功')
  } catch (error) {
    message.error('预览失败: ' + (error as any).message)
  } finally {
    previewLoading.value = false
  }
}

const handleGenerate = async () => {
  try {
    console.log('🤖 开始生成试卷，参数:', formData)
    generateLoading.value = true
    
    // 验证题型数量
    if (totalQuestionCount.value !== formData.questionCount) {
      message.warning('题型分布总数与题目数量不匹配，请调整题型分布')
      generateLoading.value = false
      return
    }
    
    // 添加文档格式需求
    formData.additionalRequirements += '\n请生成Word或PDF格式的试卷，以便于下载和打印。'
    
    console.log('📤 发送组卷请求:', formData)
    
    const response = await teacherPaperApi.generatePaper(formData)
    console.log('📥 组卷响应:', response)
    
    // 检查响应格式
    if (!response || !response.data) {
      throw new Error('响应数据为空')
    }
    
    // 处理响应数据
    const responseData = response.data
    
    if (responseData.status === 'completed') {
      // 成功生成试卷
      console.log('✅ 生成成功，题目数量:', responseData.questions?.length || 0)
      paperResult.value = responseData
      previewModalVisible.value = true // 直接打开预览弹窗
      message.success('试卷生成成功')
    } else if (responseData.status === 'pending' && responseData.taskId) {
      // 异步生成中
      console.log('⏳ 异步生成中，任务ID:', responseData.taskId)
      asyncTaskId.value = responseData.taskId
      message.info('试卷正在生成中，请稍候...')
      // 这里可以添加轮询逻辑
    } else if (responseData.status === 'failed') {
      // 生成失败
      console.error('❌ 生成失败:', responseData.errorMessage)
      message.error('生成失败: ' + (responseData.errorMessage || '未知错误'))
      
      // 设置失败结果，以便在UI中显示错误信息
      paperResult.value = responseData
    } else {
      console.warn('未知响应格式:', responseData)
      throw new Error('响应数据格式不正确')
    }
  } catch (error: any) {
    console.error('❌ 生成失败:', error)
    message.error('生成失败: ' + error.message)
  } finally {
    generateLoading.value = false
  }
}

const handleAsyncGenerate = async () => {
  try {
    asyncLoading.value = true
    
    if (totalQuestionCount.value !== formData.questionCount) {
      message.warning('题型分布总数与题目数量不匹配，请调整题型分布')
      return
    }
    
    const response = await teacherPaperApi.generatePaperAsync(formData)
    asyncTaskId.value = response.data
    message.success('异步任务已提交，任务ID: ' + response.data)
  } catch (error: any) {
    message.error('异步任务提交失败: ' + (error as any).message)
  } finally {
    asyncLoading.value = false
  }
}

const checkTaskStatus = async () => {
  if (!asyncTaskId.value) return
  
  try {
    checkingStatus.value = true
    const response = await teacherPaperApi.getTaskStatus(asyncTaskId.value)
    
    if (response.data.status === 'completed') {
      // 解析结果并显示
      message.success('任务完成！')
      // TODO: 解析并显示结果
    } else if (response.data.status === 'failed') {
      message.error('任务失败: ' + response.data.error)
    } else {
      message.info('任务进行中...')
    }
  } catch (error: any) {
    message.error('查询任务状态失败: ' + (error as any).message)
  } finally {
    checkingStatus.value = false
  }
}

const handleDownload = () => {
  if (!paperResult.value) {
    message.warning('请先生成试卷')
    return
  }
}

// 导出为文本
const exportAsText = () => {
  if (!paperResult.value) return
  
  let content = `${paperResult.value.title || '试卷'}\n\n`
  content += `总分：${formData.totalScore}分  时长：${formData.duration}分钟\n\n`
  
  // 添加试卷内容
  paperResult.value.questions.forEach((q, index) => {
    // 处理AI_OUTPUT和ERROR类型
    if (q.questionType === 'AI_OUTPUT' || q.questionType === 'ERROR') {
      content += `${index + 1}. [${q.questionType === 'AI_OUTPUT' ? 'AI输出' : '错误信息'}]\n${q.questionText}\n\n`
      return
    }
    
    // 处理常规题目类型
    const scoreText = q.score ? `(${q.score}分)` : ''
    content += `${index + 1}. ${q.questionType === 'SINGLE_CHOICE' ? '[单选题]' : 
                q.questionType === 'MULTIPLE_CHOICE' ? '[多选题]' : 
                q.questionType === 'TRUE_FALSE' ? '[判断题]' : 
                q.questionType === 'FILL_BLANK' ? '[填空题]' : '[简答题]'} ${q.questionText} ${scoreText}\n`
    
    // 添加选项
    if (q.options) {
      q.options.forEach((option, i) => {
        content += `   ${String.fromCharCode(65 + i)}. ${option}\n`
      })
    }
    
    if (q.correctAnswer) {
    content += `\n   【答案】${q.correctAnswer}\n`
    }
    
    if (q.explanation) {
      content += `   【解析】${q.explanation}\n`
    }
    
    content += '\n'
  })
  
  // 创建下载链接
  const blob = new Blob([content], { type: 'text/plain;charset=utf-8' })
  const url = URL.createObjectURL(blob)
  const link = document.createElement('a')
  link.href = url
  link.download = `${paperResult.value.title || '试卷'}.txt`
  document.body.appendChild(link)
  link.click()
  document.body.removeChild(link)
  URL.revokeObjectURL(url)
  
  message.success('试卷已导出为文本文件')
}

// 导出为Word文档
const exportAsWord = () => {
  if (!paperResult.value) return
  
  try {
    // 创建Word文档
    const doc = new Document({
      sections: [{
        properties: {},
        children: [
          // 标题
          new Paragraph({
            text: paperResult.value.title || '智能生成试卷',
            heading: HeadingLevel.HEADING_1,
            alignment: AlignmentType.CENTER
          }),
          
          // 试卷信息
          new Paragraph({
            text: `总分：${formData.totalScore}分  时长：${formData.duration}分钟`,
            alignment: AlignmentType.CENTER
          }),
          
          // 空行
          new Paragraph({}),
          
          // 题目
          ...generateQuestionParagraphs()
        ]
      }]
    })
    
    // 生成并下载文档
    Packer.toBlob(doc).then(blob => {
      saveAs(blob, `${paperResult.value?.title || '试卷'}.docx`)
      message.success('试卷已导出为Word文档')
    })
  } catch (error: any) {
    console.error('Word导出失败:', error)
    message.error('Word导出失败，将尝试导出为文本文件')
    exportAsText()
  }
}

// 生成Word文档的题目段落
const generateQuestionParagraphs = () => {
  if (!paperResult.value) return []
  
  const paragraphs: Paragraph[] = []
  
  paperResult.value.questions.forEach((q, index) => {
    // 处理AI_OUTPUT和ERROR类型
    if (q.questionType === 'AI_OUTPUT' || q.questionType === 'ERROR') {
      // 题目标题
      paragraphs.push(
        new Paragraph({
          children: [
            new TextRun({
              text: `${index + 1}. [${q.questionType === 'AI_OUTPUT' ? 'AI输出' : '错误信息'}]`,
              bold: true
            })
          ]
        })
      )
      
      // AI输出内容
      paragraphs.push(
        new Paragraph({
          text: q.questionText
        })
      )
      
      // 空行
      paragraphs.push(new Paragraph({}))
      paragraphs.push(new Paragraph({}))
      
      return
    }
    
    // 常规题目类型
    // 题目类型标签
    const questionTypeLabel = q.questionType === 'SINGLE_CHOICE' ? '[单选题]' : 
                             q.questionType === 'MULTIPLE_CHOICE' ? '[多选题]' : 
                             q.questionType === 'TRUE_FALSE' ? '[判断题]' : 
                             q.questionType === 'FILL_BLANK' ? '[填空题]' : '[简答题]'
    
    // 题目标题
    paragraphs.push(
      new Paragraph({
        children: [
          new TextRun({
            text: `${index + 1}. ${questionTypeLabel} `,
            bold: true
          }),
          new TextRun({
            text: `${q.questionText}${q.score ? ` (${q.score}分)` : ''}`
          })
        ]
      })
    )
    
    // 选项
    if (q.options) {
      q.options.forEach((option, i) => {
        paragraphs.push(
          new Paragraph({
            children: [
              new TextRun({
                text: `    ${String.fromCharCode(65 + i)}. ${option}`
              })
            ]
          })
        )
      })
    }
    
    // 空行
    paragraphs.push(new Paragraph({}))
    
    // 正确答案
    if (q.correctAnswer) {
    paragraphs.push(
      new Paragraph({
        children: [
          new TextRun({
            text: '【答案】',
            bold: true
          }),
          new TextRun({
            text: q.correctAnswer
          })
        ]
      })
    )
    }
    
    // 解析
    if (q.explanation) {
    paragraphs.push(
      new Paragraph({
        children: [
          new TextRun({
            text: '【解析】',
            bold: true
          }),
          new TextRun({
            text: q.explanation
          })
        ]
      })
    )
    }
    
    // 空行
    paragraphs.push(new Paragraph({}))
    paragraphs.push(new Paragraph({}))
  })
  
  return paragraphs
}

const handleSave = () => {
  // TODO: 实现保存到题库功能
  message.info('保存功能开发中...')
}

const handlePreviewPaper = () => {
  if (!paperResult.value) {
    message.warning('请先生成试卷')
    return
  }
  
  previewModalVisible.value = true
}

const handleRetry = () => {
  paperResult.value = null
  handleGenerate()
}

// 获取题目类型标签
const getQuestionTypeLabel = (type: string): string => {
  const typeMap: Record<string, string> = {
    'SINGLE_CHOICE': '单选题',
    'MULTIPLE_CHOICE': '多选题',
    'TRUE_FALSE': '判断题',
    'FILL_BLANK': '填空题',
    'ESSAY': '简答题',
    'AI_OUTPUT': 'AI输出',
    'ERROR': '错误信息'
  }
  return typeMap[type] || type
}

// 获取题目难度标签颜色
const getDifficultyColor = (difficulty?: string) => {
  if (!difficulty) return 'default'
  
  const colorMap: Record<string, string> = {
    'EASY': 'green',
    'MEDIUM': 'orange',
    'HARD': 'red'
  }
  return colorMap[difficulty] || 'blue'
}

// 生成本地试卷模板
const generateLocalPaperTemplate = () => {
  const courseId = formData.courseId
  const difficulty = formData.difficulty
  const questionTypes = formData.questionTypes
  
  // 根据课程ID生成不同的试卷模板
  let paperTitle = ''
  let questions: Question[] = []
  
  // 根据课程ID选择模板
  switch(courseId) {
    case 19: // Java程序设计
      paperTitle = 'Java程序设计期末考试'
      questions = generateJavaPaperTemplate(difficulty, questionTypes)
      break
    case 20: // 数据结构与算法
      paperTitle = '数据结构与算法期末考试'
      questions = generateDataStructurePaperTemplate(difficulty, questionTypes)
      break
    case 21: // Python程序基础
      paperTitle = 'Python程序设计期末考试'
      questions = generatePythonPaperTemplate(difficulty, questionTypes)
      break
    default:
      paperTitle = '课程期末考试'
      questions = generateDefaultPaperTemplate(difficulty, questionTypes)
  }
  
  // 设置试卷结果
  paperResult.value = {
    title: paperTitle,
    questions: questions,
    status: 'completed'
  }
  
  message.success('试卷模板生成成功！')
}

// 生成Java试卷模板
const generateJavaPaperTemplate = (difficulty: string, questionTypes: Record<string, number>): Question[] => {
  const questions: Question[] = []
  
  // 单选题
  if (questionTypes['SINGLE_CHOICE'] > 0) {
    questions.push({
      questionText: 'Java中，以下哪个关键字用于继承？',
      questionType: 'SINGLE_CHOICE',
      options: ['extends', 'implements', 'inherits', 'extends from'],
      correctAnswer: 'extends',
      score: 2,
      knowledgePoint: 'Java基础语法',
      difficulty: difficulty,
      explanation: '在Java中，extends关键字用于类的继承，表示一个类继承另一个类的特性。'
    })
    
    questions.push({
      questionText: '以下哪个不是Java的基本数据类型？',
      questionType: 'SINGLE_CHOICE',
      options: ['int', 'boolean', 'String', 'char'],
      correctAnswer: 'String',
      score: 2,
      knowledgePoint: 'Java基础语法',
      difficulty: difficulty,
      explanation: 'String是引用类型，不是基本数据类型。Java的基本数据类型有byte、short、int、long、float、double、char和boolean。'
    })
  }
  
  // 多选题
  if (questionTypes['MULTIPLE_CHOICE'] > 0) {
    questions.push({
      questionText: '以下哪些是Java中的集合框架接口？',
      questionType: 'MULTIPLE_CHOICE',
      options: ['List', 'Map', 'Queue', 'Array'],
      correctAnswer: 'List,Map,Queue',
      score: 4,
      knowledgePoint: '集合框架',
      difficulty: difficulty,
      explanation: 'List、Map和Queue都是Java集合框架中的接口，而Array是Java的数组类型，不是集合框架接口。'
    })
  }
  
  // 判断题
  if (questionTypes['TRUE_FALSE'] > 0) {
    questions.push({
      questionText: 'Java中的接口可以包含默认方法实现。',
      questionType: 'TRUE_FALSE',
      correctAnswer: 'true',
      score: 2,
      knowledgePoint: '面向对象编程',
      difficulty: difficulty,
      explanation: 'Java 8及以后版本中，接口可以包含默认方法实现，使用default关键字。'
    })
  }
  
  // 填空题
  if (questionTypes['FILL_BLANK'] > 0) {
    questions.push({
      questionText: 'Java中，用于处理异常的关键字有try、catch、finally、throw和_____。',
      questionType: 'FILL_BLANK',
      correctAnswer: 'throws',
      score: 3,
      knowledgePoint: '异常处理',
      difficulty: difficulty,
      explanation: 'throws关键字用于在方法签名中声明该方法可能抛出的异常类型。'
    })
  }
  
  // 简答题
  if (questionTypes['ESSAY'] > 0) {
    questions.push({
      questionText: '请简述Java中的多线程实现方式及其区别。',
      questionType: 'ESSAY',
      correctAnswer: '在Java中实现多线程有两种主要方式：\n1. 继承Thread类并重写run()方法\n2. 实现Runnable接口并实现run()方法\n\n区别：\n- 继承Thread类的方式不支持多重继承，而实现Runnable接口的方式可以继承其他类\n- 实现Runnable接口的方式更适合多个线程共享同一个目标对象的情况\n- 实现Runnable接口的方式可以更好地体现面向对象的设计思想，将线程的控制和业务逻辑分离',
      score: 10,
      knowledgePoint: '多线程',
      difficulty: difficulty,
      explanation: '这个问题考察学生对Java多线程基础概念的理解，包括实现方式和各自的优缺点。'
    })
  }
  
  return questions
}

// 生成数据结构试卷模板
const generateDataStructurePaperTemplate = (difficulty: string, questionTypes: Record<string, number>): Question[] => {
  const questions: Question[] = []
  
  // 单选题
  if (questionTypes['SINGLE_CHOICE'] > 0) {
    questions.push({
      questionText: '以下哪种数据结构是线性的？',
      questionType: 'SINGLE_CHOICE',
      options: ['树', '图', '栈', '二叉树'],
      correctAnswer: '栈',
      score: 2,
      knowledgePoint: '数据结构基础',
      difficulty: difficulty,
      explanation: '栈是一种线性数据结构，而树、图和二叉树都是非线性数据结构。'
    })
  }
  
  // 多选题
  if (questionTypes['MULTIPLE_CHOICE'] > 0) {
    questions.push({
      questionText: '以下哪些排序算法的平均时间复杂度是O(nlogn)？',
      questionType: 'MULTIPLE_CHOICE',
      options: ['快速排序', '冒泡排序', '归并排序', '插入排序'],
      correctAnswer: '快速排序,归并排序',
      score: 4,
      knowledgePoint: '排序算法',
      difficulty: difficulty,
      explanation: '快速排序和归并排序的平均时间复杂度是O(nlogn)，冒泡排序和插入排序的平均时间复杂度是O(n²)。'
    })
  }
  
  // 判断题
  if (questionTypes['TRUE_FALSE'] > 0) {
    questions.push({
      questionText: '在最坏情况下，快速排序的时间复杂度是O(n²)。',
      questionType: 'TRUE_FALSE',
      correctAnswer: 'true',
      score: 2,
      knowledgePoint: '排序算法',
      difficulty: difficulty,
      explanation: '快速排序在最坏情况下（如已排序数组）的时间复杂度是O(n²)。'
    })
  }
  
  // 填空题
  if (questionTypes['FILL_BLANK'] > 0) {
    questions.push({
      questionText: '一棵完全二叉树中，若有n个节点，则其叶子节点的个数是_____。',
      questionType: 'FILL_BLANK',
      correctAnswer: '(n+1)/2',
      score: 3,
      knowledgePoint: '树与图',
      difficulty: difficulty,
      explanation: '完全二叉树的叶子节点个数为(n+1)/2，向下取整。'
    })
  }
  
  // 简答题
  if (questionTypes['ESSAY'] > 0) {
    questions.push({
      questionText: '请详细描述红黑树的特性及其在实际应用中的优势。',
      questionType: 'ESSAY',
      correctAnswer: '红黑树特性：\n1. 每个节点要么是红色，要么是黑色\n2. 根节点是黑色\n3. 每个叶节点（NIL节点）是黑色\n4. 如果一个节点是红色，则其两个子节点都是黑色\n5. 对于每个节点，从该节点到其所有后代叶节点的简单路径上，均包含相同数目的黑色节点\n\n优势：\n1. 自平衡，保证了树的高度不会过大，查找、插入和删除操作的时间复杂度都是O(log n)\n2. 比AVL树插入和删除操作更高效，因为红黑树的平衡条件相对宽松\n3. 广泛应用于Java的TreeMap、TreeSet，C++的map、set等容器中\n4. 适用于频繁插入和删除操作的场景',
      score: 10,
      knowledgePoint: '树与图',
      difficulty: difficulty,
      explanation: '这个问题考察学生对红黑树这种高级数据结构的理解，包括其特性和实际应用价值。'
    })
  }
  
  return questions
}

// 生成Python试卷模板
const generatePythonPaperTemplate = (difficulty: string, questionTypes: Record<string, number>): Question[] => {
  const questions: Question[] = []
  
  // 单选题
  if (questionTypes['SINGLE_CHOICE'] > 0) {
    questions.push({
      questionText: 'Python中，以下哪种数据类型是不可变的？',
      questionType: 'SINGLE_CHOICE',
      options: ['列表(list)', '字典(dict)', '集合(set)', '元组(tuple)'],
      correctAnswer: '元组(tuple)',
      score: 2,
      knowledgePoint: 'Python基础语法',
      difficulty: difficulty,
      explanation: '在Python中，元组(tuple)是不可变的数据类型，而列表(list)、字典(dict)和集合(set)都是可变的。'
    })
  }
  
  // 多选题
  if (questionTypes['MULTIPLE_CHOICE'] > 0) {
    questions.push({
      questionText: '以下哪些是Python的内置函数？',
      questionType: 'MULTIPLE_CHOICE',
      options: ['map()', 'reduce()', 'filter()', 'foreach()'],
      correctAnswer: 'map(),filter()',
      score: 4,
      knowledgePoint: '函数与模块',
      difficulty: difficulty,
      explanation: 'map()和filter()是Python的内置函数，而reduce()在Python 3中被移到functools模块中，foreach()不是Python的内置函数。'
    })
  }
  
  // 判断题
  if (questionTypes['TRUE_FALSE'] > 0) {
    questions.push({
      questionText: 'Python中的列表推导式比等效的for循环执行速度更快。',
      questionType: 'TRUE_FALSE',
      correctAnswer: 'true',
      score: 2,
      knowledgePoint: 'Python基础语法',
      difficulty: difficulty,
      explanation: '列表推导式通常比等效的for循环执行速度更快，因为它是在C层面实现的，而且减少了Python解释器的开销。'
    })
  }
  
  // 填空题
  if (questionTypes['FILL_BLANK'] > 0) {
    questions.push({
      questionText: 'Python中，使用_____关键字来定义一个函数。',
      questionType: 'FILL_BLANK',
      correctAnswer: 'def',
      score: 3,
      knowledgePoint: '函数与模块',
      difficulty: difficulty,
      explanation: 'Python使用def关键字来定义函数。'
    })
  }
  
  // 简答题
  if (questionTypes['ESSAY'] > 0) {
    questions.push({
      questionText: '请解释Python中的装饰器(decorator)是什么，并给出一个简单的例子。',
      questionType: 'ESSAY',
      correctAnswer: '装饰器是Python中用于修改函数或类行为的一种特殊语法。它是一个返回函数的函数，可以在不修改原函数代码的情况下，增加额外的功能。\n\n例子：\n```python\ndef timing_decorator(func):\n    def wrapper(*args, **kwargs):\n        import time\n        start_time = time.time()\n        result = func(*args, **kwargs)\n        end_time = time.time()\n        print(f"函数 {func.__name__} 执行时间: {end_time - start_time}秒")\n        return result\n    return wrapper\n\n@timing_decorator\ndef slow_function():\n    import time\n    time.sleep(1)\n    print("函数执行完毕")\n\nslow_function()  # 输出执行时间\n```\n\n这个装饰器用于计算函数的执行时间，并在函数执行完毕后打印出来。',
      score: 10,
      knowledgePoint: '函数与模块',
      difficulty: difficulty,
      explanation: '这个问题考察学生对Python高级特性装饰器的理解和应用能力。'
    })
  }
  
  return questions
}

// 生成默认试卷模板
const generateDefaultPaperTemplate = (difficulty: string, questionTypes: Record<string, number>): Question[] => {
  const questions: Question[] = []
  
  // 单选题
  if (questionTypes['SINGLE_CHOICE'] > 0) {
    for (let i = 0; i < questionTypes['SINGLE_CHOICE']; i++) {
      questions.push({
        questionText: `单选题示例 ${i+1}`,
        questionType: 'SINGLE_CHOICE',
        options: ['选项A', '选项B', '选项C', '选项D'],
        correctAnswer: '选项A',
        score: 2,
        knowledgePoint: '基础知识点',
        difficulty: difficulty,
        explanation: '这是一个单选题示例。'
      })
    }
  }
  
  // 多选题
  if (questionTypes['MULTIPLE_CHOICE'] > 0) {
    for (let i = 0; i < questionTypes['MULTIPLE_CHOICE']; i++) {
      questions.push({
        questionText: `多选题示例 ${i+1}`,
        questionType: 'MULTIPLE_CHOICE',
        options: ['选项A', '选项B', '选项C', '选项D'],
        correctAnswer: '选项A,选项C',
        score: 4,
        knowledgePoint: '基础知识点',
        difficulty: difficulty,
        explanation: '这是一个多选题示例。'
      })
    }
  }
  
  // 判断题
  if (questionTypes['TRUE_FALSE'] > 0) {
    for (let i = 0; i < questionTypes['TRUE_FALSE']; i++) {
      questions.push({
        questionText: `判断题示例 ${i+1}`,
        questionType: 'TRUE_FALSE',
        correctAnswer: i % 2 === 0 ? 'true' : 'false',
        score: 2,
        knowledgePoint: '基础知识点',
        difficulty: difficulty,
        explanation: '这是一个判断题示例。'
      })
    }
  }
  
  // 填空题
  if (questionTypes['FILL_BLANK'] > 0) {
    for (let i = 0; i < questionTypes['FILL_BLANK']; i++) {
      questions.push({
        questionText: `填空题示例 ${i+1}：请填写____。`,
        questionType: 'FILL_BLANK',
        correctAnswer: '答案',
        score: 3,
        knowledgePoint: '基础知识点',
        difficulty: difficulty,
        explanation: '这是一个填空题示例。'
      })
    }
  }
  
  // 简答题
  if (questionTypes['ESSAY'] > 0) {
    for (let i = 0; i < questionTypes['ESSAY']; i++) {
      questions.push({
        questionText: `简答题示例 ${i+1}：请简述相关概念。`,
        questionType: 'ESSAY',
        correctAnswer: '这是一个标准答案示例，用于参考。实际评分时需要根据学生的回答内容进行评判。',
        score: 10,
        knowledgePoint: '基础知识点',
        difficulty: difficulty,
        explanation: '这是一个简答题示例。'
      })
    }
  }
  
  return questions
}

// 使用模拟知识点数据
const useDefaultKnowledgePoints = (courseId: number) => {
  if (courseId === 19) { // Java
    knowledgePointOptions.value = [
      { id: 1, name: 'Java基础语法', title: 'Java基础语法' },
      { id: 2, name: 'Java面向对象', title: 'Java面向对象' },
      { id: 3, name: 'Java集合框架', title: 'Java集合框架' },
      { id: 4, name: 'Java异常处理', title: 'Java异常处理' },
      { id: 5, name: 'Java多线程', title: 'Java多线程' }
    ]
  } else if (courseId === 20) { // 数据结构
    knowledgePointOptions.value = [
      { id: 6, name: '线性表', title: '线性表' },
      { id: 7, name: '栈与队列', title: '栈与队列' },
      { id: 8, name: '树与图', title: '树与图' },
      { id: 9, name: '查找算法', title: '查找算法' },
      { id: 10, name: '排序算法', title: '排序算法' }
    ]
  } else if (courseId === 21) { // Python
    knowledgePointOptions.value = [
      { id: 11, name: 'Python基础语法', title: 'Python基础语法' },
      { id: 12, name: 'Python函数与模块', title: 'Python函数与模块' },
      { id: 13, name: 'Python数据结构', title: 'Python数据结构' },
      { id: 14, name: 'Python文件操作', title: 'Python文件操作' },
      { id: 15, name: 'Python面向对象', title: 'Python面向对象' }
    ]
  } else {
    knowledgePointOptions.value = [
      { id: 16, name: '基础知识点1', title: '基础知识点1' },
      { id: 17, name: '基础知识点2', title: '基础知识点2' },
      { id: 18, name: '基础知识点3', title: '基础知识点3' }
    ]
  }
  console.log('🧠 使用模拟知识点数据:', knowledgePointOptions.value.length, '个知识点')
}

// 初始化
onMounted(async () => {
  console.log('🚀 智能组卷页面初始化')
  await loadTeacherCourses()
})
</script>

<style scoped>
.smart-paper-generation {
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

.generation-container {
  display: flex;
  gap: 24px;
  max-width: 1400px;
  margin: 0 auto;
}

.config-panel {
  flex: 0 0 400px;
}

.result-panel {
  flex: 1;
}

.config-card, .result-card {
  border-radius: 12px;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
}

.question-types {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.type-item {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 8px;
  background: #f8f9fa;
  border-radius: 6px;
}

.type-label {
  font-weight: 500;
}

.preview-content {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.preview-item {
  display: flex;
  align-items: center;
  gap: 8px;
}

.preview-item .label {
  font-weight: 500;
  color: #666;
}

.preview-item .value {
  color: #1890ff;
  font-weight: 600;
}

.loading-content {
  text-align: center;
  padding: 60px 20px;
}

.loading-text p {
  margin: 8px 0;
  color: #666;
  font-size: 14px;
}

.async-status {
  text-align: center;
  padding: 40px 20px;
}

.success-content {
  padding: 20px;
}

.result-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 24px;
  padding-bottom: 16px;
  border-bottom: 1px solid #f0f0f0;
}

.result-header h2 {
  margin: 0;
  color: #333;
}

.questions-list {
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.question-item {
  background: white;
  border: 1px solid #e8e8e8;
  border-radius: 8px;
  padding: 16px;
  transition: all 0.3s;
}

.question-item:hover {
  border-color: #1890ff;
  box-shadow: 0 2px 8px rgba(24, 144, 255, 0.1);
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

.score {
  margin-left: auto;
  font-weight: 600;
  color: #f5222d;
}

.question-text {
  font-size: 16px;
  line-height: 1.6;
  margin-bottom: 12px;
  color: #333;
}

.options {
  margin: 12px 0;
  padding-left: 20px;
}

.option {
  margin: 6px 0;
  color: #666;
}

.question-meta {
  display: flex;
  gap: 16px;
  margin: 12px 0;
  font-size: 14px;
  color: #666;
}

.explanation {
  margin-top: 12px;
  padding: 12px;
  background: #f6ffed;
  border-left: 3px solid #52c41a;
  border-radius: 4px;
  font-size: 14px;
  line-height: 1.6;
}

.empty-content {
  text-align: center;
  padding: 60px 20px;
}

.error-content {
  padding: 40px 20px;
}

.paper-preview {
  max-height: 600px;
  overflow-y: auto;
}

.paper-header {
  text-align: center;
  margin-bottom: 32px;
  padding-bottom: 16px;
  border-bottom: 2px solid #1890ff;
}

.paper-header h1 {
  margin: 0 0 16px 0;
  font-size: 24px;
  color: #333;
}

.paper-info {
  display: flex;
  justify-content: center;
  gap: 24px;
  color: #666;
}

.preview-question {
  margin-bottom: 24px;
  padding-bottom: 16px;
  border-bottom: 1px solid #f0f0f0;
}

.question-title {
  font-size: 16px;
  font-weight: 500;
  margin-bottom: 12px;
  line-height: 1.6;
}

.question-options {
  margin-left: 20px;
}

.question-options div {
  margin: 6px 0;
  color: #666;
}
</style> 