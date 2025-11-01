<template>
  <div class="create-assignment-page">
    <a-page-header
      title="发布作业"
      sub-title="创建新作业并分配给学生"
      @back="goBack"
    />

    <div class="content-container">
      <a-card title="作业信息" class="assignment-form-card">
        <a-form
          :model="assignmentForm"
          :label-col="{ span: 4 }"
          :wrapper-col="{ span: 20 }"
          layout="horizontal"
        >
          <!-- 基本信息 -->
          <a-form-item label="作业标题" name="title" :rules="[{ required: true, message: '请输入作业标题' }]">
            <a-input v-model:value="assignmentForm.title" placeholder="请输入作业标题" />
          </a-form-item>

          <a-form-item label="所属课程" name="courseId" :rules="[{ required: true, message: '请选择课程' }]">
            <a-select
              v-model:value="assignmentForm.courseId"
              placeholder="请选择课程"
              :loading="coursesLoading"
              @change="handleCourseChange"
            >
              <a-select-option v-for="course in courses" :key="course.id" :value="course.id">
                {{ course.title || course.name }}
              </a-select-option>
            </a-select>
          </a-form-item>

          <a-form-item label="作业说明" name="description">
            <a-textarea
              v-model:value="assignmentForm.description"
              placeholder="请输入作业说明"
              :rows="4"
            />
          </a-form-item>

          <a-form-item label="时间设置">
            <a-row :gutter="16">
              <a-col :span="12">
                <a-form-item name="startTime" :rules="[{ required: true, message: '请选择开始时间' }]">
                  <a-date-picker
                    v-model:value="assignmentForm.startTime"
                    show-time
                    placeholder="开始时间"
                    style="width: 100%"
                    :disabled-date="disablePastDates"
                  />
                </a-form-item>
              </a-col>
              <a-col :span="12">
                <a-form-item name="endTime" :rules="[{ required: true, message: '请选择结束时间' }]">
                  <a-date-picker
                    v-model:value="assignmentForm.endTime"
                    show-time
                    placeholder="结束时间"
                    style="width: 100%"
                    :disabled-date="disablePastDates"
                  />
                </a-form-item>
              </a-col>
            </a-row>
          </a-form-item>

          <a-form-item label="总分" name="totalScore" :rules="[{ required: true, message: '请设置总分' }]">
            <a-input-number
              v-model:value="assignmentForm.totalScore"
              :min="1"
              :max="100"
              style="width: 100%"
            />
          </a-form-item>

          <a-form-item label="作业模式" name="mode" :rules="[{ required: true, message: '请选择作业模式' }]">
            <a-radio-group v-model:value="assignmentForm.mode" button-style="solid">
              <a-radio-button value="question">答题模式</a-radio-button>
              <a-radio-button value="file">文件提交模式</a-radio-button>
            </a-radio-group>
          </a-form-item>

          <!-- 答题模式下的智能组卷设置 -->
          <template v-if="assignmentForm.mode === 'question'">
            <a-divider>智能组卷设置</a-divider>
            
            <!-- 知识点选择 -->
            <a-form-item label="知识点范围" name="knowledgePoints">
              <a-select
                v-model:value="assignmentForm.knowledgePoints"
                mode="multiple"
                placeholder="请选择要考查的知识点"
                :options="knowledgePointOptions"
                :loading="chaptersLoading"
              />
            </a-form-item>

            <!-- 难度级别 -->
            <a-form-item label="难度级别" name="difficulty">
              <a-radio-group v-model:value="assignmentForm.difficulty" button-style="solid">
                <a-radio-button value="EASY">简单</a-radio-button>
                <a-radio-button value="MEDIUM">中等</a-radio-button>
                <a-radio-button value="HARD">困难</a-radio-button>
              </a-radio-group>
            </a-form-item>

            <!-- 题目数量 -->
            <a-form-item label="题目数量" name="questionCount">
              <a-input-number
                v-model:value="assignmentForm.questionCount"
                :min="1"
                :max="20"
                style="width: 100%"
              />
            </a-form-item>

            <!-- 题型分布 -->
            <a-form-item label="题型分布">
              <div class="question-types">
                <div class="type-item" v-for="(type, key) in questionTypeLabels" :key="key">
                  <span class="type-label">{{ type }}:</span>
                  <a-input-number
                    v-model:value="assignmentForm.questionTypes[key]"
                    :min="0"
                    :max="10"
                    @change="() => updateTotalQuestionCount()"
                  />
                </div>
              </div>
              <div class="question-count-summary">
                当前总题数: {{ totalQuestionCount }} / {{ assignmentForm.questionCount }}
              </div>
            </a-form-item>
          </template>

          <!-- 文件提交模式下的设置 -->
          <template v-else-if="assignmentForm.mode === 'file'">
            <a-divider>文件提交设置</a-divider>
            
            <a-form-item label="允许的文件类型" name="allowedFileTypes">
              <a-select
                v-model:value="assignmentForm.allowedFileTypes"
                mode="multiple"
                placeholder="请选择允许提交的文件类型"
              >
                <a-select-option value="pdf">PDF文档</a-select-option>
                <a-select-option value="doc">Word文档</a-select-option>
                <a-select-option value="ppt">PowerPoint演示文稿</a-select-option>
                <a-select-option value="zip">ZIP压缩包</a-select-option>
                <a-select-option value="image">图片文件</a-select-option>
                <a-select-option value="code">代码文件</a-select-option>
              </a-select>
            </a-form-item>
            
            <a-form-item label="最大文件大小" name="maxFileSize">
              <a-input-number
                v-model:value="assignmentForm.maxFileSize"
                :min="1"
                :max="100"
                addonAfter="MB"
                style="width: 100%"
              />
            </a-form-item>

            <a-form-item label="参考答案" name="referenceAnswer">
              <a-textarea
                v-model:value="assignmentForm.referenceAnswer"
                placeholder="请输入参考答案，用于智能批改"
                :rows="6"
              />
            </a-form-item>
          </template>

          <a-form-item label="状态" name="status">
            <a-radio-group v-model:value="assignmentForm.status">
              <a-radio :value="0">草稿</a-radio>
              <a-radio :value="1">立即发布</a-radio>
            </a-radio-group>
          </a-form-item>

          <a-form-item :wrapper-col="{ offset: 4, span: 20 }">
            <a-space>
              <a-button type="primary" @click="handleSaveAssignment" :loading="saving">
                保存作业
              </a-button>
              <a-button @click="goBack">取消</a-button>
              <a-button v-if="assignmentForm.mode === 'question'" type="dashed" @click="handleGeneratePaper" :loading="generating">
                智能组卷
              </a-button>
            </a-space>
          </a-form-item>
        </a-form>
      </a-card>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, reactive, computed, onMounted } from 'vue'
import { useRouter, useRoute } from 'vue-router'
import { message, Modal } from 'ant-design-vue'
import axios from 'axios'
import dayjs from 'dayjs'

const router = useRouter()
const route = useRoute()

// 状态变量
const courses = ref<any[]>([])
const coursesLoading = ref(false)
const chaptersLoading = ref(false)
const knowledgePointOptions = ref<any[]>([])
const saving = ref(false)
const generating = ref(false)

// 题型标签
const questionTypeLabels = {
  'SINGLE_CHOICE': '单选题',
  'MULTIPLE_CHOICE': '多选题',
  'TRUE_FALSE': '判断题',
  'FILL_BLANK': '填空题',
  'SHORT_ANSWER': '简答题',
  'CODING': '编程题'
}

// 作业表单数据
const assignmentForm = reactive({
  id: null as number | null,
  title: '',
  courseId: null as number | null,
  description: '',
  startTime: null as any,
  endTime: null as any,
  totalScore: 100,
  status: 0, // 0: 草稿, 1: 已发布
  mode: 'question', // question: 答题模式, file: 文件提交模式
  type: 'homework', // 固定为作业类型
  
  // 智能组卷相关字段
  knowledgePoints: [] as string[],
  difficulty: 'MEDIUM',
  questionCount: 10,
  questionTypes: {
    'SINGLE_CHOICE': 5,
    'MULTIPLE_CHOICE': 2,
    'TRUE_FALSE': 3,
    'FILL_BLANK': 0,
    'SHORT_ANSWER': 0,
    'CODING': 0
  },
  
  // 文件提交相关字段
  allowedFileTypes: ['pdf', 'doc'],
  maxFileSize: 10, // MB
  referenceAnswer: ''
})

// 计算总题目数量
const totalQuestionCount = computed(() => {
  return Object.values(assignmentForm.questionTypes).reduce((sum, count) => sum + count, 0)
})

// 更新总题目数量
const updateTotalQuestionCount = () => {
  const total = totalQuestionCount.value
  if (total !== assignmentForm.questionCount) {
    message.warning(`题型分布总数(${total})与题目数量(${assignmentForm.questionCount})不匹配，请调整`)
  }
}

// 禁用过去的日期
const disablePastDates = (current: Date) => {
  return current && current < dayjs().startOf('day').toDate()
}

// 返回上一页
const goBack = () => {
  router.back()
}

// 加载教师课程列表
const loadTeacherCourses = async () => {
  try {
    console.log('📚 开始获取教师课程列表...')
    coursesLoading.value = true
    
    // 获取token
    const token = localStorage.getItem('token') || localStorage.getItem('user-token')
    
    const response = await axios.get('/api/teacher/courses', {
      headers: {
        'Authorization': token ? `Bearer ${token}` : ''
      }
    })
    console.log('📚 课程列表响应:', response)
    
    if (response.data && response.data.code === 200) {
      // 处理可能的嵌套数据结构
      let courseData = response.data.data
      
      // 检查是否有嵌套的records或list字段
      if (courseData.records) {
        courses.value = courseData.records
      } else if (courseData.list) {
        courses.value = courseData.list
      } else if (Array.isArray(courseData)) {
        courses.value = courseData
      } else {
        console.warn('未能识别的课程数据结构:', courseData)
        courses.value = []
      }
      
      console.log('✅ 成功加载课程列表，数量:', courses.value.length)
    } else {
      message.error('获取课程列表失败')
      courses.value = []
    }
  } catch (error) {
    console.error('加载课程列表失败:', error)
    message.error('获取课程列表失败，请检查网络连接')
    courses.value = []
  } finally {
    coursesLoading.value = false
  }
}

// 根据课程ID加载章节列表
const loadCourseChapters = async (courseId: number) => {
  try {
    console.log('📖 开始获取课程章节列表，课程ID:', courseId)
    chaptersLoading.value = true
    knowledgePointOptions.value = []
    assignmentForm.knowledgePoints = []
    
    // 获取token
    const token = localStorage.getItem('token') || localStorage.getItem('user-token')
    
    // 使用正确的API路径
    const response = await axios.get(`/api/teacher/chapters/course/${courseId}`, {
      headers: {
        'Authorization': token ? `Bearer ${token}` : ''
      }
    })
    console.log('📖 章节列表响应:', response)
    
    if (response.data && response.data.code === 200) {
      // 处理可能的嵌套数据结构
      let chapterData = response.data.data
      
      // 将章节转换为知识点选项
      if (Array.isArray(chapterData)) {
        knowledgePointOptions.value = chapterData.flatMap((chapter: any) => {
          // 如果有小节，使用小节作为知识点
          if (chapter.sections && chapter.sections.length > 0) {
            return chapter.sections.map((section: any) => ({
              label: `${chapter.title} - ${section.title}`,
              value: `${section.id}`
            }))
          }
          
          // 否则使用章节作为知识点
          return {
            label: chapter.title,
            value: `${chapter.id}`
          }
        })
      } else {
        console.warn('未能识别的章节数据结构:', chapterData)
        setDefaultKnowledgePoints(courseId)
      }
    } else {
      console.warn('获取章节列表返回异常:', response)
      setDefaultKnowledgePoints(courseId)
    }
  } catch (error) {
    console.error('加载章节列表失败:', error)
    setDefaultKnowledgePoints(courseId)
  } finally {
    chaptersLoading.value = false
  }
}

// 设置默认知识点
const setDefaultKnowledgePoints = (courseId: number) => {
  message.warning('获取章节列表失败，将使用默认知识点')
  
  // 根据课程ID设置不同的默认知识点
  const courseKnowledgePoints: Record<number, any[]> = {
    19: [ // Java程序设计
      { label: 'Java基础语法', value: 'java_basic' },
      { label: '面向对象编程', value: 'java_oop' },
      { label: '集合框架', value: 'java_collection' },
      { label: '异常处理', value: 'java_exception' },
      { label: '多线程', value: 'java_thread' }
    ],
    20: [ // 数据结构与算法
      { label: '数组与链表', value: 'ds_array_list' },
      { label: '栈与队列', value: 'ds_stack_queue' },
      { label: '树与图', value: 'ds_tree_graph' },
      { label: '排序算法', value: 'algo_sort' },
      { label: '查找算法', value: 'algo_search' }
    ],
    21: [ // Python程序基础
      { label: 'Python基础语法', value: 'py_basic' },
      { label: '数据类型与结构', value: 'py_data_type' },
      { label: '函数与模块', value: 'py_function' },
      { label: '文件操作', value: 'py_file' },
      { label: '异常处理', value: 'py_exception' }
    ]
  }
  
  // 获取对应课程的知识点，如果没有则使用通用知识点
  knowledgePointOptions.value = courseKnowledgePoints[courseId] || [
    { label: '基础知识点1', value: 'basic1' },
    { label: '基础知识点2', value: 'basic2' },
    { label: '基础知识点3', value: 'basic3' },
    { label: '进阶知识点1', value: 'advanced1' },
    { label: '进阶知识点2', value: 'advanced2' }
  ]
}

// 处理课程变更
const handleCourseChange = (courseId: number) => {
  console.log('课程变更:', courseId)
  loadCourseChapters(courseId)
}

// 保存作业
const handleSaveAssignment = async () => {
  // 表单验证
  if (!assignmentForm.title) {
    message.error('请输入作业名称')
    return
  }
  if (!assignmentForm.courseId) {
    message.error('请选择课程')
    return
  }
  if (!assignmentForm.startTime || !assignmentForm.endTime) {
    message.error('请选择作业时间')
    return
  }
  if (!assignmentForm.mode) {
    message.error('请选择作业模式')
    return
  }

  // 如果是答题模式，验证题目数量和题型分布
  if (assignmentForm.mode === 'question') {
    if (totalQuestionCount.value !== assignmentForm.questionCount) {
      message.warning('题型分布总数与题目数量不匹配，请调整题型分布')
      return
    }
  }

  saving.value = true
  try {
    // 获取当前用户ID
    const userInfo = localStorage.getItem('user-info')
    let userId = null
    
    if (userInfo) {
      try {
        const userObj = JSON.parse(userInfo)
        userId = userObj.id
        console.log('当前用户ID:', userId)
      } catch (e) {
        console.error('解析用户信息失败:', e)
      }
    }
    
    // 构建作业数据，添加固定字段
    const assignmentData = {
      ...assignmentForm,
      type: 'homework', // 固定值，表示作业而非考试
      userId: userId, // 添加用户ID字段
      
      // 格式化日期时间
      startTime: assignmentForm.startTime ? dayjs(assignmentForm.startTime).format('YYYY-MM-DD HH:mm:ss') : null,
      endTime: assignmentForm.endTime ? dayjs(assignmentForm.endTime).format('YYYY-MM-DD HH:mm:ss') : null
    }
    
    console.log('保存作业数据:', assignmentData)
    
    // 获取token
    const token = localStorage.getItem('token') || localStorage.getItem('user-token')
    const headers = {
      'Authorization': token ? `Bearer ${token}` : ''
    }
    
    let response: any
    if (assignmentForm.id) {
      // 编辑现有作业
      response = await axios.put(`/api/teacher/assignments/${assignmentForm.id}`, assignmentData, { headers })
      console.log('更新作业响应:', response.data)
      
      if (response.data && response.data.code === 200) {
        message.success('作业更新成功')
        
        // 如果是答题型作业，询问是否跳转到题目编辑页面
        if (assignmentForm.mode === 'question') {
          Modal.confirm({
            title: '是否编辑题目？',
            content: '作业更新成功，是否前往编辑题目？',
            okText: '是',
            cancelText: '否',
            onOk: () => {
              router.push(`/teacher/assignments/${assignmentForm.id}/edit`)
            },
            onCancel: () => {
              router.push('/teacher/assignments')
            }
          })
        } else {
          router.push('/teacher/assignments')
        }
      } else {
        message.error(response.data?.message || '更新作业失败')
      }
    } else {
      // 创建新作业
      response = await axios.post('/api/teacher/assignments', assignmentData, { headers })
      console.log('创建作业响应:', response.data)
      
      if (response.data && response.data.code === 200) {
        message.success('作业添加成功')
        
        // 如果是答题型作业，询问是否跳转到题目编辑页面
        if (assignmentForm.mode === 'question') {
          const assignmentId = response.data.data
          Modal.confirm({
            title: '是否编辑题目？',
            content: '作业添加成功，是否前往编辑题目？',
            okText: '是',
            cancelText: '否',
            onOk: () => {
              router.push(`/teacher/assignments/${assignmentId}/edit`)
            },
            onCancel: () => {
              router.push('/teacher/assignments')
            }
          })
        } else {
          router.push('/teacher/assignments')
        }
      } else {
        message.error(response.data?.message || '添加作业失败')
      }
    }
  } catch (error: any) {
    console.error('保存作业失败:', error)
    message.error(`保存作业失败: ${error.message || '未知错误'}`)
  } finally {
    saving.value = false
  }
}

// 智能组卷
const handleGeneratePaper = async () => {
  if (!assignmentForm.courseId) {
    message.error('请先选择课程')
    return
  }
  
  if (assignmentForm.knowledgePoints.length === 0) {
    message.error('请选择至少一个知识点')
    return
  }
  
  generating.value = true
  try {
    // 构建组卷请求
    const paperRequest = {
      courseId: assignmentForm.courseId,
      knowledgePoints: assignmentForm.knowledgePoints,
      difficulty: assignmentForm.difficulty,
      questionCount: assignmentForm.questionCount,
      questionTypes: assignmentForm.questionTypes,
      duration: 60, // 默认60分钟
      totalScore: assignmentForm.totalScore,
      additionalRequirements: '作业题目，难度适中，知识点覆盖全面'
    }
    
    console.log('智能组卷请求:', paperRequest)
    
    // 获取token
    const token = localStorage.getItem('token') || localStorage.getItem('user-token')
    
    // 调用智能组卷API
    const response = await axios.post('/api/teacher/assignments/generate-paper', paperRequest, {
      headers: {
        'Authorization': token ? `Bearer ${token}` : ''
      }
    })
    
    console.log('智能组卷响应:', response.data)
    
    if (response.data && response.data.code === 200) {
      const paperResult = response.data.data
      
      // 更新表单数据
      assignmentForm.title = paperResult.title || assignmentForm.title
      
      message.success('智能组卷成功，请保存作业')
      
      // 如果有题目，可以添加到作业中
      if (paperResult.questions && paperResult.questions.length > 0) {
        // 这里可以处理题目数据，但通常需要先保存作业才能添加题目
        console.log('生成的题目:', paperResult.questions)
        
        Modal.confirm({
          title: '组卷成功',
          content: `已成功生成${paperResult.questions.length}道题目，是否保存作业并编辑题目？`,
          okText: '保存并编辑题目',
          cancelText: '仅保存作业',
          onOk: () => handleSaveAssignment()
        })
      }
    } else {
      message.error(response.data?.message || '智能组卷失败')
    }
  } catch (error: any) {
    console.error('智能组卷失败:', error)
    message.error(`智能组卷失败: ${error.message || '未知错误'}`)
  } finally {
    generating.value = false
  }
}

// 初始化
onMounted(async () => {
  await loadTeacherCourses()
  
  // 如果是编辑模式，加载作业数据
  const assignmentId = route.params.id
  if (assignmentId) {
    try {
      // 获取token
      const token = localStorage.getItem('token') || localStorage.getItem('user-token')
      
      const response = await axios.get(`/api/teacher/assignments/${assignmentId}`, {
        headers: {
          'Authorization': token ? `Bearer ${token}` : ''
        }
      })
      
      if (response.data && response.data.code === 200) {
        const assignment = response.data.data
        
        // 填充表单数据
        Object.keys(assignmentForm).forEach(key => {
          if (assignment[key] !== undefined) {
            // 特殊处理日期字段
            if (key === 'startTime' || key === 'endTime') {
              assignmentForm[key] = assignment[key] ? dayjs(assignment[key]) : null
            } else {
              assignmentForm[key] = assignment[key]
            }
          }
        })
        
        // 加载课程章节
        if (assignmentForm.courseId) {
          loadCourseChapters(assignmentForm.courseId)
        }
      } else {
        message.error('加载作业数据失败')
      }
    } catch (error) {
      console.error('加载作业数据失败:', error)
      message.error('加载作业数据失败，请检查网络连接')
    }
  }
})
</script>

<style scoped>
.create-assignment-page {
  padding: 24px;
  background-color: #f0f2f5;
  min-height: 100vh;
}

.content-container {
  max-width: 1200px;
  margin: 0 auto;
}

.assignment-form-card {
  margin-top: 24px;
  border-radius: 8px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.09);
}

.question-types {
  display: flex;
  flex-wrap: wrap;
  gap: 16px;
}

.type-item {
  display: flex;
  align-items: center;
  gap: 8px;
  flex-basis: calc(33.33% - 16px);
}

.type-label {
  min-width: 80px;
}

.question-count-summary {
  margin-top: 16px;
  color: #ff4d4f;
  font-weight: 500;
}

@media (max-width: 768px) {
  .type-item {
    flex-basis: 100%;
  }
}
</style> 