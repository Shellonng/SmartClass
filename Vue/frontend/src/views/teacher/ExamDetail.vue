<template>
  <div class="exam-detail-page">
    <div class="exam-container">
      <!-- 返回按钮 -->
      <div class="back-link">
        <a-button type="link" @click="goBack">
          <arrow-left-outlined /> 返回{{ props.isAssignment ? '作业' : '考试' }}列表
        </a-button>
      </div>

      <!-- 考试头部信息 -->
      <div class="exam-header">
        <div class="exam-header-top">
          <h1 class="exam-title">{{ exam.title }}</h1>
          <div class="exam-status-tags">
            <!-- 发布状态标签 -->
            <a-tag :color="isPublished ? 'green' : 'orange'" class="status-tag">
              {{ isPublished ? '已发布' : '未发布' }}
            </a-tag>
            <!-- 考试进行状态标签 -->
            <a-tag v-if="exam.examState" :color="getStatusColor(exam.examState)" class="status-tag">
              {{ getStatusText(exam.examState) }}
            </a-tag>
          </div>
        </div>
        <div class="exam-info">
          <div class="info-item">
            <span class="info-label">题量：</span>
            <span class="info-value">{{ questions.length || '0' }}</span>
          </div>
          <div class="info-item">
            <span class="info-label">满分：</span>
            <span class="info-value">{{ exam.totalScore || '100' }}分</span>
          </div>
          <div class="info-item">
            <span class="info-label">{{ props.isAssignment ? '作业' : '考试' }}时间：</span>
            <span class="info-value">
              {{ formatDate(exam.startTime) }} ~ {{ formatDate(exam.endTime) }}
            </span>
          </div>
        </div>
        <div v-if="exam.description" class="exam-description">
          <div class="description-label">{{ props.isAssignment ? '作业' : '考试' }}说明：</div>
          <div class="description-content">{{ exam.description }}</div>
        </div>
      </div>

      <!-- 操作按钮 -->
      <div class="action-buttons">
        <a-button type="primary" @click="editExam" v-if="!props.viewOnly">
          <edit-outlined /> 编辑{{ props.isAssignment ? '作业' : '考试' }}
        </a-button>
        <a-button @click="togglePreview" :type="isPreview ? 'primary' : 'default'">
          <eye-outlined /> {{ isPreview ? '退出预览' : '预览效果' }}
        </a-button>
        <a-button v-if="!isPublished && !props.viewOnly" type="primary" @click="publishExam">
          <upload-outlined /> 发布{{ props.isAssignment ? '作业' : '考试' }}
        </a-button>
        <a-button v-if="isPublished && !props.viewOnly" type="warning" @click="unpublishExam">
          <close-circle-outlined /> 取消发布
        </a-button>
      </div>

      <!-- 题目区域 -->
      <div class="exam-content">
        <a-spin :spinning="loading">
          <div v-if="!questions || questions.length === 0" class="empty-content">
            <a-empty description="暂无题目">
              <template #description>
                <div>
                  <p style="font-size: 16px; margin-bottom: 8px;">暂无题目</p>
                  <p style="color: #999; margin-bottom: 16px;">该{{ props.isAssignment ? '作业' : '考试' }}尚未添加任何题目，请点击下方按钮开始组卷</p>
                  <a-button type="primary" @click="editExam" style="margin-top: 8px" v-if="!props.viewOnly">
                    <plus-outlined /> 添加题目
                  </a-button>
                </div>
              </template>
            </a-empty>
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
                    <div v-if="!isPreview && question.knowledgePoint" class="knowledge-point">
                      <a-tag color="blue">知识点: {{ question.knowledgePoint }}</a-tag>
                    </div>
                  </div>
                  
                  <div class="question-content">{{ question.title }}</div>
                  
                  <!-- 选择题选项 -->
                  <div v-if="['single', 'multiple'].includes(question.questionType)" class="question-options">
                    <div v-for="option in question.options" :key="option.id" class="option-item">
                      <a-radio-group v-if="question.questionType === 'single'" :value="isPreview ? null : question.correctAnswer" disabled>
                        <a-radio :value="option.optionLabel" :class="{'correct-option': !isPreview && option.optionLabel === question.correctAnswer}">
                          {{ option.optionLabel }}. {{ option.optionText }}
                        </a-radio>
                      </a-radio-group>
                      
                      <a-checkbox-group v-else-if="question.questionType === 'multiple'" :value="isPreview ? [] : question.correctAnswer?.split(',')" disabled>
                        <a-checkbox :value="option.optionLabel" :class="{'correct-option': !isPreview && question.correctAnswer?.split(',').includes(option.optionLabel)}">
                          {{ option.optionLabel }}. {{ option.optionText }}
                        </a-checkbox>
                      </a-checkbox-group>
                    </div>
                  </div>
                  
                  <!-- 判断题选项 -->
                  <div v-else-if="question.questionType === 'true_false'" class="question-options">
                    <a-radio-group :value="isPreview ? null : question.correctAnswer" disabled>
                      <a-radio value="T" :class="{'correct-option': !isPreview && question.correctAnswer === 'T'}">正确</a-radio>
                      <a-radio value="F" :class="{'correct-option': !isPreview && question.correctAnswer === 'F'}">错误</a-radio>
                    </a-radio-group>
                  </div>
                  
                  <!-- 填空题 -->
                  <div v-else-if="question.questionType === 'blank'" class="question-blank">
                    <div v-if="!isPreview" class="correct-answer">
                      <div class="answer-label">参考答案：</div>
                      <div class="answer-content">{{ question.correctAnswer }}</div>
                    </div>
                    <a-input placeholder="学生答题区域" disabled />
                  </div>
                  
                  <!-- 简答题 -->
                  <div v-else-if="question.questionType === 'short'" class="question-short">
                    <div v-if="!isPreview" class="correct-answer">
                      <div class="answer-label">参考答案：</div>
                      <div class="answer-content">{{ question.correctAnswer }}</div>
                    </div>
                    <a-textarea placeholder="学生答题区域" :rows="4" disabled />
                  </div>
                  
                  <!-- 其他题型 -->
                  <div v-else class="question-other">
                    <div v-if="!isPreview" class="correct-answer">
                      <div class="answer-label">参考答案：</div>
                      <div class="answer-content">{{ question.correctAnswer }}</div>
                    </div>
                    <a-textarea placeholder="学生答题区域" :rows="6" disabled />
                  </div>
                  
                  <!-- 答案解析 -->
                  <div v-if="!isPreview && question.explanation" class="question-explanation">
                    <div class="explanation-label">答案解析：</div>
                    <div class="explanation-content">{{ question.explanation }}</div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </a-spin>
      </div>
    </div>
  </div>
  
  <!-- 编辑考试弹窗 -->
  <a-modal 
    v-model:open="editModalVisible" 
    :title="`编辑${props.isAssignment ? '作业' : '考试'}`"
    width="900px" 
    :footer="null"
    :maskClosable="false"
  >
    <div class="edit-exam-container">
      <!-- 考试标题显示 -->
      <div class="edit-exam-header">
        <h3>{{ editingExam.title }}</h3>
        <div class="edit-exam-info">
          <span>{{ props.isAssignment ? '作业' : '考试' }}时间: {{ formatDate(editingExam.startTime) }} ~ {{ formatDate(editingExam.endTime) }}</span>
        </div>
      </div>
      
      <!-- 考试大项管理 -->
      <div class="edit-exam-sections">
        <div class="section-header">
          <h3>{{ props.isAssignment ? '作业' : '考试' }}内容管理</h3>
          <a-button type="primary" @click="addSection">
            <plus-outlined /> 添加题型
          </a-button>
        </div>
        
        <!-- 大项列表 -->
        <div v-if="editingExamSections.length === 0" class="empty-sections">
          <a-empty description="暂无题型，请添加题型" />
        </div>
        
        <div v-else class="section-list">
          <div v-for="(section, index) in editingExamSections" :key="section.id" class="section-item">
            <div class="section-item-header">
              <div class="section-title">
                {{ getRomanNumber(index + 1) }}、{{ section.title }} 
                <span class="section-summary">共{{ section.questions.length }}/{{ section.count || section.questions.length }}题，{{ section.totalScore }}分</span>
                <span v-if="section.score" class="section-per-score">(每题{{ section.score }}分)</span>
              </div>
              <div class="section-actions">
                <a-button type="primary" size="small" @click="addQuestionToSection(section)">添加题目</a-button>
                <a-button type="danger" size="small" @click="removeSection(index)">删除题型</a-button>
              </div>
            </div>
            
            <!-- 题目列表 -->
            <div v-if="section.questions.length === 0" class="empty-questions">
              <a-empty description="暂无题目，请添加题目" />
            </div>
            
            <div v-else class="question-list">
              <div v-for="(question, qIndex) in section.questions" :key="question.id" class="question-item-mini">
                <div class="question-mini-content">
                  <div class="question-mini-index">{{ qIndex + 1 }}.</div>
                  <div class="question-mini-title">{{ question.title }}</div>
                </div>
                <div class="question-mini-actions">
                  <a-tag color="blue">{{ question.score }}分</a-tag>
                  <a-button type="link" danger @click="removeQuestionFromSection(section, qIndex)">
                    <delete-outlined />
                  </a-button>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
      
      <!-- 底部操作按钮 -->
      <div class="edit-exam-footer">
        <a-button @click="editModalVisible = false">取消</a-button>
        <a-button type="primary" @click="saveEditedExam">保存</a-button>
      </div>
    </div>
  </a-modal>
  
  <!-- 添加题型弹窗 -->
  <a-modal
    v-model:open="addSectionVisible"
    title="添加题型"
    :footer="null"
    width="500px"
  >
    <div class="add-section-container">
      <a-form layout="vertical">
        <a-form-item label="题目类型" required>
          <a-select v-model:value="currentSectionType" placeholder="请选择题目类型">
            <a-select-option value="single">单选题</a-select-option>
            <a-select-option value="multiple">多选题</a-select-option>
            <a-select-option value="true_false">判断题</a-select-option>
            <a-select-option value="blank">填空题</a-select-option>
            <a-select-option value="short">简答题</a-select-option>
          </a-select>
        </a-form-item>
        
        <a-form-item label="题目数量" required>
          <a-input-number v-model:value="sectionConfig.count" :min="1" :max="50" />
        </a-form-item>
        
        <a-form-item label="每题分值" required>
          <a-input-number v-model:value="sectionConfig.score" :min="1" :max="100" />
        </a-form-item>
      </a-form>
      
      <div class="modal-footer">
        <a-button @click="addSectionVisible = false">取消</a-button>
        <a-button type="primary" @click="confirmAddSection">确定</a-button>
      </div>
    </div>
  </a-modal>
  
  <!-- 添加题目弹窗 -->
  <a-modal
    v-model:open="sectionQuestionVisible"
    :title="`添加${currentEditSection?.title || ''}题目`"
    width="900px"
    :footer="null"
    :maskClosable="false"
  >
    <div v-if="currentEditSection" class="question-select-container">
      <!-- 筛选区域 -->
      <div class="question-filter">
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
              @search="filterQuestions"
            />
          </a-form-item>
          
          <a-form-item>
            <a-button @click="resetQuestionFilters">重置筛选</a-button>
          </a-form-item>
        </a-form>
      </div>
      
      <!-- 题目列表 -->
      <div class="available-question-list">
        <a-spin :spinning="loadingQuestions">
          <a-empty v-if="availableQuestions.length === 0" description="暂无符合条件的题目" />
          
          <div v-else>
            <div v-for="question in availableQuestions" :key="question.id" class="question-select-item">
              <a-checkbox 
                :checked="question.selected" 
                @change="() => toggleQuestionSelected(question)"
              >
                <div class="question-select-content">
                  <div class="question-select-title">{{ question.title }}</div>
                  <div class="question-select-info">
                    <a-tag>难度: {{ getDifficultyStars(question.difficulty) }}</a-tag>
                    <a-tag v-if="question.knowledgePoint" color="blue">{{ question.knowledgePoint }}</a-tag>
                    <a-tag color="orange">{{ question.score || 5 }}分</a-tag>
                  </div>
                </div>
              </a-checkbox>
            </div>
          </div>
        </a-spin>
      </div>
      
      <!-- 底部操作按钮 -->
      <div class="modal-footer">
        <span class="selection-info">已选择 {{ selectedQuestions.length }} 题</span>
        <div>
          <a-button @click="sectionQuestionVisible = false">取消</a-button>
          <a-button type="primary" @click="confirmAddQuestions">确定</a-button>
        </div>
      </div>
    </div>
  </a-modal>
</template>

<script setup lang="ts">
import { ref, onMounted, computed, watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { message } from 'ant-design-vue'
import { formatDate } from '@/utils/date'
import { 
  ArrowLeftOutlined, 
  EditOutlined, 
  EyeOutlined, 
  UploadOutlined,
  PlusOutlined,
  DeleteOutlined,
  CloseCircleOutlined
} from '@ant-design/icons-vue'
import examAPI from '@/api/exam'
import axios from 'axios'
import assignmentApi from '@/api/assignment'

// 路由参数
const route = useRoute()
const router = useRouter()
const examId = computed(() => Number(route.params.id) || Number(route.query.id))

// 接收props参数
const props = defineProps({
  id: {
    type: Number,
    default: 0
  },
  isAssignment: {
    type: Boolean,
    default: false
  },
  viewOnly: {
    type: Boolean,
    default: false
  }
})

// 状态定义
const loading = ref(true)
const exam = ref<any>({})
const questions = ref<any[]>([])
const isPreview = ref(false) // 是否处于预览模式
const knowledgePoints = ref<string[]>([])

// 新增题型配置
const sectionConfig = ref({
  count: 10, // 默认题目数量
  score: 5   // 默认每题分值
})

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

// 判断是否可以发布考试
const canPublish = computed(() => {
  if (!exam.value) return false
  
  // 判断状态是否为未发布
  const status = exam.value.status
  return status !== 'in_progress' && status !== 'ended'
})

// 判断考试是否已发布
const isPublished = computed(() => {
  return exam.value && exam.value.status === 1
})

// 获取考试详情
const fetchExamDetail = async () => {
  loading.value = true
  try {
    // 调用API获取考试/作业详情
    let response;
    
    if (props.isAssignment) {
      // 获取作业详情
      const axiosResponse = await axios.get(`/api/teacher/assignments/${examId.value}`);
      response = axiosResponse.data;
    } else {
      // 获取考试详情
      response = await examAPI.getExamDetail(examId.value);
    }
    
    if (response && response.code === 200) {
      exam.value = response.data
      
      // 保存原始发布状态
      const publishStatus = exam.value.status
      
      // 设置考试进行状态
      const now = new Date().getTime()
      const startTime = new Date(exam.value.startTime).getTime()
      const endTime = new Date(exam.value.endTime).getTime()
      
      // 如果后端返回的是数字状态，则保留原值作为发布状态
      if (typeof exam.value.status === 'number') {
        // 保留原始发布状态（0未发布，1已发布）
        // 不需要转换
      } else {
        // 设置考试进行状态（不影响发布状态）
        if (now < startTime) {
          exam.value.examState = 'not_started'
        } else if (now >= startTime && now <= endTime) {
          exam.value.examState = 'in_progress'
        } else {
          exam.value.examState = 'ended'
        }
      }
      
      // 初始化问题
      if (exam.value.questions) {
        // 如果考试详情中已经有题目信息，直接使用
        console.log(`${props.isAssignment ? '作业' : '考试'}详情包含题目信息:`, exam.value.questions)
        questions.value = exam.value.questions || []
      } else {
        // 否则单独获取题目
        fetchExamQuestions()
      }
    } else {
      message.error(response?.message || `获取${props.isAssignment ? '作业' : '考试'}信息失败`)
    }
  } catch (error) {
    console.error(`获取${props.isAssignment ? '作业' : '考试'}详情失败:`, error)
    message.error(`获取${props.isAssignment ? '作业' : '考试'}详情失败`)
  } finally {
    loading.value = false
  }
}

// 获取考试题目
const fetchExamQuestions = async () => {
  loading.value = true
  try {
    // 如果考试详情中没有题目，尝试单独获取题目
    try {
      console.log(`正在获取${props.isAssignment ? '作业' : '考试'}题目:`, examId.value)
      
      let response;
      if (props.isAssignment) {
        // 获取作业题目
        const axiosResponse = await axios.get(`/api/teacher/assignments/${examId.value}/questions`);
        response = axiosResponse.data;
      } else {
        // 获取考试题目
        response = await examAPI.getTeacherExamQuestions(examId.value);
      }
      
      if (response && response.code === 200) {
        questions.value = response.data || []
        console.log('成功获取题目数据:', questions.value)
        
        // 收集知识点列表用于筛选
        const pointSet = new Set<string>()
        questions.value.forEach(q => {
          if (q.knowledgePoint) {
            pointSet.add(q.knowledgePoint)
          }
        })
        knowledgePoints.value = Array.from(pointSet)
      } else {
        console.log(`获取${props.isAssignment ? '作业' : '考试'}题目返回非200状态:`, response)
        // 不显示错误信息，保持空题目状态
        questions.value = []
      }
    } catch (error) {
      // 处理404或其他错误情况
      console.error(`单独获取${props.isAssignment ? '作业' : '考试'}题目失败:`, error)
      // 不显示错误信息，保持空题目状态
      questions.value = []
    }
  } catch (error) {
    console.error(`获取${props.isAssignment ? '作业' : '考试'}题目失败:`, error)
    // 改为不显示错误信息，以更友好的空状态展示
    questions.value = []
  } finally {
    loading.value = false
  }
}

// 将难度等级转换为星星显示
const getDifficultyStars = (difficulty: number) => {
  const stars = '★'.repeat(difficulty || 1)
  return stars
}

// 数字转罗马数字
const getRomanNumber = (num: number) => {
  const roman = ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'X']
  return roman[num - 1] || num.toString()
}

// 返回考试列表
const goBack = () => {
  if (props.isAssignment) {
    router.push('/teacher/courses/' + (exam.value.courseId || route.query.courseId) + '?view=assignments')
  } else {
    router.push('/teacher/courses/' + (exam.value.courseId || route.query.courseId) + '?view=exams')
  }
}

// 编辑考试相关状态
const editModalVisible = ref(false)
const editingExam = ref<any>({})
const editingExamSections = ref<any[]>([])
const currentEditSection = ref<any>(null)
const addSectionVisible = ref(false)
const sectionQuestionVisible = ref(false)
const currentSectionType = ref('')
const selectedQuestions = ref<any[]>([])
const availableQuestions = ref<any[]>([])
const questionFilters = ref({
  difficulty: undefined,
  knowledgePoint: undefined,
  keyword: ''
})
const loadingQuestions = ref(false)

// 编辑考试
const editExam = () => {
  // 初始化编辑数据
  editingExam.value = { ...exam.value }
  
  // 根据现有题目生成考试大项
  const sections: any[] = []
  const groupedQuestions = questionGroups.value || []
  
  groupedQuestions.forEach((group: any) => {
    if (group.questions && group.questions.length > 0) {
      sections.push({
        id: Date.now() + Math.random(),
        type: group.type,
        title: getQuestionTypeText(group.type),
        questions: [...group.questions],
        totalScore: group.totalScore
      })
    }
  })
  
  editingExamSections.value = sections
  editModalVisible.value = true
}

// 添加考试大项
const addSection = () => {
  currentSectionType.value = ''
  // 重置配置为默认值
  sectionConfig.value = {
    count: 10,
    score: 5
  }
  addSectionVisible.value = true
}

// 确认添加考试大项
const confirmAddSection = () => {
  if (!currentSectionType.value) {
    message.warning('请选择题目类型')
    return
  }
  
  if (!sectionConfig.value.count || sectionConfig.value.count <= 0) {
    message.warning('请设置有效的题目数量')
    return
  }
  
  if (!sectionConfig.value.score || sectionConfig.value.score <= 0) {
    message.warning('请设置有效的题目分值')
    return
  }
  
  // 添加新的大项
  editingExamSections.value.push({
    id: Date.now() + Math.random(),
    type: currentSectionType.value,
    title: getQuestionTypeText(currentSectionType.value),
    questions: [],
    count: sectionConfig.value.count,
    score: sectionConfig.value.score,
    totalScore: 0
  })
  
  // 清理数据并关闭弹窗
  currentSectionType.value = ''
  addSectionVisible.value = false
}

// 删除考试大项
const removeSection = (sectionIndex: number) => {
  editingExamSections.value.splice(sectionIndex, 1)
}

// 添加题目到大项
const addQuestionToSection = (section: any) => {
  currentEditSection.value = section
  // 重置选中的题目
  selectedQuestions.value = []
  // 获取该题型的可用题目
  fetchAvailableQuestions(section.type)
  // 打开题目选择弹窗
  sectionQuestionVisible.value = true
}

// 获取可用题目
const fetchAvailableQuestions = async (type: string) => {
  loadingQuestions.value = true
  try {
    // 确保传递参数格式正确
    const params = {
      courseId: exam.value.courseId,
      questionType: type,
      difficulty: questionFilters.value.difficulty,
      knowledgePoint: questionFilters.value.knowledgePoint,
      keyword: questionFilters.value.keyword
    }
    
    console.log('获取可用题目，参数:', params)
    const response = await examAPI.getCourseQuestionList(params)
    
    if (response && response.code === 200 && response.data) {
      // 处理返回的题目数据
      const questions = response.data.records || []
      
      // 标记已选题目
      const currentSectionQuestionIds = currentEditSection.value.questions.map((q: any) => q.id)
      
      availableQuestions.value = questions.map((q: any) => ({
        ...q,
        selected: currentSectionQuestionIds.includes(q.id)
      }))
      
      // 如果没有可用题目，显示提示
      if (availableQuestions.value.length === 0) {
        message.info(`没有找到${getQuestionTypeText(type)}题型的题目，请先创建题目`)
      }
    } else {
      message.error(response?.message || '获取题目失败')
      availableQuestions.value = []
    }
  } catch (error) {
    console.error('获取题目失败:', error)
    message.error('获取题目失败')
    availableQuestions.value = []
  } finally {
    loadingQuestions.value = false
  }
}

// 筛选题目
const filterQuestions = () => {
  fetchAvailableQuestions(currentEditSection.value.type)
}

// 重置筛选条件
const resetQuestionFilters = () => {
  questionFilters.value = {
    difficulty: undefined,
    knowledgePoint: undefined,
    keyword: ''
  }
  filterQuestions()
}

// 切换题目选择状态
const toggleQuestionSelected = (question: any) => {
  question.selected = !question.selected
  
  if (question.selected) {
    selectedQuestions.value.push(question)
  } else {
    const index = selectedQuestions.value.findIndex(q => q.id === question.id)
    if (index > -1) {
      selectedQuestions.value.splice(index, 1)
    }
  }
}

// 确认添加题目
const confirmAddQuestions = () => {
  if (selectedQuestions.value.length === 0) {
    message.warning('请至少选择一个题目')
    return
  }
  
  // 将选中的题目添加到当前编辑的大项中，并应用设置的分数
  const selectedQuestionsWithScore = selectedQuestions.value.map(q => ({
    ...q,
    score: currentEditSection.value.score || q.score || 5 // 应用大项设置的分数
  }));
  
  currentEditSection.value.questions = [
    ...currentEditSection.value.questions,
    ...selectedQuestionsWithScore.filter(sq => 
      !currentEditSection.value.questions.some((q: any) => q.id === sq.id)
    )
  ]
  
  // 更新总分
  currentEditSection.value.totalScore = currentEditSection.value.questions.reduce(
    (total: number, q: any) => total + (q.score || 0), 0
  )
  
  // 关闭弹窗
  sectionQuestionVisible.value = false
}

// 从大项中移除题目
const removeQuestionFromSection = (section: any, questionIndex: number) => {
  section.questions.splice(questionIndex, 1)
  
  // 更新总分
  section.totalScore = section.questions.reduce(
    (total: number, q: any) => total + (q.score || 0), 0
  )
}

// 保存编辑的考试
const saveEditedExam = async () => {
  // 检查每个大项是否有题目
  const emptySections = editingExamSections.value.filter(s => !s.questions || s.questions.length === 0)
  if (emptySections.length > 0) {
    message.warning(`有${emptySections.length}个题型没有添加题目，请添加题目或删除该题型`)
    return
  }
  
  // 准备数据
  const examData = {
    ...editingExam.value,
    totalScore: editingExamSections.value.reduce((total, s) => total + s.totalScore, 0),
    // 确保状态是数字类型，而不是字符串
    status: typeof editingExam.value.status === 'string' 
      ? getStatusNumber(editingExam.value.status) 
      : editingExam.value.status
  }
  
  console.log('保存考试数据:', examData)
  
  try {
    // 更新考试基本信息 - 使用正确的API路径
    const updateResponse = await examAPI.updateExam(examId.value, examData)
    
    if (updateResponse && updateResponse.code === 200) {
      // 收集所有题目ID和分值
      const allQuestionIds: number[] = []
      const allScores: number[] = []
      
      editingExamSections.value.forEach(section => {
        section.questions.forEach((q: any) => {
          allQuestionIds.push(q.id)
          allScores.push(q.score || 0)
        })
      })
      
      // 检查是否有题目需要关联
      if (allQuestionIds.length === 0) {
        message.success(`${props.isAssignment ? '作业' : '考试'}基本信息更新成功`)
        // 刷新数据
        fetchExamDetail()
        // 关闭弹窗
        editModalVisible.value = false
        return
      }
      
      // 保存题目关联 - 使用正确的API路径
      console.log('保存题目关联数据:', {
        examId: examId.value,
        questionIds: allQuestionIds,
        scores: allScores
      })
      
      try {
        const questionsResponse = await examAPI.selectQuestions(examId.value, allQuestionIds, allScores)
        
        if (questionsResponse && questionsResponse.code === 200) {
          message.success(`${props.isAssignment ? '作业' : '考试'}更新成功`)
          // 刷新数据
          fetchExamDetail()
          // 关闭弹窗
          editModalVisible.value = false
        } else {
          message.error(questionsResponse?.message || '题目关联保存失败')
        }
      } catch (error) {
        console.error('题目关联保存失败:', error)
        message.error('题目关联保存失败，请检查题目数据')
      }
    } else {
      message.error(updateResponse?.message || `${props.isAssignment ? '作业' : '考试'}更新失败`)
    }
  } catch (error) {
    console.error(`保存${props.isAssignment ? '作业' : '考试'}失败:`, error)
    message.error(`保存${props.isAssignment ? '作业' : '考试'}失败`)
  }
}

// 切换预览模式
const togglePreview = () => {
  isPreview.value = !isPreview.value
}

// 在新窗口预览考试
const previewExam = () => {
  // 在新窗口中打开预览
  window.open(`/student/exams/${examId.value}?preview=true`, '_blank')
}

// 发布考试
const publishExam = async () => {
  try {
    // 如果考试状态是字符串，先转换为数字
    if (exam.value && typeof exam.value.status === 'string') {
      const statusNumber = getStatusNumber(exam.value.status);
      console.log('发布前转换状态:', exam.value.status, '->', statusNumber);
    }
    
    let response;
    if (props.isAssignment) {
      // 调用作业发布API
      const axiosResponse = await axios.put(`/api/teacher/assignments/${examId.value}/publish`);
      response = axiosResponse.data;
    } else {
      // 调用考试发布API
      response = await examAPI.publishExam(examId.value);
    }
    
    if (response && response.code === 200) {
      message.success(`${props.isAssignment ? '作业' : '考试'}发布成功`);
      // 刷新信息
      fetchExamDetail();
    } else {
      message.error(response?.message || `发布${props.isAssignment ? '作业' : '考试'}失败`);
    }
  } catch (error) {
    console.error(`发布${props.isAssignment ? '作业' : '考试'}失败:`, error);
    message.error(`发布${props.isAssignment ? '作业' : '考试'}失败`);
  }
}

// 取消发布考试
const unpublishExam = async () => {
  try {
    // 创建一个包含必要字段的对象
    const examInfo = {
      id: examId.value,
      title: exam.value.title,
      courseId: exam.value.courseId,
      userId: exam.value.userId,
      description: exam.value.description || '',
      startTime: exam.value.startTime,
      endTime: exam.value.endTime,
      status: 0 // 将状态设置为0，表示未发布
    };
    
    console.log(`取消发布${props.isAssignment ? '作业' : '考试'}，数据:`, examInfo);
    
    let response;
    if (props.isAssignment) {
      // 调用作业取消发布API
      const response = await assignmentApi.unpublishAssignment(examId.value);
      if (response.code === 200) {
        message.success('作业取消发布成功');
        // 刷新信息
        fetchExamDetail();
      } else {
        message.error(response.message || '取消发布失败');
      }
    } else {
      // 调用考试取消发布API
      const response = await examAPI.updateExam(examId.value, examInfo);
      if (response && response.code === 200) {
        message.success('考试取消发布成功');
        // 刷新信息
        fetchExamDetail();
      } else {
        message.error(response?.message || '取消发布失败');
      }
    }
  } catch (error) {
    console.error(`取消发布${props.isAssignment ? '作业' : '考试'}失败:`, error);
    message.error(`取消发布${props.isAssignment ? '作业' : '考试'}失败`);
  }
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

// 获取状态文本映射
const statusTextMap: Record<string, string> = {
  'not_started': '未开始',
  'in_progress': '进行中',
  'ended': '已结束'
}

// 获取状态数字映射（后端使用）
const statusNumberMap: Record<string, number> = {
  'not_started': 0,  // 未开始状态为0
  'in_progress': 1,  // 进行中状态为1
  'ended': 2         // 已结束状态为2
}

// 获取状态显示文本
const getStatusText = (status: string): string => {
  return statusTextMap[status] || '未知状态'
}

// 字符串状态转数字
const getStatusNumber = (status: string): number => {
  return statusNumberMap[status] || 0
}

// 获取状态标签颜色
const getStatusColor = (status: string | number): string => {
  // 如果传入的是数字，先转为字符串状态
  if (typeof status === 'number') {
    status = Object.keys(statusNumberMap).find(key => statusNumberMap[key] === status) || 'not_started';
  }
  
  const colorMap: Record<string, string> = {
    'not_started': 'blue',
    'in_progress': 'green',
    'ended': 'gray'
  }
  return colorMap[status as string] || 'default'
}

// 生命周期钩子
onMounted(() => {
  fetchExamDetail()
})
</script>

<style scoped>
.exam-detail-page {
  padding: 24px;
  background-color: #f5f7fa;
  min-height: 100vh;
  display: flex;
  justify-content: center;
}

.exam-container {
  width: 100%;
  max-width: 1000px;
  background-color: #fff;
  padding: 30px;
  border-radius: 8px;
  box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
}

.back-link {
  margin-bottom: 20px;
}

.exam-header {
  margin-bottom: 24px;
  padding-bottom: 20px;
  border-bottom: 1px solid #e8e8e8;
}

.exam-header-top {
  display: flex;
  align-items: center;
  margin-bottom: 16px;
}

.exam-title {
  font-size: 24px;
  font-weight: 600;
  margin: 0;
  margin-right: 16px;
}

.exam-status-tags {
  display: flex;
  gap: 8px;
}

.status-tag {
  font-size: 14px;
  padding: 2px 12px;
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

.action-buttons {
  display: flex;
  gap: 12px;
  margin-bottom: 24px;
}

.exam-content {
  margin-top: 24px;
}

.empty-content {
  margin: 60px 0;
  padding: 40px;
  background-color: #f9f9f9;
  border-radius: 8px;
  border: 1px dashed #e8e8e8;
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
  background-color: #f5f7fa;
  padding: 10px 16px;
  border-radius: 4px;
}

.questions {
  margin-left: 12px;
}

.question-item {
  margin-bottom: 24px;
  padding: 20px;
  border: 1px solid #e8e8e8;
  border-radius: 8px;
  background-color: #fff;
  box-shadow: 0 1px 4px rgba(0, 0, 0, 0.03);
}

.question-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 16px;
  flex-wrap: wrap;
}

.question-index {
  font-weight: 600;
  font-size: 16px;
}

.knowledge-point {
  margin-top: 8px;
}

.question-content {
  margin-bottom: 16px;
  line-height: 1.6;
  font-size: 15px;
}

.question-options {
  margin-left: 8px;
}

.option-item {
  margin-bottom: 8px;
}

.correct-option {
  color: #52c41a;
  font-weight: 600;
}

.question-blank,
.question-short,
.question-other {
  margin-top: 16px;
}

.correct-answer {
  margin-bottom: 12px;
  padding: 10px 16px;
  background-color: #f9f9f9;
  border-radius: 4px;
  border-left: 4px solid #1890ff;
}

.answer-label {
  font-weight: 600;
  margin-bottom: 4px;
  color: #1890ff;
}

.answer-content {
  white-space: pre-line;
}

.question-explanation {
  margin-top: 16px;
  padding: 10px 16px;
  background-color: #f6ffed;
  border-radius: 4px;
  border-left: 4px solid #52c41a;
}

.explanation-label {
  font-weight: 600;
  margin-bottom: 4px;
  color: #52c41a;
}

.explanation-content {
  white-space: pre-line;
}

/* 编辑考试弹窗样式 */
.edit-exam-container {
  display: flex;
  flex-direction: column;
  gap: 20px;
}

.edit-exam-header {
  padding-bottom: 16px;
  border-bottom: 1px solid #e8e8e8;
}

.edit-exam-info {
  color: #666;
  margin-top: 8px;
}

.section-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 16px;
}

.section-list {
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.section-item {
  border: 1px solid #e8e8e8;
  border-radius: 8px;
  padding: 16px;
}

.section-item-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 16px;
}

.section-title {
  font-weight: 600;
}

.section-summary {
  font-weight: normal;
  color: #666;
  margin-left: 8px;
}

.section-per-score {
  font-weight: normal;
  color: #1890ff;
  margin-left: 8px;
  font-size: 12px;
}

.section-actions {
  display: flex;
  gap: 8px;
}

.empty-sections, .empty-questions {
  margin: 20px 0;
}

.question-item-mini {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 10px 16px;
  border-bottom: 1px solid #f0f0f0;
}

.question-mini-content {
  display: flex;
  align-items: flex-start;
  gap: 8px;
  width: 80%;
}

.question-mini-index {
  font-weight: 600;
  min-width: 24px;
}

.question-mini-title {
  overflow: hidden;
  text-overflow: ellipsis;
  display: -webkit-box;
  -webkit-line-clamp: 1;
  -webkit-box-orient: vertical;
}

.question-mini-actions {
  display: flex;
  align-items: center;
  gap: 8px;
}

.edit-exam-footer {
  display: flex;
  justify-content: flex-end;
  gap: 12px;
  margin-top: 16px;
  padding-top: 16px;
  border-top: 1px solid #e8e8e8;
}

.add-section-container, .question-select-container {
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.modal-footer {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-top: 16px;
}

.selection-info {
  font-weight: 600;
  color: #1890ff;
}

.question-filter {
  margin-bottom: 16px;
  padding-bottom: 16px;
  border-bottom: 1px solid #e8e8e8;
}

.available-question-list {
  max-height: 400px;
  overflow-y: auto;
}

.question-select-item {
  padding: 12px;
  border-bottom: 1px solid #f0f0f0;
}

.question-select-content {
  margin-left: 8px;
}

.question-select-title {
  font-weight: 500;
  margin-bottom: 8px;
}

.question-select-info {
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
}
</style> 