<template>
  <div class="assignment-management">
    <div class="page-header">
      <h2>作业管理</h2>
      <a-button type="primary" @click="showAddAssignmentModal">
        <PlusOutlined />
        添加作业
      </a-button>
    </div>

    <div class="filter-section">
      <div class="filter-row">
        <div class="filter-left">
          <div class="filter-item">
            <span class="filter-label">状态：</span>
            <a-select 
              v-model:value="filters.status" 
              style="width: 120px" 
              placeholder="全部状态"
              allowClear
              @change="handleFilterChange"
            >
              <a-select-option value="not_started">未开始</a-select-option>
              <a-select-option value="in_progress">进行中</a-select-option>
              <a-select-option value="ended">已结束</a-select-option>
            </a-select>
          </div>
          <div class="filter-item">
            <span class="filter-label">课程：</span>
            <a-select 
              v-model:value="filters.courseId" 
              style="width: 180px" 
              placeholder="全部课程"
              allowClear
              @change="handleFilterChange"
            >
              <a-select-option v-for="course in courses" :key="course.id" :value="course.id">{{ course.name }}</a-select-option>
            </a-select>
          </div>
        </div>
        <div class="filter-right">
          <div class="filter-item search-box">
            <a-input-search
              v-model:value="filters.keyword"
              placeholder="搜索作业名称"
              style="width: 250px"
              @search="handleSearch"
              enter-button
            />
          </div>
          <a-button @click="resetFilters" size="middle">
            重置筛选
          </a-button>
        </div>
      </div>
    </div>

    <div class="assignment-content">
      <a-spin :spinning="loading">
        <a-empty v-if="assignments.length === 0" description="暂无作业" />
        
        <a-table
          v-else
          :dataSource="assignments"
          :columns="columns"
          :pagination="pagination"
          :rowKey="(record) => record.id"
          @change="handleTableChange"
        >
          <!-- 作业名称 -->
          <template #bodyCell="{ column, record }">
            <template v-if="column.dataIndex === 'title'">
              <div class="assignment-title">
                <span>{{ record.title }}</span>
                <a-tag v-if="record.type" class="assignment-type-tag">{{ getTypeText(record.type) }}</a-tag>
              </div>
            </template>

            <!-- 作业状态 -->
            <template v-else-if="column.dataIndex === 'status'">
              <a-tag :color="getStatusColor(record.status)">{{ getStatusText(record.status) }}</a-tag>
            </template>

            <!-- 作业时间 -->
            <template v-else-if="column.dataIndex === 'assignmentTime'">
              <div>
                <div>开始：{{ formatDate(record.startTime) }}</div>
                <div>截止：{{ formatDate(record.endTime) }}</div>
              </div>
            </template>

            <!-- 操作 -->
            <template v-else-if="column.dataIndex === 'action'">
              <div class="assignment-actions">
                <a-tooltip title="查看">
                  <a-button type="link" @click="viewAssignment(record)">
                    <EyeOutlined />
                  </a-button>
                </a-tooltip>
                <a-tooltip title="编辑">
                  <a-button type="link" @click="editAssignment(record)">
                    <EditOutlined />
                  </a-button>
                </a-tooltip>
                <a-tooltip title="删除">
                  <a-popconfirm
                    title="确定要删除这个作业吗？"
                    @confirm="handleDeleteAssignment(record.id)"
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

    <!-- 添加/编辑作业弹窗 -->
    <a-modal
      v-model:open="assignmentModalVisible"
      :title="isEditing ? '编辑作业' : '添加作业'"
      :maskClosable="false"
      @ok="handleSaveAssignment"
      :okButtonProps="{ loading: saving }"
      :okText="saving ? '保存中...' : '保存'"
      width="700px"
    >
      <a-form :model="assignmentForm" layout="vertical">
        <a-form-item label="作业名称" required>
          <a-input v-model:value="assignmentForm.title" placeholder="请输入作业名称" />
        </a-form-item>
        <a-form-item label="所属课程" required>
          <a-select v-model:value="assignmentForm.courseId" placeholder="请选择课程">
            <a-select-option v-for="course in courses" :key="course.id" :value="course.id">{{ course.name }}</a-select-option>
          </a-select>
        </a-form-item>
        <a-form-item label="作业时间" required>
          <a-range-picker 
            v-model:value="assignmentTimeRange" 
            :show-time="{ format: 'HH:mm' }" 
            format="YYYY-MM-DD HH:mm"
            @change="handleTimeRangeChange"
          />
        </a-form-item>
        <a-form-item label="作业类型" required>
          <a-select v-model:value="assignmentForm.type" placeholder="请选择作业类型">
            <a-select-option value="QUIZ">测验</a-select-option>
            <a-select-option value="ESSAY">论文</a-select-option>
            <a-select-option value="PROJECT">项目</a-select-option>
            <a-select-option value="OTHER">其他</a-select-option>
          </a-select>
        </a-form-item>
        <a-form-item label="总分值" required>
          <a-input-number v-model:value="assignmentForm.totalScore" :min="1" :max="100" />
        </a-form-item>
        <a-form-item label="作业说明">
          <a-textarea v-model:value="assignmentForm.description" placeholder="请输入作业说明" :rows="4" />
        </a-form-item>
        <a-form-item label="题目添加">
          <a-button type="primary" @click="showQuestionSelectionModal">
            <PlusOutlined />
            添加题目
          </a-button>
          <div v-if="assignmentForm.questions && assignmentForm.questions.length > 0" class="question-list">
            <div v-for="(question, index) in assignmentForm.questions" :key="index" class="question-item">
              <div class="question-info">
                <div class="question-type">【{{ getQuestionTypeText(question.questionType) }}】</div>
                <div class="question-title">{{ question.title }}</div>
              </div>
              <div class="question-actions">
                <a-input-number
                  v-model:value="question.score"
                  :min="1"
                  :max="100"
                  placeholder="分值"
                  style="width: 80px"
                />
                <a-button type="text" danger @click="removeQuestion(index)">
                  <DeleteOutlined />
                </a-button>
              </div>
            </div>
          </div>
        </a-form-item>
      </a-form>
    </a-modal>

    <!-- 查看作业详情弹窗 -->
    <a-modal
      v-model:open="viewModalVisible"
      title="作业详情"
      :footer="null"
      width="700px"
    >
      <div v-if="currentAssignment" class="assignment-detail">
        <div class="assignment-detail-header">
          <a-tag :color="getStatusColor(currentAssignment.status)" class="status-tag">
            {{ getStatusText(currentAssignment.status) }}
          </a-tag>
          <a-tag class="type-tag">{{ getTypeText(currentAssignment.type) }}</a-tag>
        </div>
        
        <div class="assignment-detail-item">
          <div class="assignment-detail-label">作业名称：</div>
          <div class="assignment-detail-value">{{ currentAssignment.title }}</div>
        </div>
        
        <div class="assignment-detail-item">
          <div class="assignment-detail-label">所属课程：</div>
          <div class="assignment-detail-value">{{ currentAssignment.courseName }}</div>
        </div>
        
        <div class="assignment-detail-item">
          <div class="assignment-detail-label">作业时间：</div>
          <div class="assignment-detail-value">
            <div>开始时间：{{ formatDate(currentAssignment.startTime) }}</div>
            <div>截止时间：{{ formatDate(currentAssignment.endTime) }}</div>
          </div>
        </div>
        
        <div class="assignment-detail-item">
          <div class="assignment-detail-label">总分值：</div>
          <div class="assignment-detail-value">{{ currentAssignment.totalScore }} 分</div>
        </div>
        
        <div class="assignment-detail-item" v-if="currentAssignment.description">
          <div class="assignment-detail-label">作业说明：</div>
          <div class="assignment-detail-value">{{ currentAssignment.description }}</div>
        </div>
        
        <div class="assignment-detail-actions">
          <a-button type="primary" @click="editAssignment(currentAssignment)">编辑作业</a-button>
          <a-button @click="viewModalVisible = false">关闭</a-button>
        </div>
      </div>
    </a-modal>

    <!-- 题目选择弹窗 -->
    <a-modal
      v-model:open="questionSelectionVisible"
      title="选择题目"
      width="800px"
      @ok="handleConfirmQuestions"
      :okButtonProps="{ loading: loadingQuestions }"
    >
      <div class="question-filter">
        <a-form layout="inline">
          <a-form-item label="题目类型">
            <a-select v-model:value="questionFilters.type" style="width: 120px" allowClear>
              <a-select-option value="single">单选题</a-select-option>
              <a-select-option value="multiple">多选题</a-select-option>
              <a-select-option value="true_false">判断题</a-select-option>
              <a-select-option value="blank">填空题</a-select-option>
              <a-select-option value="short">简答题</a-select-option>
              <a-select-option value="code">编程题</a-select-option>
            </a-select>
          </a-form-item>
          <a-form-item label="难度">
            <a-select v-model:value="questionFilters.difficulty" style="width: 120px" allowClear>
              <a-select-option value="1">1星</a-select-option>
              <a-select-option value="2">2星</a-select-option>
              <a-select-option value="3">3星</a-select-option>
              <a-select-option value="4">4星</a-select-option>
              <a-select-option value="5">5星</a-select-option>
            </a-select>
          </a-form-item>
          <a-form-item>
            <a-button type="primary" @click="searchQuestions">搜索</a-button>
            <a-button style="margin-left: 8px" @click="resetQuestionFilters">重置</a-button>
          </a-form-item>
        </a-form>
      </div>
      
      <a-table
        :dataSource="availableQuestions"
        :columns="questionColumns"
        :rowKey="(record) => record.id"
        :rowSelection="{ 
          selectedRowKeys: selectedQuestionIds, 
          onChange: onSelectChange 
        }"
        :loading="loadingQuestions"
        size="small"
      >
        <template #bodyCell="{ column, record }">
          <template v-if="column.dataIndex === 'questionType'">
            {{ getQuestionTypeText(record.questionType) }}
          </template>
          <template v-else-if="column.dataIndex === 'difficulty'">
            <a-rate :value="record.difficulty" disabled />
          </template>
        </template>
      </a-table>
    </a-modal>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted, computed } from 'vue'
import { message } from 'ant-design-vue'
import { 
  PlusOutlined, 
  EyeOutlined, 
  EditOutlined, 
  DeleteOutlined,
  CheckOutlined
} from '@ant-design/icons-vue'
import { formatDate } from '@/utils/date'
import dayjs from 'dayjs'
import type { Dayjs } from 'dayjs'
import axios from 'axios'

// 作业状态
const assignmentStatus = {
  NOT_STARTED: 'not_started',
  IN_PROGRESS: 'in_progress',
  ENDED: 'ended'
}

// 作业类型
const assignmentType = {
  QUIZ: 'QUIZ',
  ESSAY: 'ESSAY',
  PROJECT: 'PROJECT',
  OTHER: 'OTHER'
}

// 状态定义
const assignments = ref<any[]>([])
const courses = ref<any[]>([])
const loading = ref(false)
const pagination = ref({
  current: 1,
  pageSize: 10,
  total: 0,
  showSizeChanger: true,
  showTotal: (total: number) => `共 ${total} 条`
})

// 筛选条件
const filters = ref({
  status: undefined as string | undefined,
  courseId: undefined as number | undefined,
  keyword: ''
})

// 表格列定义
const columns = [
  {
    title: '作业名称',
    dataIndex: 'title',
    key: 'title',
    ellipsis: true
  },
  {
    title: '课程',
    dataIndex: 'courseName',
    key: 'courseName',
    width: 180
  },
  {
    title: '状态',
    dataIndex: 'status',
    key: 'status',
    width: 100
  },
  {
    title: '作业时间',
    dataIndex: 'assignmentTime',
    key: 'assignmentTime',
    width: 200
  },
  {
    title: '操作',
    dataIndex: 'action',
    key: 'action',
    width: 150,
    fixed: 'right'
  }
]

// 表单和编辑状态
const assignmentModalVisible = ref(false)
const isEditing = ref(false)
const saving = ref(false)
const assignmentForm = ref<any>({
  id: undefined,
  title: '',
  courseId: undefined,
  startTime: '',
  endTime: '',
  totalScore: 100,
  type: 'QUIZ',
  description: '',
  status: 0,
  questions: []
})

// 时间范围选择器的值
const assignmentTimeRange = ref<[Dayjs, Dayjs] | null>(null)

// 查看作业相关状态
const viewModalVisible = ref(false)
const currentAssignment = ref<any | null>(null)

// 题目选择相关
const questionSelectionVisible = ref(false)
const availableQuestions = ref<any[]>([])
const selectedQuestionIds = ref<number[]>([])
const loadingQuestions = ref(false)
const questionFilters = ref({
  type: undefined as string | undefined,
  difficulty: undefined as string | undefined,
  keyword: ''
})

// 题目表格列定义
const questionColumns = [
  {
    title: '题目',
    dataIndex: 'title',
    key: 'title',
    ellipsis: true
  },
  {
    title: '题型',
    dataIndex: 'questionType',
    key: 'questionType',
    width: 100
  },
  {
    title: '难度',
    dataIndex: 'difficulty',
    key: 'difficulty',
    width: 120
  },
  {
    title: '知识点',
    dataIndex: 'knowledgePoint',
    key: 'knowledgePoint',
    width: 150,
    ellipsis: true
  }
]

// 初始化
onMounted(() => {
  // 加载课程列表
  fetchCourses()
  
  // 加载作业列表
  fetchAssignments()
})

// 加载课程列表
const fetchCourses = async () => {
  try {
    // 模拟数据，实际项目中替换为API调用
    courses.value = [
      { id: 1, name: 'Java编程基础' },
      { id: 2, name: '数据结构与算法' },
      { id: 3, name: '软件工程导论' },
      { id: 4, name: '操作系统原理' }
    ]
    
    // 实际项目中的API调用示例：
    // const res = await axios.get('/api/teacher/courses')
    // if (res.data.code === 200) {
    //   courses.value = res.data.data
    // }
  } catch (error) {
    console.error('加载课程列表失败:', error)
    message.error('加载课程列表失败')
  }
}

// 加载作业列表
const fetchAssignments = async () => {
  loading.value = true
  try {
    // 模拟数据，实际项目中替换为API调用
    setTimeout(() => {
      assignments.value = [
        {
          id: 1,
          title: 'Java基础编程作业1',
          courseId: 1,
          courseName: 'Java编程基础',
          type: 'QUIZ',
          status: 'in_progress',
          startTime: '2025-06-20 10:00:00',
          endTime: '2025-06-30 23:59:59',
          totalScore: 100,
          description: '完成Java面向对象编程相关题目',
          submissionRate: 0.7,
          submittedCount: 35,
          totalCount: 50
        },
        {
          id: 2,
          title: '算法设计与分析作业',
          courseId: 2,
          courseName: '数据结构与算法',
          type: 'PROJECT',
          status: 'not_started',
          startTime: '2025-07-01 00:00:00',
          endTime: '2025-07-10 23:59:59',
          totalScore: 100,
          description: '完成一个图算法的实现与分析',
          submissionRate: 0,
          submittedCount: 0,
          totalCount: 45
        },
        {
          id: 3,
          title: '软件需求分析报告',
          courseId: 3,
          courseName: '软件工程导论',
          type: 'ESSAY',
          status: 'ended',
          startTime: '2025-05-15 00:00:00',
          endTime: '2025-06-15 23:59:59',
          totalScore: 100,
          description: '编写一份完整的软件需求分析报告',
          submissionRate: 0.95,
          submittedCount: 38,
          totalCount: 40
        }
      ]
      
      pagination.value.total = assignments.value.length
      loading.value = false
    }, 800)
    
    // 实际项目中的API调用示例：
    // const params = {
    //   page: pagination.value.current,
    //   size: pagination.value.pageSize,
    //   ...filters.value
    // }
    // const res = await axios.get('/api/teacher/assignments', { params })
    // if (res.data.code === 200) {
    //   assignments.value = res.data.data.records
    //   pagination.value.total = res.data.data.total
    // }
  } catch (error) {
    console.error('加载作业列表失败:', error)
    message.error('加载作业列表失败')
  } finally {
    loading.value = false
  }
}

// 筛选变化事件
const handleFilterChange = () => {
  pagination.value.current = 1
  fetchAssignments()
}

// 搜索事件
const handleSearch = () => {
  pagination.value.current = 1
  fetchAssignments()
}

// 重置筛选条件
const resetFilters = () => {
  filters.value = {
    status: undefined,
    courseId: undefined,
    keyword: ''
  }
  pagination.value.current = 1
  fetchAssignments()
}

// 表格变化事件
const handleTableChange = (pag: any) => {
  pagination.value = {
    ...pagination.value,
    current: pag.current,
    pageSize: pag.pageSize
  }
  fetchAssignments()
}

// 查看作业详情
const viewAssignment = (record: any) => {
  currentAssignment.value = record
  viewModalVisible.value = true
}

// 编辑作业
const editAssignment = (record: any) => {
  isEditing.value = true
  assignmentForm.value = { ...record }
  
  if (record.startTime && record.endTime) {
    assignmentTimeRange.value = [
      dayjs(record.startTime),
      dayjs(record.endTime)
    ]
  }
  
  assignmentModalVisible.value = true
}

// 显示添加作业弹窗
const showAddAssignmentModal = () => {
  isEditing.value = false
  assignmentForm.value = {
    id: undefined,
    title: '',
    courseId: undefined,
    startTime: '',
    endTime: '',
    totalScore: 100,
    type: 'QUIZ',
    description: '',
    status: 0,
    questions: []
  }
  assignmentTimeRange.value = null
  assignmentModalVisible.value = true
}

// 处理时间范围变化
const handleTimeRangeChange = (dates: any) => {
  if (dates && dates.length === 2) {
    assignmentForm.value.startTime = dates[0].format('YYYY-MM-DD HH:mm:ss')
    assignmentForm.value.endTime = dates[1].format('YYYY-MM-DD HH:mm:ss')
  } else {
    assignmentForm.value.startTime = ''
    assignmentForm.value.endTime = ''
  }
}

// 保存作业
const handleSaveAssignment = async () => {
  // 表单验证
  if (!assignmentForm.value.title) {
    message.error('请输入作业名称')
    return
  }
  if (!assignmentForm.value.courseId) {
    message.error('请选择所属课程')
    return
  }
  if (!assignmentForm.value.startTime || !assignmentForm.value.endTime) {
    message.error('请选择作业时间')
    return
  }
  if (!assignmentForm.value.type) {
    message.error('请选择作业类型')
    return
  }

  saving.value = true
  try {
    // 模拟API请求
    setTimeout(() => {
      if (isEditing.value) {
        message.success('作业更新成功')
      } else {
        message.success('作业添加成功')
      }
      assignmentModalVisible.value = false
      fetchAssignments()
      saving.value = false
    }, 1000)
    
    // 实际项目中的API调用示例
    // if (isEditing.value) {
    //   await axios.put(`/api/teacher/assignments/${assignmentForm.value.id}`, assignmentForm.value)
    //   message.success('作业更新成功')
    // } else {
    //   await axios.post('/api/teacher/assignments', assignmentForm.value)
    //   message.success('作业添加成功')
    // }
    // assignmentModalVisible.value = false
    // fetchAssignments()
  } catch (error) {
    console.error('保存作业失败:', error)
    message.error('保存作业失败，请稍后再试')
  } finally {
    saving.value = false
  }
}

// 删除作业
const handleDeleteAssignment = async (id: number) => {
  try {
    // 模拟API请求
    setTimeout(() => {
      message.success('作业已删除')
      fetchAssignments()
    }, 1000)
    
    // 实际项目中的API调用示例
    // await axios.delete(`/api/teacher/assignments/${id}`)
    // message.success('作业已删除')
    // fetchAssignments()
  } catch (error) {
    console.error('删除作业失败:', error)
    message.error('删除作业失败，请稍后再试')
  }
}

// 获取状态显示文本
const getStatusText = (status: string) => {
  switch (status) {
    case assignmentStatus.NOT_STARTED:
      return '未开始'
    case assignmentStatus.IN_PROGRESS:
      return '进行中'
    case assignmentStatus.ENDED:
      return '已结束'
    default:
      return '未知状态'
  }
}

// 获取状态对应的颜色
const getStatusColor = (status: string) => {
  switch (status) {
    case assignmentStatus.NOT_STARTED:
      return 'blue'
    case assignmentStatus.IN_PROGRESS:
      return 'green'
    case assignmentStatus.ENDED:
      return 'gray'
    default:
      return 'default'
  }
}

// 获取作业类型显示文本
const getTypeText = (type: string) => {
  switch (type) {
    case 'QUIZ':
      return '测验'
    case 'ESSAY':
      return '论文'
    case 'PROJECT':
      return '项目'
    case 'OTHER':
      return '其他'
    default:
      return '未知类型'
  }
}

// 获取题目类型显示文本
const getQuestionTypeText = (type: string) => {
  switch (type) {
    case 'single':
      return '单选题'
    case 'multiple':
      return '多选题'
    case 'true_false':
      return '判断题'
    case 'blank':
      return '填空题'
    case 'short':
      return '简答题'
    case 'code':
      return '编程题'
    default:
      return '未知类型'
  }
}

// 提交状态颜色
const getSubmissionStatus = (rate: number) => {
  if (rate >= 0.8) return 'success'
  if (rate >= 0.5) return 'normal'
  return 'exception'
}

// 显示题目选择弹窗
const showQuestionSelectionModal = () => {
  if (!assignmentForm.value.courseId) {
    message.error('请先选择课程')
    return
  }
  
  questionSelectionVisible.value = true
  loadQuestions()
  
  // 初始化已选题目
  selectedQuestionIds.value = assignmentForm.value.questions.map((q: any) => q.id)
}

// 加载题目
const loadQuestions = async () => {
  loadingQuestions.value = true
  try {
    // 模拟数据，实际项目中替换为API调用
    setTimeout(() => {
      availableQuestions.value = [
        {
          id: 1,
          title: '下列关于Java中基本数据类型的说法，正确的是？',
          questionType: 'single',
          difficulty: 2,
          knowledgePoint: 'Java基础'
        },
        {
          id: 2,
          title: 'Java中以下哪些是集合框架的接口？（多选）',
          questionType: 'multiple',
          difficulty: 3,
          knowledgePoint: 'Java集合'
        },
        {
          id: 3,
          title: 'Java中，String类是不可变的。',
          questionType: 'true_false',
          difficulty: 1,
          knowledgePoint: 'Java字符串'
        },
        {
          id: 4,
          title: '请简述Java垃圾回收机制的原理。',
          questionType: 'short',
          difficulty: 4,
          knowledgePoint: 'Java内存管理'
        }
      ]
      loadingQuestions.value = false
    }, 800)
    
    // 实际项目中的API调用示例：
    // const params = {
    //   courseId: assignmentForm.value.courseId,
    //   ...questionFilters.value
    // }
    // const res = await axios.get('/api/teacher/questions', { params })
    // if (res.data.code === 200) {
    //   availableQuestions.value = res.data.data
    // }
  } catch (error) {
    console.error('加载题目失败:', error)
    message.error('加载题目失败')
  } finally {
    loadingQuestions.value = false
  }
}

// 搜索题目
const searchQuestions = () => {
  loadQuestions()
}

// 重置题目筛选
const resetQuestionFilters = () => {
  questionFilters.value = {
    type: undefined,
    difficulty: undefined,
    keyword: ''
  }
  loadQuestions()
}

// 选择题目变化
const onSelectChange = (selectedRowKeys: any[]) => {
  selectedQuestionIds.value = selectedRowKeys
}

// 确认选择题目
const handleConfirmQuestions = () => {
  // 获取所有选中的题目详情
  const selectedQuestions = availableQuestions.value.filter(q => selectedQuestionIds.value.includes(q.id))
  
  // 将新选择的题目添加到作业表单中
  assignmentForm.value.questions = selectedQuestions.map(q => ({
    ...q,
    score: 10 // 设置默认分值
  }))
  
  questionSelectionVisible.value = false
  message.success(`已选择${selectedQuestions.length}道题目`)
}

// 从作业中移除题目
const removeQuestion = (index: number) => {
  assignmentForm.value.questions.splice(index, 1)
}
</script>

<style scoped>
.assignment-management {
  padding: 10px;
}

.page-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 20px;
}

.filter-section {
  margin-bottom: 20px;
  padding: 16px;
  background: #f5f5f5;
  border-radius: 4px;
}

.filter-row {
  display: flex;
  justify-content: space-between;
}

.filter-left,
.filter-right {
  display: flex;
  gap: 16px;
}

.filter-label {
  margin-right: 8px;
}

.assignment-content {
  background: #fff;
  padding: 20px;
  border-radius: 4px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
}

.assignment-title {
  display: flex;
  align-items: center;
  gap: 8px;
}

.assignment-type-tag {
  margin-left: 8px;
}

.assignment-actions {
  display: flex;
  justify-content: center;
}

.assignment-detail {
  padding: 10px;
}

.assignment-detail-header {
  display: flex;
  gap: 10px;
  margin-bottom: 20px;
}

.assignment-detail-item {
  margin-bottom: 15px;
  display: flex;
}

.assignment-detail-label {
  font-weight: bold;
  width: 100px;
  flex-shrink: 0;
}

.assignment-detail-value {
  flex: 1;
}

.assignment-detail-actions {
  margin-top: 30px;
  display: flex;
  gap: 10px;
  justify-content: flex-end;
  border-top: 1px solid #f0f0f0;
  padding-top: 15px;
}

.submission-count {
  font-size: 12px;
  color: #999;
  text-align: center;
  margin-top: 5px;
}

.question-list {
  margin-top: 15px;
  max-height: 300px;
  overflow-y: auto;
  border: 1px solid #eee;
  padding: 10px;
  border-radius: 4px;
}

.question-item {
  display: flex;
  justify-content: space-between;
  padding: 10px;
  border-bottom: 1px solid #f0f0f0;
}

.question-item:last-child {
  border-bottom: none;
}

.question-info {
  flex: 1;
  display: flex;
  align-items: center;
}

.question-type {
  margin-right: 10px;
  color: #1890ff;
  white-space: nowrap;
}

.question-title {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.question-actions {
  display: flex;
  align-items: center;
  gap: 10px;
}

.question-filter {
  margin-bottom: 20px;
  padding-bottom: 15px;
  border-bottom: 1px solid #f0f0f0;
}
</style> 