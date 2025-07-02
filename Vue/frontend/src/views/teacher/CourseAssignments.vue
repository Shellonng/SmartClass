<template>
  <div class="course-assignments">
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
            <span class="filter-label">类型：</span>
            <a-select 
              v-model:value="filters.type" 
              style="width: 120px" 
              placeholder="全部类型"
              allowClear
              @change="handleFilterChange"
            >
              <a-select-option value="QUIZ">测验</a-select-option>
              <a-select-option value="ESSAY">论文</a-select-option>
              <a-select-option value="PROJECT">项目</a-select-option>
              <a-select-option value="OTHER">其他</a-select-option>
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
          :rowKey="(record: any) => record.id"
          @change="handleTableChange"
        >
          <!-- 作业名称 -->
          <template #bodyCell="{ column, record }">
            <template v-if="column.dataIndex === 'title'">
              <div class="assignment-title">
                <span>{{ record.title }}</span>
              </div>
            </template>

            <!-- 作业状态 -->
            <template v-else-if="column.dataIndex === 'status'">
              <a-tag :color="getStatusColor(record.status)">{{ getStatusText(record.status) }}</a-tag>
            </template>

            <!-- 作业模式 -->
            <template v-else-if="column.dataIndex === 'mode'">
              <a-tag :color="getModeColor(record.mode)">{{ getModeText(record.mode) }}</a-tag>
            </template>

            <!-- 作业时间 -->
            <template v-else-if="column.dataIndex === 'assignmentTime'">
              <div>
                <div>开始：{{ formatDate(record.startTime) }}</div>
                <div>截止：{{ formatDate(record.endTime) }}</div>
              </div>
            </template>

            <!-- 提交情况 -->
            <template v-else-if="column.dataIndex === 'submissionRate'">
              <a-progress 
                :percent="record.submissionRate * 100" 
                :format="(percent: number) => `${Math.round(percent)}%`"
                :status="getSubmissionStatus(record.submissionRate)"
              />
              <div class="submission-count">{{ record.submittedCount }}/{{ record.totalCount }}</div>
            </template>

            <!-- 操作 -->
            <template v-else-if="column.dataIndex === 'action'">
              <div class="assignment-actions">
                <a-tooltip title="查看">
                  <a-button type="link" @click="viewAssignment(record)">
                    <EyeOutlined />
                  </a-button>
                </a-tooltip>
                <a-tooltip title="批改">
                  <a-button type="link" @click="reviewAssignment(record)">
                    <CheckOutlined />
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
        <a-form-item label="作业模式" required>
          <a-radio-group v-model:value="assignmentForm.mode">
            <a-radio :value="assignmentMode.FILE">上传文件型</a-radio>
            <a-radio :value="assignmentMode.QUESTION">答题型</a-radio>
          </a-radio-group>
        </a-form-item>
        <a-form-item label="作业时间" required>
          <a-range-picker 
            v-model:value="assignmentTimeRange" 
            :show-time="{ format: 'HH:mm' }" 
            format="YYYY-MM-DD HH:mm"
            @change="handleTimeRangeChange"
          />
        </a-form-item>
        <a-form-item label="作业说明">
          <a-textarea v-model:value="assignmentForm.description" placeholder="请输入作业说明" :rows="4" />
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
          <div class="assignment-detail-label">作业模式：</div>
          <div class="assignment-detail-value">{{ getModeText(currentAssignment.mode) }}</div>
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
        
        <div class="assignment-detail-item">
          <div class="assignment-detail-label">提交状态：</div>
          <div class="assignment-detail-value">
            <a-progress 
              :percent="currentAssignment.submissionRate * 100" 
              :format="(percent: number) => `${Math.round(percent)}% (${currentAssignment.submittedCount}/${currentAssignment.totalCount})`" 
              :status="getSubmissionStatus(currentAssignment.submissionRate)"
            />
          </div>
        </div>
        
        <div class="assignment-detail-item">
          <div class="assignment-detail-label">创建时间：</div>
          <div class="assignment-detail-value">{{ formatDate(currentAssignment.createTime) }}</div>
        </div>
        
        <div class="assignment-detail-actions">
          <a-button type="primary" @click="reviewAssignment(currentAssignment)">批改作业</a-button>
          <a-button type="default" @click="editAssignment(currentAssignment)">编辑作业</a-button>
          <a-button @click="viewModalVisible = false">关闭</a-button>
        </div>
      </div>
    </a-modal>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted, watch, defineProps, defineComponent } from 'vue'
import { message, Modal } from 'ant-design-vue'
import {
  PlusOutlined,
  EyeOutlined,
  EditOutlined,
  DeleteOutlined,
  CheckOutlined
} from '@ant-design/icons-vue'
import { formatDate } from '@/utils/date'
import axios from 'axios'
import type { Dayjs } from 'dayjs'
import { useRouter } from 'vue-router'

// 定义组件名称，便于调试
defineComponent({
  name: 'CourseAssignments'
})

const props = defineProps({
  courseId: {
    type: Number,
    required: true
  }
})

// 作业状态
const assignmentStatus = {
  NOT_STARTED: 0,  // 修改为整数类型，0 代表未开始
  IN_PROGRESS: 1,  // 修改为整数类型，1 代表进行中
  ENDED: 2         // 修改为整数类型，2 代表已结束
}

// 作业类型
const assignmentType = {
  QUIZ: 'QUIZ',
  ESSAY: 'ESSAY',
  PROJECT: 'PROJECT',
  OTHER: 'OTHER'
}

// 作业模式
const assignmentMode = {
  QUESTION: 'question', // 答题型
  FILE: 'file'          // 上传文件型
}

// 状态定义
const assignments = ref<any[]>([])
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
  type: undefined as string | undefined,
  keyword: ''
})

// 表格列定义
const columns = [
  {
    title: '作业名称',
    dataIndex: 'title',
    key: 'title',
    ellipsis: true,
    width: '20%'
  },
  {
    title: '状态',
    dataIndex: 'status',
    key: 'status',
    width: '10%'
  },
  {
    title: '模式',
    dataIndex: 'mode',
    key: 'mode',
    width: '10%'
  },
  {
    title: '作业时间',
    dataIndex: 'assignmentTime',
    key: 'assignmentTime',
    width: '20%'
  },
  {
    title: '提交情况',
    dataIndex: 'submissionRate',
    key: 'submissionRate',
    width: '15%'
  },
  {
    title: '操作',
    dataIndex: 'action',
    key: 'action',
    width: '15%'
  }
]

// 添加/编辑作业相关状态
const assignmentModalVisible = ref(false)
const isEditing = ref(false)
const saving = ref(false)
const assignmentTimeRange = ref<[Dayjs, Dayjs] | null>(null)
const assignmentForm = ref({
  id: undefined as number | undefined,
  title: '',
  courseId: props.courseId,
  startTime: '',
  endTime: '',
  description: '',
  status: assignmentStatus.NOT_STARTED,
  userId: undefined as number | undefined,
  mode: assignmentMode.FILE // 默认为上传文件型
})

// 查看作业相关状态
const viewModalVisible = ref(false)
const currentAssignment = ref<any | null>(null)

// 路由实例
const router = useRouter()

// 监听课程ID变化
watch(() => props.courseId, (newId) => {
  if (newId) {
    fetchAssignments()
  }
})

// 生命周期钩子
onMounted(() => {
  fetchAssignments()
})

// 获取作业列表
const fetchAssignments = async () => {
  loading.value = true
  console.log('开始获取课程作业列表，课程ID:', props.courseId)
  
  try {
    // 获取用户Token
    const token = localStorage.getItem('user-token') || localStorage.getItem('token')
    const userInfo = localStorage.getItem('user-info')
    let userId = ''
    
    if (userInfo) {
      try {
        const userObj = JSON.parse(userInfo)
        userId = userObj.id || ''
        console.log('当前用户ID:', userId)
      } catch (e) {
        console.error('解析用户信息失败:', e)
      }
    }
    
    // 构建认证头
    const authToken = userId ? `Bearer token-${userId}` : (token ? `Bearer ${token}` : '')
    console.log('使用认证Token:', authToken)
    
    // 构建API请求参数
    const params = {
      pageNum: pagination.value.current,
      pageSize: pagination.value.pageSize,
      courseId: props.courseId,
      ...filters.value
    }
    console.log('请求参数:', params)
    
    // 发送API请求
    console.log(`请求URL: /api/teacher/assignments，参数包括courseId=${props.courseId}`)
    const response = await axios.get('/api/teacher/assignments', {
      params,
      headers: {
        'Authorization': authToken
      }
    })
    
    console.log('API响应:', response.data)
    
    if (response.data && response.data.code === 200) {
      assignments.value = response.data.data.records || []
      pagination.value.total = response.data.data.total || 0
      console.log('成功获取作业列表:', assignments.value)
    } else {
      console.error('获取作业列表失败:', response.data)
      message.error(response.data?.message || '获取作业列表失败')
      // 使用备用方案，加载模拟数据
      loadMockData()
    }
  } catch (error: any) {
    console.error('获取作业列表异常:', error)
    if (error.response) {
      console.error('错误响应:', error.response.data)
      console.error('状态码:', error.response.status)
      console.error('响应头:', error.response.headers)
      message.error(`获取作业列表失败: ${error.response.status} - ${error.response.data?.message || '未知错误'}`)
    } else if (error.request) {
      console.error('请求未收到响应:', error.request)
      message.error('获取作业列表失败: 服务器未响应')
    } else {
      console.error('请求配置错误:', error.message)
      message.error(`获取作业列表失败: ${error.message}`)
    }
    
    // 发生错误时，使用备用方案，加载模拟数据
    loadMockData()
  } finally {
    loading.value = false
  }
}

// 加载模拟数据（备用方案）
const loadMockData = () => {
  console.log('使用模拟数据')
  assignments.value = [
    {
      id: 1,
      title: '第一次作业',
      courseId: props.courseId,
      startTime: '2025-07-01 09:00:00',
      endTime: '2025-07-07 23:59:59',
      totalScore: 100,
      type: assignmentType.QUIZ,
      mode: assignmentMode.QUESTION,
      description: '完成课本第一章习题',
      status: assignmentStatus.IN_PROGRESS,
      createTime: '2025-06-20 14:30:00',
      submissionRate: 0.75,
      submittedCount: 30,
      totalCount: 40
    },
    {
      id: 2,
      title: '第二次作业',
      courseId: props.courseId,
      startTime: '2025-07-08 09:00:00',
      endTime: '2025-07-14 23:59:59',
      totalScore: 100,
      type: assignmentType.ESSAY,
      mode: assignmentMode.FILE,
      description: '网络协议分析报告',
      status: assignmentStatus.NOT_STARTED,
      createTime: '2025-06-22 10:15:00',
      submissionRate: 0,
      submittedCount: 0,
      totalCount: 40
    }
  ]
  pagination.value.total = assignments.value.length
}

// 筛选变化处理
const handleFilterChange = () => {
  pagination.value.current = 1
  fetchAssignments()
}

// 搜索处理
const handleSearch = () => {
  pagination.value.current = 1
  fetchAssignments()
}

// 重置筛选条件
const resetFilters = () => {
  filters.value = {
    status: undefined,
    type: undefined,
    keyword: ''
  }
  pagination.value.current = 1
  fetchAssignments()
}

// 表格变化事件
const handleTableChange = (pag: any) => {
  pagination.value.current = pag.current;
  pagination.value.pageSize = pag.pageSize;
  fetchAssignments();
}

// 作业时间范围变化
const handleTimeRangeChange = (dates: [Dayjs, Dayjs] | null) => {
  if (dates) {
    assignmentForm.value.startTime = dates[0].format('YYYY-MM-DD HH:mm:ss')
    assignmentForm.value.endTime = dates[1].format('YYYY-MM-DD HH:mm:ss')
  } else {
    assignmentForm.value.startTime = ''
    assignmentForm.value.endTime = ''
  }
}

// 显示添加作业弹窗
const showAddAssignmentModal = () => {
  isEditing.value = false
  
  // 获取当前用户ID
  const userInfo = localStorage.getItem('user-info')
  let userId = undefined as number | undefined
  
  if (userInfo) {
    try {
      const userObj = JSON.parse(userInfo)
      userId = userObj.id
      console.log('当前用户ID:', userId)
    } catch (e) {
      console.error('解析用户信息失败:', e)
    }
  }
  
  // 如果无法获取用户ID，使用默认值
  if (!userId) {
    userId = 6 // 使用测试教师ID作为默认值
    console.log('使用默认用户ID:', userId)
  }
  
  assignmentForm.value = {
    id: undefined,
    title: '',
    courseId: props.courseId,
    startTime: '',
    endTime: '',
    description: '',
    status: assignmentStatus.NOT_STARTED,
    userId: userId,
    mode: assignmentMode.FILE // 默认为上传文件型
  }
  assignmentTimeRange.value = null
  assignmentModalVisible.value = true
}

// 查看作业
const viewAssignment = (assignment: any) => {
  // 跳转到作业详情页并显示提交记录选项卡
  router.push({
    path: `/teacher/assignments/${assignment.id}`,
    query: { tab: 'submissions' }
  })
}

// 编辑作业
const editAssignment = async (assignment: any) => {
  isEditing.value = true
  
  try {
    // 从API获取完整的作业详情
    const response = await axios.get(`/api/teacher/assignments/${assignment.id}`)
    
    if (response.data.code === 200) {
      const data = response.data.data
      
      // 如果是答题型作业，直接跳转到组卷设置页面
      if (data.mode === assignmentMode.QUESTION) {
        console.log('编辑答题型作业，跳转到组卷设置页面')
        router.push(`/teacher/assignments/${data.id}/edit`)
        return
      }
      
      // 如果是文件上传型作业，显示普通编辑弹窗
      // 只保留需要的字段
      assignmentForm.value = {
        id: data.id,
        title: data.title,
        courseId: data.courseId,
        startTime: data.startTime,
        endTime: data.endTime,
        description: data.description || '',
        status: data.status,
        userId: data.userId, // 保留userId字段
        mode: data.mode || assignmentMode.FILE // 保留mode字段
      }
      
      console.log('编辑作业表单:', assignmentForm.value)
      
      // 如果有开始时间和结束时间，设置日期选择器的值
      if (data.startTime && data.endTime) {
        const dayjs = (await import('dayjs')).default
        assignmentTimeRange.value = [
          dayjs(data.startTime),
          dayjs(data.endTime)
        ] as [Dayjs, Dayjs]
      } else {
        assignmentTimeRange.value = null
      }
      
      assignmentModalVisible.value = true
      
      // 如果是从详情弹窗点击的编辑按钮，关闭详情弹窗
      if (viewModalVisible.value) {
        viewModalVisible.value = false
      }
    } else {
      message.error(response.data?.message || '获取作业详情失败')
    }
  } catch (error: any) {
    console.error('获取作业详情失败:', error)
    message.error('获取作业详情失败，请稍后再试')
    
    // 获取当前用户ID
    const userInfo = localStorage.getItem('user-info')
    let userId = null
    
    if (userInfo) {
      try {
        const userObj = JSON.parse(userInfo)
        userId = userObj.id
      } catch (e) {
        console.error('解析用户信息失败:', e)
      }
    }
    
    // 如果无法获取用户ID，使用默认值
    if (!userId) {
      userId = 6 // 使用测试教师ID作为默认值
    }
    
    // 退回到使用传入的基本信息
    assignmentForm.value = {
      id: assignment.id,
      title: assignment.title,
      courseId: assignment.courseId || props.courseId,
      startTime: assignment.startTime || '',
      endTime: assignment.endTime || '',
      description: assignment.description || '',
      status: assignment.status || assignmentStatus.NOT_STARTED,
      userId: assignment.userId || userId, // 使用传入的userId或默认值
      mode: assignment.mode || assignmentMode.FILE // 使用传入的mode或默认值
    }
    assignmentModalVisible.value = true
  }
}

// 批改作业
const reviewAssignment = (assignment: any) => {
  message.info(`暂未实现批改功能：${assignment.title}`)
  // 这里应该跳转到批改页面，或者打开批改弹窗
}

// 删除作业
const handleDeleteAssignment = async (id: number) => {
  try {
    const response = await axios.delete(`/api/teacher/assignments/${id}`)
    
    if (response.data.code === 200) {
      message.success('作业删除成功')
      fetchAssignments()
    } else {
      message.error(response.data?.message || '删除作业失败')
      console.error('删除作业失败:', response.data)
    }
  } catch (error: any) {
    console.error('作业删除失败:', error)
    if (error.response) {
      console.error('错误响应:', error.response.data)
      console.error('状态码:', error.response.status)
      message.error(`删除作业失败: ${error.response.status} - ${error.response.data?.message || '未知错误'}`)
    } else if (error.request) {
      console.error('请求未收到响应:', error.request)
      message.error('删除作业失败: 服务器未响应')
    } else {
      message.error(`删除作业失败: ${error.message}`)
    }
  }
}

// 保存作业
const handleSaveAssignment = async () => {
  // 表单验证
  if (!assignmentForm.value.title) {
    message.error('请输入作业名称')
    return
  }
  if (!assignmentForm.value.startTime || !assignmentForm.value.endTime) {
    message.error('请选择作业时间')
    return
  }
  if (!assignmentForm.value.mode) {
    message.error('请选择作业模式')
    return
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
    
    // 如果无法获取用户ID，使用默认值
    if (!userId) {
      userId = 6 // 使用测试教师ID作为默认值
      console.log('使用默认用户ID:', userId)
    }
    
    // 构建作业数据，添加固定字段
    const assignmentData = {
      ...assignmentForm.value,
      type: 'homework', // 固定值，表示作业而非考试
      status: assignmentForm.value.status, // 确保状态值是整数
      userId: userId, // 添加用户ID字段
      mode: assignmentForm.value.mode // 添加mode字段
    }
    
    console.log('保存作业数据:', assignmentData)
    console.log('作业状态值类型:', typeof assignmentData.status)
    console.log('作业状态值:', assignmentData.status)
    console.log('作业模式:', assignmentData.mode)
    
    let response: any
    if (isEditing.value && assignmentData.id) {
      // 编辑现有作业
      response = await axios.put(`/api/teacher/assignments/${assignmentData.id}`, assignmentData)
      console.log('更新作业响应:', response.data)
      
      if (response.data.code === 200) {
        message.success('作业更新成功')
        
        // 如果是答题型作业，询问是否跳转到题目编辑页面
        if (assignmentData.mode === assignmentMode.QUESTION) {
          Modal.confirm({
            title: '是否编辑题目？',
            content: '作业更新成功，是否前往编辑题目？',
            okText: '是',
            cancelText: '否',
            onOk: () => {
              router.push(`/teacher/assignments/${assignmentData.id}/edit`)
            }
          })
        }
      } else {
        message.error(response.data?.message || '更新作业失败')
        console.error('更新作业失败:', response.data)
        return
      }
    } else {
      // 创建新作业
      response = await axios.post('/api/teacher/assignments', assignmentData)
      console.log('创建作业响应:', response.data)
      
      if (response.data.code === 200) {
        message.success('作业添加成功')
        
        // 如果是答题型作业，询问是否跳转到题目编辑页面
        if (assignmentData.mode === assignmentMode.QUESTION) {
          Modal.confirm({
            title: '是否编辑题目？',
            content: '作业添加成功，是否前往编辑题目？',
            okText: '是',
            cancelText: '否',
            onOk: () => {
              const assignmentId = response.data.data
              router.push(`/teacher/assignments/${assignmentId}/edit`)
            }
          })
        }
      } else {
        message.error(response.data?.message || '添加作业失败')
        console.error('添加作业失败:', response.data)
        return
      }
    }
    
    // 成功后关闭弹窗并刷新列表
    assignmentModalVisible.value = false
    fetchAssignments()
  } catch (error: any) {
    console.error('保存作业失败:', error)
    if (error.response) {
      console.error('错误响应:', error.response.data)
      console.error('状态码:', error.response.status)
      message.error(`保存作业失败: ${error.response.status} - ${error.response.data?.message || '未知错误'}`)
    } else if (error.request) {
      console.error('请求未收到响应:', error.request)
      message.error('保存作业失败: 服务器未响应')
    } else {
      message.error(`保存作业失败: ${error.message}`)
    }
  } finally {
    saving.value = false
  }
}

// 获取提交状态
const getSubmissionStatus = (rate: number): string => {
  if (rate === 0) return 'exception';
  if (rate < 0.5) return 'active';
  if (rate < 1) return 'normal';
  return 'success';
}

// 获取状态显示文本
const getStatusText = (status: number): string => {
  const statusMap: Record<number, string> = {
    [assignmentStatus.NOT_STARTED]: '未开始',
    [assignmentStatus.IN_PROGRESS]: '进行中',
    [assignmentStatus.ENDED]: '已结束'
  }
  return statusMap[status] || '未知状态'
}

// 获取状态标签颜色
const getStatusColor = (status: number): string => {
  const colorMap: Record<number, string> = {
    [assignmentStatus.NOT_STARTED]: 'blue',
    [assignmentStatus.IN_PROGRESS]: 'green',
    [assignmentStatus.ENDED]: 'gray'
  }
  return colorMap[status] || 'default'
}

// 获取作业类型文本
const getTypeText = (type: string): string => {
  const typeMap: Record<string, string> = {
    [assignmentType.QUIZ]: '测验',
    [assignmentType.ESSAY]: '论文',
    [assignmentType.PROJECT]: '项目',
    [assignmentType.OTHER]: '其他'
  }
  return typeMap[type] || ''
}

// 获取作业模式文本
const getModeText = (mode: string): string => {
  const modeMap: Record<string, string> = {
    [assignmentMode.QUESTION]: '答题型',
    [assignmentMode.FILE]: '上传文件型'
  }
  return modeMap[mode] || '未知模式'
}

// 获取作业模式颜色
const getModeColor = (mode: string): string => {
  const colorMap: Record<string, string> = {
    [assignmentMode.QUESTION]: 'blue',
    [assignmentMode.FILE]: 'green'
  }
  return colorMap[mode] || 'default'
}
</script>

<style scoped>
.course-assignments {
  padding: 24px;
  background-color: #fff;
}

.page-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 24px;
}

.page-header h2 {
  margin: 0;
  font-size: 20px;
  font-weight: 600;
}

.filter-section {
  background-color: #f5f7fa;
  padding: 16px;
  border-radius: 8px;
  margin-bottom: 24px;
}

.filter-row {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  justify-content: space-between;
}

.filter-left {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
}

.filter-right {
  display: flex;
  align-items: center;
  gap: 16px;
}

.filter-item {
  display: flex;
  align-items: center;
  margin-right: 24px;
  margin-bottom: 8px;
}

.filter-label {
  margin-right: 8px;
  white-space: nowrap;
}

.search-box {
  flex-grow: 1;
}

.assignment-content {
  background-color: #fff;
}

.assignment-title {
  display: flex;
  align-items: center;
  gap: 8px;
}

.assignment-type-tag {
  font-size: 12px;
  line-height: 1;
  padding: 2px 6px;
  border-radius: 4px;
  background-color: #f5f5f5;
  color: #666;
}

.assignment-actions {
  display: flex;
  gap: 8px;
}

.submission-count {
  font-size: 12px;
  color: #999;
  margin-top: 4px;
  text-align: center;
}

.assignment-detail {
  padding: 16px;
}

.assignment-detail-header {
  display: flex;
  align-items: center;
  gap: 16px;
  margin-bottom: 20px;
  padding-bottom: 12px;
  border-bottom: 1px solid #f0f0f0;
}

.status-tag, .type-tag {
  font-size: 14px;
  padding: 2px 12px;
}

.assignment-detail-item {
  margin-bottom: 16px;
}

.assignment-detail-label {
  font-weight: 600;
  margin-bottom: 8px;
}

.assignment-detail-value {
  white-space: pre-line;
}

.assignment-detail-actions {
  display: flex;
  justify-content: flex-end;
  gap: 12px;
  margin-top: 24px;
  padding-top: 16px;
  border-top: 1px solid #f0f0f0;
}
</style> 