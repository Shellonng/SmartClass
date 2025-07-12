<template>
  <div class="assignment-list-page">
    <div class="page-header">
      <div class="header-left">
        <h1>作业管理</h1>
        <p class="description">管理课程作业、批改学生提交</p>
      </div>
      <div class="header-right">
        <a-button type="primary" @click="createNewAssignment">
          <template #icon><PlusOutlined /></template>
          创建作业
        </a-button>
      </div>
    </div>

    <div class="content-container">
      <!-- 筛选区域 -->
      <a-card class="filter-card">
        <a-form layout="inline">
          <a-form-item label="课程">
            <a-select
              v-model:value="filters.courseId"
              placeholder="选择课程"
              style="width: 200px"
              :options="courseOptions"
              :loading="coursesLoading"
              allowClear
              @change="handleFilterChange"
            />
          </a-form-item>
          
          <a-form-item label="状态">
            <a-select
              v-model:value="filters.status"
              placeholder="选择状态"
              style="width: 150px"
              :options="statusOptions"
              allowClear
              @change="handleFilterChange"
            />
          </a-form-item>
          
          <a-form-item label="搜索">
            <a-input-search
              v-model:value="filters.keyword"
              placeholder="搜索作业标题"
              style="width: 250px"
              @search="handleSearch"
            />
          </a-form-item>
        </a-form>
      </a-card>

      <!-- 作业列表 -->
      <a-card class="list-card">
        <a-table
          :columns="columns"
          :data-source="assignments"
          :loading="loading"
          :pagination="pagination"
          @change="handleTableChange"
          row-key="id"
        >
          <!-- 标题列 -->
          <template #bodyCell="{ column, record }">
            <template v-if="column.key === 'title'">
              <a @click="viewAssignmentDetail(record.id)">{{ record.title }}</a>
            </template>
            
            <!-- 状态列 -->
            <template v-if="column.key === 'status'">
              <a-tag :color="getStatusColor(record.status)">
                {{ getStatusText(record.status) }}
              </a-tag>
            </template>
            
            <!-- 提交率列 -->
            <template v-if="column.key === 'submissionRate'">
              <a-progress
                :percent="record.submissionRate || 0"
                size="small"
                :status="getSubmissionRateStatus(record.submissionRate)"
              />
            </template>
            
            <!-- 操作列 -->
            <template v-if="column.key === 'action'">
              <a-space>
                <a-button type="link" size="small" @click="viewAssignmentDetail(record.id)">
                  查看
                </a-button>
                <a-button type="link" size="small" @click="editAssignment(record.id)">
                  编辑
                </a-button>
                <a-button
                  v-if="record.status === 0"
                  type="link"
                  size="small"
                  @click="publishAssignment(record.id)"
                >
                  发布
                </a-button>
                <a-popconfirm
                  title="确定删除此作业吗？"
                  @confirm="deleteAssignment(record.id)"
                  okText="确定"
                  cancelText="取消"
                >
                  <a-button type="link" size="small" danger>
                    删除
                  </a-button>
                </a-popconfirm>
              </a-space>
            </template>
          </template>
        </a-table>
      </a-card>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, reactive, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { message } from 'ant-design-vue'
import { PlusOutlined } from '@ant-design/icons-vue'
import axios from 'axios'
import { useAuthStore } from '@/stores/auth'

// 定义Assignment接口
interface Assignment {
  id: number;
  title: string;
  courseName?: string;
  courseId?: number;
  startTime?: string;
  endTime?: string;
  status: number;
  submissionRate?: number;
  totalStudents?: number;
  submittedCount?: number;
  gradedCount?: number;
  [key: string]: any;
}

const router = useRouter()

// 状态变量
const assignments = ref<Assignment[]>([])
const courseOptions = ref<{ value: number; label: string }[]>([])
const loading = ref(false)
const coursesLoading = ref(false)

// 筛选条件
const filters = reactive({
  courseId: undefined as number | undefined,
  status: undefined as number | undefined,
  keyword: ''
})

// 分页配置
const pagination = reactive({
  current: 1,
  pageSize: 10,
  total: 0,
  showSizeChanger: true,
  showQuickJumper: true
})

// 状态选项
const statusOptions = [
  { value: 0, label: '草稿' },
  { value: 1, label: '已发布' }
]

// 表格列定义
const columns = [
  {
    title: '作业标题',
    dataIndex: 'title',
    key: 'title',
    ellipsis: true,
    width: '25%'
  },
  {
    title: '所属课程',
    dataIndex: 'courseName',
    key: 'courseName',
    ellipsis: true,
    width: '15%'
  },
  {
    title: '开始时间',
    dataIndex: 'startTime',
    key: 'startTime',
    width: '15%'
  },
  {
    title: '结束时间',
    dataIndex: 'endTime',
    key: 'endTime',
    width: '15%'
  },
  {
    title: '状态',
    dataIndex: 'status',
    key: 'status',
    width: '10%'
  },
  {
    title: '提交率',
    dataIndex: 'submissionRate',
    key: 'submissionRate',
    width: '10%'
  },
  {
    title: '操作',
    key: 'action',
    width: '10%'
  }
]

// 获取状态文本
const getStatusText = (status: number) => {
  switch (status) {
    case 0: return '草稿'
    case 1: return '已发布'
    default: return '未知'
  }
}

// 获取状态颜色
const getStatusColor = (status: number) => {
  switch (status) {
    case 0: return 'orange'
    case 1: return 'green'
    default: return 'default'
  }
}

// 获取提交率状态
const getSubmissionRateStatus = (rate: number) => {
  if (rate === undefined || rate === null) return 'normal'
  if (rate < 30) return 'exception'
  if (rate < 80) return 'normal'
  return 'success'
}

// 加载作业列表
const loadAssignments = async () => {
  loading.value = true
  try {
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
    
    // 构建查询参数
    const params = {
      page: pagination.current,
      size: pagination.pageSize,
      courseId: filters.courseId,
      status: filters.status,
      keyword: filters.keyword || undefined
    }
    
    const response = await axios.get('/api/teacher/assignments', {
      params,
      headers: {
        'Authorization': token ? `Bearer ${token}` : ''
      }
    })
    
    if (response.data && response.data.code === 200) {
      const data = response.data.data
      
      // 使用类型安全的方式创建新数组
      const newAssignments: Assignment[] = Array.isArray(data.records) 
        ? data.records.map((item: any): Assignment => ({
            id: item.id,
            title: item.title,
            courseName: item.courseName || '',
            courseId: item.courseId,
            startTime: item.startTime ? new Date(item.startTime).toLocaleString() : undefined,
            endTime: item.endTime ? new Date(item.endTime).toLocaleString() : undefined,
            status: item.status,
            submissionRate: item.submissionRate || 0,
            totalStudents: item.totalStudents || 0,
            submittedCount: item.submittedCount || 0,
            gradedCount: item.gradedCount || 0
          }))
        : [];
      
      // 直接赋值新数组，而不是修改现有数组
      assignments.value = newAssignments;
      pagination.total = data.total || 0
    } else {
      message.error('获取作业列表失败')
    }
  } catch (error) {
    console.error('加载作业列表失败:', error)
    message.error('获取作业列表失败，请检查网络连接')
  } finally {
    loading.value = false
  }
}

// 加载课程列表
const loadCourses = async () => {
  coursesLoading.value = true
  try {
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
    
    const response = await axios.get('/api/teacher/courses', {
      headers: {
        'Authorization': token ? `Bearer ${token}` : ''
      }
    })
    
    if (response.data && response.data.code === 200) {
      // 处理可能的嵌套数据结构
      let courseData = response.data.data
      
      let courses: any[] = []
      if (courseData.records) {
        courses = courseData.records
      } else if (courseData.list) {
        courses = courseData.list
      } else if (Array.isArray(courseData)) {
        courses = courseData
      }
      
      // 转换为下拉选项格式
      courseOptions.value = courses.map((course: any) => ({
        value: course.id,
        label: course.title || course.name
      }))
    } else {
      message.error('获取课程列表失败')
    }
  } catch (error) {
    console.error('加载课程列表失败:', error)
    message.error('获取课程列表失败，请检查网络连接')
  } finally {
    coursesLoading.value = false
  }
}

// 处理筛选条件变化
const handleFilterChange = () => {
  pagination.current = 1 // 重置到第一页
  loadAssignments()
}

// 处理搜索
const handleSearch = () => {
  pagination.current = 1 // 重置到第一页
  loadAssignments()
}

// 处理表格变化（分页、排序等）
const handleTableChange = (pag: any) => {
  pagination.current = pag.current
  pagination.pageSize = pag.pageSize
  loadAssignments()
}

// 创建新作业
const createNewAssignment = () => {
  router.push('/teacher/assignments/create')
}

// 查看作业详情
const viewAssignmentDetail = (id: number) => {
  router.push(`/teacher/assignments/${id}`)
}

// 编辑作业
const editAssignment = (id: number) => {
  router.push(`/teacher/assignments/${id}/edit`)
}

// 发布作业
const publishAssignment = async (id: number) => {
  try {
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
    
    const response = await axios.put(`/api/teacher/assignments/${id}/publish`, {}, {
      headers: {
        'Authorization': token ? `Bearer ${token}` : ''
      }
    })
    
    if (response.data && response.data.code === 200) {
      message.success('作业发布成功')
      loadAssignments() // 重新加载列表
    } else {
      message.error(response.data?.message || '发布作业失败')
    }
  } catch (error) {
    console.error('发布作业失败:', error)
    message.error('发布作业失败，请检查网络连接')
  }
}

// 删除作业
const deleteAssignment = async (id: number) => {
  try {
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
    
    const response = await axios.delete(`/api/teacher/assignments/${id}`, {
      headers: {
        'Authorization': token ? `Bearer ${token}` : ''
      }
    })
    
    if (response.data && response.data.code === 200) {
      message.success('作业删除成功')
      loadAssignments() // 重新加载列表
    } else {
      message.error(response.data?.message || '删除作业失败')
    }
  } catch (error) {
    console.error('删除作业失败:', error)
    message.error('删除作业失败，请检查网络连接')
  }
}

// 初始化
onMounted(() => {
  loadCourses()
  loadAssignments()
})
</script>

<style scoped>
.assignment-list-page {
  padding: 24px;
  background-color: #f0f2f5;
  min-height: 100vh;
}

.page-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 24px;
}

.header-left h1 {
  margin-bottom: 8px;
  font-size: 24px;
  font-weight: 600;
}

.description {
  color: rgba(0, 0, 0, 0.45);
}

.content-container {
  max-width: 1200px;
  margin: 0 auto;
}

.filter-card {
  margin-bottom: 24px;
  border-radius: 8px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.09);
}

.list-card {
  border-radius: 8px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.09);
}
</style> 