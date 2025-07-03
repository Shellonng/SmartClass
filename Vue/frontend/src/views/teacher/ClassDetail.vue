<template>
  <div class="class-detail">
    <a-spin :spinning="loading">
      <!-- 班级详情头部 -->
      <div class="detail-header">
        <div class="header-left">
          <h1 class="class-title">{{ classInfo?.name || '班级详情' }}</h1>
          <a-tag v-if="classInfo?.courseId" color="blue">
            {{ courseName }}
          </a-tag>
          <a-tag v-if="classInfo?.isDefault" color="green">默认班级</a-tag>
        </div>
        <div class="header-actions">
          <a-button @click="goBack">返回班级列表</a-button>
        </div>
      </div>

      <!-- 班级基本信息 -->
      <a-card title="班级信息" class="info-card">
        <div class="class-info">
          <div class="info-item">
            <span class="label">班级名称:</span>
            <span>{{ classInfo?.name || '暂无' }}</span>
          </div>
          <div class="info-item">
            <span class="label">创建时间:</span>
            <span>{{ formatDate(classInfo?.createTime) }}</span>
          </div>
          <div class="info-item">
            <span class="label">班级描述:</span>
            <span>{{ classInfo?.description || '暂无描述' }}</span>
          </div>
          <div class="info-item">
            <span class="label">学生数量:</span>
            <span>{{ students.length }} 人</span>
          </div>
        </div>
      </a-card>

      <!-- 绑定课程信息 -->
      <a-card title="绑定课程信息" class="info-card" v-if="classInfo?.courseId && courseInfo">
        <div class="course-info">
          <div class="info-item">
            <span class="label">课程名称:</span>
            <span>{{ courseInfo.title || courseInfo.courseName }}</span>
          </div>
          <div class="info-item">
            <span class="label">课程状态:</span>
            <a-tag :color="getCourseStatusColor(courseInfo.status)">{{ courseInfo.status }}</a-tag>
          </div>
          <div class="info-item">
            <span class="label">课程类型:</span>
            <span>{{ courseInfo.courseType || '未设置' }}</span>
          </div>
          <div class="info-item">
            <span class="label">开始时间:</span>
            <span>{{ formatDate(courseInfo.startTime) }}</span>
          </div>
          <div class="info-item">
            <span class="label">结束时间:</span>
            <span>{{ formatDate(courseInfo.endTime) }}</span>
          </div>
          <div class="info-item full-width">
            <span class="label">课程描述:</span>
            <span>{{ courseInfo.description || '暂无描述' }}</span>
          </div>
        </div>
      </a-card>

      <!-- 学生列表 -->
      <a-card title="学生列表" class="info-card">
        <!-- 搜索栏 -->
        <div class="search-bar">
          <a-input-search
            v-model:value="searchKeyword"
            placeholder="搜索学号或姓名"
            style="width: 300px"
            @search="handleSearch"
          />
          <a-button @click="refreshStudentList" style="margin-left: 10px">
            <template #icon><ReloadOutlined /></template>
            刷新
          </a-button>
        </div>

        <!-- 学生表格 -->
        <a-table
          :dataSource="filteredStudents"
          :columns="columns"
          :pagination="pagination"
          rowKey="id"
          :loading="studentsLoading"
        >
          <template #bodyCell="{ column, record }">
            <!-- 头像列 -->
            <template v-if="column.key === 'avatar'">
              <a-avatar :src="record.user?.avatar || 'https://joeschmoe.io/api/v1/random'">
                {{ record.user?.realName?.substring(0, 1) || 'U' }}
              </a-avatar>
            </template>
            
            <!-- 姓名列 -->
            <template v-if="column.key === 'name'">
              {{ record.user?.realName || '未设置' }}
            </template>
            
            <!-- 用户名列 -->
            <template v-if="column.key === 'username'">
              {{ record.user?.username || '未设置' }}
            </template>
            
            <!-- 邮箱列 -->
            <template v-if="column.key === 'email'">
              {{ record.user?.email || '未设置' }}
            </template>
            
            <!-- 学籍状态列 -->
            <template v-if="column.key === 'status'">
              <a-tag :color="getStatusColor(record.enrollmentStatus)">
                {{ getStatusText(record.enrollmentStatus) }}
              </a-tag>
            </template>
            
            <!-- 操作列 -->
            <template v-if="column.key === 'action'">
              <a-button type="link" @click="viewStudentDetail(record.id)">
                <template #icon><EyeOutlined /></template>
                查看
              </a-button>
            </template>
          </template>
        </a-table>
      </a-card>
    </a-spin>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted, computed } from 'vue'
import { useRouter, useRoute } from 'vue-router'
import { message } from 'ant-design-vue'
import type { TablePaginationConfig } from 'ant-design-vue'
import { 
  ReloadOutlined, 
  EyeOutlined,
} from '@ant-design/icons-vue'
import axios from 'axios'

// API导入
import { 
  getClasses, 
  type Class,
  type Course,
} from '@/api/teacher'

import type { Student } from '@/api/student'

// 接收参数
const route = useRoute()
const classId = computed(() => Number(route.params.id))

// 状态定义
const router = useRouter()
const loading = ref(false)
const studentsLoading = ref(false)
const classInfo = ref<Class | null>(null)
const courseInfo = ref<Course | null>(null)
const students = ref<Student[]>([])
const searchKeyword = ref('')

// 计算属性
const courseName = computed(() => {
  return courseInfo.value 
    ? (courseInfo.value.title || courseInfo.value.courseName || '未知课程')
    : '未绑定课程'
})

// 学生过滤逻辑
const filteredStudents = computed(() => {
  if (!searchKeyword.value.trim()) {
    return students.value
  }
  
  const keyword = searchKeyword.value.toLowerCase().trim()
  return students.value.filter(student => {
    // 按学号(ID)搜索
    if (student.id && String(student.id).includes(keyword)) {
      return true
    }
    
    // 按姓名搜索
    if (student.user?.realName && student.user.realName.toLowerCase().includes(keyword)) {
      return true
    }
    
    // 按用户名搜索
    if (student.user?.username && student.user.username.toLowerCase().includes(keyword)) {
      return true
    }
    
    return false
  })
})

// 分页配置
const pagination = ref<TablePaginationConfig>({
  current: 1,
  pageSize: 10,
  total: 0,
  showSizeChanger: true,
  showTotal: (total) => `共 ${total} 条记录`
})

// 表格列定义
const columns = [
  { title: '头像', key: 'avatar', width: 80 },
  { title: '学号', dataIndex: 'id', key: 'id', sorter: (a: Student, b: Student) => a.id - b.id },
  { title: '姓名', key: 'name' },
  { title: '用户名', key: 'username' },
  { title: '邮箱', key: 'email' },
  { title: '学籍状态', key: 'status' },
  { title: '操作', key: 'action', width: 120 }
]

// 获取班级详情
const fetchClassDetail = async () => {
  try {
    loading.value = true
    console.log('开始获取班级详情, ID:', classId.value)
    const res = await getClassDetail(classId.value)
    console.log('班级详情API返回:', res)
    
    if (res && res.data) {
      // 处理不同的响应格式
      if (res.data.code === 200 && res.data.data) {
        classInfo.value = res.data.data
      } else if (typeof res.data === 'object' && !res.data.code) {
        // 直接返回的对象
        classInfo.value = res.data
      } else {
        console.error('无法识别的班级数据格式:', res.data)
        message.error('获取班级信息失败：格式错误')
        classInfo.value = null
      }
      
      console.log('处理后的班级信息:', classInfo.value)
      
      // 如果班级绑定了课程，获取课程信息
      if (classInfo.value && classInfo.value.courseId) {
        fetchCourseDetail(classInfo.value.courseId)
      }
    } else {
      message.error('获取班级详情失败')
    }
  } catch (error: any) {
    console.error('获取班级详情失败:', error)
    message.error(`获取班级详情失败: ${error.response?.data?.message || error.message || '请稍后重试'}`)
  } finally {
    loading.value = false
  }
}

// 获取课程详情
const fetchCourseDetail = async (courseId: number) => {
  try {
    console.log('开始获取课程详情, ID:', courseId)
    const res = await getCourseDetail(courseId)
    console.log('课程详情API返回:', res)
    
    if (res && res.data) {
      // 处理不同的响应格式
      if (res.data.code === 200 && res.data.data) {
        courseInfo.value = res.data.data
      } else if (typeof res.data === 'object' && !res.data.code) {
        // 直接返回的对象
        courseInfo.value = res.data
      } else {
        console.error('无法识别的课程数据格式:', res.data)
        courseInfo.value = null
      }
      
      console.log('处理后的课程信息:', courseInfo.value)
    } else {
      console.error('获取课程详情失败:', res)
      courseInfo.value = null
    }
  } catch (error: any) {
    console.error('获取课程详情失败:', error)
    message.error(`获取课程详情失败: ${error.response?.data?.message || error.message || '请稍后重试'}`)
    courseInfo.value = null
  }
}

// 获取班级学生列表
const fetchStudentList = async () => {
  try {
    studentsLoading.value = true
    const params = {
      page: 1,
      size: 100, // 获取较大数量以便本地搜索
      keyword: ''
    }
    
    console.log('开始获取班级学生列表, 班级ID:', classId.value)
    const res = await getClassStudents(classId.value, params)
    console.log('班级学生API返回:', res)
    
    if (res && res.data) {
      // 处理不同的响应格式
      let studentsData = null
      
      if (Array.isArray(res.data)) {
        studentsData = res.data
      } else if (res.data.records) {
        studentsData = res.data.records
      } else if (res.data.content) {
        studentsData = res.data.content
      } else if (res.data.data && Array.isArray(res.data.data)) {
        studentsData = res.data.data
      } else if (res.data.code === 200 && Array.isArray(res.data.data)) {
        // 处理包装在Result中的数据
        studentsData = res.data.data
      } else if (res.data.code === 200 && res.data.data && (res.data.data.records || res.data.data.content)) {
        // 处理包装在Result和Page中的数据
        studentsData = res.data.data.records || res.data.data.content
      } else {
        console.error('无法解析学生数据:', res.data)
        studentsData = []
      }
      
      console.log('解析后的学生数据:', studentsData)
      
      if (studentsData) {
        // 规范化学生数据结构
        students.value = studentsData.map((student: any) => {
          // 确保基本字段存在
          const processedStudent: any = {
            id: student.id,
            user: {},
            enrollmentStatus: student.enrollmentStatus || 'UNKNOWN'
          }
          
          // 处理用户信息 - 有些接口返回嵌套的user对象，有些接口返回平铺的字段
          if (student.user) {
            // 直接使用嵌套的user对象
            processedStudent.user = student.user
          } else {
            // 构建user对象
            processedStudent.user = {
              id: student.userId,
              username: student.username,
              realName: student.realName,
              email: student.email,
              avatar: student.avatar
            }
          }
          
          // 拷贝其他字段
          Object.keys(student).forEach(key => {
            if (key !== 'user' && !processedStudent.hasOwnProperty(key)) {
              processedStudent[key] = student[key]
            }
          })
          
          return processedStudent
        })
      } else {
        students.value = []
      }
      
      // 更新分页信息
      pagination.value.total = students.value.length
      console.log('处理后的学生列表:', students.value)
    } else {
      console.error('获取学生列表失败:', res)
      students.value = []
    }
  } catch (error) {
    console.error('获取学生列表失败:', error)
    message.error('获取学生列表失败，请稍后重试')
    students.value = []
  } finally {
    studentsLoading.value = false
  }
}

// 处理搜索
const handleSearch = () => {
  // 本地搜索，不需要调用API
  pagination.value.current = 1
}

// 刷新学生列表
const refreshStudentList = () => {
  fetchStudentList()
}

// 查看学生详情
const viewStudentDetail = (id: number) => {
  router.push(`/teacher/students/${id}`)
}

// 返回班级列表页面
const goBack = () => {
  router.push('/teacher/classes')
}

// 格式化日期
const formatDate = (dateString?: string) => {
  if (!dateString) return '未设置'
  
  try {
    const date = new Date(dateString)
    return date.toLocaleString('zh-CN', {
      year: 'numeric',
      month: '2-digit',
      day: '2-digit',
      hour: '2-digit',
      minute: '2-digit'
    })
  } catch (e) {
    return dateString
  }
}

// 获取课程状态颜色
const getCourseStatusColor = (status?: string) => {
  if (!status) return 'default'
  
  switch (status) {
    case '进行中': return 'green'
    case '未开始': return 'blue'
    case '已结束': return 'gray'
    default: return 'default'
  }
}

// 获取学籍状态文本
const getStatusText = (status?: string) => {
  if (!status) return '未设置'
  
  switch (status) {
    case 'ENROLLED': return '在读'
    case 'SUSPENDED': return '休学'
    case 'GRADUATED': return '毕业'
    case 'DROPPED_OUT': return '退学'
    default: return status
  }
}

// 获取学籍状态颜色
const getStatusColor = (status?: string) => {
  if (!status) return 'default'
  
  switch (status) {
    case 'ENROLLED': return 'green'
    case 'SUSPENDED': return 'orange'
    case 'GRADUATED': return 'blue'
    case 'DROPPED_OUT': return 'red'
    default: return 'default'
  }
}

// 重新定义API调用函数，确保路径正确并添加授权token
const getClassDetail = (classId: number) => {
  // 获取token和用户ID
  const token = localStorage.getItem('user-token') || localStorage.getItem('token')
  const userInfo = localStorage.getItem('user-info')
  let userId = ''
  
  if (userInfo) {
    try {
      const userObj = JSON.parse(userInfo)
      userId = userObj.id || ''
    } catch (e) {
      console.error('解析用户信息失败:', e)
    }
  }
  
  // 使用token构建授权头
  const authToken = userId ? `Bearer token-${userId}` : (token ? `Bearer ${token}` : '')
  
  return axios.get(`/api/teacher/classes/${classId}`, {
    headers: {
      'Authorization': authToken
    }
  })
}

const getCourseDetail = (courseId: number) => {
  // 获取token和用户ID
  const token = localStorage.getItem('user-token') || localStorage.getItem('token')
  const userInfo = localStorage.getItem('user-info')
  let userId = ''
  
  if (userInfo) {
    try {
      const userObj = JSON.parse(userInfo)
      userId = userObj.id || ''
    } catch (e) {
      console.error('解析用户信息失败:', e)
    }
  }
  
  // 使用token构建授权头
  const authToken = userId ? `Bearer token-${userId}` : (token ? `Bearer ${token}` : '')
  
  return axios.get(`/api/teacher/courses/${courseId}`, {
    headers: {
      'Authorization': authToken
    }
  })
}

const getClassStudents = (classId: number, params?: { page?: number; size?: number; keyword?: string }) => {
  // 获取token和用户ID
  const token = localStorage.getItem('user-token') || localStorage.getItem('token')
  const userInfo = localStorage.getItem('user-info')
  let userId = ''
  
  if (userInfo) {
    try {
      const userObj = JSON.parse(userInfo)
      userId = userObj.id || ''
    } catch (e) {
      console.error('解析用户信息失败:', e)
    }
  }
  
  // 使用token构建授权头
  const authToken = userId ? `Bearer token-${userId}` : (token ? `Bearer ${token}` : '')
  
  console.log('调用获取班级学生API:', classId, params, authToken ? '已添加授权' : '无授权')
  
  return axios.get(`/api/teacher/classes/${classId}/students`, {
    params,
    headers: {
      'Authorization': authToken
    }
  })
}

// 组件挂载时获取数据
onMounted(() => {
  fetchClassDetail()
  fetchStudentList()
})
</script>

<style scoped>
.class-detail {
  padding: 24px;
}

.detail-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 24px;
}

.header-left {
  display: flex;
  align-items: center;
}

.class-title {
  margin: 0;
  margin-right: 16px;
}

.info-card {
  margin-bottom: 24px;
}

.class-info, .course-info {
  display: flex;
  flex-wrap: wrap;
}

.info-item {
  width: 50%;
  margin-bottom: 16px;
}

.info-item.full-width {
  width: 100%;
}

.label {
  font-weight: 500;
  margin-right: 8px;
}

.search-bar {
  margin-bottom: 16px;
  display: flex;
}

@media screen and (max-width: 768px) {
  .info-item {
    width: 100%;
  }
}
</style> 