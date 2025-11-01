<template>
  <div class="student-courses">
    <!-- 页面头部 -->
    <div class="page-header">
      <div class="header-content">
        <h1 class="page-title">
          <BookOutlined />
          我的课程
        </h1>
        <p class="page-description">探索知识海洋，开启学习之旅</p>
      </div>
      <div class="header-actions">
        <a-button type="primary" @click="showJoinCourseModal">
          <PlusOutlined />
          加入课程
        </a-button>
      </div>
    </div>

    <!-- 课程概览 -->
    <div class="courses-overview">
      <div class="overview-card total">
        <div class="card-icon">
          <BookOutlined />
        </div>
        <div class="card-content">
          <div class="card-title">总课程数</div>
          <div class="card-value">{{ totalCourses }}</div>
          <div class="card-subtitle">本学期 {{ activeCourses }} 门</div>
        </div>
      </div>

      <div class="overview-card students">
        <div class="card-icon">
          <TeamOutlined />
        </div>
        <div class="card-content">
          <div class="card-title">进行中</div>
          <div class="card-value">{{ activeCourses }}</div>
          <div class="card-subtitle">总计 {{ totalCourses }} 门</div>
        </div>
      </div>

      <div class="overview-card progress">
        <div class="card-icon">
          <ClockCircleOutlined />
        </div>
        <div class="card-content">
          <div class="card-title">平均进度</div>
          <div class="card-value">{{ averageProgress }}%</div>
          <div class="card-subtitle">{{ completedLessons }}/{{ totalLessons }} 课时</div>
        </div>
      </div>

      <div class="overview-card performance">
        <div class="card-icon">
          <TrophyOutlined />
        </div>
        <div class="card-content">
          <div class="card-title">平均成绩</div>
          <div class="card-value">{{ averageScore }}</div>
          <div class="card-subtitle">本学期学习成绩</div>
        </div>
      </div>
    </div>
    
    <!-- 课程列表 -->
    <div class="courses-content">
      <!-- 筛选和搜索 -->
      <div class="filter-section">
        <div class="filter-controls">
          <a-select
            v-model:value="semesterFilter"
            placeholder="选择学期"
            style="width: 150px"
            @change="handleFilter"
          >
            <a-select-option value="">全部学期</a-select-option>
            <a-select-option v-for="semester in semesters" :key="semester" :value="semester">
              {{ semester }}
            </a-select-option>
          </a-select>

          <a-select
            v-model:value="statusFilter"
            placeholder="课程状态"
            style="width: 120px"
            @change="handleFilter"
          >
            <a-select-option value="">全部状态</a-select-option>
            <a-select-option value="进行中">进行中</a-select-option>
            <a-select-option value="已结束">已结束</a-select-option>
            <a-select-option value="未开始">未开始</a-select-option>
          </a-select>

          <a-select
            v-model:value="typeFilter"
            placeholder="课程类型"
            style="width: 120px"
            @change="handleFilter"
          >
            <a-select-option value="">全部类型</a-select-option>
            <a-select-option value="必修课">必修课</a-select-option>
            <a-select-option value="选修课">选修课</a-select-option>
          </a-select>

          <a-input-search
            v-model:value="searchKeyword"
            placeholder="搜索课程名称或代码..."
            style="width: 250px"
            @search="handleSearch"
          />
        </div>

        <div class="view-controls">
          <a-radio-group v-model:value="viewMode" @change="handleViewChange">
            <a-radio-button value="table">
              <TableOutlined />
              表格视图
            </a-radio-button>
            <a-radio-button value="card">
              <AppstoreOutlined />
              卡片视图
            </a-radio-button>
          </a-radio-group>
        </div>
      </div>
      
      <!-- 视图容器 - 固定宽度 -->
      <div class="fixed-width-container">
        <!-- 表格视图 -->
        <div v-if="viewMode === 'table'" class="view-container table-view">
          <a-table
            :columns="columns"
            :data-source="courseList"
            :loading="loading"
            :pagination="pagination"
            :scroll="{ x: 'max-content' }"
            row-key="id"
            @change="handleTableChange"
          >
            <template #bodyCell="{ column, record }">
              <template v-if="column.key === 'courseName'">
                <div class="course-info">
                  <div class="course-name">{{ record.title || record.courseName }}</div>
                  <div class="course-meta">{{ record.courseType || record.category }} · {{ record.credit }}学分</div>
                </div>
              </template>

              <template v-else-if="column.key === 'semester'">
                {{ record.term || record.semester || '-' }}
              </template>

              <template v-else-if="column.key === 'status'">
                <a-tag :class="getStatusClass(record.status)">
                  {{ getStatusText(record.status) }}
                </a-tag>
              </template>

              <template v-else-if="column.key === 'students'">
                <div class="students-cell">
                  <div class="students-count">{{ record.studentCount || 0 }}</div>
                  <div class="students-text">名学生</div>
                </div>
              </template>

              <template v-else-if="column.key === 'progress'">
                <div class="progress-cell">
                  <a-progress 
                    :percent="calculateProgress(record.startTime, record.endTime)" 
                    size="small"
                    :stroke-color="getProgressColor(calculateProgress(record.startTime, record.endTime))"
                  />
                  <div class="progress-text">{{ calculateProgress(record.startTime, record.endTime) }}%</div>
                </div>
              </template>
              
              <template v-else-if="column.key === 'performance'">
                <div class="performance-cell">
                  <div class="average-score">{{ record.averageScore || '-' }}</div>
                </div>
              </template>
              
              <template v-else-if="column.key === 'startTime'">
                {{ formatDate(record.startTime) }}
              </template>

              <template v-else-if="column.key === 'action'">
                <a-button-group size="small">
                  <a-button @click="viewCourse(record)">
                    <EyeOutlined />
                    查看
                  </a-button>
                  <a-button @click.stop="continueLearning(record)">
                    <PlayCircleOutlined />
                    学习
                  </a-button>
                </a-button-group>
              </template>
            </template>
          </a-table>
        </div>
        
        <!-- 卡片视图 -->
        <div v-else class="view-container card-view">
          <a-empty v-if="courseList.length === 0" description="暂无符合条件的课程" />
          <div v-else class="cards-grid">
            <div 
              v-for="course in courseList" 
              :key="course.id" 
              class="course-card"
              @click="viewCourse(course)"
            >
              <div class="card-cover">
                <img 
                  v-if="course.coverImage" 
                  :src="course.coverImage" 
                  :alt="course.title || course.courseName" 
                  class="cover-image" 
                />
                <div v-else class="cover-placeholder">
                  <BookOutlined style="font-size: 48px" />
                </div>
                
                <div class="card-tag" :class="getStatusClass(course.status)">
                  {{ getStatusText(course.status) }}
                </div>
              </div>
              
              <div class="card-content">
                <div class="card-header">
                  <h3 class="card-title">{{ course.title || course.courseName }}</h3>
                  <a-tag color="blue">{{ course.courseType || course.category }}</a-tag>
                </div>
                
                <div class="card-meta">
                  <div class="meta-item">
                    <TeamOutlined />
                    <span>{{ course.studentCount || 0 }} 名学生</span>
                  </div>
                  <div class="meta-item">
                    <ClockCircleOutlined />
                    <span>{{ course.credit || 0 }} 学分</span>
                  </div>
                </div>
                
                <div class="card-progress">
                  <div class="progress-label">
                    <span>课程进度</span>
                    <span>{{ calculateProgress(course.startTime, course.endTime) }}%</span>
                  </div>
                  <a-progress 
                    :percent="calculateProgress(course.startTime, course.endTime)" 
                    :stroke-color="getProgressColor(calculateProgress(course.startTime, course.endTime))"
                    :show-info="false" 
                    size="small" 
                  />
                </div>
                
                <div class="card-actions">
                  <a-button type="primary" size="small" @click.stop="continueLearning(course)">
                    <PlayCircleOutlined />
                    继续学习
                  </a-button>
                  <a-button size="small" @click.stop="toggleFavorite(course)">
                    <HeartOutlined v-if="!course.isFavorite" />
                    <HeartFilled v-else style="color: #ff4d4f" />
                    {{ course.isFavorite ? '已收藏' : '收藏' }}
                  </a-button>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- 加入课程弹窗 -->
    <a-modal
      v-model:open="joinCourseModalVisible"
      title="加入课程"
      :width="600"
      @ok="handleJoinCourse"
      @cancel="handleCancelJoin"
      :confirm-loading="joinLoading"
    >
      <div class="join-course-content">
        <a-form
          ref="joinFormRef"
          :model="joinCourseForm"
          :rules="joinCourseRules"
          layout="vertical"
        >
          <a-form-item label="课程代码" name="courseCode">
            <a-input 
              v-model:value="joinCourseForm.courseCode"
              placeholder="请输入课程邀请代码"
              size="large"
              @pressEnter="handleJoinCourse"
            />
            <div class="form-tip">
              课程代码由任课教师提供，通常为6-8位字母数字组合
            </div>
          </a-form-item>
          <a-form-item label="加入密码（可选）" name="password">
            <a-input-password 
              v-model:value="joinCourseForm.password"
              placeholder="请输入课程密码（如有）"
              size="large"
              @pressEnter="handleJoinCourse"
            />
          </a-form-item>
        </a-form>
      </div>
    </a-modal>
  </div>
</template>

<script setup lang="ts">
import { ref, reactive, computed, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { message, Modal } from 'ant-design-vue'
import { useAuthStore } from '@/stores/auth'
import {
  BookOutlined,
  PlusOutlined,
  TeamOutlined,
  ClockCircleOutlined,
  TrophyOutlined,
  TableOutlined,
  AppstoreOutlined,
  EyeOutlined,
  PlayCircleOutlined,
  HeartOutlined,
  HeartFilled,
  DownloadOutlined
} from '@ant-design/icons-vue'
import { getStudentEnrolledCourses, type PageResponse, type Course as ApiCourse } from '@/api/course'
import dayjs from 'dayjs'

// 扩展Course类型，添加前端需要的额外属性
interface Course extends ApiCourse {
  isFavorite?: boolean;
}

const router = useRouter()
const authStore = useAuthStore()

// 响应式数据
const loading = ref(false)
const viewMode = ref('card')
const searchKeyword = ref('')
const semesterFilter = ref('')
const statusFilter = ref('')
const typeFilter = ref('')
const joinLoading = ref(false)
const joinCourseModalVisible = ref(false)

// 课程数据
const courseList = ref<Course[]>([])
const pagination = ref({
  current: 1,
  pageSize: 10,
  total: 0,
  showSizeChanger: true,
  showQuickJumper: true,
  showTotal: (total: number) => `共 ${total} 条记录`
})

// 表单引用
const joinFormRef = ref()

// 表单数据
const joinCourseForm = reactive({
  courseCode: '',
  password: ''
})

const joinCourseRules = {
  courseCode: [
    { required: true, message: '请输入课程代码', trigger: 'blur' },
    { min: 6, max: 8, message: '课程代码长度为6-8位', trigger: 'blur' }
  ]
}

const semesters = ref(['2024-2025-1', '2024-2025-2', '2023-2024-1', '2023-2024-2'])

// 表格列定义
const columns = [
  {
    title: '课程名称',
    dataIndex: 'title',
    key: 'courseName',
    width: 180,
    ellipsis: true
  },
  {
    title: '学分',
    dataIndex: 'credit',
    key: 'credit',
    width: 60,
    align: 'center'
  },
  {
    title: '学期',
    dataIndex: 'term',
    key: 'semester',
    width: 100
  },
  {
    title: '状态',
    dataIndex: 'status',
    key: 'status',
    width: 80
  },
  {
    title: '学生数',
    dataIndex: 'studentCount',
    key: 'students',
    width: 80,
    align: 'center'
  },
  {
    title: '课程进度',
    key: 'progress',
    width: 180
  },
  {
    title: '平均成绩',
    dataIndex: 'averageScore',
    key: 'performance',
    width: 80,
    align: 'center'
  },
  {
    title: '开课时间',
    dataIndex: 'startTime',
    key: 'startTime',
    width: 100
  },
  {
    title: '操作',
    key: 'action',
    width: 180,
    fixed: 'right',
    className: 'action-column'
  }
]

// 计算属性
const totalCourses = computed(() => courseList.value.length)
const activeCourses = computed(() => courseList.value.filter(c => c.status === '进行中').length)
const totalStudents = computed(() => courseList.value.reduce((sum, c) => sum + (c.studentCount || 0), 0))
const averageProgress = computed(() => {
  // 计算所有课程进度的平均值
  if (courseList.value.length === 0) return 0
  
  const totalProgress = courseList.value.reduce((sum, course) => {
    return sum + calculateProgress(course.startTime, course.endTime)
  }, 0)
  
  return Math.round(totalProgress / courseList.value.length)
})
const completedLessons = computed(() => {
  // 这里可以根据实际需求计算完成的课时
  return courseList.value.reduce((sum, c) => sum + (c.chapterCount || 0), 0)
})
const totalLessons = computed(() => {
  // 这里可以根据实际需求计算总课时
  return completedLessons.value + 10 // 临时计算
})
const averageScore = computed(() => {
  const validScores = courseList.value.filter(c => c.averageScore !== undefined && c.averageScore !== null)
  if (validScores.length === 0) return '0.0'
  const total = validScores.reduce((sum, c) => sum + (c.averageScore || 0), 0)
  return (total / validScores.length).toFixed(1)
})

// 方法
const loadCourses = async () => {
  try {
    loading.value = true
    const params = {
      page: pagination.value.current - 1, // 后端分页从0开始
      size: pagination.value.pageSize,
      keyword: searchKeyword.value || undefined,
      status: statusFilter.value || undefined,
      term: semesterFilter.value || undefined
    }
    
    console.log('请求参数:', params)
    
    // 获取学生课程
    const response = await getStudentEnrolledCourses(params)
    
    console.log('API返回原始数据:', response)
    
    if (response && response.data) {
      console.log('API返回data:', response.data)
      
      // 将response.data视为any类型处理
      const responseData: any = response.data
      
      // 尝试从不同位置获取课程列表
      let courses = []
      if (responseData.content && Array.isArray(responseData.content)) {
        console.log('使用content字段获取课程列表')
        courses = responseData.content
      } else if (responseData.records && Array.isArray(responseData.records)) {
        console.log('使用records字段获取课程列表')
        courses = responseData.records
      } else if (responseData.data && responseData.data.content && Array.isArray(responseData.data.content)) {
        console.log('使用data.content字段获取课程列表')
        courses = responseData.data.content
      } else if (responseData.data && responseData.data.records && Array.isArray(responseData.data.records)) {
        console.log('使用data.records字段获取课程列表')
        courses = responseData.data.records
      }
      
      console.log('解析后的课程列表:', courses)
      
      // 处理编码问题
      courses = courses.map((course: any) => {
        // 确保课程对象有所有必要的字段
        return {
          ...course,
          title: course.title || course.courseName || '未命名课程',
          isFavorite: false // 添加收藏状态字段
        }
      })
      
      courseList.value = courses
      
      // 获取总数
      let total = 0
      if (typeof responseData.totalElements === 'number') {
        total = responseData.totalElements
      } else if (typeof responseData.total === 'number') {
        total = responseData.total
      } else if (responseData.data && typeof responseData.data.totalElements === 'number') {
        total = responseData.data.totalElements
      } else if (responseData.data && typeof responseData.data.total === 'number') {
        total = responseData.data.total
      }
      
      pagination.value.total = total
      
      console.log('处理后的课程列表:', courseList.value)
      console.log('总课程数:', total)
    } else {
      message.error('获取课程列表失败')
      courseList.value = []
    }
  } catch (error) {
    console.error('获取课程列表失败:', error)
    message.error('获取课程列表失败，请检查网络连接')
    courseList.value = []
  } finally {
    loading.value = false
  }
}

const handleTableChange = (pag: any) => {
  pagination.value.current = pag.current
  pagination.value.pageSize = pag.pageSize
  loadCourses()
}

const handleFilter = () => {
  pagination.value.current = 1
  loadCourses()
}

const handleSearch = () => {
  pagination.value.current = 1
  loadCourses()
}

const handleViewChange = () => {
  // 视图切换不需要重新加载数据
  console.log('视图切换为:', viewMode.value)
}

const viewCourse = (course: Course) => {
  router.push(`/student/courses/${course.id}`)
}

const continueLearning = (course: Course) => {
  router.push(`/student/courses/${course.id}`)
}

const showJoinCourseModal = () => {
  joinCourseModalVisible.value = true
}

const handleJoinCourse = async () => {
  try {
    await joinFormRef.value.validate()
    joinLoading.value = true
    
    // TODO: 实现加入课程的API调用
    await new Promise(resolve => setTimeout(resolve, 1500))
    
    message.success('加入课程成功！')
    joinCourseModalVisible.value = false
    joinFormRef.value.resetFields()
    loadCourses() // 重新加载课程列表
  } catch (error) {
    console.error('加入课程失败:', error)
  } finally {
    joinLoading.value = false
  }
}

const handleCancelJoin = () => {
  joinCourseModalVisible.value = false
  joinFormRef.value.resetFields()
}

const toggleFavorite = (course: Course) => {
  // 临时处理，实际需要调用收藏API
  course.isFavorite = !course.isFavorite
  message.success(course.isFavorite ? '课程已加入收藏' : '已取消收藏')
}

// 工具方法
const calculateProgress = (startTime?: string, endTime?: string): number => {
  if (!startTime || !endTime) return 0
  
  const now = dayjs()
  const start = dayjs(startTime)
  const end = dayjs(endTime)
  
  // 课程未开始
  if (now.isBefore(start)) return 0
  
  // 课程已结束
  if (now.isAfter(end)) return 100
  
  // 计算进度
  const totalDays = end.diff(start, 'day')
  const passedDays = now.diff(start, 'day')
  
  return Math.min(100, Math.max(0, Math.round((passedDays / totalDays) * 100)))
}

const formatDate = (date?: string) => {
  if (!date) return '-'
  return dayjs(date).format('YYYY-MM-DD')
}

const getStatusText = (status?: string) => {
  if (!status) return '未知'
  
  const statusMap: Record<string, string> = {
    'NOT_STARTED': '未开始',
    'IN_PROGRESS': '进行中',
    'FINISHED': '已结束',
    '未开始': '未开始',
    '进行中': '进行中',
    '已结束': '已结束'
  }
  
  return statusMap[status] || status
}

const getStatusClass = (status?: string) => {
  if (!status) return 'status-unknown'
  
  const normalizedStatus = getStatusText(status).toLowerCase()
  
  if (normalizedStatus.includes('进行')) return 'status-active'
  if (normalizedStatus.includes('结束')) return 'status-completed'
  if (normalizedStatus.includes('未开')) return 'status-pending'
  
  return 'status-unknown'
}

const getProgressColor = (progress: number) => {
  if (progress >= 80) return '#52c41a'
  if (progress >= 60) return '#1890ff'
  if (progress >= 40) return '#faad14'
  return '#ff4d4f'
}

// 页面初始化
onMounted(() => {
  loadCourses()
})
</script>

<style scoped>
.student-courses {
  padding: 24px;
}

/* 页面头部 */
.page-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 24px;
}

.page-title {
  font-size: 24px;
  font-weight: 600;
  margin: 0;
  display: flex;
  align-items: center;
  gap: 8px;
}

.page-description {
  margin-top: 8px;
  color: #666;
}

/* 课程概览卡片 */
.courses-overview {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 20px;
  margin-bottom: 24px;
}

.overview-card {
  background-color: white;
  border-radius: 8px;
  padding: 20px;
  display: flex;
  align-items: center;
  gap: 16px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);
}

.card-icon {
  width: 48px;
  height: 48px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 24px;
  border-radius: 12px;
}

.total .card-icon {
  background-color: #e6f7ff;
  color: #1890ff;
}

.students .card-icon {
  background-color: #fff7e6;
  color: #fa8c16;
}

.progress .card-icon {
  background-color: #f6ffed;
  color: #52c41a;
}

.performance .card-icon {
  background-color: #fff0f6;
  color: #eb2f96;
}

.card-content {
  flex: 1;
}

.card-title {
  font-size: 14px;
  color: #666;
  margin: 0 0 8px 0;
}

.card-value {
  font-size: 24px;
  font-weight: 600;
  color: #333;
  line-height: 1.2;
  margin: 0;
}

.card-subtitle {
  margin-top: 4px;
  font-size: 12px;
  color: #999;
}

/* 筛选和搜索 */
.filter-section {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 24px;
  flex-wrap: wrap;
  gap: 16px;
}

.filter-controls {
  display: flex;
  gap: 12px;
  flex-wrap: wrap;
}

/* 课程内容区域 */
.courses-content {
  background-color: white;
  border-radius: 8px;
  padding: 24px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);
}

/* 表格视图样式 */
.table-view {
  min-height: 400px;
}

.course-info {
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.course-name {
  font-weight: 600;
  color: #333;
}

.course-meta {
  font-size: 12px;
  color: #666;
}

.students-cell {
  display: flex;
  flex-direction: column;
  align-items: center;
}

.students-count {
  font-weight: 600;
  color: #1890ff;
}

.students-text {
  font-size: 12px;
  color: #666;
}

.progress-cell {
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.progress-text {
  font-size: 12px;
  text-align: center;
  color: #666;
}

.performance-cell {
  text-align: center;
}

.average-score {
  font-weight: 600;
  color: #52c41a;
}

/* 状态样式 */
.status-active {
  background-color: #1890ff !important;
  color: white;
}

.status-completed {
  background-color: #52c41a !important;
  color: white;
}

.status-pending {
  background-color: #faad14 !important;
  color: white;
}

.status-unknown {
  background-color: #d9d9d9 !important;
  color: #666;
}

/* 卡片视图样式 */
.cards-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
  gap: 20px;
}

.course-card {
  background-color: white;
  border-radius: 8px;
  overflow: hidden;
  border: 1px solid #eee;
  transition: all 0.3s ease;
  cursor: pointer;
  height: 100%;
  display: flex;
  flex-direction: column;
}

.course-card:hover {
  box-shadow: 0 4px 16px rgba(0, 0, 0, 0.1);
  transform: translateY(-4px);
  border-color: #1890ff;
}

.card-cover {
  height: 160px;
  position: relative;
  overflow: hidden;
}

.cover-image {
  width: 100%;
  height: 100%;
  object-fit: cover;
}

.cover-placeholder {
  width: 100%;
  height: 100%;
  display: flex;
  align-items: center;
  justify-content: center;
  background-color: #f5f5f5;
  color: #999;
}

.card-tag {
  position: absolute;
  top: 12px;
  right: 12px;
  padding: 4px 8px;
  border-radius: 4px;
  font-size: 12px;
}

.card-content {
  padding: 16px;
  flex: 1;
  display: flex;
  flex-direction: column;
}

.card-header {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  margin-bottom: 12px;
}

.card-title {
  margin: 0;
  font-size: 16px;
  font-weight: 600;
  flex: 1;
}

.card-meta {
  display: flex;
  justify-content: space-between;
  margin-bottom: 16px;
}

.meta-item {
  display: flex;
  align-items: center;
  gap: 4px;
  color: #666;
  font-size: 12px;
}

.card-progress {
  margin-bottom: 16px;
}

.progress-label {
  display: flex;
  justify-content: space-between;
  margin-bottom: 4px;
  font-size: 12px;
  color: #666;
}

.card-actions {
  margin-top: auto;
  display: flex;
  gap: 8px;
}

/* 加入课程弹窗 */
.join-course-content {
  padding: 16px 0;
}

.form-tip {
  font-size: 12px;
  color: #999;
  margin-top: 4px;
}

/* 响应式调整 */
@media (max-width: 768px) {
  .courses-overview {
    grid-template-columns: repeat(2, 1fr);
  }

  .filter-section {
    flex-direction: column;
    align-items: stretch;
  }

  .cards-grid {
    grid-template-columns: 1fr;
  }
}

/* 空状态 */
.empty-state {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 60px 0;
  color: #666;
}
</style> 