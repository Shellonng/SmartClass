<template>
  <div class="student-courses">
    <!-- 页面头部 -->
    <div class="page-header">
      <div class="header-content">
        <h1 class="page-title">我的课程</h1>
        <p class="page-description">探索知识海洋，开启学习之旅</p>
      </div>
      <div class="header-actions">
        <a-button type="primary" @click="showJoinCourseModal">
          <PlusOutlined />
          加入课程
        </a-button>
      </div>
    </div>

    <!-- 课程统计 -->
    <div class="course-stats">
      <div class="stat-card total">
        <div class="stat-icon">
          <BookOutlined />
        </div>
        <div class="stat-content">
          <div class="stat-value">{{ courseStats.total }}</div>
          <div class="stat-label">总课程数</div>
        </div>
      </div>
      
      <div class="stat-card ongoing">
        <div class="stat-icon">
          <PlayCircleOutlined />
        </div>
        <div class="stat-content">
          <div class="stat-value">{{ courseStats.ongoing }}</div>
          <div class="stat-label">进行中</div>
        </div>
      </div>
      
      <div class="stat-card completed">
        <div class="stat-icon">
          <CheckCircleOutlined />
        </div>
        <div class="stat-content">
          <div class="stat-value">{{ courseStats.completed }}</div>
          <div class="stat-label">已完成</div>
        </div>
      </div>
      
      <div class="stat-card progress">
        <div class="stat-icon">
          <LineChartOutlined />
        </div>
        <div class="stat-content">
          <div class="stat-value">{{ courseStats.avgProgress }}%</div>
          <div class="stat-label">平均进度</div>
        </div>
      </div>
    </div>

    <!-- 搜索和筛选 -->
    <div class="filters-section">
      <div class="search-box">
        <a-input-search
          v-model:value="searchKeyword"
          placeholder="搜索课程名称、教师姓名..."
          size="large"
          @search="handleSearch"
          class="search-input"
        >
          <template #prefix>
            <SearchOutlined />
          </template>
        </a-input-search>
      </div>
      <div class="filter-controls">
        <a-radio-group v-model:value="activeTab" @change="handleTabChange" size="large">
          <a-radio-button value="all">全部课程</a-radio-button>
          <a-radio-button value="ongoing">进行中</a-radio-button>
          <a-radio-button value="completed">已结束</a-radio-button>
          <a-radio-button value="favorite">我的收藏</a-radio-button>
        </a-radio-group>
        <a-select
          v-model:value="sortBy"
          placeholder="排序方式"
          style="width: 150px"
          @change="handleSort"
        >
          <a-select-option value="recent">最近学习</a-select-option>
          <a-select-option value="progress">学习进度</a-select-option>
          <a-select-option value="name">课程名称</a-select-option>
          <a-select-option value="startTime">开课时间</a-select-option>
        </a-select>
      </div>
    </div>

    <!-- 课程列表 -->
    <div class="courses-content">
      <a-spin :spinning="loading" tip="加载中...">
        <div class="courses-grid" v-if="filteredCourses.length > 0">
          <div 
            v-for="course in filteredCourses" 
            :key="course.id"
            class="course-card"
            @click="goToCourseDetail(course.id)"
          >
            <!-- 课程封面 -->
            <div class="course-cover">
              <img :src="course.cover" :alt="course.name" />
              <div class="course-overlay">
                <div class="course-actions">
                  <a-button type="primary" ghost size="small">
                    <PlayCircleOutlined />
                    继续学习
                  </a-button>
                  <a-button 
                    type="text" 
                    size="small"
                    @click.stop="toggleFavorite(course)"
                    class="favorite-btn"
                    :class="{ active: course.isFavorite }"
                  >
                    <HeartOutlined v-if="!course.isFavorite" />
                    <HeartFilled v-else />
                  </a-button>
                </div>
              </div>
              <div class="course-status" :class="course.status">
                {{ getStatusText(course.status) }}
              </div>
              <div class="course-difficulty" :class="course.difficulty">
                {{ getDifficultyText(course.difficulty) }}
              </div>
            </div>
            
            <!-- 课程信息 -->
            <div class="course-info">
              <div class="course-category">{{ course.category }}</div>
              <h3 class="course-name">{{ course.name }}</h3>
              <p class="course-description">{{ course.description }}</p>
              
              <!-- 教师信息 -->
              <div class="course-meta">
                <div class="teacher-info">
                  <a-avatar :size="28" :src="course.teacher.avatar">
                    {{ course.teacher.name.charAt(0) }}
                  </a-avatar>
                  <span class="teacher-name">{{ course.teacher.name }}</span>
                </div>
                <div class="course-stats">
                  <span class="stat-item">
                    <UsersOutlined />
                    {{ course.studentCount }}人
                  </span>
                  <span class="stat-item">
                    <ClockCircleOutlined />
                    {{ course.duration }}h
                  </span>
                </div>
              </div>
              
              <!-- 学习进度 -->
              <div class="course-progress">
                <div class="progress-header">
                  <span class="progress-label">学习进度</span>
                  <span class="progress-percentage">{{ course.progress }}%</span>
                </div>
                <a-progress 
                  :percent="course.progress" 
                  :stroke-color="getProgressColor(course.progress)"
                  :show-info="false"
                  :stroke-width="6"
                />
                <div class="progress-details">
                  已完成 {{ course.completedLessons }}/{{ course.totalLessons }} 课时
                  <span v-if="course.lastStudyTime" class="last-study">
                    · 上次学习：{{ formatLastStudyTime(course.lastStudyTime) }}
                  </span>
                </div>
              </div>
              
              <!-- 课程标签 -->
              <div class="course-tags">
                <a-tag v-for="tag in course.tags" :key="tag" size="small" :color="getTagColor(tag)">
                  {{ tag }}
                </a-tag>
              </div>

              <!-- 课程操作按钮 -->
              <div class="course-footer">
                <a-button 
                  type="primary" 
                  block
                  @click.stop="continueLearning(course)"
                  :disabled="course.status === 'completed'"
                >
                  <PlayCircleOutlined v-if="course.progress === 0" />
                  <ReloadOutlined v-else />
                  {{ course.progress === 0 ? '开始学习' : '继续学习' }}
                </a-button>
              </div>
            </div>
          </div>
        </div>
        
        <!-- 空状态 -->
        <div v-else class="empty-state">
          <a-empty 
            :image="Empty.PRESENTED_IMAGE_SIMPLE"
            description="暂无符合条件的课程"
          >
            <a-button type="primary" @click="showJoinCourseModal">
              <PlusOutlined />
              加入课程
            </a-button>
          </a-empty>
        </div>
      </a-spin>

      <!-- 分页 -->
      <div class="pagination-wrapper" v-if="filteredCourses.length > 0">
        <a-pagination
          v-model:current="currentPage"
          v-model:page-size="pageSize"
          :total="totalCourses"
          :show-size-changer="true"
          :show-quick-jumper="true"
          :show-total="(total, range) => `共 ${total} 门课程`"
          @change="handlePageChange"
        />
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
import { message, Empty } from 'ant-design-vue'
import {
  PlusOutlined,
  SearchOutlined,
  BookOutlined,
  PlayCircleOutlined,
  CheckCircleOutlined,
  LineChartOutlined,
  HeartOutlined,
  HeartFilled,
  UsersOutlined,
  ClockCircleOutlined,
  ReloadOutlined
} from '@ant-design/icons-vue'

interface Course {
  id: number
  name: string
  description: string
  cover: string
  category: string
  difficulty: 'beginner' | 'intermediate' | 'advanced'
  status: 'ongoing' | 'completed' | 'pending'
  progress: number
  completedLessons: number
  totalLessons: number
  duration: number
  studentCount: number
  isFavorite: boolean
  tags: string[]
  teacher: {
    id: number
    name: string
    avatar: string
    title: string
  }
  joinTime: string
  lastStudyTime?: string
  rating: number
  reviewCount: number
}

const router = useRouter()

// 页面状态
const loading = ref(false)
const joinLoading = ref(false)
const searchKeyword = ref('')
const activeTab = ref('all')
const sortBy = ref('recent')
const joinCourseModalVisible = ref(false)
const currentPage = ref(1)
const pageSize = ref(12)
const totalCourses = ref(0)

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

// 课程统计
const courseStats = ref({
  total: 8,
  ongoing: 5,
  completed: 3,
  avgProgress: 73
})

// 模拟课程数据
const courses = ref<Course[]>([
  {
    id: 1,
    name: '高等数学（上）',
    description: '掌握函数、极限、导数、积分等基本概念和计算方法，为后续专业课程打下坚实的数学基础',
    cover: 'https://via.placeholder.com/300x200/1890ff/white?text=数学',
    category: '数学',
    difficulty: 'intermediate',
    status: 'ongoing',
    progress: 85,
    completedLessons: 17,
    totalLessons: 20,
    duration: 45,
    studentCount: 156,
    isFavorite: true,
    tags: ['必修', '重点', '基础'],
    teacher: {
      id: 1,
      name: '张教授',
      avatar: '',
      title: '数学系教授'
    },
    joinTime: '2024-01-15',
    lastStudyTime: '2024-01-20 14:30',
    rating: 4.8,
    reviewCount: 234
  },
  {
    id: 2,
    name: '大学英语四级',
    description: '系统提升英语听说读写能力，针对性训练四级考试技巧，帮助学生顺利通过CET-4考试',
    cover: 'https://via.placeholder.com/300x200/52c41a/white?text=英语',
    category: '英语',
    difficulty: 'beginner',
    status: 'ongoing',
    progress: 62,
    completedLessons: 15,
    totalLessons: 24,
    duration: 60,
    studentCount: 203,
    isFavorite: false,
    tags: ['四级', '必修'],
    teacher: {
      id: 2,
      name: '李老师',
      avatar: '',
      title: '外语系副教授'
    },
    joinTime: '2024-01-10',
    lastStudyTime: '2024-01-19 16:45',
    rating: 4.6,
    reviewCount: 189
  },
  {
    id: 3,
    name: '计算机程序设计（C++）',
    description: '从基础语法到面向对象编程，全面学习C++编程语言，培养程序设计思维和解决问题的能力',
    cover: 'https://via.placeholder.com/300x200/722ed1/white?text=C%2B%2B',
    category: '计算机',
    difficulty: 'advanced',
    status: 'completed',
    progress: 100,
    completedLessons: 30,
    totalLessons: 30,
    duration: 80,
    studentCount: 98,
    isFavorite: true,
    tags: ['编程', '必修', '实践'],
    teacher: {
      id: 3,
      name: '王教授',
      avatar: '',
      title: '计算机系教授'
    },
    joinTime: '2023-09-01',
    lastStudyTime: '2023-12-20 10:15',
    rating: 4.9,
    reviewCount: 156
  },
  {
    id: 4,
    name: '线性代数',
    description: '学习向量、矩阵、线性方程组等核心概念，掌握线性代数的基本理论和计算方法',
    cover: 'https://via.placeholder.com/300x200/fa8c16/white?text=线代',
    category: '数学',
    difficulty: 'intermediate',
    status: 'ongoing',
    progress: 45,
    completedLessons: 9,
    totalLessons: 20,
    duration: 40,
    studentCount: 142,
    isFavorite: false,
    tags: ['必修', '数学'],
    teacher: {
      id: 4,
      name: '陈教授',
      avatar: '',
      title: '数学系副教授'
    },
    joinTime: '2024-01-05',
    lastStudyTime: '2024-01-18 09:20',
    rating: 4.5,
    reviewCount: 87
  }
])

// 计算属性
const filteredCourses = computed(() => {
  let filtered = courses.value

  // 按标签筛选
  if (activeTab.value !== 'all') {
    switch (activeTab.value) {
      case 'ongoing':
        filtered = filtered.filter(course => course.status === 'ongoing')
        break
      case 'completed':
        filtered = filtered.filter(course => course.status === 'completed')
        break
      case 'favorite':
        filtered = filtered.filter(course => course.isFavorite)
        break
    }
  }

  // 按关键词搜索
  if (searchKeyword.value) {
    const keyword = searchKeyword.value.toLowerCase()
    filtered = filtered.filter(course => 
      course.name.toLowerCase().includes(keyword) ||
      course.teacher.name.toLowerCase().includes(keyword) ||
      course.category.toLowerCase().includes(keyword) ||
      course.description.toLowerCase().includes(keyword)
    )
  }

  // 排序
  filtered.sort((a, b) => {
    switch (sortBy.value) {
      case 'recent':
        return new Date(b.lastStudyTime || b.joinTime).getTime() - new Date(a.lastStudyTime || a.joinTime).getTime()
      case 'progress':
        return b.progress - a.progress
      case 'name':
        return a.name.localeCompare(b.name, 'zh-CN')
      case 'startTime':
        return new Date(b.joinTime).getTime() - new Date(a.joinTime).getTime()
      default:
        return 0
    }
  })

  totalCourses.value = filtered.length
  return filtered.slice((currentPage.value - 1) * pageSize.value, currentPage.value * pageSize.value)
})

// 方法
const handleSearch = () => {
  currentPage.value = 1
}

const handleTabChange = () => {
  currentPage.value = 1
}

const handleSort = () => {
  currentPage.value = 1
}

const handlePageChange = () => {
  // 分页处理已在计算属性中完成
}

const goToCourseDetail = (courseId: number) => {
  router.push(`/student/courses/${courseId}`)
}

const continueLearning = (course: Course) => {
  if (course.status === 'completed') {
    message.info('该课程已完成学习')
    return
  }
  message.success(`继续学习：${course.name}`)
  // 这里可以跳转到具体的学习页面
  router.push(`/student/courses/${course.id}/learn`)
}

const toggleFavorite = (course: Course) => {
  course.isFavorite = !course.isFavorite
  message.success(course.isFavorite ? '已加入收藏' : '已取消收藏')
  // 这里可以调用API更新收藏状态
}

const showJoinCourseModal = () => {
  joinCourseModalVisible.value = true
}

const handleJoinCourse = async () => {
  try {
    await joinFormRef.value.validate()
    joinLoading.value = true
    
    // 模拟API调用
    await new Promise(resolve => setTimeout(resolve, 1500))
    
    message.success('加入课程成功！')
    joinCourseModalVisible.value = false
    
    // 重置表单
    joinFormRef.value.resetFields()
    
    // 刷新课程列表
    // await loadCourses()
    
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

// 工具方法
const getStatusText = (status: string) => {
  const statusMap = {
    ongoing: '进行中',
    completed: '已完成',
    pending: '未开始'
  }
  return statusMap[status as keyof typeof statusMap] || '未知'
}

const getDifficultyText = (difficulty: string) => {
  const difficultyMap = {
    beginner: '入门',
    intermediate: '进阶',
    advanced: '高级'
  }
  return difficultyMap[difficulty as keyof typeof difficultyMap] || '未知'
}

const getProgressColor = (progress: number) => {
  if (progress >= 80) return '#52c41a'
  if (progress >= 60) return '#1890ff'
  if (progress >= 40) return '#faad14'
  return '#ff4d4f'
}

const getTagColor = (tag: string) => {
  const colorMap: { [key: string]: string } = {
    '必修': 'red',
    '选修': 'blue',
    '重点': 'orange',
    '基础': 'green',
    '进阶': 'purple',
    '实践': 'cyan',
    '理论': 'geekblue',
    '编程': 'magenta',
    '数学': 'volcano',
    '英语': 'lime',
    '四级': 'gold'
  }
  return colorMap[tag] || 'default'
}

const formatLastStudyTime = (time: string) => {
  const now = new Date()
  const studyTime = new Date(time)
  const diffTime = now.getTime() - studyTime.getTime()
  const diffDays = Math.floor(diffTime / (1000 * 60 * 60 * 24))
  
  if (diffDays === 0) return '今天'
  if (diffDays === 1) return '昨天'
  if (diffDays < 7) return `${diffDays}天前`
  return studyTime.toLocaleDateString('zh-CN')
}

// 页面初始化
onMounted(() => {
  loading.value = true
  // 模拟加载数据
  setTimeout(() => {
    loading.value = false
  }, 1000)
  
  console.log('学生课程页面初始化完成')
})
</script>

<style scoped>
.student-courses {
  padding: 24px;
  min-height: 100vh;
  background: #f5f7fa;
}

/* 页面头部 */
.page-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 32px;
  padding: 32px;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  border-radius: 20px;
  color: white;
  box-shadow: 0 12px 40px rgba(102, 126, 234, 0.3);
  position: relative;
  overflow: hidden;
}

.page-header::before {
  content: '';
  position: absolute;
  top: -50%;
  right: -50%;
  width: 200%;
  height: 200%;
  background: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><circle cx="50" cy="50" r="1" fill="white" opacity="0.1"/></svg>') repeat;
  animation: float 20s linear infinite;
}

@keyframes float {
  0% { transform: translate(0, 0) rotate(0deg); }
  100% { transform: translate(-50px, -50px) rotate(360deg); }
}

.header-content {
  flex: 1;
  position: relative;
  z-index: 2;
}

.page-title {
  font-size: 32px;
  font-weight: 700;
  margin: 0 0 8px 0;
  background: linear-gradient(45deg, #ffffff, #e3f2fd);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}

.page-description {
  font-size: 16px;
  margin: 0;
  opacity: 0.9;
}

.header-actions {
  position: relative;
  z-index: 2;
}

.header-actions .ant-btn {
  background: rgba(255, 255, 255, 0.2);
  border-color: rgba(255, 255, 255, 0.3);
  color: white;
  backdrop-filter: blur(10px);
  height: 48px;
  padding: 0 24px;
  border-radius: 12px;
  font-weight: 600;
}

.header-actions .ant-btn:hover {
  background: rgba(255, 255, 255, 0.3);
  transform: translateY(-2px);
  box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
}

/* 课程统计 */
.course-stats {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 20px;
  margin-bottom: 32px;
}

.stat-card {
  background: white;
  border-radius: 16px;
  padding: 24px;
  display: flex;
  align-items: center;
  gap: 16px;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.06);
  border: 1px solid #f0f0f0;
  transition: all 0.3s ease;
}

.stat-card:hover {
  transform: translateY(-4px);
  box-shadow: 0 8px 30px rgba(0, 0, 0, 0.12);
}

.stat-icon {
  width: 56px;
  height: 56px;
  border-radius: 14px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 20px;
  color: white;
}

.stat-card.total .stat-icon {
  background: linear-gradient(135deg, #667eea, #764ba2);
}

.stat-card.ongoing .stat-icon {
  background: linear-gradient(135deg, #1890ff, #36cfc9);
}

.stat-card.completed .stat-icon {
  background: linear-gradient(135deg, #52c41a, #73d13d);
}

.stat-card.progress .stat-icon {
  background: linear-gradient(135deg, #faad14, #ffc53d);
}

.stat-content {
  flex: 1;
}

.stat-value {
  font-size: 28px;
  font-weight: 700;
  color: #333;
  margin-bottom: 4px;
}

.stat-label {
  font-size: 14px;
  color: #666;
}

/* 搜索和筛选 */
.filters-section {
  background: white;
  border-radius: 16px;
  padding: 24px;
  margin-bottom: 32px;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.06);
  border: 1px solid #f0f0f0;
}

.search-box {
  margin-bottom: 20px;
}

.search-input {
  max-width: 400px;
  border-radius: 12px;
}

.filter-controls {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 16px;
}

/* 课程网格 */
.courses-content {
  background: white;
  border-radius: 16px;
  padding: 32px;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.06);
  border: 1px solid #f0f0f0;
}

.courses-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(380px, 1fr));
  gap: 24px;
  margin-bottom: 32px;
}

.course-card {
  background: white;
  border-radius: 16px;
  border: 1px solid #f0f0f0;
  overflow: hidden;
  transition: all 0.3s ease;
  cursor: pointer;
}

.course-card:hover {
  transform: translateY(-8px);
  box-shadow: 0 12px 40px rgba(0, 0, 0, 0.15);
  border-color: #1890ff;
}

.course-cover {
  position: relative;
  height: 200px;
  overflow: hidden;
}

.course-cover img {
  width: 100%;
  height: 100%;
  object-fit: cover;
  transition: transform 0.3s ease;
}

.course-card:hover .course-cover img {
  transform: scale(1.05);
}

.course-overlay {
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: rgba(0, 0, 0, 0.4);
  display: flex;
  align-items: center;
  justify-content: center;
  opacity: 0;
  transition: all 0.3s ease;
}

.course-card:hover .course-overlay {
  opacity: 1;
}

.course-actions {
  display: flex;
  gap: 12px;
  align-items: center;
}

.favorite-btn {
  color: white !important;
  border-color: white !important;
}

.favorite-btn.active {
  color: #ff4d4f !important;
}

.course-status {
  position: absolute;
  top: 12px;
  right: 12px;
  padding: 4px 12px;
  border-radius: 20px;
  font-size: 12px;
  font-weight: 500;
  color: white;
  backdrop-filter: blur(10px);
}

.course-status.ongoing {
  background: rgba(24, 144, 255, 0.8);
}

.course-status.completed {
  background: rgba(82, 196, 26, 0.8);
}

.course-status.pending {
  background: rgba(250, 173, 20, 0.8);
}

.course-difficulty {
  position: absolute;
  top: 12px;
  left: 12px;
  padding: 4px 8px;
  border-radius: 12px;
  font-size: 11px;
  font-weight: 500;
  color: white;
  backdrop-filter: blur(10px);
}

.course-difficulty.beginner {
  background: rgba(82, 196, 26, 0.8);
}

.course-difficulty.intermediate {
  background: rgba(250, 173, 20, 0.8);
}

.course-difficulty.advanced {
  background: rgba(255, 77, 79, 0.8);
}

.course-info {
  padding: 20px;
}

.course-category {
  color: #1890ff;
  font-size: 12px;
  font-weight: 500;
  margin-bottom: 8px;
}

.course-name {
  font-size: 18px;
  font-weight: 600;
  color: #333;
  margin: 0 0 8px 0;
  line-height: 1.4;
}

.course-description {
  font-size: 14px;
  color: #666;
  margin: 0 0 16px 0;
  line-height: 1.5;
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
  overflow: hidden;
}

.course-meta {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 16px;
}

.teacher-info {
  display: flex;
  align-items: center;
  gap: 8px;
}

.teacher-name {
  font-size: 14px;
  color: #333;
  font-weight: 500;
}

.course-stats {
  display: flex;
  gap: 16px;
}

.stat-item {
  display: flex;
  align-items: center;
  gap: 4px;
  font-size: 12px;
  color: #666;
}

.course-progress {
  margin-bottom: 16px;
}

.progress-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 8px;
}

.progress-label {
  font-size: 14px;
  color: #333;
  font-weight: 500;
}

.progress-percentage {
  font-size: 14px;
  color: #1890ff;
  font-weight: 600;
}

.progress-details {
  font-size: 12px;
  color: #999;
  margin-top: 4px;
}

.last-study {
  color: #666;
}

.course-tags {
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
  margin-bottom: 16px;
}

.course-footer {
  margin-top: 16px;
}

/* 分页 */
.pagination-wrapper {
  display: flex;
  justify-content: center;
  margin-top: 32px;
}

/* 空状态 */
.empty-state {
  text-align: center;
  padding: 80px 20px;
}

/* 加入课程弹窗 */
.join-course-content {
  padding: 20px 0;
}

.form-tip {
  font-size: 12px;
  color: #999;
  margin-top: 4px;
}

/* 响应式设计 */
@media (max-width: 1200px) {
  .course-stats {
    grid-template-columns: repeat(2, 1fr);
  }
  
  .courses-grid {
    grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
  }
  
  .filter-controls {
    flex-direction: column;
    align-items: stretch;
    gap: 12px;
  }
}

@media (max-width: 768px) {
  .student-courses {
    padding: 16px;
  }
  
  .page-header {
    flex-direction: column;
    gap: 16px;
    text-align: center;
    padding: 24px;
  }
  
  .course-stats {
    grid-template-columns: repeat(2, 1fr);
    gap: 12px;
  }
  
  .stat-card {
    padding: 16px;
  }
  
  .courses-grid {
    grid-template-columns: 1fr;
    gap: 16px;
  }
  
  .courses-content {
    padding: 20px;
  }
  
  .filters-section {
    padding: 16px;
  }
}

@media (max-width: 480px) {
  .course-stats {
    grid-template-columns: 1fr;
  }
  
  .course-meta {
    flex-direction: column;
    align-items: flex-start;
    gap: 8px;
  }
  
  .page-title {
    font-size: 24px;
  }
}
</style> 