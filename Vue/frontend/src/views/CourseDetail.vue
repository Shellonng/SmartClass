<script setup lang="ts">
import { ref, onMounted, computed, watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { useAuthStore } from '@/stores/auth'
import { message } from 'ant-design-vue'
import { 
  getCourseDetail,
  getRelatedCourses,
  joinCourse
} from '@/api/course'
import { 
  PlayCircleOutlined, 
  UserOutlined, 
  StarOutlined, 
  ClockCircleOutlined,
  BookOutlined,
  TeamOutlined,
  CalendarOutlined,
  CheckCircleOutlined
} from '@ant-design/icons-vue'

const route = useRoute()
const router = useRouter()
const authStore = useAuthStore()

const courseId = computed(() => route.params.id as string)
const loading = ref(false)
const enrolling = ref(false)

// 课程详情数据
const course = ref<any>(null)

// 默认课程数据（作为后备）
const defaultCourse = ref({
  id: 1,
  title: '高等数学A',
  instructor: '张教授',
  university: '清华大学',
  category: '数学',
  level: 'beginner',
  students: 15420,
  rating: 4.8,
  reviewCount: 1250,
  duration: '16周',
  effort: '每周4-6小时',
  language: '中文',
  image: '',
  description: '本课程系统讲解高等数学的基本概念、理论和方法，包括极限、导数、积分等内容。适合理工科学生学习，为后续专业课程打下坚实的数学基础。',
  longDescription: `
    <h3>课程简介</h3>
    <p>高等数学是理工科学生的重要基础课程，本课程将系统地介绍微积分的基本概念、理论和方法。通过本课程的学习，学生将掌握极限、连续、导数、积分等核心概念，培养数学思维和解决实际问题的能力。</p>
    
    <h3>学习目标</h3>
    <ul>
      <li>掌握极限的概念和计算方法</li>
      <li>理解导数的定义和几何意义</li>
      <li>熟练运用导数解决实际问题</li>
      <li>掌握积分的概念和计算技巧</li>
      <li>能够运用微积分解决物理、工程等领域的问题</li>
    </ul>
    
    <h3>适合人群</h3>
    <p>本课程适合理工科专业的大学生，以及对数学感兴趣的学习者。建议具备高中数学基础。</p>
  `,
  tags: ['数学', '基础课程', '理工科', '微积分'],
  price: 0,
  originalPrice: 299,
  startDate: '2024-03-01',
  endDate: '2024-06-15',
  enrolled: false,
  certificate: true,
  prerequisites: ['高中数学基础'],
  skills: ['微积分', '数学分析', '问题解决'],
  instructorInfo: {
    name: '张教授',
    title: '数学系教授',
    university: '清华大学',
    bio: '清华大学数学系教授，博士生导师。主要研究方向为数学分析和应用数学，发表学术论文50余篇。',
    avatar: '',
    courses: 8,
    students: 45000
  }
})

// 课程大纲
const syllabus = ref([
  {
    week: 1,
    title: '函数与极限',
    topics: ['函数的概念', '数列极限', '函数极限'],
    duration: '4小时',
    completed: false
  },
  {
    week: 2,
    title: '连续性',
    topics: ['函数的连续性', '间断点', '连续函数的性质'],
    duration: '4小时',
    completed: false
  },
  {
    week: 3,
    title: '导数与微分',
    topics: ['导数的定义', '求导法则', '微分的概念'],
    duration: '5小时',
    completed: false
  },
  {
    week: 4,
    title: '导数的应用',
    topics: ['中值定理', '函数的单调性', '极值问题'],
    duration: '5小时',
    completed: false
  },
  {
    week: 5,
    title: '不定积分',
    topics: ['原函数与不定积分', '换元积分法', '分部积分法'],
    duration: '5小时',
    completed: false
  },
  {
    week: 6,
    title: '定积分',
    topics: ['定积分的定义', '牛顿-莱布尼茨公式', '定积分的应用'],
    duration: '5小时',
    completed: false
  }
])

// 学生评价
const reviews = ref([
  {
    id: 1,
    student: '李同学',
    avatar: '',
    rating: 5,
    date: '2024-01-15',
    content: '张教授讲解非常清晰，例题丰富，对理解概念很有帮助。课程安排合理，循序渐进。'
  },
  {
    id: 2,
    student: '王同学',
    avatar: '',
    rating: 4,
    date: '2024-01-10',
    content: '内容很全面，但有些地方讲得比较快，需要反复观看。总体来说是很好的课程。'
  },
  {
    id: 3,
    student: '陈同学',
    avatar: '',
    rating: 5,
    date: '2024-01-08',
    content: '作为数学基础课程，这门课程设计得很好。老师的教学方法很适合初学者。'
  }
])

// 相关课程
const relatedCourses = ref([
  {
    id: 2,
    title: '线性代数',
    instructor: '赵教授',
    university: '中科大',
    rating: 4.6,
    students: 9800,
    image: '',
    price: 0
  },
  {
    id: 3,
    title: '概率论与数理统计',
    instructor: '刘教授',
    university: '北京大学',
    rating: 4.7,
    students: 12500,
    image: '',
    price: 199
  }
])

const activeTab = ref('overview')

// 加载课程详情
const loadCourseDetail = async () => {
  if (!courseId.value) return
  
  loading.value = true
  try {
    const response = await getCourseDetail(parseInt(courseId.value))
    course.value = response.data
  } catch (error) {
    console.error('加载课程详情失败:', error)
    message.error('加载课程详情失败，显示默认数据')
    course.value = defaultCourse.value
  } finally {
    loading.value = false
  }
}

// 加载相关课程
const loadRelatedCourses = async () => {
  if (!courseId.value) return
  
  try {
    const response = await getRelatedCourses(parseInt(courseId.value))
    relatedCourses.value = response.data
  } catch (error) {
    console.error('加载相关课程失败:', error)
    // 保持默认数据
  }
}

// 报名课程
const enrollCourse = async () => {
  if (!authStore.isAuthenticated) {
    router.push('/login')
    return
  }
  
  if (!course.value) {
    message.error('课程信息未加载')
    return
  }
  
  enrolling.value = true
  try {
    await joinCourse(course.value.id)
    course.value.enrolled = true
    course.value.students += 1
    message.success('报名成功！')
  } catch (error) {
    console.error('报名失败:', error)
    message.error('报名失败，请稍后重试')
  } finally {
    enrolling.value = false
  }
}

// 获取难度标签颜色
const getLevelColor = (level: string) => {
  switch (level) {
    case 'beginner': return 'green'
    case 'intermediate': return 'orange'
    case 'advanced': return 'red'
    default: return 'default'
  }
}

// 获取难度标签文本
const getLevelText = (level: string) => {
  switch (level) {
    case 'beginner': return '初级'
    case 'intermediate': return '中级'
    case 'advanced': return '高级'
    default: return '未知'
  }
}

// 查看相关课程
const viewRelatedCourse = (courseId: number) => {
  router.push(`/course/${courseId}`)
}

// 格式化价格
const formatPrice = (price: number) => {
  return price === 0 ? '免费' : `¥${price}`
}

// 监听路由参数变化
watch(courseId, (newId) => {
  if (newId) {
    loadCourseDetail()
    loadRelatedCourses()
  }
}, { immediate: true })

// 组件挂载时加载数据
onMounted(async () => {
  if (courseId.value) {
    await loadCourseDetail()
    await loadRelatedCourses()
  }
})
</script>

<template>
  <div class="course-detail-page">
    <!-- 加载状态 -->
    <div v-if="loading" class="loading-container">
      <a-spin size="large" tip="加载课程详情中..." />
    </div>
    
    <!-- 课程内容 -->
    <div v-else-if="course">
      <!-- 课程头部信息 -->
      <div class="course-header">
        <div class="container">
          <div class="course-hero">
            <div class="course-info">
              <div class="breadcrumb">
                <a-breadcrumb>
                  <a-breadcrumb-item>
                    <router-link to="/">首页</router-link>
                  </a-breadcrumb-item>
                  <a-breadcrumb-item>
                    <router-link to="/courses">课程</router-link>
                  </a-breadcrumb-item>
                  <a-breadcrumb-item>{{ course.title }}</a-breadcrumb-item>
                </a-breadcrumb>
              </div>
            
              <h1 class="course-title">{{ course.title }}</h1>
              <p class="course-subtitle">{{ course.description }}</p>
              
              <div class="course-meta">
                <div class="meta-item">
                  <UserOutlined />
                  <span>{{ course.instructor }} · {{ course.university }}</span>
                </div>
                <div class="meta-item">
                  <StarOutlined />
                  <span>{{ course.rating }}分 ({{ course.reviewCount }}评价)</span>
                </div>
                <div class="meta-item">
                  <TeamOutlined />
                  <span>{{ course.students.toLocaleString() }}人学习</span>
                </div>
                <div class="meta-item">
                  <a-tag :color="getLevelColor(course.level)">
                    {{ getLevelText(course.level) }}
                  </a-tag>
                </div>
              </div>
              
              <div class="course-tags">
                <a-tag v-for="tag in course.tags" :key="tag" class="course-tag">
                  {{ tag }}
                </a-tag>
              </div>
            </div>
            
            <div class="course-media">
              <div class="course-image">
                <div v-if="!course.image" class="placeholder-course-detail-image">
                  <div class="placeholder-content">
                    <BookOutlined style="font-size: 48px; color: #1890ff;" />
                    <span>课程封面</span>
                  </div>
                </div>
                <img v-else :src="course.image" :alt="course.title" />
                <div class="play-button">
                  <PlayCircleOutlined />
                </div>
              </div>
              
              <div class="enrollment-card">
                <div class="price-info">
                  <span v-if="course.price === 0" class="current-price free">免费</span>
                  <template v-else>
                    <span class="current-price">¥{{ course.price }}</span>
                    <span v-if="course.originalPrice" class="original-price">¥{{ course.originalPrice }}</span>
                  </template>
                </div>
                
                <a-button 
                  v-if="!course.enrolled"
                  type="primary" 
                  size="large" 
                  block
                  :loading="enrolling"
                  @click="enrollCourse"
                >
                  {{ course.price === 0 ? '免费报名' : '立即购买' }}
                </a-button>
                
                <a-button v-else type="default" size="large" block disabled>
                  <CheckCircleOutlined /> 已报名
                </a-button>
                
                <div class="course-details">
                  <div class="detail-item">
                    <ClockCircleOutlined />
                    <span>{{ course.duration }}</span>
                  </div>
                  <div class="detail-item">
                    <BookOutlined />
                    <span>{{ course.effort }}</span>
                  </div>
                  <div class="detail-item">
                    <CalendarOutlined />
                    <span>{{ course.startDate }} 开课</span>
                  </div>
                  <div v-if="course.certificate" class="detail-item">
                    <CheckCircleOutlined />
                    <span>提供证书</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      <!-- 课程内容 -->
      <div class="course-content">
        <div class="container">
          <div class="content-layout">
            <!-- 主要内容 -->
            <div class="main-content">
              <a-tabs v-model:activeKey="activeTab" size="large">
                <a-tab-pane key="overview" tab="课程概述">
                  <div class="overview-content">
                    <div v-html="course.longDescription" class="description"></div>
                    
                    <div class="course-info-grid">
                      <div class="info-section">
                        <h3>学习收获</h3>
                        <ul>
                          <li v-for="skill in course.skills" :key="skill">
                            <CheckCircleOutlined class="check-icon" />
                            {{ skill }}
                          </li>
                        </ul>
                      </div>
                      
                      <div class="info-section">
                        <h3>先修要求</h3>
                        <ul>
                          <li v-for="prerequisite in course.prerequisites" :key="prerequisite">
                            <CheckCircleOutlined class="check-icon" />
                            {{ prerequisite }}
                          </li>
                        </ul>
                      </div>
                    </div>
                  </div>
                </a-tab-pane>
                
                <a-tab-pane key="syllabus" tab="课程大纲">
                  <div class="syllabus-content">
                    <div v-for="item in syllabus" :key="item.week" class="syllabus-item">
                      <div class="week-header">
                        <h3>第{{ item.week }}周：{{ item.title }}</h3>
                        <span class="duration">{{ item.duration }}</span>
                      </div>
                      <ul class="topics-list">
                        <li v-for="topic in item.topics" :key="topic">
                          {{ topic }}
                        </li>
                      </ul>
                    </div>
                  </div>
                </a-tab-pane>
                
                <a-tab-pane key="instructor" tab="讲师介绍">
                  <div class="instructor-content">
                    <div class="instructor-card">
                      <div class="instructor-avatar">
                        <a-avatar v-if="!course.instructorInfo.avatar" size="large">
                          {{ course.instructorInfo.name.charAt(0) }}
                        </a-avatar>
                        <img v-else :src="course.instructorInfo.avatar" :alt="course.instructorInfo.name" />
                      </div>
                      <div class="instructor-info">
                        <h3>{{ course.instructorInfo.name }}</h3>
                        <p class="instructor-title">{{ course.instructorInfo.title }}</p>
                        <p class="instructor-university">{{ course.instructorInfo.university }}</p>
                        <div class="instructor-stats">
                          <span>{{ course.instructorInfo.courses }}门课程</span>
                          <span>{{ course.instructorInfo.students.toLocaleString() }}名学生</span>
                        </div>
                      </div>
                    </div>
                    <div class="instructor-bio">
                      <p>{{ course.instructorInfo.bio }}</p>
                    </div>
                  </div>
                </a-tab-pane>
                
                <a-tab-pane key="reviews" tab="学员评价">
                  <div class="reviews-content">
                    <div class="reviews-summary">
                      <div class="rating-overview">
                        <div class="rating-score">
                          <span class="score">{{ course.rating }}</span>
                          <div class="stars">
                            <a-rate :value="course.rating" disabled allow-half />
                          </div>
                          <span class="review-count">{{ course.reviewCount }}条评价</span>
                        </div>
                      </div>
                    </div>
                    
                    <div class="reviews-list">
                      <div v-for="review in reviews" :key="review.id" class="review-item">
                        <div class="review-header">
                          <div class="reviewer-info">
                            <a-avatar v-if="!review.avatar" size="small">
                              {{ review.student.charAt(0) }}
                            </a-avatar>
                            <img v-else :src="review.avatar" :alt="review.student" class="reviewer-avatar" />
                            <div>
                              <div class="reviewer-name">{{ review.student }}</div>
                              <div class="review-date">{{ review.date }}</div>
                            </div>
                          </div>
                          <a-rate :value="review.rating" disabled size="small" />
                        </div>
                        <p class="review-content">{{ review.content }}</p>
                      </div>
                    </div>
                  </div>
                </a-tab-pane>
              </a-tabs>
            </div>
            
            <!-- 侧边栏 -->
            <div class="sidebar">
              <div class="related-courses">
                <h3>相关课程</h3>
                <div class="related-course-list">
                  <div 
                    v-for="relatedCourse in relatedCourses" 
                    :key="relatedCourse.id" 
                    class="related-course-item"
                    @click="viewRelatedCourse(relatedCourse.id)"
                  >
                    <div v-if="!relatedCourse.image" class="placeholder-related-image">
                      <BookOutlined style="font-size: 16px; color: #1890ff;" />
                    </div>
                    <img v-else :src="relatedCourse.image" :alt="relatedCourse.title" />
                    <div class="related-course-info">
                      <h4>{{ relatedCourse.title }}</h4>
                      <p>{{ relatedCourse.instructor }} · {{ relatedCourse.university }}</p>
                      <div class="related-course-meta">
                        <span class="rating">
                          <StarOutlined /> {{ relatedCourse.rating }}
                        </span>
                        <span class="price">
                          {{ relatedCourse.price === 0 ? '免费' : `¥${relatedCourse.price}` }}
                        </span>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
    
    <!-- 空数据状态 -->
    <div v-else class="empty-container">
      <a-empty description="课程不存在或已下架">
        <a-button type="primary" @click="$router.push('/courses')">
          返回课程列表
        </a-button>
      </a-empty>
    </div>
  </div>
</template>

<style scoped>
.course-detail-page {
  min-height: 100vh;
  background: #f5f5f5;
}

/* 课程头部 */
.course-header {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  padding: 40px 20px 60px;
}

.container {
  max-width: 1200px;
  margin: 0 auto;
}

.course-hero {
  display: grid;
  grid-template-columns: 1fr 400px;
  gap: 60px;
  align-items: start;
}

.breadcrumb {
  margin-bottom: 20px;
}

.breadcrumb :deep(.ant-breadcrumb a) {
  color: rgba(255, 255, 255, 0.8);
}

.course-title {
  font-size: 2.5rem;
  font-weight: bold;
  margin-bottom: 15px;
  line-height: 1.2;
}

.course-subtitle {
  font-size: 1.1rem;
  margin-bottom: 25px;
  opacity: 0.9;
  line-height: 1.6;
}

.course-meta {
  display: flex;
  flex-wrap: wrap;
  gap: 20px;
  margin-bottom: 20px;
}

.meta-item {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 0.95rem;
}

.course-tags {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}

.course-tag {
  background: rgba(255, 255, 255, 0.2);
  color: white;
  border: none;
}

/* 课程媒体区域 */
.course-media {
  display: flex;
  flex-direction: column;
  gap: 20px;
}

.course-image {
  position: relative;
  border-radius: 12px;
  overflow: hidden;
  cursor: pointer;
}

.course-image img {
  width: 100%;
  height: auto;
}

.play-button {
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  font-size: 4rem;
  color: white;
  opacity: 0.8;
  transition: opacity 0.3s ease;
}

.course-image:hover .play-button {
  opacity: 1;
}

/* 报名卡片 */
.enrollment-card {
  background: white;
  padding: 25px;
  border-radius: 12px;
  box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1);
}

.price-info {
  text-align: center;
  margin-bottom: 20px;
}

.current-price {
  font-size: 2rem;
  font-weight: bold;
  color: #ff4d4f;
}

.current-price.free {
  color: #52c41a;
}

.original-price {
  font-size: 1.2rem;
  color: #999;
  text-decoration: line-through;
  margin-left: 10px;
}

.course-details {
  margin-top: 20px;
  padding-top: 20px;
  border-top: 1px solid #f0f0f0;
}

.detail-item {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 10px;
  color: #666;
  font-size: 0.9rem;
}

/* 课程内容区域 */
.course-content {
  padding: 40px 20px;
}

.content-layout {
  display: grid;
  grid-template-columns: 1fr 300px;
  gap: 40px;
}

.main-content {
  background: white;
  border-radius: 12px;
  padding: 30px;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
}

/* 概述内容 */
.overview-content .description {
  margin-bottom: 30px;
  line-height: 1.8;
}

.overview-content .description :deep(h3) {
  color: #333;
  margin-top: 25px;
  margin-bottom: 15px;
}

.course-info-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 30px;
}

.info-section h3 {
  color: #333;
  margin-bottom: 15px;
  font-size: 1.2rem;
}

.info-section ul {
  list-style: none;
  padding: 0;
}

.info-section li {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 10px;
  color: #666;
}

.check-icon {
  color: #52c41a;
}

/* 课程大纲 */
.syllabus-item {
  margin-bottom: 30px;
  padding: 20px;
  background: #f9f9f9;
  border-radius: 8px;
}

.week-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 15px;
}

.week-header h3 {
  color: #333;
  margin: 0;
}

.duration {
  color: #666;
  font-size: 0.9rem;
}

.topics-list {
  list-style: none;
  padding: 0;
}

.topics-list li {
  padding: 8px 0;
  color: #666;
  border-bottom: 1px solid #eee;
}

.topics-list li:last-child {
  border-bottom: none;
}

/* 讲师介绍 */
.instructor-card {
  display: flex;
  gap: 20px;
  margin-bottom: 25px;
  padding: 20px;
  background: #f9f9f9;
  border-radius: 8px;
}

.instructor-avatar img {
  width: 80px;
  height: 80px;
  border-radius: 50%;
  object-fit: cover;
}

.instructor-info h3 {
  color: #333;
  margin-bottom: 5px;
}

.instructor-title {
  color: #666;
  margin-bottom: 5px;
}

.instructor-university {
  color: #999;
  margin-bottom: 10px;
}

.instructor-stats {
  display: flex;
  gap: 15px;
  font-size: 0.9rem;
  color: #666;
}

.instructor-bio {
  line-height: 1.8;
  color: #666;
}

/* 评价内容 */
.reviews-summary {
  margin-bottom: 30px;
  padding: 20px;
  background: #f9f9f9;
  border-radius: 8px;
}

.rating-overview {
  text-align: center;
}

.rating-score .score {
  font-size: 3rem;
  font-weight: bold;
  color: #faad14;
  display: block;
}

.stars {
  margin: 10px 0;
}

.review-count {
  color: #666;
  font-size: 0.9rem;
}

.review-item {
  margin-bottom: 25px;
  padding: 20px;
  background: #f9f9f9;
  border-radius: 8px;
}

.review-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 15px;
}

.reviewer-info {
  display: flex;
  align-items: center;
  gap: 12px;
}

.reviewer-avatar {
  width: 40px;
  height: 40px;
  border-radius: 50%;
  object-fit: cover;
}

.reviewer-name {
  font-weight: 500;
  color: #333;
}

.review-date {
  font-size: 0.8rem;
  color: #999;
}

.review-content {
  line-height: 1.6;
  color: #666;
  margin: 0;
}

/* 侧边栏 */
.sidebar {
  display: flex;
  flex-direction: column;
  gap: 30px;
}

.related-courses {
  background: white;
  padding: 25px;
  border-radius: 12px;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
}

.related-courses h3 {
  color: #333;
  margin-bottom: 20px;
  font-size: 1.2rem;
}

.related-course-item {
  display: flex;
  gap: 12px;
  margin-bottom: 20px;
  cursor: pointer;
  transition: transform 0.2s ease;
}

.related-course-item:hover {
  transform: translateY(-2px);
}

.related-course-item img {
  width: 80px;
  height: 60px;
  border-radius: 6px;
  object-fit: cover;
}

.related-course-info h4 {
  font-size: 0.9rem;
  color: #333;
  margin-bottom: 5px;
  line-height: 1.3;
}

.related-course-info p {
  font-size: 0.8rem;
  color: #666;
  margin-bottom: 8px;
}

.related-course-meta {
  display: flex;
  justify-content: space-between;
  align-items: center;
  font-size: 0.8rem;
}

.related-course-meta .rating {
  color: #faad14;
}

.related-course-meta .price {
  color: #ff4d4f;
  font-weight: 500;
}

/* 响应式设计 */
@media (max-width: 1024px) {
  .course-hero {
    grid-template-columns: 1fr;
    gap: 40px;
  }
  
  .content-layout {
    grid-template-columns: 1fr;
    gap: 30px;
  }
  
  .course-info-grid {
    grid-template-columns: 1fr;
  }
}

/* 加载和空状态样式 */
.loading-container {
  display: flex;
  justify-content: center;
  align-items: center;
  min-height: 400px;
  padding: 60px 20px;
}

.empty-container {
  display: flex;
  justify-content: center;
  align-items: center;
  min-height: 400px;
  padding: 60px 20px;
}

@media (max-width: 768px) {
  .course-title {
    font-size: 2rem;
  }
  
  .course-meta {
    flex-direction: column;
    gap: 10px;
  }
  
  .instructor-card {
    flex-direction: column;
    text-align: center;
  }
  
  .review-header {
    flex-direction: column;
    align-items: flex-start;
    gap: 10px;
  }
}

/* 占位符图片样式 */
.placeholder-course-detail-image {
  width: 100%;
  height: 300px;
  background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
  border: 2px dashed #1890ff;
  border-radius: 12px;
  display: flex;
  align-items: center;
  justify-content: center;
}

.placeholder-course-detail-image .placeholder-content {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 12px;
  text-align: center;
  color: #1890ff;
}

.placeholder-course-detail-image .placeholder-content span {
  font-size: 16px;
  font-weight: 500;
}

.placeholder-related-image {
  width: 80px;
  height: 60px;
  background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
  border: 1px solid #e2e8f0;
  border-radius: 6px;
  display: flex;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;
}
</style>