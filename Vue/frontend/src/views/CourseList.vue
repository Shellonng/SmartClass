<script setup lang="ts">
import { ref, onMounted, computed, watch } from 'vue'
import { useRouter } from 'vue-router'
import { message } from 'ant-design-vue'
import { SearchOutlined, FilterOutlined, UserOutlined, StarOutlined, BookOutlined } from '@ant-design/icons-vue'
import { getCourseList, getCourseCategories, type Course, type CourseListParams } from '@/api/course'

const router = useRouter()

// 搜索和筛选状态
const searchKeyword = ref('')
const selectedCategory = ref('all')
const selectedLevel = ref('all')
const sortBy = ref('popular')
const loading = ref(false)
const currentPage = ref(1)
const pageSize = ref(12)
const total = ref(0)

// 课程数据
const courses = ref<Course[]>([])
const categories = ref<string[]>([])

// 加载课程列表
const loadCourses = async () => {
  loading.value = true
  try {
    const params: CourseListParams = {
      page: currentPage.value,
      size: pageSize.value,
      keyword: searchKeyword.value || undefined,
      category: selectedCategory.value === 'all' ? undefined : selectedCategory.value,
      level: selectedLevel.value === 'all' ? undefined : selectedLevel.value,
      sortBy: sortBy.value
    }
    
    const response = await getCourseList(params)
    if (response.code === 200) {
      courses.value = response.data.courses
      total.value = response.data.total
    } else {
      message.error('加载课程列表失败')
    }
  } catch (error) {
    console.error('加载课程列表失败:', error)
    message.error('加载课程列表失败')
    // 使用模拟数据作为后备
    courses.value = getMockCourses()
    total.value = 50
  } finally {
    loading.value = false
  }
}

// 加载课程分类
const loadCategories = async () => {
  try {
    const response = await getCourseCategories()
    if (response.code === 200) {
      categories.value = response.data
    }
  } catch (error) {
    console.error('加载课程分类失败:', error)
    // 使用默认分类
    categories.value = ['数学', '计算机', '物理', '化学', '生物', '经济学', '管理学']
  }
}

// 模拟数据（作为后备）
const getMockCourses = (): Course[] => [
  {
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
    image: '',
    description: '本课程系统讲解高等数学的基本概念、理论和方法，包括极限、导数、积分等内容。',
    tags: ['数学', '基础课程', '理工科'],
    price: 0,
    originalPrice: 299,
    startDate: '2024-03-01'
  },
  {
    id: 2,
    title: '计算机程序设计基础',
    instructor: '李教授',
    university: '北京大学',
    category: '计算机',
    level: 'beginner',
    students: 12350,
    rating: 4.9,
    reviewCount: 980,
    duration: '12周',
    effort: '每周3-5小时',
    image: '',
    description: '面向零基础学员的编程入门课程，通过Python语言学习编程思维和基本技能。',
    tags: ['编程', 'Python', '入门'],
    price: 199,
    startDate: '2024-03-15'
  },
  {
    id: 3,
    title: '大学英语综合教程',
    instructor: '王教授',
    university: '复旦大学',
    category: '语言',
    level: 'intermediate',
    students: 18900,
    rating: 4.7,
    reviewCount: 1560,
    duration: '20周',
    effort: '每周2-4小时',
    image: '',
    description: '提升英语听说读写综合能力，涵盖语法、词汇、阅读理解和写作技巧。',
    tags: ['英语', '语言学习', '四六级'],
    price: 299,
    startDate: '2024-02-20'
  },
  {
    id: 4,
    title: '线性代数',
    instructor: '赵教授',
    university: '中国科学技术大学',
    category: '数学',
    level: 'intermediate',
    students: 9800,
    rating: 4.6,
    reviewCount: 720,
    duration: '14周',
    effort: '每周4-6小时',
    image: '',
    description: '深入学习线性代数的核心概念，包括矩阵运算、向量空间、特征值等。',
    tags: ['数学', '线性代数', '理工科'],
    price: 0,
    startDate: '2024-03-10'
  },
  {
    id: 5,
    title: '数据结构与算法',
    instructor: '陈教授',
    university: '上海交通大学',
    category: '计算机',
    level: 'advanced',
    students: 8750,
    rating: 4.9,
    reviewCount: 650,
    duration: '18周',
    effort: '每周6-8小时',
    image: '',
    description: '系统学习常用数据结构和算法设计技巧，提升编程能力和问题解决能力。',
    tags: ['算法', '数据结构', '编程'],
    price: 399,
    startDate: '2024-04-01'
  },
  {
    id: 6,
    title: '微观经济学原理',
    instructor: '刘教授',
    university: '北京大学',
    category: '经济',
    level: 'beginner',
    students: 11200,
    rating: 4.5,
    reviewCount: 890,
    duration: '16周',
    effort: '每周3-4小时',
    image: '',
    description: '介绍微观经济学的基本理论，包括供需关系、市场结构、消费者行为等。',
    tags: ['经济学', '微观经济', '商科'],
    price: 199,
    startDate: '2024-03-20'
  }
];

// 分类选项
const categoryOptions = computed(() => [
  { label: '全部', value: 'all' },
  ...categories.value.map(cat => ({ label: cat, value: cat }))
])

// 难度选项
const levels = [
  { value: 'all', label: '全部难度' },
  { value: 'beginner', label: '初级' },
  { value: 'intermediate', label: '中级' },
  { value: 'advanced', label: '高级' }
]

// 排序选项
const sortOptions = [
  { value: 'popular', label: '最受欢迎' },
  { value: 'rating', label: '评分最高' },
  { value: 'newest', label: '最新发布' },
  { value: 'price_low', label: '价格从低到高' },
  { value: 'price_high', label: '价格从高到低' }
]

// 筛选后的课程
const filteredCourses = computed(() => {
  let result = courses.value

  // 关键词搜索
  if (searchKeyword.value) {
    const keyword = searchKeyword.value.toLowerCase()
    result = result.filter(course => 
      course.title.toLowerCase().includes(keyword) ||
      course.instructor.toLowerCase().includes(keyword) ||
      course.university.toLowerCase().includes(keyword) ||
      course.description.toLowerCase().includes(keyword) ||
      course.tags.some(tag => tag.toLowerCase().includes(keyword))
    )
  }

  // 分类筛选
  if (selectedCategory.value !== 'all') {
    result = result.filter(course => course.category === selectedCategory.value)
  }

  // 难度筛选
  if (selectedLevel.value !== 'all') {
    result = result.filter(course => course.level === selectedLevel.value)
  }

  // 排序
  switch (sortBy.value) {
    case 'popular':
      result.sort((a, b) => b.students - a.students)
      break
    case 'rating':
      result.sort((a, b) => b.rating - a.rating)
      break
    case 'newest':
      result.sort((a, b) => {
        const dateA = a.startDate ? new Date(a.startDate).getTime() : 0
        const dateB = b.startDate ? new Date(b.startDate).getTime() : 0
        return dateB - dateA
      })
      break
    case 'price_low':
      result.sort((a, b) => a.price - b.price)
      break
    case 'price_high':
      result.sort((a, b) => b.price - a.price)
      break
  }

  return result
})

// 搜索课程
const searchCourses = () => {
  currentPage.value = 1
  loadCourses()
}

// 重置筛选
const resetFilters = () => {
  searchKeyword.value = ''
  selectedCategory.value = 'all'
  selectedLevel.value = 'all'
  sortBy.value = 'popular'
  currentPage.value = 1
  loadCourses()
}

// 查看课程详情
const viewCourse = (courseId: number) => {
  router.push(`/course/${courseId}`)
}

// 分页变化
const onPageChange = (page: number) => {
  currentPage.value = page
  loadCourses()
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

// 格式化价格
const formatPrice = (price: number) => {
  return price === 0 ? '免费' : `¥${price}`
}

// 监听筛选条件变化
watch([selectedCategory, selectedLevel, sortBy], () => {
  currentPage.value = 1
  loadCourses()
})

// 组件挂载时加载数据
onMounted(async () => {
  await loadCategories()
  await loadCourses()
})
</script>

<template>
  <div class="course-list-page">
    <!-- 页面头部 -->
    <div class="page-header">
      <div class="container">
        <h1>课程中心</h1>
        <p>发现优质课程，开启学习之旅</p>
      </div>
    </div>

    <!-- 搜索和筛选区域 -->
    <div class="search-filter-section">
      <div class="container">
        <div class="search-bar">
          <a-input-search
            v-model:value="searchKeyword"
            placeholder="搜索课程、教师或大学"
            size="large"
            @search="searchCourses"
          >
            <template #prefix>
              <SearchOutlined />
            </template>
          </a-input-search>
        </div>
        
        <div class="filter-bar">
          <div class="filter-item">
            <label>分类：</label>
            <a-select v-model:value="selectedCategory" style="width: 150px">
              <a-select-option v-for="category in categoryOptions" :key="category.value" :value="category.value">
                {{ category.label }}
              </a-select-option>
            </a-select>
          </div>
          
          <div class="filter-item">
            <label>难度：</label>
            <a-select v-model:value="selectedLevel" style="width: 120px">
              <a-select-option v-for="level in levels" :key="level.value" :value="level.value">
                {{ level.label }}
              </a-select-option>
            </a-select>
          </div>
          
          <div class="filter-item">
            <label>排序：</label>
            <a-select v-model:value="sortBy" style="width: 150px">
              <a-select-option v-for="option in sortOptions" :key="option.value" :value="option.value">
                {{ option.label }}
              </a-select-option>
            </a-select>
          </div>
        </div>
      </div>
    </div>

    <!-- 课程列表 -->
    <div class="courses-section">
      <div class="container">
        <div class="courses-header">
          <h2>共找到 {{ filteredCourses.length }} 门课程</h2>
        </div>
        
        <div class="courses-grid">
          <div 
            v-for="course in filteredCourses" 
            :key="course.id" 
            class="course-card"
            @click="viewCourse(course.id)"
          >
            <div class="course-image">
              <div v-if="!course.image" class="placeholder-course-image">
                <div class="placeholder-content">
                  <BookOutlined style="font-size: 24px; color: #1890ff;" />
                  <span>{{ course.category }}</span>
                </div>
              </div>
              <img v-else :src="course.image" :alt="course.title" />
              <div class="course-price">
                {{ course.price === 0 ? '免费' : `¥${course.price}` }}
              </div>
            </div>
            
            <div class="course-content">
              <div class="course-header">
                <h3 class="course-title">{{ course.title }}</h3>
                <a-tag :color="getLevelColor(course.level)" class="level-tag">
                  {{ getLevelText(course.level) }}
                </a-tag>
              </div>
              
              <p class="course-instructor">{{ course.instructor }} · {{ course.university }}</p>
              <p class="course-description">{{ course.description }}</p>
              
              <div class="course-tags">
                <a-tag v-for="tag in course.tags" :key="tag" class="course-tag">
                  {{ tag }}
                </a-tag>
              </div>
              
              <div class="course-stats">
                <div class="stat-item">
                  <UserOutlined />
                  <span>{{ course.students.toLocaleString() }}人学习</span>
                </div>
                <div class="stat-item">
                  <StarOutlined />
                  <span>{{ course.rating }}分</span>
                </div>
                <div class="stat-item">
                  <span>{{ course.duration }}</span>
                </div>
              </div>
            </div>
          </div>
        </div>
        
        <!-- 空状态 -->
        <div v-if="filteredCourses.length === 0" class="empty-state">
          <a-empty description="暂无符合条件的课程">
            <a-button type="primary" @click="() => { searchKeyword = ''; selectedCategory = 'all'; selectedLevel = 'all' }">
              重置筛选条件
            </a-button>
          </a-empty>
        </div>
        
        <!-- 分页 -->
        <div class="pagination-container">
          <a-pagination
            v-model:current="currentPage"
            :total="total"
            :page-size="pageSize"
            :show-size-changer="false"
            :show-quick-jumper="true"
            :show-total="(total: number, range: [number, number]) => `共 ${total} 门课程，当前显示 ${range[0]}-${range[1]} 门`"
            @change="onPageChange"
          />
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.course-list-page {
  min-height: 100vh;
  background: #f5f5f5;
}

/* 页面头部 */
.page-header {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  padding: 60px 20px;
  text-align: center;
}

.page-header h1 {
  font-size: 2.5rem;
  margin-bottom: 10px;
}

.page-header p {
  font-size: 1.1rem;
  opacity: 0.9;
}

.container {
  max-width: 1200px;
  margin: 0 auto;
}

/* 搜索筛选区域 */
.search-filter-section {
  background: white;
  padding: 30px 20px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
}

.search-bar {
  margin-bottom: 20px;
}

.filter-bar {
  display: flex;
  gap: 20px;
  align-items: center;
  flex-wrap: wrap;
}

.filter-item {
  display: flex;
  align-items: center;
  gap: 8px;
}

.filter-item label {
  font-weight: 500;
  color: #666;
}

/* 课程列表区域 */
.courses-section {
  padding: 40px 20px;
}

.courses-header {
  margin-bottom: 30px;
}

.courses-header h2 {
  color: #333;
  font-size: 1.5rem;
}

.courses-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(350px, 1fr));
  gap: 30px;
}

/* 课程卡片 */
.course-card {
  background: white;
  border-radius: 12px;
  overflow: hidden;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
  transition: all 0.3s ease;
  cursor: pointer;
}

.course-card:hover {
  transform: translateY(-5px);
  box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
}

.course-image {
  position: relative;
  height: 200px;
  overflow: hidden;
}

.course-image img {
  width: 100%;
  height: 100%;
  object-fit: cover;
}

.course-price {
  position: absolute;
  top: 15px;
  right: 15px;
  background: #ff4d4f;
  color: white;
  padding: 5px 12px;
  border-radius: 20px;
  font-weight: bold;
  font-size: 0.9rem;
}

/* 占位符图片样式 */
.placeholder-course-image {
  width: 100%;
  height: 200px;
  background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
  border: 1px solid #e2e8f0;
  border-radius: 8px;
  display: flex;
  align-items: center;
  justify-content: center;
  position: relative;
}

.placeholder-course-image .placeholder-content {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 8px;
  text-align: center;
}

.placeholder-course-image .placeholder-content span {
  font-size: 12px;
  font-weight: 500;
  color: #64748b;
}

.course-content {
  padding: 20px;
}

.course-header {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  margin-bottom: 10px;
}

.course-title {
  font-size: 1.3rem;
  font-weight: bold;
  color: #333;
  margin: 0;
  flex: 1;
  margin-right: 10px;
}

.level-tag {
  flex-shrink: 0;
}

.course-instructor {
  color: #666;
  margin-bottom: 10px;
  font-size: 0.9rem;
}

.course-description {
  color: #888;
  font-size: 0.9rem;
  line-height: 1.5;
  margin-bottom: 15px;
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
  line-clamp: 2;
  overflow: hidden;
}

.course-tags {
  margin-bottom: 15px;
}

.course-tag {
  margin-right: 5px;
  margin-bottom: 5px;
  font-size: 0.8rem;
}

.course-stats {
  display: flex;
  justify-content: space-between;
  align-items: center;
  color: #999;
  font-size: 0.9rem;
}

.stat-item {
  display: flex;
  align-items: center;
  gap: 4px;
}

/* 空状态 */
.empty-state {
  text-align: center;
  padding: 60px 20px;
  color: #999;
}

.empty-state .ant-empty-description {
  color: #999;
}

.pagination-container {
  margin-top: 40px;
  text-align: center;
  padding: 20px 0;
  border-top: 1px solid #f0f0f0;
}

/* 响应式设计 */
@media (max-width: 768px) {
  .page-header h1 {
    font-size: 2rem;
  }
  
  .filter-bar {
    flex-direction: column;
    align-items: flex-start;
    gap: 15px;
  }
  
  .courses-grid {
    grid-template-columns: 1fr;
  }
  
  .course-header {
    flex-direction: column;
    align-items: flex-start;
    gap: 10px;
  }
  
  .course-stats {
    flex-direction: column;
    align-items: flex-start;
    gap: 8px;
  }
}
</style>