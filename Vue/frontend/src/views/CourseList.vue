<script setup lang="ts">
import { ref, onMounted, computed, watch } from 'vue'
import { useRouter } from 'vue-router'
import { message } from 'ant-design-vue'
import { SearchOutlined, FilterOutlined, UserOutlined, StarOutlined, BookOutlined, TeamOutlined, AppstoreOutlined, BarsOutlined } from '@ant-design/icons-vue'
import { getPublicCourseList, getCourseCategories, enrollCourse as apiEnrollCourse, getEnrolledCourses, type Course, type CourseCategory, type CourseListParams, getCourseInstructor } from '@/api/course'
import { useAuthStore } from '@/stores/auth'
import axios from 'axios'

// 模拟课程数据
const mockCourses = [
  {
    id: 1,
    title: '高等数学（上）',
    description: '本课程系统讲解高等数学的基本概念、理论和方法，包括函数、极限、导数、微分、积分等内容。',
    coverImage: 'https://img.freepik.com/free-vector/hand-drawn-mathematics-background_23-2148157511.jpg',
    instructor: '张教授',
    instructorId: 101,
    category: '数学',
    categoryId: 1,
    level: 'intermediate',
    price: 0,
    rating: 4.8,
    students: 15420,
    duration: '16周',
    tags: ['数学', '微积分', '理工科基础'],
    type: 'FEATURED',
    createdAt: '2023-09-01',
    updatedAt: '2024-01-15'
  },
  {
    id: 2,
    title: 'Python编程基础',
    description: '零基础入门Python编程，掌握Python基本语法、数据类型、控制结构、函数和模块等核心内容。',
    coverImage: 'https://img.freepik.com/free-vector/programming-concept-illustration_114360-1351.jpg',
    instructor: '李教授',
    instructorId: 102,
    category: '计算机科学',
    categoryId: 2,
    level: 'beginner',
    price: 99,
    rating: 4.9,
    students: 23150,
    duration: '12周',
    tags: ['Python', '编程', '入门'],
    type: 'FEATURED',
    createdAt: '2023-10-15',
    updatedAt: '2024-02-20'
  },
  {
    id: 3,
    title: '大学物理（力学部分）',
    description: '系统讲解经典力学的基本概念、定律和方法，包括牛顿力学、刚体力学、振动和波动等内容。',
    coverImage: 'https://img.freepik.com/free-vector/physics-concept-illustration_114360-3972.jpg',
    instructor: '王教授',
    instructorId: 103,
    category: '物理',
    categoryId: 3,
    level: 'intermediate',
    price: 0,
    rating: 4.7,
    students: 12680,
    duration: '14周',
    tags: ['物理', '力学', '理工科基础'],
    createdAt: '2023-09-10',
    updatedAt: '2024-01-10'
  },
  {
    id: 4,
    title: '数据结构与算法',
    description: '深入学习常用数据结构和算法设计技巧，包括数组、链表、栈、队列、树、图以及各种排序和搜索算法。',
    coverImage: 'https://img.freepik.com/free-vector/programming-concept-illustration_114360-1213.jpg',
    instructor: '陈教授',
    instructorId: 104,
    category: '计算机科学',
    categoryId: 2,
    level: 'advanced',
    price: 199,
    rating: 4.9,
    students: 9850,
    duration: '16周',
    tags: ['数据结构', '算法', '计算机科学'],
    createdAt: '2023-11-05',
    updatedAt: '2024-02-28'
  },
  {
    id: 5,
    title: '大学英语综合教程',
    description: '提升英语听说读写综合能力，涵盖语法、词汇、阅读理解和写作技巧，为四六级考试做准备。',
    coverImage: 'https://img.freepik.com/free-vector/english-school-landing-page-template_23-2148475038.jpg',
    instructor: '刘教授',
    instructorId: 105,
    category: '外语',
    categoryId: 4,
    level: 'beginner',
    price: 129,
    rating: 4.6,
    students: 18760,
    duration: '20周',
    tags: ['英语', '四六级', '语言学习'],
    createdAt: '2023-08-20',
    updatedAt: '2024-01-05'
  },
  {
    id: 6,
    title: '微观经济学原理',
    description: '介绍微观经济学的基本理论，包括供需关系、市场结构、消费者行为、生产理论等内容。',
    coverImage: 'https://img.freepik.com/free-vector/economy-concept-illustration_114360-7385.jpg',
    instructor: '赵教授',
    instructorId: 106,
    category: '经济学',
    categoryId: 5,
    level: 'intermediate',
    price: 149,
    rating: 4.5,
    students: 11200,
    duration: '15周',
    tags: ['经济学', '微观经济', '商科基础'],
    createdAt: '2023-10-01',
    updatedAt: '2024-02-10'
  },
  {
    id: 7,
    title: '线性代数',
    description: '系统学习线性代数的基本概念和方法，包括矩阵运算、行列式、向量空间、特征值和特征向量等内容。',
    coverImage: 'https://img.freepik.com/free-vector/mathematics-concept-illustration_114360-3972.jpg',
    instructor: '张教授',
    instructorId: 101,
    category: '数学',
    categoryId: 1,
    level: 'intermediate',
    price: 0,
    rating: 4.7,
    students: 13580,
    duration: '12周',
    tags: ['数学', '线性代数', '理工科基础'],
    createdAt: '2023-09-15',
    updatedAt: '2024-01-20'
  },
  {
    id: 8,
    title: 'Java程序设计',
    description: '从零开始学习Java编程语言，掌握面向对象编程思想和Java核心技术，为开发企业级应用打下基础。',
    coverImage: 'https://img.freepik.com/free-vector/programming-concept-illustration_114360-1670.jpg',
    instructor: '李教授',
    instructorId: 102,
    category: '计算机科学',
    categoryId: 2,
    level: 'intermediate',
    price: 199,
    rating: 4.8,
    students: 16420,
    duration: '18周',
    tags: ['Java', '编程', '面向对象'],
    type: 'FEATURED',
    createdAt: '2023-11-10',
    updatedAt: '2024-03-01'
  }
];

// 模拟课程分类数据
const mockCategories = [
  { id: 1, name: '数学' },
  { id: 2, name: '计算机科学' },
  { id: 3, name: '物理' },
  { id: 4, name: '外语' },
  { id: 5, name: '经济学' },
  { id: 6, name: '文学艺术' }
];

const router = useRouter()
const authStore = useAuthStore()

// 状态
const loading = ref(false)
const courses = ref<Course[]>([])
const categories = ref<CourseCategory[]>([])
const searchKeyword = ref('')
const selectedCategory = ref('')
const sortBy = ref('latest')
const viewMode = ref('card')
const enrolledCourses = ref<number[]>([])
const teacherNames = ref<Record<number, string>>({}) // 存储教师ID到真实姓名的映射

// 分页配置
const pagination = {
  pageSize: 12,
  current: 1,
  total: 0,
  onChange: (page: number) => {
    pagination.current = page
    fetchCourses()
  }
}

// 过滤和排序后的课程列表
const filteredCourses = computed(() => {
  return courses.value
})

// 获取课程教师的真实姓名
const fetchTeacherNames = async () => {
  const teacherIds = courses.value
    .map(course => course.instructorId)
    .filter((id, index, self) => id && self.indexOf(id) === index) // 去重
  
  try {
    for (const teacherId of teacherIds) {
      if (!teacherNames.value[teacherId]) {
        try {
          const response = await getCourseInstructor(teacherId)
          if (response && response.data && response.data.name) {
            teacherNames.value[teacherId] = response.data.name
          }
        } catch (error) {
          console.error(`获取教师 ${teacherId} 信息失败:`, error)
        }
      }
    }
  } catch (error) {
    console.error('获取教师信息失败:', error)
  }
}

// 获取课程列表
const fetchCourses = async () => {
  loading.value = true
  
  try {
    const params: CourseListParams = {
      page: pagination.current - 1,
      size: pagination.pageSize,
      keyword: searchKeyword.value,
      categoryId: selectedCategory.value || undefined,
      sortBy: sortBy.value
    }
    
    const response = await getPublicCourseList(params)
    courses.value = response.content || []
    pagination.total = response.totalElements || 0
    
    // 获取教师真实姓名
    fetchTeacherNames()
    
    console.log('从后端获取课程列表成功:', response)
  } catch (error) {
    console.error('获取课程列表失败:', error)
    message.error('获取课程列表失败，请稍后重试')
    courses.value = []
    pagination.total = 0
  } finally {
    loading.value = false
  }
}

// 获取课程分类
const fetchCategories = async () => {
  try {
    const response = await getCourseCategories()
    categories.value = response || mockCategories
    console.log('获取课程分类成功:', categories.value)
  } catch (error) {
    console.error('获取课程分类失败:', error)
    categories.value = mockCategories
  }
}

// 获取用户已加入的课程
const fetchEnrolledCourses = async () => {
  if (!authStore.isAuthenticated) return
  
  try {
    const response = await getEnrolledCourses()
    enrolledCourses.value = response.map(course => course.id)
    console.log('获取已加入课程成功:', enrolledCourses.value)
  } catch (error) {
    console.error('获取已加入课程失败:', error)
    enrolledCourses.value = []
  }
}

// 判断用户是否已加入课程
const isEnrolled = (courseId: number) => {
  return enrolledCourses.value.includes(courseId)
}

// 搜索处理
const handleSearch = () => {
  pagination.current = 1
  fetchCourses()
}

// 分类变更处理
const handleCategoryChange = () => {
  pagination.current = 1
  fetchCourses()
}

// 排序变更处理
const handleSortChange = () => {
  pagination.current = 1
  fetchCourses()
}

// 跳转到课程详情页
const goToCourseDetail = (courseId: number) => {
  router.push(`/courses/${courseId}`)
}

// 加入课程
const enrollCourse = async (courseId: number) => {
  if (!authStore.isAuthenticated) {
    message.warning('请先登录后再加入课程')
    router.push('/login?redirect=' + encodeURIComponent(router.currentRoute.value.fullPath))
    return
  }

  if (isEnrolled(courseId)) {
    // 已加入课程，直接进入学习
    if (authStore.user?.role === 'STUDENT') {
      router.push(`/student/courses/${courseId}`)
    } else if (authStore.user?.role === 'TEACHER') {
      router.push(`/teacher/courses/${courseId}`)
    }
    return
}

  // 模拟加入课程
  message.success('成功加入课程')
  enrolledCourses.value.push(courseId)
}

// 文本截断
const truncateText = (text: string, maxLength: number) => {
  if (!text) return ''
  return text.length > maxLength ? text.substring(0, maxLength) + '...' : text
}

// 获取难度标签颜色
const getLevelColor = (level: string) => {
  switch (level) {
    case 'beginner': return 'green'
    case 'intermediate': return 'blue'
    case 'advanced': return 'red'
    default: return 'default'
  }
}

// 获取难度文本
const getLevelText = (level: string) => {
  switch (level) {
    case 'beginner': return '初级'
    case 'intermediate': return '中级'
    case 'advanced': return '高级'
    default: return '未知'
  }
}

/**
 * 处理课程封面图片URL
 * 将/files/开头的路径转换为API访问路径
 */
const processImageUrl = (url: string | undefined): string => {
  if (!url) return '/default-course.jpg';
  
  // 如果是旧格式的URL(/files/开头)，则尝试转换为新格式
  if (url.startsWith('/files/')) {
    // 检查是否是课程封面路径
    if (url.includes('/courses/covers/')) {
      // 提取年月和文件名部分
      const parts = url.split('/');
      if (parts.length >= 5) {
        const yearMonth = parts[parts.length - 2];
        const filename = parts[parts.length - 1];
        return `/api/photo/${yearMonth}/${filename}`;
      }
    }
    // 如果不是课程封面或无法解析，使用通用文件访问API
    return '/api/common/files/get/' + url.substring(7);
  }
  
  // 如果已经是新格式(/api/photo/开头)或外部URL，直接返回
  return url;
};

// 生成渐变背景色
const getRandomGradient = (() => {
  // 预定义一些漂亮的渐变色组合
  const gradients = [
    'linear-gradient(135deg, #F6D365 0%, #FDA085 100%)',
    'linear-gradient(135deg, #5EFCE8 0%, #736EFE 100%)',
    'linear-gradient(135deg, #FCCF31 0%, #F55555 100%)',
    'linear-gradient(135deg, #43CBFF 0%, #9708CC 100%)',
    'linear-gradient(135deg, #FFAF7B 0%, #D76D77 100%)',
    'linear-gradient(135deg, #A6C0FE 0%, #F68084 100%)',
    'linear-gradient(135deg, #6A11CB 0%, #2575FC 100%)',
    'linear-gradient(135deg, #FF9A9E 0%, #FAD0C4 100%)',
    'linear-gradient(135deg, #FF0844 0%, #FFB199 100%)',
    'linear-gradient(135deg, #8BC6EC 0%, #9599E2 100%)',
  ];
  
  // 使用闭包保存已使用的索引
  let lastIndices: number[] = [];
  
  return (courseId: number) => {
    // 基于courseId生成索引，但确保颜色分布均匀
    let index = courseId % gradients.length;
    
    // 如果这个索引最近被使用过，选择另一个
    if (lastIndices.includes(index)) {
      for (let i = 0; i < gradients.length; i++) {
        const newIndex = (index + i) % gradients.length;
        if (!lastIndices.includes(newIndex)) {
          index = newIndex;
          break;
        }
      }
    }
    
    // 更新最近使用的索引（保持最多3个）
    lastIndices.push(index);
    if (lastIndices.length > 3) {
      lastIndices.shift();
    }
    
    return gradients[index];
  };
})();

// 获取课程状态显示文本
const getCourseStatus = (course: Course) => {
  if (!course.status) return '未开始';
  return course.status;
};

// 根据教师ID获取教师姓名
const getTeacherName = (instructorId: number, fallbackName: string): string => {
  return teacherNames.value[instructorId] || fallbackName
}

// 生命周期钩子
onMounted(() => {
  fetchCourses()
  fetchCategories()
  
  if (authStore.isAuthenticated) {
    fetchEnrolledCourses()
  }
})
</script>

<template>
  <div class="course-list-container">
    <div class="page-header">
      <h1 class="page-title">全部课程</h1>
      <div class="filter-section">
          <a-input-search
            v-model:value="searchKeyword"
          placeholder="搜索课程名称或教师"
          style="width: 250px"
          @search="handleSearch"
        />
        <a-select
          v-model:value="selectedCategory"
          style="width: 150px"
          placeholder="课程分类"
          @change="handleCategoryChange"
        >
          <a-select-option value="">全部分类</a-select-option>
          <a-select-option v-for="category in categories" :key="category.id" :value="category.id.toString()">
            {{ category.name }}
              </a-select-option>
            </a-select>
        <a-select
          v-model:value="sortBy"
          style="width: 150px"
          @change="handleSortChange"
        >
          <a-select-option value="latest">最新发布</a-select-option>
          <a-select-option value="popular">最受欢迎</a-select-option>
          <a-select-option value="rating">评分最高</a-select-option>
          <a-select-option value="price_low">价格从低到高</a-select-option>
          <a-select-option value="price_high">价格从高到低</a-select-option>
            </a-select>
      </div>
    </div>

    <!-- 加载状态 -->
    <div v-if="loading" class="loading-container">
      <a-spin size="large" />
      <p>正在加载课程数据...</p>
        </div>
        
    <!-- 卡片视图 -->
    <div v-else-if="viewMode === 'card'" class="course-grid">
      <a-row :gutter="[24, 24]">
        <a-col :xs="24" :sm="12" :md="8" :lg="6" v-for="course in filteredCourses" :key="course.id">
          <a-card hoverable class="course-card" @click="goToCourseDetail(course.id)">
            <template #cover>
              <div class="course-cover" :style="{ background: course.coverImage ? 'none' : getRandomGradient(course.id) }">
                <img v-if="course.coverImage" :src="processImageUrl(course.coverImage)" :alt="course.title" />
                <div class="no-image-title" v-else>{{ course.title }}</div>
                <div class="course-badge" v-if="course.type === 'FEATURED'">精品课程</div>
              </div>
            </template>
            <a-card-meta :title="course.title">
              <template #description>
                <div class="course-info">
                  <div class="course-teacher">
                    <UserOutlined /> {{ getTeacherName(course.instructorId, course.instructor) }}
                  </div>
                  <div class="course-stats">
                    <span class="course-rating">
                      <StarOutlined /> 学分{{ course.credit || course.rating || '0' }}
                    </span>
                    <span class="course-students">
                      <TeamOutlined /> {{ course.students?.toLocaleString() || '0' }} 名学生
                    </span>
                  </div>
                  <div class="course-description">{{ truncateText(course.description, 60) }}</div>
                  <div class="course-tag-list">
                    <a-tag v-if="course.courseType || course.category" size="small">{{ course.courseType || course.category }}</a-tag>
                    <a-tag v-for="tag in course.tags?.slice(0, 1)" :key="tag" size="small">{{ tag }}</a-tag>
                  </div>
                </div>
              </template>
            </a-card-meta>
            <div class="course-footer">
              <span class="course-status" :class="getCourseStatus(course).toLowerCase().replace(/\s/g, '-')">
                {{ getCourseStatus(course) }}
              </span>
              <a-button type="primary" size="small" @click.stop="enrollCourse(course.id)">
                {{ isEnrolled(course.id) ? '继续学习' : '加入学习' }}
              </a-button>
            </div>
          </a-card>
        </a-col>
      </a-row>
      
      <!-- 卡片视图分页 -->
      <div class="pagination-container">
        <a-pagination
          v-model:current="pagination.current"
          :total="pagination.total"
          :pageSize="pagination.pageSize"
          @change="pagination.onChange"
          show-quick-jumper
          show-size-changer
          :pageSizeOptions="['12', '24', '36', '48']"
          @showSizeChange="(current: number, size: number) => { pagination.pageSize = size; fetchCourses(); }"
        />
      </div>
    </div>
    
    <!-- 列表视图 -->
    <div v-else class="course-list">
      <a-list
        :data-source="filteredCourses"
        :pagination="pagination"
      >
        <template #renderItem="{ item: course }">
          <a-list-item class="course-list-item" @click="goToCourseDetail(course.id)">
            <a-list-item-meta>
              <template #avatar>
                <div class="course-list-image">
                  <img :src="processImageUrl(course.coverImage)" :alt="course.title" />
                </div>
              </template>
              <template #title>
                <div class="course-list-title">
                  {{ course.title }}
                  <a-tag color="blue" v-if="course.type === 'FEATURED'">精品课程</a-tag>
                  <a-tag :color="getLevelColor(course.level)">{{ getLevelText(course.level) }}</a-tag>
                </div>
              </template>
              <template #description>
                <div class="course-list-info">
                  <div class="course-list-description">{{ truncateText(course.description, 100) }}</div>
                  <div class="course-list-meta">
                    <span class="course-teacher">
                      <UserOutlined /> {{ getTeacherName(course.instructorId, course.instructor) }}
                    </span>
                    <span class="course-rating">
                      <StarOutlined /> {{ course.rating || '暂无评分' }}
                    </span>
                    <span class="course-students">
                      <TeamOutlined /> {{ course.students?.toLocaleString() || '0' }} 名学生
                    </span>
                    <span class="course-duration">
                      {{ course.duration || '16周' }}
                    </span>
                  </div>
                  <div class="course-tag-list">
                    <a-tag v-if="course.category" size="small">{{ course.category }}</a-tag>
                    <a-tag v-for="tag in course.tags || []" :key="tag" size="small">{{ tag }}</a-tag>
                  </div>
                </div>
              </template>
            </a-list-item-meta>
            <div class="course-list-actions">
              <span class="course-price" v-if="course.price > 0">¥{{ course.price }}</span>
              <span class="course-free" v-else>免费</span>
              <a-button type="primary" @click.stop="enrollCourse(course.id)">
                {{ isEnrolled(course.id) ? '继续学习' : '加入学习' }}
              </a-button>
            </div>
          </a-list-item>
        </template>
      </a-list>
    </div>
        
    <!-- 空状态 -->
    <div v-if="!loading && filteredCourses.length === 0" class="empty-state">
      <a-empty description="暂无符合条件的课程" />
      <a-button type="primary" @click="searchKeyword = ''; selectedCategory = ''; sortBy = 'latest'">
              重置筛选条件
            </a-button>
        </div>
        
    <!-- 视图切换按钮 -->
    <div class="view-toggle">
      <a-button-group>
        <a-button 
          type="text" 
          :class="{ active: viewMode === 'card' }" 
          @click="viewMode = 'card'"
        >
          <AppstoreOutlined />
        </a-button>
        <a-button 
          type="text" 
          :class="{ active: viewMode === 'list' }" 
          @click="viewMode = 'list'"
        >
          <BarsOutlined />
        </a-button>
      </a-button-group>
    </div>
  </div>
</template>

<style scoped>
.course-list-container {
  max-width: 1200px;
  margin: 0 auto;
  padding: 24px;
  position: relative;
}

.page-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 24px;
  flex-wrap: wrap;
  gap: 16px;
}

.page-title {
  font-size: 24px;
  font-weight: 600;
  color: #333;
  margin: 0;
}

.filter-section {
  display: flex;
  gap: 12px;
  flex-wrap: wrap;
}

/* 卡片视图样式 */
.course-grid {
  margin-bottom: 40px;
}

.course-card {
  height: 100%;
  transition: all 0.3s;
  display: flex;
  flex-direction: column;
}

.course-card:hover {
  transform: translateY(-5px);
  box-shadow: 0 8px 16px rgba(0, 0, 0, 0.1);
}

.course-cover {
  position: relative;
  height: 160px;
  overflow: hidden;
  display: flex;
  justify-content: center;
  align-items: center;
  color: white;
  font-weight: bold;
  text-shadow: 0 1px 3px rgba(0,0,0,0.5);
}

.no-image-title {
  font-size: 20px;
  padding: 20px;
  text-align: center;
}

.course-badge {
  position: absolute;
  top: 10px;
  right: 10px;
  background-color: #ff4d4f;
  color: white;
  padding: 2px 8px;
  border-radius: 4px;
  font-size: 12px;
}

.course-info {
  margin-top: 8px;
}

.course-teacher {
  font-size: 13px;
  color: #666;
  margin-bottom: 4px;
}

.course-stats {
  display: flex;
  justify-content: space-between;
  font-size: 12px;
  color: #999;
  margin-bottom: 8px;
}

.course-rating {
  color: #faad14;
}

.course-description {
  font-size: 12px;
  color: #666;
  line-height: 1.5;
  margin-top: 8px;
  margin-bottom: 8px;
  height: 36px;
  overflow: hidden;
}

.course-tag-list {
  margin-top: 8px;
}

.course-footer {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-top: auto;
  padding-top: 16px;
  border-top: 1px solid #f0f0f0;
}

.course-price {
  font-size: 16px;
  font-weight: 600;
  color: #ff4d4f;
}

.course-free {
  font-size: 16px;
  font-weight: 600;
  color: #52c41a;
}

/* 列表视图样式 */
.course-list-item {
  padding: 16px;
  border-radius: 8px;
  transition: all 0.3s;
  cursor: pointer;
}

.course-list-item:hover {
  background-color: #f5f5f5;
}

.course-list-image {
  width: 120px;
  height: 80px;
  overflow: hidden;
  border-radius: 4px;
}

.course-list-image img {
  width: 100%;
  height: 100%;
  object-fit: cover;
}

.course-list-title {
  font-size: 16px;
  font-weight: 500;
  display: flex;
  align-items: center;
  gap: 8px;
}

.course-list-info {
  max-width: 600px;
}

.course-list-description {
  font-size: 13px;
  color: #666;
  margin-bottom: 8px;
}

.course-list-meta {
  display: flex;
  gap: 16px;
  font-size: 12px;
  color: #999;
  flex-wrap: wrap;
}

.course-list-actions {
  display: flex;
  flex-direction: column;
  align-items: flex-end;
  gap: 8px;
}

/* 加载和空状态 */
.loading-container {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 60px 0;
  color: #999;
}

.empty-state {
  padding: 60px 0;
  text-align: center;
}

/* 视图切换按钮 */
.view-toggle {
  position: fixed;
  bottom: 24px;
  right: 24px;
  background: white;
  border-radius: 8px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.15);
  z-index: 10;
}

.view-toggle .ant-btn {
  padding: 8px 12px;
}

.view-toggle .ant-btn.active {
  color: #1890ff;
  background-color: #e6f7ff;
}

/* 响应式调整 */
@media (max-width: 768px) {
  .page-header {
    flex-direction: column;
    align-items: flex-start;
  }
  
  .filter-section {
    width: 100%;
  }
  
  .course-list-image {
    width: 80px;
    height: 60px;
  }
}

.course-status {
  font-size: 16px;
  font-weight: 600;
  padding: 2px 8px;
  border-radius: 4px;
}

.course-status.未开始 {
  color: #1890ff;
}

.course-status.进行中 {
  color: #52c41a;
}

.course-status.已结束 {
  color: #faad14;
}

/* 分页容器 */
.pagination-container {
  margin-top: 24px;
  text-align: center;
}
</style>