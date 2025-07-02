import axios from 'axios'

// 通用API响应类型
export interface ApiResponse<T = any> {
  code: number
  message: string
  data: T
}

// 课程相关API接口
export interface CourseListParams {
  page?: number
  size?: number
  keyword?: string
  categoryId?: string | number
  level?: string
  sortBy?: string
}

// 课程类型定义
export interface Course {
  id: number
  title: string
  description: string
  coverImage?: string
  instructor: string
  instructorId: number
  category: string
  categoryId: number
  level: string
  price: number
  rating: number
  students: number
  duration: string
  tags: string[]
  type?: string
  createdAt: string
  updatedAt: string
  status?: string        // 课程状态：未开始/进行中/已结束
  courseType?: string    // 课程类型：必修课/选修课
  credit?: number        // 课程学分
}

// 课程分类类型定义
export interface CourseCategory {
  id: number
  name: string
}

// 分页响应类型
export interface PageResponse<T> {
  content: T[]
  totalElements: number
  totalPages: number
  size: number
  number: number
  first?: boolean
  last?: boolean
  empty?: boolean
}

// 课程列表响应类型
export interface CourseListResponse {
  content: Course[];
  totalElements: number;
  totalPages: number;
  size: number;
  number: number;
  first: boolean;
  last: boolean;
  empty: boolean;
}

// 课程讲师类型定义
export interface Instructor {
  id: number;
  name: string;
  title: string;
  avatar?: string;
  description?: string;
}

// 课程章节类型定义
export interface Chapter {
  id: number;
  courseId: number;
  title: string;
  description?: string;
  orderNum: number;
  sections?: any[];
}

// 课程资源类型定义
export interface Resource {
  id: number;
  courseId: number;
  name: string;
  fileType: string;
  fileSize: number;
  fileUrl: string;
  description?: string;
}

// 模拟课程数据
const mockCourses: Course[] = [
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
]

// 模拟课程分类
const mockCategories: CourseCategory[] = [
  { id: 1, name: '数学' },
  { id: 2, name: '计算机科学' },
  { id: 3, name: '物理' },
  { id: 4, name: '外语' },
  { id: 5, name: '经济学' },
  { id: 6, name: '文学艺术' }
]

/**
 * 适配后端Course对象到前端Course对象
 * 由于后端和前端的数据结构可能不完全一致，需要进行适配
 */
export function adaptBackendCourse(backendCourse: any): Course {
  return {
    id: backendCourse.id,
    title: backendCourse.title || backendCourse.courseName || '',
    description: backendCourse.description || '',
    coverImage: backendCourse.coverImage || '',
    instructor: backendCourse.teacherName || '',
    instructorId: backendCourse.teacherId || 0,
    category: backendCourse.category || backendCourse.courseType || '',
    categoryId: backendCourse.categoryId || 0,
    level: backendCourse.level || 'intermediate',
    price: backendCourse.price || 0,
    rating: backendCourse.rating || backendCourse.averageScore || 4.5,
    students: backendCourse.studentCount || 0,
    duration: backendCourse.duration || '16周',
    tags: backendCourse.tags || [backendCourse.courseType || '课程'], // 设置默认标签
    createdAt: backendCourse.createTime || '',
    updatedAt: backendCourse.updateTime || '',
    status: backendCourse.status || '未开始',
    courseType: backendCourse.courseType || '',
    credit: backendCourse.credit || 0
  }
}

/**
 * 获取公共课程列表（不需要认证）
 */
export async function getPublicCourseList(params: CourseListParams = {}): Promise<PageResponse<Course>> {
  // 添加详细的请求日志
  console.log('调用getPublicCourseList API，参数:', params);
  
  // 调用真实API获取课程数据
  return axios.get('/api/courses/public', { params })
    .then(response => {
      console.log('API响应成功:', response.data);
      
      if (response.data && response.data.code === 200) {
        // 适配数据结构
        const backendData = response.data.data;
        console.log('后端返回的原始数据:', backendData);
        
        const adaptedContent = (backendData.records || []).map((item: any) => {
          console.log('适配前的课程数据:', item);
          const adapted = adaptBackendCourse(item);
          console.log('适配后的课程数据:', adapted);
          return adapted;
        });
        
        const result = {
          content: adaptedContent,
          totalElements: backendData.total || 0,
          totalPages: backendData.pages || 0,
          size: backendData.pageSize || params.size || 10,
          number: backendData.current || params.page || 0,
          first: (backendData.current || 0) === 0,
          last: (backendData.current || 0) === (backendData.pages || 1) - 1,
          empty: !adaptedContent.length
        };
        
        console.log('返回给前端的最终数据:', result);
        return result;
      } else {
        console.error('获取课程列表失败:', response.data);
        // 发生错误时返回空数据
        return {
          content: [],
          totalElements: 0,
          totalPages: 0,
          size: params.size || 10,
          number: params.page || 0,
          first: true,
          last: true,
          empty: true
        };
      }
    })
    .catch(error => {
      console.error('获取课程列表异常:', error);
      // 发生异常时返回空数据
      return {
        content: [],
        totalElements: 0,
        totalPages: 0,
        size: params.size || 10,
        number: params.page || 0,
        first: true,
        last: true,
        empty: true
      };
    });
}

/**
 * 加入课程
 */
export async function enrollCourse(courseId: number): Promise<void> {
  // 模拟API调用
  console.log('模拟加入课程:', courseId)
  return new Promise(resolve => {
    setTimeout(resolve, 300)
  })
}

/**
 * 获取已加入的课程
 */
export async function getEnrolledCourses(): Promise<Course[]> {
  // 模拟已加入的课程
  console.log('使用模拟数据替代API调用: getEnrolledCourses')
  return new Promise(resolve => {
    setTimeout(() => {
      // 返回ID为1、3、7的课程
      const enrolledCourses = mockCourses.filter(course => [1, 3, 7].includes(course.id))
      resolve(enrolledCourses)
    }, 300)
  })
}

/**
 * 检查是否已加入课程
 */
export async function checkEnrollment(courseId: number): Promise<boolean> {
  // 模拟检查是否已加入课程
  console.log('使用模拟数据替代API调用: checkEnrollment', courseId)
  return new Promise(resolve => {
    setTimeout(() => {
      // 假设ID为1、3、7的课程已加入
      resolve([1, 3, 7].includes(courseId))
    }, 200)
  })
}

/**
 * 获取课程分类
 */
export async function getCourseCategories(): Promise<CourseCategory[]> {
  // 模拟获取课程分类
  console.log('使用模拟数据替代API调用: getCourseCategories')
  return new Promise(resolve => {
    setTimeout(() => {
      resolve(mockCategories)
    }, 200)
  })
}

/**
 * 获取课程详情
 */
export async function getCourseDetail(courseId: number): Promise<Course> {
  // 模拟获取课程详情
  console.log('使用模拟数据替代API调用: getCourseDetail', courseId)
  return new Promise((resolve, reject) => {
    setTimeout(() => {
      const course = mockCourses.find(c => c.id === courseId)
      if (course) {
        resolve(course)
      } else {
        reject(new Error('课程不存在'))
      }
    }, 300)
  })
}

// 搜索课程
export const searchCourses = (params: CourseListParams & { q: string }): Promise<ApiResponse<CourseListResponse>> => {
  return axios.get('/api/courses/search', { params })
}

// 获取热门课程
export const getPopularCourses = (limit: number = 8): Promise<ApiResponse<Course[]>> => {
  return axios.get('/api/courses/popular', { params: { limit } })
}

// 获取相关课程
export const getRelatedCourses = (courseId: number, limit: number = 4): Promise<ApiResponse<Course[]>> => {
  return axios.get(`/api/courses/${courseId}/related`, { params: { limit } })
}

// 获取课程讲师信息
export const getCourseInstructor = (courseId: number): Promise<ApiResponse<Instructor>> => {
  return axios.get(`/api/courses/${courseId}/instructor`)
}

// 获取课程章节
export const getCourseChapters = (courseId: number): Promise<ApiResponse<Chapter[]>> => {
  return axios.get(`/api/courses/${courseId}/chapters`)
}

// 获取章节详情
export const getChapterDetail = (chapterId: number): Promise<ApiResponse<Chapter>> => {
  return axios.get(`/api/chapters/${chapterId}`)
}

// 获取课程资源
export const getCourseResources = (courseId: number): Promise<ApiResponse<Resource[]>> => {
  return axios.get(`/api/courses/${courseId}/resources`)
}

// 学生课程相关API
export interface JoinCourseRequest {
  inviteCode?: string
  courseId?: number
}

export interface CourseProgress {
  courseId: number
  progress: number
  completedLessons: number
  totalLessons: number
  lastAccessTime: string
}

// 加入课程
export const joinCourse = (data: JoinCourseRequest): Promise<ApiResponse<void>> => {
  return axios.post('/api/student/courses/join', data)
}

// 退出课程
export const leaveCourse = (courseId: number): Promise<ApiResponse<void>> => {
  return axios.delete(`/api/student/courses/${courseId}/leave`)
}

// 获取我的课程
export const getMyCourses = (params: CourseListParams & { status?: string }): Promise<ApiResponse<CourseListResponse>> => {
  return axios.get('/api/student/courses', { params })
}

// 获取课程学习进度
export const getCourseProgress = (courseId: number): Promise<ApiResponse<CourseProgress>> => {
  return axios.get(`/api/student/courses/${courseId}/progress`)
}

// 更新学习进度
export const updateLearningProgress = (courseId: number, lessonId: number): Promise<ApiResponse<void>> => {
  return axios.post(`/api/student/courses/${courseId}/lessons/${lessonId}/complete`)
}

// 收藏课程
export const favoriteCourse = (courseId: number): Promise<ApiResponse<void>> => {
  return axios.post(`/api/student/courses/${courseId}/favorite`)
}

// 取消收藏
export const unfavoriteCourse = (courseId: number): Promise<ApiResponse<void>> => {
  return axios.delete(`/api/student/courses/${courseId}/favorite`)
}

// 获取收藏的课程
export const getFavoriteCourses = (params?: CourseListParams): Promise<ApiResponse<CourseListResponse>> => {
  return axios.get('/api/student/courses/favorites', { params })
}

// 评价课程
export const rateCourse = (courseId: number, data: { rating: number; comment?: string }): Promise<ApiResponse<void>> => {
  return axios.post(`/api/student/courses/${courseId}/rate`, data)
}

// 获取课程评价
export const getCourseReviews = (courseId: number, params?: { page?: number; size?: number }): Promise<ApiResponse<any>> => {
  return axios.get(`/api/courses/${courseId}/reviews`, { params })
}

// 报告课程问题
export const reportCourse = (courseId: number, data: { reason: string; description?: string }): Promise<ApiResponse<void>> => {
  return axios.post(`/api/courses/${courseId}/report`, data)
}

// 评论相关接口
export function getSectionComments(sectionId: number, page = 1, size = 10) {
  return axios.get(`/api/sections/${sectionId}/comments`, {
    params: { page, size }
  })
}

export function getSectionCommentReplies(sectionId: number, commentId: number, page = 1, size = 50) {
  return axios.get(`/api/sections/${sectionId}/comments/${commentId}/replies`, {
    params: { page, size }
  })
}

export function getCourseComments(courseId: number, page = 1, size = 10) {
  return axios.get(`/api/courses/${courseId}/comments`, {
    params: { page, size }
  })
}

export function createSectionComment(sectionId: number, data: {
  content: string
  parentId?: number
}) {
  return axios.post(`/api/sections/${sectionId}/comments`, data)
}

export function updateSectionComment(sectionId: number, commentId: number, data: {
  content: string
}) {
  return axios.put(`/api/sections/${sectionId}/comments/${commentId}`, data)
}

export function deleteSectionComment(sectionId: number, commentId: number) {
  return axios.delete(`/api/sections/${sectionId}/comments/${commentId}`)
}

// 课程资源相关接口
export interface CourseResource {
  id: number
  courseId: number
  name: string
  fileType: string
  fileSize: number
  formattedSize: string
  fileUrl: string
  description?: string
  downloadCount: number
  uploadUserId: number
  uploadUserName: string
  createTime: string
}

// 获取课程资源列表
export function getTeacherCourseResources(courseId: number) {
  return axios.get(`/api/teacher/courses/${courseId}/resources`)
}

// 分页获取课程资源
export function getTeacherCourseResourcesPage(courseId: number, page = 1, size = 10) {
  console.log('调用getTeacherCourseResourcesPage，参数:', { courseId, page, size })
  
  // 确保courseId是有效数字
  if (isNaN(courseId) || courseId <= 0) {
    console.error('无效的courseId:', courseId)
    return Promise.reject(new Error('Invalid courseId'))
  }
  
  return axios.get(`/api/teacher/courses/${courseId}/resources/page`, {
    params: { page, size }
  })
}

// 上传课程资源
export function uploadCourseResource(courseId: number, file: File, name?: string, description?: string) {
  const formData = new FormData()
  formData.append('file', file)
  if (name) formData.append('name', name)
  if (description) formData.append('description', description)
  
  // 确保courseId是有效数字
  if (isNaN(courseId) || courseId <= 0) {
    console.error('Invalid courseId:', courseId)
    return Promise.reject(new Error('Invalid courseId'))
  }
  
  return axios.post(`/api/teacher/courses/${courseId}/resources/upload`, formData, {
    headers: {
      'Content-Type': 'multipart/form-data'
    }
  })
}

// 删除课程资源
export function deleteCourseResource(resourceId: number) {
  return axios.delete(`/api/teacher/courses/resources/${resourceId}`)
}

// 获取资源详情
export function getResourceDetail(resourceId: number) {
  return axios.get(`/api/teacher/courses/resources/${resourceId}`)
}

// 获取资源下载链接
export function getResourceDownloadUrl(resourceId: number) {
  return `/api/teacher/courses/resources/${resourceId}/download`
}

// 获取资源预览链接
export function getResourcePreviewUrl(resourceId: number) {
  return `/api/teacher/courses/resources/${resourceId}/preview`
}

// 直接下载资源（返回blob）
export function downloadResourceDirectly(resourceId: number) {
  return axios.get(`/api/teacher/courses/resources/${resourceId}/download`, {
    responseType: 'blob'
  })
}

// 直接预览资源（返回blob）
export function previewResourceDirectly(resourceId: number) {
  return axios.get(`/api/teacher/courses/resources/${resourceId}/preview`, {
    responseType: 'blob'
  })
}