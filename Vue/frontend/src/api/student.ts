import axios from 'axios'
import request from '@/utils/request'
import type { ApiResponse } from '@/utils/request'

// 定义API响应结构
export interface ApiResult<T> {
  code: number;
  message: string;
  data: T;
  timestamp?: any;
  error?: boolean;
  success?: boolean;
}

// 学生相关接口
export interface Assignment {
  id: number
  title: string
  description: string
  dueDate: string
  status: 'pending' | 'submitted' | 'graded'
  score?: number
  maxScore: number
  courseName: string
}

export interface Grade {
  id: number
  assignmentTitle: string
  courseName: string
  score: number
  maxScore: number
  submitTime: string
  gradeTime: string
  feedback?: string
}

export interface Course {
  id: number
  name: string
  description: string
  teacherName: string
  progress: number
  totalLessons: number
  completedLessons: number
}

export interface StudyPlan {
  id: number
  title: string
  description: string
  startTime: string
  endTime: string
  status: 'pending' | 'in-progress' | 'completed'
  priority: 'low' | 'medium' | 'high'
}

export interface LearningResource {
  id: number
  title: string
  type: 'video' | 'document' | 'link' | 'book'
  url: string
  description: string
  category: string
  rating: number
}

// 学生实体接口
export interface Student {
  id: number
  userId: number
  studentId: string
  enrollmentStatus?: string
  gpa?: number
  gpaLevel?: string
  createTime?: string
  updateTime?: string
  user?: {
    id: number
    username: string
    email?: string
    realName?: string
    avatar?: string
    role: string
    status: string
  }
  classes?: Class[]
  courses?: Course[]
}

// 班级实体接口
export interface Class {
  id: number
  name: string
  description?: string
  courseId?: number
  teacherId: number
  isDefault?: boolean
  createTime?: string
  studentCount?: number
  course?: {
    id: number
    title: string
  }
}

// 学生搜索结果接口
export interface StudentSearchResult {
  id: number
  userId: number
  studentId: string
  realName: string
}

// 选课申请实体接口
export interface EnrollmentRequest {
  id: number
  studentId: number
  courseId: number
  status: number // 0=待审核 1=已通过 2=已拒绝
  reason?: string
  reviewComment?: string
  submitTime?: string
  reviewTime?: string
  student?: {
    id: number
    studentId: string
    name: string
  }
  course?: {
    id: number
    title: string
  }
}

// 班级相关接口

/**
 * 班级信息接口
 */
export interface ClassInfo {
  id: number
  name: string
  description?: string
  courseId?: number
  teacherId?: number
  isDefault?: boolean
  createTime?: string
  studentCount?: number
  course?: {
    id: number
    title: string
    description?: string
    coverImage?: string
  }
}

/**
 * 班级学生接口
 */
export interface ClassStudent {
  id: number
  userId: number
  enrollmentStatus?: string
  gpa?: number
  gpaLevel?: string
  createTime?: string
  updateTime?: string
  user?: {
    id: number
    username: string
    email?: string
    realName?: string
    avatar?: string
    role?: string
    status?: string
  }
}

// 获取学生仪表板数据
export const getDashboardData = () => {
  return axios.get('/api/student/dashboard')
}

// 获取学习统计数据
export const getStudyStatistics = (timeRange?: string) => {
  return axios.get('/api/student/dashboard/statistics', { params: { timeRange } })
}

// 获取今日任务列表
export const getTodayTasks = () => {
  return axios.get('/api/student/dashboard/today-tasks')
}

// 获取学习进度概览
export const getProgressOverview = () => {
  return axios.get('/api/student/dashboard/progress-overview')
}

// 获取近期成绩
export const getRecentGrades = (limit: number = 10) => {
  return axios.get('/api/student/dashboard/recent-grades', { params: { limit } })
}

// 获取学习建议
export const getLearningSuggestions = () => {
  return axios.get('/api/student/dashboard/learning-suggestions')
}

// 获取待完成作业列表
export const getPendingAssignments = () => {
  return axios.get<Assignment[]>('/api/student/tasks/todo')
}

// 获取所有作业列表
export const getAssignments = (params?: { page?: number; size?: number; status?: string }) => {
  return axios.get<Assignment[]>('/api/student/tasks', { params })
}

// 提交作业
export const submitAssignment = (assignmentId: number, data: FormData) => {
  return axios.post(`/api/student/tasks/${assignmentId}/submit`, data, {
    headers: {
      'Content-Type': 'multipart/form-data'
    }
  })
}

// 获取成绩列表
export const getGrades = (params?: { page?: number; size?: number; courseId?: number }) => {
  return axios.get<Grade[]>('/api/student/grades', { params })
}

// 获取课程列表
export const getCourses = () => {
  return axios.get<Course[]>('/api/student/courses')
}

// 获取课程详情
export const getCourseDetail = (courseId: number) => {
  return axios.get(`/api/student/courses/${courseId}`)
}

// 获取学习计划
export const getStudyPlans = () => {
  return axios.get<StudyPlan[]>('/api/student/study-plans')
}

// 创建学习计划
export const createStudyPlan = (data: Omit<StudyPlan, 'id'>) => {
  return axios.post('/api/student/study-plans', data)
}

// 更新学习计划
export const updateStudyPlan = (id: number, data: Partial<StudyPlan>) => {
  return axios.put(`/api/student/study-plans/${id}`, data)
}

// 删除学习计划
export const deleteStudyPlan = (id: number) => {
  return axios.delete(`/api/student/study-plans/${id}`)
}

// 获取学习资源
export const getLearningResources = (params?: { category?: string; type?: string }) => {
  return axios.get<LearningResource[]>('/api/student/resources', { params })
}

// 获取学习进度
export const getLearningProgress = () => {
  return axios.get('/api/student/progress')
}

// 更新学习进度
export const updateLearningProgress = (courseId: number, lessonId: number) => {
  return axios.post(`/api/student/progress/${courseId}/${lessonId}`)
}

// AI学习助手相关接口
export const getPersonalRecommendations = (params?: { courseId?: number; learningGoal?: string; limit?: number }) => {
  return axios.get('/api/student/ai-learning/recommendations', { params })
}

export const askAIQuestion = (data: { question: string; questionType?: string; courseId?: number }) => {
  return axios.post('/api/student/ai-learning/question-answer', data)
}

export const getAbilityAnalysis = (params?: { courseId?: number; timeRange?: string }) => {
  return axios.get('/api/student/ai-learning/ability-analysis', { params })
}

export const generateStudyPlan = (data: { goal: string; planDays: number; courseId?: number }) => {
  return axios.post('/api/student/ai-learning/study-plan', data)
}

export const getKnowledgeMastery = (courseId: number, chapterId?: number) => {
  return axios.get('/api/student/ai-learning/knowledge-mastery', { params: { courseId, chapterId } })
}

export const getEfficiencyAnalysis = (timeRange?: string) => {
  return axios.get('/api/student/ai-learning/efficiency-analysis', { params: { timeRange } })
}

export const getPracticeRecommendations = (data: { subjectId: number; difficultyLevel: string }) => {
  return axios.post('/api/student/ai-learning/practice-recommendations', data)
}

export const getLearningReport = (params?: { reportType?: string; timeRange?: string }) => {
  return axios.get('/api/student/ai-learning/learning-report', { params })
}

export const setLearningGoals = (data: { goalType: string; goals: any[] }) => {
  return axios.post('/api/student/ai-learning/learning-goals', data)
}

export const getLearningHistory = (params?: { courseId?: number; page?: number; size?: number }) => {
  return axios.get('/api/student/ai-learning/learning-history', { params })
}

export const submitAIFeedback = (data: { feedbackType: string; rating: number; comment?: string }) => {
  return axios.post('/api/student/ai-learning/feedback', data)
}

// 获取学生列表
export const getStudents = (params: {
  current: number
  size: number
  keyword?: string
  classId?: number
  courseId?: number
}) => {
  return request.get('/api/teacher/students', { params })
}

// 获取学生详情
export const getStudentDetail = (id: number) => {
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
  
  console.log('调用获取学生详情API, ID:', id)
  
  return axios.get(`/api/teacher/students/${id}`, {
    headers: {
      'Authorization': authToken
    }
  })
}

// 获取班级列表
export const getTeacherClasses = () => {
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
  
  console.log('调用获取班级列表API')
  
  return axios.get('/api/teacher/classes', {
    headers: {
      'Authorization': authToken
    }
  })
  .then(response => {
    console.log('获取班级列表响应:', response)
    return response
  })
  .catch(error => {
    console.error('获取班级列表失败:', error)
    throw error
  })
}

// 添加学生到班级
export const addStudentToClass = (studentId: number, classId: number) => {
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
  
  console.log('调用添加学生到班级API, 学生ID:', studentId, '班级ID:', classId)
  
  return axios.post('/api/teacher/students/add-to-class', 
    { studentId, classId },
    {
      headers: {
        'Authorization': authToken,
        'Content-Type': 'application/json'
      }
    }
  )
}

// 从班级移除学生
export const removeStudentFromClass = (studentId: number, classId: number) => {
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
  
  console.log('调用从班级移除学生API, 学生ID:', studentId, '班级ID:', classId)
  
  return axios.delete('/api/teacher/students/remove-from-class', {
    data: { studentId, classId },
    headers: {
      'Authorization': authToken,
      'Content-Type': 'application/json'
    }
  })
}

// 添加学生到课程
export const addStudentToCourse = (studentId: number, courseId: number) => {
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
  
  console.log('调用添加学生到课程API, 学生ID:', studentId, '课程ID:', courseId)
  
  return axios.post('/api/teacher/students/add-to-course', 
    { studentId, courseId },
    {
      headers: {
        'Authorization': authToken,
        'Content-Type': 'application/json'
      }
    }
  )
}

// 从课程移除学生
export const removeStudentFromCourse = (studentId: number, courseId: number) => {
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
  
  console.log('调用从课程移除学生API, 学生ID:', studentId, '课程ID:', courseId)
  
  return axios.delete('/api/teacher/students/remove-from-course', {
    data: { studentId, courseId },
    headers: {
      'Authorization': authToken,
      'Content-Type': 'application/json'
    }
  })
}

// 处理选课申请
export const processEnrollmentRequest = (requestId: number, approved: boolean, comment?: string) => {
  return request.post('/api/teacher/students/process-enrollment-request', {
    requestId,
    approved,
    comment
  })
}

// 获取选课申请列表
export const getEnrollmentRequests = (params: {
  current: number
  size: number
  courseId?: number
}) => {
  return request.get('/api/teacher/students/enrollment-requests', { params })
}

// 创建学生账户
export const createStudent = (data: {
  username: string
  password?: string
  email?: string
  realName?: string
}) => {
  return request.post('/api/teacher/students/create', data)
}

// 搜索学生，用于添加学生到课程或班级
export const searchStudents = (keyword?: string) => {
  console.log('调用searchStudents API, 关键词:', keyword);
  
  // 获取token和用户ID
  const token = localStorage.getItem('user-token') || localStorage.getItem('token');
  const userInfo = localStorage.getItem('user-info');
  let userId = '';
  
  if (userInfo) {
    try {
      const userObj = JSON.parse(userInfo);
      userId = userObj.id || '';
    } catch (e) {
      console.error('解析用户信息失败:', e);
    }
  }
  
  // 使用简化的token格式
  const authToken = userId ? `Bearer token-${userId}` : (token ? `Bearer ${token}` : '');
  
  return axios.get<ApiResult<StudentSearchResult[]>>('/api/teacher/students/search', { 
    params: { keyword },
    headers: {
      'Authorization': authToken
    }
  })
    .then(response => {
      console.log('搜索学生响应:', response);
      return response;
    })
    .catch(error => {
      console.error('搜索学生错误:', error);
      throw error;
    });
}

// 更新学生信息
export const updateStudent = (id: number, data: Student) => {
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
  
  console.log('调用更新学生API, ID:', id)
  
  return axios.put(`/api/teacher/students/${id}`, data, {
    headers: {
      'Authorization': authToken,
      'Content-Type': 'application/json'
    }
  })
}

/**
 * 获取班级信息
 */
export const getClassInfo = async (): Promise<ClassInfo> => {
  try {
    const response = await axios.get('/api/student/classes/info')
    if (response.data && response.data.code === 200) {
      return response.data.data
    } else {
      throw new Error(response.data?.message || '获取班级信息失败')
    }
  } catch (error) {
    console.error('获取班级信息API错误:', error)
    throw error
  }
}

/**
 * 获取班级同学列表
 */
export const getClassMembers = async (): Promise<{classId: number, students: ClassStudent[]}> => {
  try {
    const response = await axios.get('/api/student/classes/members')
    if (response.data && response.data.code === 200) {
      return response.data.data
    } else {
      throw new Error(response.data?.message || '获取班级同学列表失败')
    }
  } catch (error) {
    console.error('获取班级同学列表API错误:', error)
    throw error
  }
}