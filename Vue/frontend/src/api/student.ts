import axios from 'axios'

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

// 获取学生仪表板数据
export const getDashboardData = () => {
  return axios.get('/student/dashboard')
}

// 获取待完成作业列表
export const getPendingAssignments = () => {
  return axios.get<Assignment[]>('/student/assignments/pending')
}

// 获取所有作业列表
export const getAssignments = (params?: { page?: number; size?: number; status?: string }) => {
  return axios.get<Assignment[]>('/student/assignments', { params })
}

// 提交作业
export const submitAssignment = (assignmentId: number, data: FormData) => {
  return axios.post(`/student/assignments/${assignmentId}/submit`, data, {
    headers: {
      'Content-Type': 'multipart/form-data'
    }
  })
}

// 获取成绩列表
export const getGrades = (params?: { page?: number; size?: number; courseId?: number }) => {
  return axios.get<Grade[]>('/student/grades', { params })
}

// 获取课程列表
export const getCourses = () => {
  return axios.get<Course[]>('/student/courses')
}

// 获取课程详情
export const getCourseDetail = (courseId: number) => {
  return axios.get(`/student/courses/${courseId}`)
}

// 获取学习计划
export const getStudyPlans = () => {
  return axios.get<StudyPlan[]>('/student/study-plans')
}

// 创建学习计划
export const createStudyPlan = (data: Omit<StudyPlan, 'id'>) => {
  return axios.post('/student/study-plans', data)
}

// 更新学习计划
export const updateStudyPlan = (id: number, data: Partial<StudyPlan>) => {
  return axios.put(`/student/study-plans/${id}`, data)
}

// 删除学习计划
export const deleteStudyPlan = (id: number) => {
  return axios.delete(`/student/study-plans/${id}`)
}

// 获取学习资源
export const getLearningResources = (params?: { category?: string; type?: string }) => {
  return axios.get<LearningResource[]>('/student/resources', { params })
}

// 获取学习进度
export const getLearningProgress = () => {
  return axios.get('/student/progress')
}

// 更新学习进度
export const updateLearningProgress = (courseId: number, lessonId: number) => {
  return axios.post(`/student/progress/${courseId}/${lessonId}`)
}