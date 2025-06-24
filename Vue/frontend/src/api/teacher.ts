import axios from 'axios'

// 教师相关接口
export interface Student {
  id: number
  username: string
  realName: string
  email: string
  avatar?: string
  enrollDate: string
  status: 'active' | 'inactive'
}

export interface Assignment {
  id: number
  title: string
  description: string
  courseId: number
  courseName: string
  dueDate: string
  maxScore: number
  status: 'draft' | 'published' | 'closed'
  submissionCount: number
  totalStudents: number
}

export interface Submission {
  id: number
  studentId: number
  studentName: string
  assignmentId: number
  assignmentTitle: string
  submitTime: string
  content: string
  attachments?: string[]
  score?: number
  feedback?: string
  status: 'submitted' | 'graded'
}

export interface Course {
  id: number
  name: string
  description: string
  teacherId: number
  studentCount: number
  status: 'active' | 'inactive'
  createTime: string
}

export interface Lesson {
  id: number
  title: string
  content: string
  courseId: number
  orderIndex: number
  duration: number
  videoUrl?: string
  attachments?: string[]
}

// 获取教师仪表板数据
export const getDashboardData = () => {
  return axios.get('/teacher/dashboard')
}

// 学生管理
export const getStudents = (params?: { page?: number; size?: number; keyword?: string }) => {
  return axios.get<Student[]>('/teacher/students', { params })
}

export const getStudentDetail = (studentId: number) => {
  return axios.get(`/teacher/students/${studentId}`)
}

export const updateStudent = (studentId: number, data: Partial<Student>) => {
  return axios.put(`/teacher/students/${studentId}`, data)
}

// 课程管理
export const getCourses = () => {
  return axios.get<Course[]>('/teacher/courses')
}

export const createCourse = (data: Omit<Course, 'id' | 'teacherId' | 'studentCount' | 'createTime'>) => {
  return axios.post('/teacher/courses', data)
}

export const updateCourse = (courseId: number, data: Partial<Course>) => {
  return axios.put(`/teacher/courses/${courseId}`, data)
}

export const deleteCourse = (courseId: number) => {
  return axios.delete(`/teacher/courses/${courseId}`)
}

// 课程内容管理
export const getLessons = (courseId: number) => {
  return axios.get<Lesson[]>(`/teacher/courses/${courseId}/lessons`)
}

export const createLesson = (courseId: number, data: Omit<Lesson, 'id' | 'courseId'>) => {
  return axios.post(`/teacher/courses/${courseId}/lessons`, data)
}

export const updateLesson = (lessonId: number, data: Partial<Lesson>) => {
  return axios.put(`/teacher/lessons/${lessonId}`, data)
}

export const deleteLesson = (lessonId: number) => {
  return axios.delete(`/teacher/lessons/${lessonId}`)
}

// 作业管理
export const getAssignments = (params?: { page?: number; size?: number; courseId?: number }) => {
  return axios.get<Assignment[]>('/teacher/assignments', { params })
}

export const createAssignment = (data: Omit<Assignment, 'id' | 'submissionCount' | 'totalStudents'>) => {
  return axios.post('/teacher/assignments', data)
}

export const updateAssignment = (assignmentId: number, data: Partial<Assignment>) => {
  return axios.put(`/teacher/assignments/${assignmentId}`, data)
}

export const deleteAssignment = (assignmentId: number) => {
  return axios.delete(`/teacher/assignments/${assignmentId}`)
}

// 作业提交管理
export const getSubmissions = (assignmentId: number) => {
  return axios.get<Submission[]>(`/teacher/assignments/${assignmentId}/submissions`)
}

export const gradeSubmission = (submissionId: number, data: { score: number; feedback?: string }) => {
  return axios.post(`/teacher/submissions/${submissionId}/grade`, data)
}

// 统计数据
export const getStatistics = () => {
  return axios.get('/teacher/statistics')
}

// 文件上传
export const uploadFile = (file: File, type: 'lesson' | 'assignment' | 'avatar') => {
  const formData = new FormData()
  formData.append('file', file)
  formData.append('type', type)
  
  return axios.post('/teacher/upload', formData, {
    headers: {
      'Content-Type': 'multipart/form-data'
    }
  })
}