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

export interface Course {
  id: number
  name: string
  description: string
  studentCount: number
  status: 'active' | 'inactive'
  createTime: string
}

export interface Assignment {
  id: number
  title: string
  description: string
  dueDate: string
  status: 'draft' | 'published' | 'closed'
  submissionCount: number
  totalStudents: number
}

export interface Class {
  id: number
  name: string
  description: string
  studentCount: number
  status: 'active' | 'inactive'
  createTime: string
}

export interface Grade {
  id: number
  studentName: string
  assignmentTitle: string
  score: number
  maxScore: number
  submitTime: string
  gradeTime?: string
  status: 'submitted' | 'graded'
}

// 获取教师仪表板数据
export const getDashboardData = () => {
  return axios.get('/api/teacher/dashboard')
}

// 获取教学统计数据
export const getTeachingStatistics = (timeRange?: string) => {
  return axios.get('/api/teacher/dashboard/statistics', { params: { timeRange } })
}

// 获取待处理任务
export const getPendingTasks = () => {
  return axios.get('/api/teacher/dashboard/pending-tasks')
}

// 获取课程概览
export const getCourseOverview = () => {
  return axios.get('/api/teacher/dashboard/course-overview')
}

// 获取学生表现分析
export const getStudentPerformance = (params?: { courseId?: number; classId?: number }) => {
  return axios.get('/api/teacher/dashboard/student-performance', { params })
}

// 获取教学建议
export const getTeachingSuggestions = () => {
  return axios.get('/api/teacher/dashboard/teaching-suggestions')
}

// 获取近期活动
export const getRecentActivities = (limit: number = 10) => {
  return axios.get('/api/teacher/dashboard/recent-activities', { params: { limit } })
}

// 学生管理
export const getStudents = (params?: { page?: number; size?: number; keyword?: string }) => {
  return axios.get<Student[]>('/api/teacher/students', { params })
}

export const getStudentDetail = (studentId: number) => {
  return axios.get(`/api/teacher/students/${studentId}`)
}

export const updateStudent = (studentId: number, data: Partial<Student>) => {
  return axios.put(`/api/teacher/students/${studentId}`, data)
}

// 课程管理
export const getCourses = (params?: { page?: number; size?: number; keyword?: string }) => {
  return axios.get<Course[]>('/api/teacher/courses', { params })
}

export const createCourse = (data: Omit<Course, 'id' | 'createTime'>) => {
  return axios.post('/api/teacher/courses', data)
}

export const updateCourse = (courseId: number, data: Partial<Course>) => {
  return axios.put(`/api/teacher/courses/${courseId}`, data)
}

export const deleteCourse = (courseId: number) => {
  return axios.delete(`/api/teacher/courses/${courseId}`)
}

export const getCourseDetail = (courseId: number) => {
  return axios.get(`/api/teacher/courses/${courseId}`)
}

// 作业管理
export const getAssignments = (params?: { page?: number; size?: number; courseId?: number; status?: string }) => {
  return axios.get<Assignment[]>('/api/teacher/tasks', { params })
}

export const createAssignment = (data: Omit<Assignment, 'id' | 'submissionCount' | 'totalStudents'>) => {
  return axios.post('/api/teacher/tasks', data)
}

export const updateAssignment = (assignmentId: number, data: Partial<Assignment>) => {
  return axios.put(`/api/teacher/tasks/${assignmentId}`, data)
}

export const deleteAssignment = (assignmentId: number) => {
  return axios.delete(`/api/teacher/tasks/${assignmentId}`)
}

export const publishAssignment = (assignmentId: number) => {
  return axios.post(`/api/teacher/tasks/${assignmentId}/publish`)
}

export const getAssignmentDetail = (assignmentId: number) => {
  return axios.get(`/api/teacher/tasks/${assignmentId}`)
}

export const getAssignmentSubmissions = (assignmentId: number, params?: { page?: number; size?: number }) => {
  return axios.get(`/api/teacher/tasks/${assignmentId}/submissions`, { params })
}

// 班级管理
export const getClasses = (params?: { page?: number; size?: number; keyword?: string }) => {
  return axios.get<Class[]>('/api/teacher/classes', { params })
}

export const createClass = (data: Omit<Class, 'id' | 'createTime'>) => {
  return axios.post('/api/teacher/classes', data)
}

export const updateClass = (classId: number, data: Partial<Class>) => {
  return axios.put(`/api/teacher/classes/${classId}`, data)
}

export const deleteClass = (classId: number) => {
  return axios.delete(`/api/teacher/classes/${classId}`)
}

export const getClassDetail = (classId: number) => {
  return axios.get(`/api/teacher/classes/${classId}`)
}

export const getClassStudents = (classId: number, params?: { page?: number; size?: number; keyword?: string }) => {
  return axios.get(`/api/teacher/classes/${classId}/students`, { params })
}

export const addStudentsToClass = (classId: number, studentIds: number[]) => {
  return axios.post(`/api/teacher/classes/${classId}/students`, { studentIds })
}

export const removeStudentFromClass = (classId: number, studentId: number) => {
  return axios.delete(`/api/teacher/classes/${classId}/students/${studentId}`)
}

// 成绩管理
export const getGrades = (params?: { page?: number; size?: number; courseId?: number; classId?: number; taskId?: number }) => {
  return axios.get<Grade[]>('/api/teacher/grades', { params })
}

export const createGrade = (data: { studentId: number; taskId: number; score: number; feedback?: string }) => {
  return axios.post('/api/teacher/grades', data)
}

export const batchCreateGrades = (data: { taskId: number; grades: Array<{ studentId: number; score: number; feedback?: string }> }) => {
  return axios.post('/api/teacher/grades/batch', data)
}

export const updateGrade = (gradeId: number, data: { score: number; feedback?: string }) => {
  return axios.put(`/api/teacher/grades/${gradeId}`, data)
}

export const getGradeStatistics = (params?: { courseId?: number; classId?: number; timeRange?: string }) => {
  return axios.get('/api/teacher/grades/statistics', { params })
}

export const exportGrades = (params?: { courseId?: number; classId?: number; format?: string }) => {
  return axios.get('/api/teacher/grades/export', { params, responseType: 'blob' })
}

// 资源管理
export const getResources = (params?: { page?: number; size?: number; category?: string; keyword?: string }) => {
  return axios.get('/api/teacher/resources', { params })
}

export const uploadResource = (data: FormData) => {
  return axios.post('/api/teacher/resources/upload', data, {
    headers: {
      'Content-Type': 'multipart/form-data'
    }
  })
}

export const updateResource = (resourceId: number, data: { name?: string; description?: string; category?: string }) => {
  return axios.put(`/api/teacher/resources/${resourceId}`, data)
}

export const deleteResource = (resourceId: number) => {
  return axios.delete(`/api/teacher/resources/${resourceId}`)
}

export const getResourceDetail = (resourceId: number) => {
  return axios.get(`/api/teacher/resources/${resourceId}`)
}

// AI工具相关接口
export const intelligentGrading = (data: { taskId: number; submissionId: number; gradingCriteria?: string }) => {
  return axios.post('/api/teacher/ai/grade', data)
}

export const batchIntelligentGrading = (data: { taskId: number; submissionIds: number[]; gradingCriteria?: string }) => {
  return axios.post('/api/teacher/ai/batch-grade', data)
}

export const generateRecommendations = (data: { studentId: number; courseId: number; analysisType?: string }) => {
  return axios.post('/api/teacher/ai/recommend', data)
}

export const analyzeStudentAbility = (data: { studentId: number; analysisDimensions: string[]; timeRange?: string }) => {
  return axios.post('/api/teacher/ai/ability-analysis', data)
}

export const generateKnowledgeGraph = (data: { courseId: number; chapterCount: number; autoGenerate?: boolean }) => {
  return axios.post('/api/teacher/ai/knowledge-graph', data)
}

export const generateQuestions = (data: { knowledgePoints: string[]; questionType: string; questionCount: number; difficulty?: string }) => {
  return axios.post('/api/teacher/ai/generate-questions', data)
}

export const optimizeLearningPath = (data: { studentId: number; targetSkills: string[]; timeConstraint?: number }) => {
  return axios.post('/api/teacher/ai/optimize-path', data)
}

export const analyzeClassroomPerformance = (data: { classId: number; timeRange: string; analysisType?: string }) => {
  return axios.post('/api/teacher/ai/classroom-analysis', data)
}

export const generateTeachingSuggestions = (data: { courseId: number; studentGroup: string; teachingGoals?: string[] }) => {
  return axios.post('/api/teacher/ai/teaching-suggestions', data)
}

export const analyzeDocument = (file: File, analysisType: string, courseId?: number) => {
  const formData = new FormData()
  formData.append('file', file)
  formData.append('analysisType', analysisType)
  if (courseId) {
    formData.append('courseId', courseId.toString())
  }
  return axios.post('/api/teacher/ai/analyze-document', formData, {
    headers: {
      'Content-Type': 'multipart/form-data'
    }
  })
}

export const getAIAnalysisHistory = (params?: { type?: string; courseId?: number; page?: number; size?: number }) => {
  return axios.get('/api/teacher/ai/analysis-history', { params })
}