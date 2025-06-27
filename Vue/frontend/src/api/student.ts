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