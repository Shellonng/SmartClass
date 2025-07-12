import request from '@/utils/request'

// 组卷相关接口
export interface PaperGenerationRequest {
  courseId: number
  knowledgePoints: string[]
  difficulty: 'EASY' | 'MEDIUM' | 'HARD'
  questionCount: number
  questionTypes: Record<string, number>
  duration: number
  totalScore: number
  additionalRequirements?: string
}

export interface GeneratedQuestion {
  questionText: string
  questionType: string
  options?: string[]
  correctAnswer: string
  score: number
  knowledgePoint: string
  difficulty: string
  explanation: string
}

export interface PaperGenerationResponse {
  title: string
  questions: GeneratedQuestion[]
  status: string
  taskId?: string
  errorMessage?: string
}

// 自动批改相关接口
export interface StudentAnswer {
  questionId: number
  questionText: string
  questionType: string
  correctAnswer: string
  studentAnswer: string
  totalScore: number
}

export interface AutoGradingRequest {
  assignmentId: number
  studentId: number
  answers: StudentAnswer[]
  gradingType: 'OBJECTIVE' | 'SUBJECTIVE' | 'MIXED'
  gradingCriteria?: string
}

export interface GradingResult {
  questionId: number
  isCorrect: boolean
  score: number
  totalScore: number
  comment: string
  errorType?: string
  suggestion?: string
}

export interface AutoGradingResponse {
  results: GradingResult[]
  totalScore: number
  earnedScore: number
  percentage: number
  overallComment: string
  status: string
  taskId?: string
  studentId?: number
}

// 教师端组卷API
export const teacherPaperApi = {
  // 智能组卷
  generatePaper: (data: PaperGenerationRequest) => 
    request.post<PaperGenerationResponse>('/api/teacher/paper/generate', data, {
      timeout: 480000 // 显式设置480秒超时（8分钟）
    }),
  
  // 异步智能组卷
  generatePaperAsync: (data: PaperGenerationRequest) => 
    request.post<string>('/api/teacher/paper/generate-async', data, {
      timeout: 480000 // 显式设置480秒超时（8分钟）
    }),
  
  // 查询组卷任务状态
  getTaskStatus: (taskId: string) => 
    request.get(`/api/teacher/paper/task/${taskId}`),
  
  // 预览组卷参数
  previewPaper: (data: PaperGenerationRequest) => 
    request.post('/api/teacher/paper/preview', data)
}

// 教师端批改API
export const teacherGradingApi = {
  // 智能批改
  autoGrade: (data: AutoGradingRequest) => 
    request.post<AutoGradingResponse>('/api/teacher/grading/auto-grade', data),
  
  // 批量智能批改
  batchAutoGrade: (data: {
    assignmentId: number
    studentSubmissions: Array<{
      studentId: number
      answers: StudentAnswer[]
    }>
    gradingType: string
    gradingCriteria?: string
  }) => request.post('/api/teacher/grading/batch-grade', data),
  
  // 查询批改任务状态
  getTaskStatus: (taskId: string) => 
    request.get(`/api/teacher/grading/task/${taskId}`),
  
  // 获取批改统计
  getStatistics: (assignmentId: number) => 
    request.get(`/api/teacher/grading/statistics/${assignmentId}`),
  
  // 设置批改标准
  setCriteria: (data: {
    assignmentId: number
    criteria: string
    parameters?: Record<string, any>
  }) => request.post('/api/teacher/grading/criteria', data)
}

// 学生端组卷API
export const studentPaperApi = {
  // 生成个性化练习
  generatePractice: (data: {
    courseId: number
    weakKnowledgePoints: string[]
    abilityLevel: 'LOW' | 'MEDIUM' | 'HIGH'
    questionCount: number
    preferredQuestionTypes: Record<string, number>
  }) => request.post<PaperGenerationResponse>('/api/student/paper/generate-practice', data),
  
  // 生成错题重练
  generateRetry: (data: {
    courseId: number
    errorKnowledgePoints: string[]
    errorTypes: string[]
    retryCount: number
  }) => request.post<PaperGenerationResponse>('/api/student/paper/generate-retry', data),
  
  // 智能推荐练习
  recommendPractice: (courseId: number, count: number = 10) => 
    request.get<PaperGenerationResponse>(`/api/student/paper/recommend/${courseId}?count=${count}`),
  
  // 获取练习历史
  getPracticeHistory: (page: number = 1, size: number = 10) => 
    request.get(`/api/student/paper/history?page=${page}&size=${size}`)
}

// 通用Dify API
export const difyApi = {
  // 获取任务状态（通用）
  getTaskStatus: (taskId: string, appType: string) => 
    request.get(`/api/dify/task/${taskId}?appType=${appType}`)
} 

// 默认导出所有API
export default {
  teacherPaperApi,
  teacherGradingApi,
  studentPaperApi,
  difyApi
} 