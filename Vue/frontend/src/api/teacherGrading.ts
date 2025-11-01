import request from '@/utils/request'

export interface ApiResponse<T> {
  code: number
  data: T
  message: string
}

export interface AutoGradingResponse {
  submissionId: number
  status: string
  totalScore: number
  earnedScore: number
  overallComment: string
  results: GradingResult[]
  errorMessage?: string
  taskId?: string
}

export interface GradingResult {
  questionId: number
  questionType: string
  isCorrect: boolean
  score: number
  maxScore: number
  comment: string
}

export interface AutoGradingRequest {
  submissionId?: number
  assignmentId: number
  studentId?: number
  gradingCriteria?: string
  maxScore?: number
  questions: {
    questionId: number
    questionText: string
    questionType: string
    correctAnswer: string
    totalScore?: number
  }[]
  studentAnswers: {
    questionId: number
    studentAnswer: string
  }[]
}

export interface StatisticsResponse {
  assignmentId: number
  totalSubmissions: number
  gradedSubmissions: number
  averageScore: number
  highestScore: number
  lowestScore: number
  scoreDistribution: Record<string, number>
  commonErrors: string[]
  knowledgePointMastery: Record<string, number>
}

export default {
  /**
   * 智能批改单个作业
   * @param data 批改请求数据
   */
  autoGrade(data: AutoGradingRequest) {
    return request.post<ApiResponse<AutoGradingResponse>>('/api/teacher/grading/single', data)
  },

  /**
   * 批量智能批改
   * @param data 批量批改请求数据
   */
  batchAutoGrade(data: any) {
    return request.post<ApiResponse<AutoGradingResponse[]>>('/api/teacher/grading/batch', data)
  },

  /**
   * 获取批改进度
   * @param taskId 任务ID
   */
  getGradingProgress(taskId: string) {
    return request.get<ApiResponse<any>>(`/api/teacher/grading/progress/${taskId}`)
  },

  /**
   * 获取批改统计
   * @param assignmentId 作业ID
   */
  getStatistics(assignmentId: number) {
    return request.get<ApiResponse<StatisticsResponse>>(`/api/teacher/grading/statistics/${assignmentId}`)
  },

  /**
   * 设置批改标准
   * @param data 批改标准数据
   */
  setGradingCriteria(data: any) {
    return request.post<ApiResponse<string>>('/api/teacher/grading/criteria', data)
  },
  
  /**
   * 获取教师关联的作业及提交情况
   * @returns 教师关联的作业列表和提交情况
   */
  getTeacherAssignments() {
    return request.get<ApiResponse<any[]>>('/api/teacher/grading/teacher-assignments')
  }
} 