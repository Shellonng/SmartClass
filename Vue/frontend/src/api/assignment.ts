import request from '@/utils/request'
import type { ApiResponse } from '@/utils/request'

/**
 * 任务相关API
 */
export default {
  /**
   * 获取教师课程列表
   * @returns 课程列表
   */
  getTeacherCourses(): Promise<ApiResponse> {
    return request({
      url: `/api/teacher/assignments/courses`,
      method: 'get'
    })
  },

  /**
   * 获取任务列表
   * @param params 查询参数
   * @returns 任务列表
   */
  getAssignmentList(params: any): Promise<ApiResponse> {
    console.log('调用getAssignmentList API，参数:', params)
    return request({
      url: `/api/teacher/assignments`,
      method: 'get',
      params: {
        courseId: params.courseId,
        userId: params.userId,
        status: params.status,
        keyword: params.keyword,
        current: params.current,
        pageSize: params.pageSize
      }
    })
  },

  /**
   * 获取任务详情
   * @param id 任务ID
   * @returns 任务详情
   */
  getAssignmentDetail(id: number): Promise<ApiResponse> {
    return request({
      url: `/api/teacher/assignments/${id}`,
      method: 'get'
    })
  },

  /**
   * 创建任务
   * @param data 任务信息
   * @returns 创建结果
   */
  createAssignment(data: any): Promise<ApiResponse> {
    return request({
      url: '/api/teacher/assignments',
      method: 'post',
      data
    })
  },

  /**
   * 更新任务
   * @param id 任务ID
   * @param data 任务信息
   * @returns 更新结果
   */
  updateAssignment(id: number, data: any): Promise<ApiResponse> {
    return request({
      url: `/api/teacher/assignments/${id}`,
      method: 'put',
      data
    })
  },

  /**
   * 删除任务
   * @param id 任务ID
   * @returns 删除结果
   */
  deleteAssignment(id: number): Promise<ApiResponse> {
    return request({
      url: `/api/teacher/assignments/${id}`,
      method: 'delete'
    })
  },

  /**
   * 发布任务
   * @param id 任务ID
   * @returns 发布结果
   */
  publishAssignment(id: number): Promise<ApiResponse> {
    return request({
      url: `/api/teacher/assignments/${id}/publish`,
      method: 'put'
    })
  },

  /**
   * 取消发布任务
   * @param id 任务ID
   * @returns 取消发布结果
   */
  unpublishAssignment(id: number): Promise<ApiResponse> {
    return request({
      url: `/api/teacher/assignments/${id}/unpublish`,
      method: 'put'
    })
  },

  /**
   * 自动组卷
   * @param data 组卷配置
   * @returns 组卷结果
   */
  generateAssignmentPaper(data: any): Promise<ApiResponse> {
    return request({
      url: '/api/teacher/assignments/generate-paper',
      method: 'post',
      data
    })
  },

  /**
   * 手动选题
   * @param assignmentId 任务ID
   * @param questionIds 题目ID列表
   * @param scores 分值列表
   * @returns 选题结果
   */
  selectQuestions(assignmentId: number, questionIds: number[], scores: number[]): Promise<ApiResponse> {
    return request({
      url: `/api/teacher/assignments/${assignmentId}/select-questions`,
      method: 'post',
      data: {
        assignmentId: assignmentId,
        questionIds: questionIds,
        scores: scores
      }
    })
  },

  /**
   * 获取知识点列表
   * @param courseId 课程ID
   * @param createdBy 创建者ID
   * @returns 知识点列表
   */
  getKnowledgePoints(courseId: number, createdBy: number): Promise<ApiResponse<string[]>> {
    console.log('调用getKnowledgePoints API，参数:', { courseId, createdBy })
    return request({
      url: `/api/teacher/assignments/questions/knowledge-points`,
      method: 'get',
      params: {
        courseId: courseId,
        createdBy: createdBy
      }
    })
  },
  
  /**
   * 获取题目（按题型分类）
   * @param courseId 课程ID
   * @param questionType 题目类型（可选）
   * @param difficulty 难度（可选）
   * @param knowledgePoint 知识点（可选）
   * @param createdBy 创建者ID（可选）
   * @param keyword 关键词（可选）
   * @returns 题目列表（按题型分类）
   */
  getQuestionsByType(
    courseId: number, 
    questionType?: string, 
    difficulty?: number, 
    knowledgePoint?: string, 
    createdBy?: number,
    keyword?: string
  ): Promise<ApiResponse> {
    console.log('调用getQuestionsByType API，参数:', { 
      courseId, 
      questionType, 
      difficulty, 
      knowledgePoint, 
      createdBy,
      keyword
    })
    return request({
      url: `/api/teacher/assignments/questions`,
      method: 'get',
      params: {
        courseId,
        questionType,
        difficulty,
        knowledgePoint,
        createdBy,
        keyword
      }
    })
  },

  /**
   * 获取任务题目 (学生端)
   * @param assignmentId 任务ID
   * @returns 任务题目列表
   */
  getAssignmentQuestions(assignmentId: number): Promise<any[]> {
    return request({
      url: `/api/student/assignments/${assignmentId}/questions`,
      method: 'get'
    }).then(response => response.data || [])
  },

  /**
   * 获取任务题目 (教师端)
   * @param assignmentId 任务ID
   * @returns 任务题目列表（包含答案）
   */
  getTeacherAssignmentQuestions(assignmentId: number): Promise<ApiResponse> {
    return request({
      url: `/api/teacher/assignments/${assignmentId}/questions`,
      method: 'get'
    })
  },
  
  /**
   * 获取任务提交记录列表
   * @param assignmentId 任务ID
   * @param params 查询参数
   * @returns 提交记录列表
   */
  getAssignmentSubmissions(assignmentId: number, params?: any): Promise<ApiResponse> {
    return request({
      url: `/api/teacher/assignments/${assignmentId}/submissions`,
      method: 'get',
      params: {
        current: params?.current,
        pageSize: params?.pageSize,
        status: params?.status
      }
    })
  },
  
  /**
   * 批改任务提交
   * @param submissionId 提交记录ID
   * @param data 批改信息
   * @returns 批改结果
   */
  gradeSubmission(submissionId: number, data: any): Promise<ApiResponse> {
    return request({
      url: `/api/teacher/submissions/${submissionId}/grade`,
      method: 'put',
      data
    })
  },
  
  /**
   * 删除任务提交记录
   * @param submissionId 提交记录ID
   * @returns 删除结果
   */
  deleteSubmission(submissionId: number): Promise<ApiResponse> {
    return request({
      url: `/api/teacher/submissions/${submissionId}`,
      method: 'delete'
    })
  },

  /**
   * 获取任务详情（学生端）
   * @param id 任务ID
   * @returns 任务详情
   */
  getStudentAssignmentDetail(id: number): Promise<ApiResponse> {
    return request({
      url: `/api/student/assignments/${id}`,
      method: 'get'
    })
  },

  /**
   * 提交任务答案（学生端）
   * @param assignmentId 任务ID
   * @param answers 答案数据
   * @returns 提交结果
   */
  submitAssignmentAnswers(assignmentId: number, answers: any): Promise<ApiResponse> {
    return request({
      url: `/api/student/assignments/${assignmentId}/submit`,
      method: 'post',
      data: answers
    })
  },

  /**
   * 提交考试答案（学生端）
   * @param examId 考试ID
   * @param answers 答案数据
   * @returns 提交结果
   */
  submitExamAnswers(examId: number, answers: any): Promise<ApiResponse> {
    return request({
      url: `/api/student/exams/${examId}/submit`,
      method: 'post',
      data: answers
    })
  },

  /**
   * 提交文件任务（学生端）
   * @param assignmentId 任务ID
   * @param formData 包含文件的FormData
   * @returns 提交结果
   */
  submitAssignmentFile(assignmentId: number, formData: FormData): Promise<ApiResponse> {
    return request({
      url: `/api/student/assignments/${assignmentId}/submit-file`,
      method: 'post',
      data: formData,
      headers: {
        'Content-Type': 'multipart/form-data'
      }
    })
  },

  /**
   * 获取学生提交记录（学生端）
   * @param assignmentId 任务ID
   * @returns 提交记录
   */
  getStudentSubmission(assignmentId: number): Promise<ApiResponse> {
    return request({
      url: `/api/student/assignments/${assignmentId}/submission`,
      method: 'get'
    })
  },

  /**
   * 获取任务列表
   * @param params 查询参数
   * @returns 任务列表
   */
  getAssignmentListStudent(params: any): Promise<ApiResponse> {
    return request({
      url: '/api/student/assignments',
      method: 'get',
      params
    })
  },

  /**
   * 获取任务详情
   * @param id 任务ID
   * @returns 任务详情
   */
  getAssignmentDetailStudent(id: number): Promise<ApiResponse> {
    return request({
      url: `/api/student/assignments/${id}`,
      method: 'get'
    })
  },

  /**
   * 获取任务题目
   * @param id 任务ID
   * @returns 任务题目
   */
  getAssignmentQuestionsStudent(id: number): Promise<ApiResponse> {
    return request({
      url: `/api/student/assignments/${id}/questions`,
      method: 'get'
    })
  },

  /**
   * 提交任务/考试答案
   * @param id 任务ID
   * @param data 答案数据
   * @returns 提交结果
   */
  submitAssignment(id: number, data: any): Promise<ApiResponse> {
    return request({
      url: `/api/student/assignments/${id}/submit`,
      method: 'post',
      data
    })
  },

  /**
   * 获取已提交的任务列表
   * @param params 查询参数
   * @returns 已提交的任务列表
   */
  getSubmittedAssignments(params: any): Promise<ApiResponse> {
    return request({
      url: '/api/student/submitted-assignments',
      method: 'get',
      params
    })
  },

  /**
   * 保存单题答案
   * @param id 任务ID
   * @param questionId 题目ID
   * @param data 答案数据
   * @returns 保存结果
   */
  saveQuestionAnswer(id: number, questionId: number, data: any): Promise<ApiResponse> {
    return request({
      url: `/api/student/assignments/${id}/questions/${questionId}/save`,
      method: 'post',
      data
    })
  }
} 