import request from '@/utils/request'
import type { ApiResponse } from '@/utils/request'

/**
 * 考试相关API
 */
export default {
  /**
   * 获取考试列表
   * @param params 查询参数
   * @returns 考试列表
   */
  getExamList(params: any): Promise<ApiResponse> {
    console.log('调用getExamList API，参数:', params)
    return request({
      url: `/api/teacher/exams`,
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
   * 获取考试详情
   * @param id 考试ID
   * @returns 考试详情
   */
  getExamDetail(id: number): Promise<ApiResponse> {
    return request({
      url: `/api/teacher/exams/${id}`,
      method: 'get'
    })
  },

  /**
   * 创建考试
   * @param data 考试信息
   * @returns 创建结果
   */
  createExam(data: any): Promise<ApiResponse> {
    return request({
      url: '/api/teacher/exams',
      method: 'post',
      data
    })
  },

  /**
   * 更新考试
   * @param id 考试ID
   * @param data 考试信息
   * @returns 更新结果
   */
  updateExam(id: number, data: any): Promise<ApiResponse> {
    return request({
      url: `/api/teacher/exams/${id}`,
      method: 'put',
      data
    })
  },

  /**
   * 删除考试
   * @param id 考试ID
   * @returns 删除结果
   */
  deleteExam(id: number): Promise<ApiResponse> {
    return request({
      url: `/api/teacher/exams/${id}`,
      method: 'delete'
    })
  },

  /**
   * 发布考试
   * @param id 考试ID
   * @returns 发布结果
   */
  publishExam(id: number): Promise<ApiResponse> {
    return request({
      url: `/api/teacher/exams/${id}/publish`,
      method: 'put'
    })
  },

  /**
   * 取消发布考试
   * @param id 考试ID
   * @returns 取消发布结果
   */
  unpublishExam(id: number): Promise<ApiResponse> {
    return request({
      url: `/api/teacher/exams/${id}`,
      method: 'put',
      data: {
        status: 0 // 将状态设置为0，表示未发布
      }
    })
  },

  /**
   * 自动组卷
   * @param data 组卷配置
   * @returns 组卷结果
   */
  generateExamPaper(data: any): Promise<ApiResponse> {
    return request({
      url: '/api/teacher/exams/generate-paper',
      method: 'post',
      data
    })
  },

  /**
   * 手动选题
   * @param examId 考试ID
   * @param questionIds 题目ID列表
   * @param scores 分值列表
   * @returns 选题结果
   */
  selectQuestions(examId: number, questionIds: number[], scores: number[]): Promise<ApiResponse> {
    return request({
      url: `/api/teacher/exams/${examId}/select-questions`,
      method: 'post',
      data: {
        examId: examId,
        questionIds: questionIds,
        scores: scores
      }
    })
  },

  /**
   * 获取课程题目列表
   * @param params 查询参数
   * @returns 题目列表
   */
  getCourseQuestionList(params: any): Promise<ApiResponse> {
    console.log('调用getCourseQuestionList API，参数:', params)
    return request({
      url: `/api/teacher/questions/list`,
      method: 'get',
      params: {
        courseId: params.courseId,
        questionType: params.questionType,
        difficulty: params.difficulty,
        knowledgePoint: params.knowledgePoint,
        onlyKnowledgePoints: params.onlyKnowledgePoints,
        createdBy: params.createdBy, // 添加创建者筛选
        pageSize: params.pageSize,
        pageNum: params.current || 1 // 使用pageNum而不是current
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
      url: `/api/teacher/exams/questions/knowledge-points`,
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
      url: `/api/teacher/exams/questions`,
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
   * 获取考试题目 (学生端)
   * @param examId 考试ID
   * @returns 考试题目列表
   */
  getExamQuestions(examId: number): Promise<ApiResponse> {
    return request({
      url: `/api/student/exams/${examId}/questions`,
      method: 'get'
    })
  },

  /**
   * 获取考试题目 (教师端)
   * @param examId 考试ID
   * @returns 考试题目列表（包含答案）
   */
  getTeacherExamQuestions(examId: number): Promise<ApiResponse> {
    return request({
      url: `/api/teacher/exams/${examId}/questions`,
      method: 'get'
    })
  },

  /**
   * 保存考试答案
   * @param examId 考试ID
   * @param questionId 题目ID
   * @param answer 答案内容
   * @returns 保存结果
   */
  saveExamAnswer(examId: number, questionId: number, answer: any): Promise<ApiResponse> {
    return request({
      url: `/api/student/exams/${examId}/answer`,
      method: 'post',
      data: {
        examId,
        questionId,
        answer
      }
    })
  },


} 