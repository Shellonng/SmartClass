import request from '@/utils/request'
import type { ApiResponse } from '@/utils/request'

/**
 * 作业相关API
 */
export default {
  /**
   * 获取作业列表
   * @param params 查询参数
   * @returns 作业列表
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
   * 获取作业详情
   * @param id 作业ID
   * @returns 作业详情
   */
  getAssignmentDetail(id: number): Promise<ApiResponse> {
    return request({
      url: `/api/teacher/assignments/${id}`,
      method: 'get'
    })
  },

  /**
   * 创建作业
   * @param data 作业信息
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
   * 更新作业
   * @param id 作业ID
   * @param data 作业信息
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
   * 删除作业
   * @param id 作业ID
   * @returns 删除结果
   */
  deleteAssignment(id: number): Promise<ApiResponse> {
    return request({
      url: `/api/teacher/assignments/${id}`,
      method: 'delete'
    })
  },

  /**
   * 发布作业
   * @param id 作业ID
   * @returns 发布结果
   */
  publishAssignment(id: number): Promise<ApiResponse> {
    return request({
      url: `/api/teacher/assignments/${id}/publish`,
      method: 'put'
    })
  },

  /**
   * 取消发布作业
   * @param id 作业ID
   * @returns 取消发布结果
   */
  unpublishAssignment(id: number): Promise<ApiResponse> {
    return request({
      url: `/api/teacher/assignments/${id}`,
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
  generateAssignmentPaper(data: any): Promise<ApiResponse> {
    return request({
      url: '/api/teacher/assignments/generate-paper',
      method: 'post',
      data
    })
  },

  /**
   * 手动选题
   * @param assignmentId 作业ID
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
   * 获取作业题目 (学生端)
   * @param assignmentId 作业ID
   * @returns 作业题目列表
   */
  getAssignmentQuestions(assignmentId: number): Promise<ApiResponse> {
    return request({
      url: `/api/student/assignments/${assignmentId}/questions`,
      method: 'get'
    })
  },

  /**
   * 获取作业题目 (教师端)
   * @param assignmentId 作业ID
   * @returns 作业题目列表（包含答案）
   */
  getTeacherAssignmentQuestions(assignmentId: number): Promise<ApiResponse> {
    return request({
      url: `/api/teacher/assignments/${assignmentId}/questions`,
      method: 'get'
    })
  },
  
  /**
   * 获取作业提交记录列表
   * @param assignmentId 作业ID
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
   * 批改作业提交
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
   * 删除作业提交记录
   * @param submissionId 提交记录ID
   * @returns 删除结果
   */
  deleteSubmission(submissionId: number): Promise<ApiResponse> {
    return request({
      url: `/api/teacher/submissions/${submissionId}`,
      method: 'delete'
    })
  }
} 