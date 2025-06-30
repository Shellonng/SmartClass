import axios from 'axios'

// 题目类型
export enum QuestionType {
  SINGLE = 'single',
  MULTIPLE = 'multiple',
  TRUE_FALSE = 'true_false',
  BLANK = 'blank',
  SHORT = 'short',
  CODE = 'code'
}

// 题目类型描述
export const QuestionTypeDesc = {
  [QuestionType.SINGLE]: '单选题',
  [QuestionType.MULTIPLE]: '多选题',
  [QuestionType.TRUE_FALSE]: '判断题',
  [QuestionType.BLANK]: '填空题',
  [QuestionType.SHORT]: '简答题',
  [QuestionType.CODE]: '编程题'
}

// 题目难度级别
export const DifficultyLevels = [
  { value: 1, label: '简单' },
  { value: 2, label: '较简单' },
  { value: 3, label: '中等' },
  { value: 4, label: '较难' },
  { value: 5, label: '困难' }
]

// 题目选项接口
export interface QuestionOption {
  id?: number
  questionId?: number
  optionLabel: string
  optionText: string
}

// 题目图片接口
export interface QuestionImage {
  id?: number
  questionId?: number
  imageUrl: string
  description?: string
  sequence?: number
  uploadTime?: string
}

// 题目接口
export interface Question {
  id?: number
  title: string
  questionType: string
  questionTypeDesc?: string
  difficulty: number
  correctAnswer?: string
  explanation?: string
  knowledgePoint?: string
  courseId: number
  courseName?: string
  chapterId: number
  chapterName?: string
  createdBy?: number
  teacherName?: string
  createTime?: string
  updateTime?: string
  options?: QuestionOption[]
  images?: QuestionImage[]
}

// 分页请求接口
export interface PageRequest {
  pageNum: number
  pageSize: number
}

// 分页响应接口
export interface PageResponse<T> {
  total: number
  pages: number
  pageNum: number
  pageSize: number
  records: T[]
}

// 题目查询请求
export interface QuestionQueryRequest extends PageRequest {
  courseId?: number
  chapterId?: number
  questionType?: string
  difficulty?: number
  knowledgePoint?: string
  keyword?: string
}

/**
 * 添加题目
 * @param question 题目信息
 * @returns 题目ID
 */
export function addQuestion(question: Question) {
  return axios.post<{code: number, data: Question, message: string}>('/api/teacher/questions', question)
    .then(response => {
      // 直接返回响应，无论是否成功，让调用方决定如何处理
      return response;
    })
    .catch(error => {
      console.error("添加题目请求失败:", error);
      // 返回一个统一格式的错误响应
      return {
        data: {
          code: 500,
          data: null as unknown as Question, // 类型转换以满足接口要求
          message: error.response?.data?.message || '网络错误，添加题目失败'
        }
      };
    });
}

/**
 * 更新题目
 * @param question 题目信息
 * @returns 是否成功
 */
export function updateQuestion(question: Question) {
  return axios.put<{code: number, data: Question, message: string}>('/api/teacher/questions', question)
    .then(response => {
      return response;
    })
    .catch(error => {
      console.error("更新题目请求失败:", error);
      return {
        data: {
          code: 500,
          data: null as unknown as Question,
          message: error.response?.data?.message || '网络错误，更新题目失败'
        }
      };
    });
}

/**
 * 删除题目
 * @param id 题目ID
 * @returns 是否成功
 */
export function deleteQuestion(id: number) {
  return axios.delete<boolean>(`/api/teacher/questions/${id}`)
}

/**
 * 获取题目详情
 * @param id 题目ID
 * @returns 题目详情
 */
export function getQuestionDetail(id: number) {
  return axios.get<{code: number, data: Question, message: string}>(`/api/teacher/questions/${id}`)
    .then(response => {
      return response;
    })
    .catch(error => {
      console.error("获取题目详情失败:", error);
      return {
        data: {
          code: 500,
          data: {} as Question, // 类型转换以满足接口要求
          message: error.response?.data?.message || '网络错误，获取题目详情失败'
        }
      };
    });
}

/**
 * 分页查询题目
 * @param params 查询参数
 * @returns 分页结果
 */
export function getQuestionPage(params: QuestionQueryRequest) {
  return axios.get<PageResponse<Question>>('/api/teacher/questions/list', { params })
}

/**
 * 根据课程ID查询题目列表
 * @param courseId 课程ID
 * @returns 题目列表
 */
export function getQuestionsByCourse(courseId: number) {
  return axios.get<Question[] | { code: number, data: Question[], message: string }>(`/api/teacher/questions/course/${courseId}`)
    .then(response => {
      // 处理直接返回数组的情况
      if (Array.isArray(response.data)) {
        return { data: response.data }
      }
      // 处理返回Result包装对象的情况
      if (response.data && typeof response.data === 'object' && 'code' in response.data) {
        if (response.data.code === 200) {
          return { data: response.data.data || [] }
        } else {
          throw new Error(response.data.message || '获取题目列表失败')
        }
      }
      // 处理其他情况
      return { data: [] }
    })
    .catch(error => {
      console.error('获取课程题目列表请求失败:', error)
      // 返回空数组，确保前端不会因为错误而卡死
      return { data: [] }
    })
}

/**
 * 根据章节ID查询题目列表
 * @param chapterId 章节ID
 * @returns 题目列表
 */
export function getQuestionsByChapter(chapterId: number) {
  return axios.get<Question[]>(`/api/teacher/questions/chapter/${chapterId}`)
} 