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
  title: string
  courseName?: string
  description: string
  coverImage?: string
  credit?: number
  credits?: number
  category?: string
  courseType?: string
  difficulty?: string
  status: string
  startTime?: string
  endTime?: string
  createTime: string
  updateTime?: string
  teacherName?: string
  teacherId?: number
  studentCount?: number
  averageScore?: number
  progress?: number
  term?: string
  semester?: string
  objectives?: string
  requirements?: string
  chapterCount?: number
  taskCount?: number
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
    description?: string
  courseId?: number
  teacherId?: number
  studentCount?: number
  status?: 'active' | 'inactive'
  isDefault?: boolean
  createTime?: string
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

// 课程创建请求接口
export interface CourseCreateRequest {
  title?: string
  courseName?: string
  description?: string
  coverImage?: string
  credit?: number
  category?: string
  courseType?: string
  difficulty?: string
  startTime?: string
  endTime?: string
  term?: string
  semester?: string
}

// 课程更新请求接口
export interface CourseUpdateRequest {
  title?: string
  courseName?: string
  description?: string
  coverImage?: string
  credit?: number
  category?: string
  courseType?: string
  difficulty?: string
  startTime?: string
  endTime?: string
  term?: string
  semester?: string
}

// 定义通用分页响应接口
interface PageResponse<T> {
  // 兼容不同的数据字段名
  records?: T[];
  content?: T[];
  list?: T[];
  
  // 兼容不同的总数字段名
  total?: number;
  totalElements?: number;
  
  // 其他可能的分页信息
  size?: number;
  current?: number;
  pages?: number;
  totalPages?: number;
  pageNum?: number;
  pageSize?: number;
  
  // 状态标识
  first?: boolean;
  last?: boolean;
  empty?: boolean;
  number?: number;
}

// 获取教师仪表板数据
export const getDashboardData = () => {
  return axios.get('/teacher/dashboard')
}

// 获取教学统计数据
export const getTeachingStatistics = (timeRange?: string) => {
  return axios.get('/teacher/dashboard/statistics', { params: { timeRange } })
}

// 获取待处理任务
export const getPendingTasks = () => {
  return axios.get('/teacher/dashboard/pending-tasks')
}

// 获取课程概览
export const getCourseOverview = () => {
  return axios.get('/teacher/dashboard/course-overview')
}

// 获取学生表现分析
export const getStudentPerformance = (params?: { courseId?: number; classId?: number }) => {
  return axios.get('/teacher/dashboard/student-performance', { params })
}

// 获取教学建议
export const getTeachingSuggestions = () => {
  return axios.get('/teacher/dashboard/teaching-suggestions')
}

// 获取近期活动
export const getRecentActivities = (limit: number = 10) => {
  return axios.get('/teacher/dashboard/recent-activities', { params: { limit } })
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

// 课程管理 - 更新为与后端匹配的接口
export const getCourses = (params?: { page?: number; size?: number; keyword?: string; status?: string; term?: string }) => {
  // 获取token和用户ID
  const token = localStorage.getItem('user-token') || localStorage.getItem('token');
  const userInfo = localStorage.getItem('user-info');
  let userId = '';
  
  if (userInfo) {
    try {
      const userObj = JSON.parse(userInfo);
      userId = userObj.id || '';
    } catch (e) {
      console.error('解析用户信息失败:', e);
    }
  }
  
  // 使用简化的token格式
  const authToken = userId ? `Bearer token-${userId}` : (token ? `Bearer ${token}` : '');
  
  return axios.get('/api/teacher/courses', { 
    params,
    headers: {
      'Authorization': authToken
    }
  })
    .then(response => {
      console.log('获取课程列表响应:', response);
      return response;
    })
    .catch(error => {
      console.error('获取课程列表错误:', error);
      throw error;
    });
}

export const createCourse = (data: CourseCreateRequest) => {
  // 确保数据格式正确
  const formattedData = {
    title: data.courseName || data.title, // 确保title字段有值
    courseName: data.courseName || data.title, // 兼容前端字段
    description: data.description || '',
    coverImage: data.coverImage || '',
    credit: data.credit || 3,
    courseType: data.courseType || '必修课',
    category: data.category || data.courseType || '必修课', // 兼容前端字段
    startTime: data.startTime, 
    endTime: data.endTime,
    term: data.semester || data.term || '2024-2025-1',
    semester: data.semester || data.term || '2024-2025-1', // 兼容前端字段
    status: '未开始'
  };
  
  console.log('发送创建课程请求，数据:', JSON.stringify(formattedData));
  
  // 获取token和用户ID
  const token = localStorage.getItem('user-token') || localStorage.getItem('token');
  const userInfo = localStorage.getItem('user-info');
  let userId = '';
  
  if (userInfo) {
    try {
      const userObj = JSON.parse(userInfo);
      userId = userObj.id || '';
      console.log('从用户信息中获取到用户ID:', userId);
    } catch (e) {
      console.error('解析用户信息失败:', e);
    }
  }
  
  // 使用简化的token格式
  const authToken = userId ? `Bearer token-${userId}` : (token ? `Bearer ${token}` : '');
  console.log('使用的认证头:', authToken);
  
  return axios.post('/api/teacher/courses', formattedData, {
    withCredentials: true, // 确保发送凭证（Cookie）
    headers: {
      'Content-Type': 'application/json',
      'Authorization': authToken
    }
  })
    .then(response => {
      console.log('创建课程响应:', response);
      return response;
    })
    .catch(error => {
      console.error('创建课程错误:', error);
      console.error('错误详情:', {
        message: error.message,
        status: error.response?.status,
        data: error.response?.data,
        config: {
          url: error.config?.url,
          baseURL: error.config?.baseURL,
          method: error.config?.method,
          headers: error.config?.headers
        }
      });
      throw error;
    });
}

export const updateCourse = (courseId: number, data: CourseUpdateRequest) => {
  return axios.put(`/teacher/courses/${courseId}`, data)
}

export const deleteCourse = (courseId: number | string) => {
  // 确保courseId是有效的
  if (!courseId) {
    console.error('无效的课程ID:', courseId);
    return Promise.reject(new Error('无效的课程ID'));
  }
  
  // 将courseId转换为字符串，避免JavaScript的数值精度问题
  const courseIdStr = String(courseId);
  console.log(`处理课程ID: ${courseId}, 转换为字符串: ${courseIdStr}`);

  // 获取token和用户ID
  const token = localStorage.getItem('user-token') || localStorage.getItem('token');
  const userInfo = localStorage.getItem('user-info');
  let userId = '';
  
  if (userInfo) {
    try {
      const userObj = JSON.parse(userInfo);
      userId = userObj.id || '';
      console.log('从用户信息中获取到用户ID:', userId);
    } catch (e) {
      console.error('解析用户信息失败:', e);
    }
  }
  
  // 使用简化的token格式
  const authToken = userId ? `Bearer token-${userId}` : (token ? `Bearer ${token}` : '');
  console.log('使用的认证头:', authToken);
  console.log(`发送删除课程请求，课程ID: ${courseIdStr}, URL: /api/teacher/courses/${courseIdStr}`);
  
  return axios.delete(`/api/teacher/courses/${courseIdStr}`, {
    headers: {
      'Authorization': authToken
    }
  })
    .then(response => {
      console.log('删除课程响应:', response);
      return response;
    })
    .catch(error => {
      console.error('删除课程错误:', error);
      console.error('错误详情:', {
        message: error.message,
        status: error.response?.status,
        data: error.response?.data,
        config: {
          url: error.config?.url,
          baseURL: error.config?.baseURL,
          method: error.config?.method,
          headers: error.config?.headers
        }
      });
      throw error;
    });
}

export const getCourseDetail = (courseId: number) => {
  return axios.get(`/teacher/courses/${courseId}`)
}

export const publishCourse = (courseId: number) => {
  return axios.post(`/teacher/courses/${courseId}/publish`)
}

export const unpublishCourse = (courseId: number) => {
  return axios.post(`/teacher/courses/${courseId}/unpublish`)
}

export const getCourseChapters = (courseId: number) => {
  return axios.get(`/teacher/courses/${courseId}/chapters`)
}

export const getCourseStatistics = (courseId: number) => {
  return axios.get(`/teacher/courses/${courseId}/statistics`)
}

export const copyCourse = (courseId: number, newCourseName: string) => {
  return axios.post(`/teacher/courses/${courseId}/copy`, { newCourseName })
}

export const exportCourse = (courseId: number) => {
  return axios.get(`/teacher/courses/${courseId}/export`)
}

// 作业管理
export const getAssignments = (params?: { page?: number; size?: number; courseId?: number; status?: string }) => {
  return axios.get<Assignment[]>('/teacher/tasks', { params })
}

export const createAssignment = (data: Omit<Assignment, 'id' | 'submissionCount' | 'totalStudents'>) => {
  return axios.post('/teacher/tasks', data)
}

export const updateAssignment = (assignmentId: number, data: Partial<Assignment>) => {
  return axios.put(`/teacher/tasks/${assignmentId}`, data)
}

export const deleteAssignment = (assignmentId: number) => {
  return axios.delete(`/teacher/tasks/${assignmentId}`)
}

export const publishAssignment = (assignmentId: number) => {
  return axios.post(`/teacher/tasks/${assignmentId}/publish`)
}

export const getAssignmentDetail = (assignmentId: number) => {
  return axios.get(`/teacher/tasks/${assignmentId}`)
}

export const getAssignmentSubmissions = (assignmentId: number, params?: { page?: number; size?: number }) => {
  return axios.get(`/teacher/tasks/${assignmentId}/submissions`, { params })
}

// 班级管理
export const getClasses = (params?: { page?: number; size?: number; keyword?: string; courseId?: number }) => {
  console.log('调用getClasses，参数:', params)
  
  // 获取token和用户ID
  const token = localStorage.getItem('user-token') || localStorage.getItem('token')
  const userInfo = localStorage.getItem('user-info')
  let userId = ''
  
  if (userInfo) {
    try {
      const userObj = JSON.parse(userInfo)
      userId = userObj.id || ''
    } catch (e) {
      console.error('解析用户信息失败:', e)
    }
  }
  
  // 使用简化的token格式
  const authToken = userId ? `Bearer token-${userId}` : (token ? `Bearer ${token}` : '')
  
  return axios.get('/api/teacher/classes', { 
    params,
    headers: {
      'Authorization': authToken
    }
  })
    .then(response => {
      console.log('获取班级列表响应:', response)
      // 检查响应中的数据格式
      const responseData = response.data || {};
      
      // 如果响应没有直接包含数据数组，尝试从data字段获取
      if (response.data && typeof response.data === 'object') {
        if (Array.isArray(responseData.records)) {
          return responseData;
        } else if (Array.isArray(responseData.content)) {
          return responseData;
        } else if (Array.isArray(responseData.list)) {
          return responseData;
        } else if (responseData.data && Array.isArray(responseData.data.content)) {
          return responseData.data;
        } else if (responseData.data && Array.isArray(responseData.data.records)) {
          return responseData.data;
        } else if (responseData.data && Array.isArray(responseData.data.list)) {
          return responseData.data;
        } else if (responseData.data && Array.isArray(responseData.data)) {
          // 将普通数组转换为分页格式
          return {
            content: responseData.data,
            records: responseData.data,
            list: responseData.data,
            total: responseData.data.length,
            size: responseData.data.length,
            current: 1,
            pages: 1
          };
        }
      }
      return response;
    })
    .catch(error => {
      console.error('获取班级列表错误:', error)
      throw error
    })
}

export const createClass = (data: { 
  name: string; 
  description?: string; 
  courseId?: number | null;
  isDefault?: boolean;
}) => {
  return axios.post('/api/teacher/classes', data)
}

export const updateClass = (classId: number, data: Partial<Class>) => {
  return axios.put(`/api/teacher/classes/${classId}`, data)
}

export const deleteClass = (classId: number) => {
  return axios.delete(`/api/teacher/classes/${classId}`)
}

export const getClassDetail = (classId: number) => {
  return axios.get(`/teacher/classes/${classId}`)
}

export const getClassStudents = (classId: number, params?: { page?: number; size?: number; keyword?: string }) => {
  return axios.get(`/teacher/classes/${classId}/students`, { params })
}

export const addStudentsToClass = (classId: number, studentIds: number[]) => {
  return axios.post(`/teacher/classes/${classId}/students`, { studentIds })
}

export const removeStudentFromClass = (classId: number, studentId: number) => {
  return axios.delete(`/teacher/classes/${classId}/students/${studentId}`)
}

// 成绩管理
export const getGrades = (params?: { page?: number; size?: number; courseId?: number; classId?: number; taskId?: number }) => {
  return axios.get<Grade[]>('/teacher/grades', { params })
}

export const createGrade = (data: { studentId: number; taskId: number; score: number; feedback?: string }) => {
  return axios.post('/teacher/grades', data)
}

export const batchCreateGrades = (data: { taskId: number; grades: Array<{ studentId: number; score: number; feedback?: string }> }) => {
  return axios.post('/teacher/grades/batch', data)
}

export const updateGrade = (gradeId: number, data: { score: number; feedback?: string }) => {
  return axios.put(`/teacher/grades/${gradeId}`, data)
}

export const getGradeStatistics = (params?: { courseId?: number; classId?: number; timeRange?: string }) => {
  return axios.get('/teacher/grades/statistics', { params })
}

export const exportGrades = (params?: { courseId?: number; classId?: number; format?: string }) => {
  return axios.get('/teacher/grades/export', { params, responseType: 'blob' })
}

// 资源管理
export const getResources = (params?: { page?: number; size?: number; category?: string; keyword?: string }) => {
  return axios.get('/teacher/resources', { params })
}

export const uploadResource = (data: FormData) => {
  return axios.post('/teacher/resources/upload', data, {
    headers: {
      'Content-Type': 'multipart/form-data'
    }
  })
}

export const updateResource = (resourceId: number, data: { name?: string; description?: string; category?: string }) => {
  return axios.put(`/teacher/resources/${resourceId}`, data)
}

export const deleteResource = (resourceId: number) => {
  return axios.delete(`/teacher/resources/${resourceId}`)
}

export const getResourceDetail = (resourceId: number) => {
  return axios.get(`/teacher/resources/${resourceId}`)
}

// AI工具相关接口
export const intelligentGrading = (data: { taskId: number; submissionId: number; gradingCriteria?: string }) => {
  return axios.post('/teacher/ai/grade', data)
}

export const batchIntelligentGrading = (data: { taskId: number; submissionIds: number[]; gradingCriteria?: string }) => {
  return axios.post('/teacher/ai/batch-grade', data)
}

export const generateRecommendations = (data: { studentId: number; courseId: number; analysisType?: string }) => {
  return axios.post('/teacher/ai/recommend', data)
}

export const analyzeStudentAbility = (data: { studentId: number; analysisDimensions: string[]; timeRange?: string }) => {
  return axios.post('/teacher/ai/ability-analysis', data)
}

export const generateKnowledgeGraph = (data: { courseId: number; chapterCount: number; autoGenerate?: boolean }) => {
  return axios.post('/teacher/ai/knowledge-graph', data)
}

export const generateQuestions = (data: { knowledgePoints: string[]; questionType: string; questionCount: number; difficulty?: string }) => {
  return axios.post('/teacher/ai/generate-questions', data)
}

export const optimizeLearningPath = (data: { studentId: number; targetSkills: string[]; timeConstraint?: number }) => {
  return axios.post('/teacher/ai/optimize-path', data)
}

export const analyzeClassroomPerformance = (data: { classId: number; timeRange: string; analysisType?: string }) => {
  return axios.post('/teacher/ai/classroom-analysis', data)
}

export const generateTeachingSuggestions = (data: { courseId: number; studentGroup: string; teachingGoals?: string[] }) => {
  return axios.post('/teacher/ai/teaching-suggestions', data)
}

export const analyzeDocument = (file: File, analysisType: string, courseId?: number) => {
  const formData = new FormData()
  formData.append('file', file)
  formData.append('analysisType', analysisType)
  if (courseId) {
    formData.append('courseId', courseId.toString())
  }
  return axios.post('/teacher/ai/analyze-document', formData, {
    headers: {
      'Content-Type': 'multipart/form-data'
    }
  })
}

export const getAIAnalysisHistory = (params?: { type?: string; courseId?: number; page?: number; size?: number }) => {
  return axios.get('/teacher/ai/analysis-history', { params })
}

// Chapter interfaces
export interface Chapter {
  id?: number;
  courseId: number;
  title: string;
  description?: string;
  sortOrder?: number;
  sections?: Section[];
  createTime?: string;
  updateTime?: string;
}

// Section interfaces
export interface Section {
  id?: number
  chapterId: number
  title: string
  description?: string
  videoUrl?: string
  duration?: number
  sortOrder?: number
  createTime?: string
  updateTime?: string
}

// Chapter API functions
export const getChaptersByCourseId = (courseId: number) => {
  return axios.get(`/teacher/chapters/course/${courseId}`);
}

export const createChapter = (chapter: Chapter) => {
  return axios.post('/teacher/chapters', chapter);
}

export const updateChapter = (id: number, chapter: Chapter) => {
  return axios.put(`/teacher/chapters/${id}`, chapter);
}

export const deleteChapter = (id: number) => {
  return axios.delete(`/teacher/chapters/${id}`);
}

// Section API functions
export const getSectionsByChapterId = (chapterId: number) => {
  return axios.get(`/teacher/sections/chapter/${chapterId}`)
}

export const getSectionById = (sectionId: number) => {
  return axios.get(`/teacher/sections/${sectionId}`)
}

export const createSection = (section: Section) => {
  return axios.post('/teacher/sections', section)
}

export const updateSection = (id: number, section: Section) => {
  return axios.put(`/teacher/sections/${id}`, section)
}

export const deleteSection = (id: number) => {
  return axios.delete(`/teacher/sections/${id}`)
}

// 上传小节视频
export const uploadSectionVideo = (sectionId: number, file: File) => {
  const formData = new FormData();
  formData.append('file', file);
  return axios.post(`/teacher/sections/${sectionId}/upload-video`, formData, {
    headers: {
      'Content-Type': 'multipart/form-data'
    }
  });
};

// 获取当前教师的课程列表（用于班级绑定）
export const getTeacherCourses = (params?: { page?: number; size?: number }) => {
  console.log('调用getTeacherCourses，参数:', params)
  
  // 获取token和用户ID
  const token = localStorage.getItem('user-token') || localStorage.getItem('token')
  const userInfo = localStorage.getItem('user-info')
  let userId = ''
  
  if (userInfo) {
    try {
      const userObj = JSON.parse(userInfo)
      userId = userObj.id || ''
      console.log('从用户信息中获取到用户ID:', userId)
    } catch (e) {
      console.error('解析用户信息失败:', e)
    }
  }
  
  // 使用简化的token格式
  const authToken = userId ? `Bearer token-${userId}` : (token ? `Bearer ${token}` : '')
  console.log('使用的认证头:', authToken)
  
  return axios.get('/api/teacher/classes/courses', { 
    params,
    headers: {
      'Authorization': authToken
    }
  })
    .then(response => {
      console.log('获取教师课程列表响应:', response)
      
      // 处理不同格式的响应
      if (!response || !response.data) {
        console.warn('未获取到课程数据或数据结构不符合预期')
        return { data: [] }
      }

      // 处理不同的响应结构
      const responseData = response.data

      // 1. 响应就是数组格式
      if (Array.isArray(responseData)) {
        console.log('教师课程数据直接是数组格式')
        return { data: responseData }
      }
      
      // 2. Result包装类型
      if (responseData.code === 200 && responseData.data) {
        // 2.1 Result中的data就是数组
        if (Array.isArray(responseData.data)) {
          console.log('教师课程数据在Result.data中')
          return { data: responseData.data }
        }
        
        // 2.2 Result中的data是分页对象
        if (responseData.data.records || responseData.data.content || responseData.data.list) {
          console.log('教师课程数据在Result.data的分页对象中')
          return { 
            data: responseData.data.records || 
                  responseData.data.content || 
                  responseData.data.list || []
          }
        }
      }
      
      // 3. 分页响应直接在外层
      if (responseData.records || responseData.content || responseData.list) {
        console.log('教师课程数据在分页对象中')
        return { 
          data: responseData.records || 
                responseData.content || 
                responseData.list || [] 
        }
      }
      
      // 4. 其他未知格式
      console.warn('未能识别的教师课程数据结构:', responseData)
      return { data: [] }
    })
    .catch(error => {
      console.error('获取教师课程列表错误:', error)
      console.error('错误详情:', {
        message: error.message,
        status: error.response?.status,
        data: error.response?.data,
        config: error.config && {
          url: error.config.url,
          method: error.config.method,
          baseURL: error.config.baseURL,
          headers: error.config.headers
        }
      })
      return { data: [] } // 返回空数组而不是抛出异常
    })
}