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
export const getClasses = (params?: { page?: number; size?: number; keyword?: string }) => {
  return axios.get<Class[]>('/teacher/classes', { params })
}

export const createClass = (data: Omit<Class, 'id' | 'createTime'>) => {
  return axios.post('/teacher/classes', data)
}

export const updateClass = (classId: number, data: Partial<Class>) => {
  return axios.put(`/teacher/classes/${classId}`, data)
}

export const deleteClass = (classId: number) => {
  return axios.delete(`/teacher/classes/${classId}`)
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