import axios from 'axios'

// 通用API响应类型
export interface ApiResponse<T = any> {
  code: number
  message: string
  data: T
}

// 课程相关API接口
export interface CourseListParams {
  page?: number
  size?: number
  keyword?: string
  category?: string
  level?: string
  sortBy?: string
  priceType?: string
}

export interface Course {
  id: number
  name: string
  description: string
  instructor: {
    id: number
    name: string
    avatar?: string
    title?: string
  }
  category: string
  level: string
  rating: number
  studentCount: number
  duration: number
  price: number
  originalPrice?: number
  coverImage?: string
  tags: string[]
  createdAt: string
  updatedAt: string
  status: 'published' | 'draft' | 'archived'
  chapters?: Chapter[]
}

export interface Chapter {
  id: number
  title: string
  description?: string
  orderIndex: number
  duration: number
  lessons: Lesson[]
}

export interface Lesson {
  id: number
  title: string
  description?: string
  type: 'video' | 'text' | 'quiz' | 'assignment'
  orderIndex: number
  duration: number
  videoUrl?: string
  content?: string
  resources?: Resource[]
}

export interface Resource {
  id: number
  name: string
  type: 'document' | 'video' | 'audio' | 'image' | 'other'
  url: string
  size?: number
}

export interface CourseListResponse {
  courses: Course[]
  total: number
  page: number
  size: number
  totalPages: number
}

export interface Instructor {
  id: number
  name: string
  avatar?: string
  title?: string
  bio?: string
  rating: number
  studentCount: number
  courseCount: number
}

// 获取课程列表
export const getCourseList = (params: CourseListParams): Promise<ApiResponse<CourseListResponse>> => {
  return axios.get('/api/courses', { params })
}

// 获取课程详情
export const getCourseDetail = (courseId: number) => {
  return axios.get(`/teacher/courses/${courseId}`)
}

// 搜索课程
export const searchCourses = (params: CourseListParams & { q: string }): Promise<ApiResponse<CourseListResponse>> => {
  return axios.get('/api/courses/search', { params })
}

// 获取热门课程
export const getPopularCourses = (limit: number = 8): Promise<ApiResponse<Course[]>> => {
  return axios.get('/api/courses/popular', { params: { limit } })
}

// 获取课程分类
export const getCourseCategories = (): Promise<ApiResponse<string[]>> => {
  return axios.get('/api/courses/categories')
}

// 获取相关课程
export const getRelatedCourses = (courseId: number, limit: number = 4): Promise<ApiResponse<Course[]>> => {
  return axios.get(`/api/courses/${courseId}/related`, { params: { limit } })
}

// 获取课程讲师信息
export const getCourseInstructor = (courseId: number): Promise<ApiResponse<Instructor>> => {
  return axios.get(`/api/courses/${courseId}/instructor`)
}

// 获取课程章节
export const getCourseChapters = (courseId: number): Promise<ApiResponse<Chapter[]>> => {
  return axios.get(`/api/courses/${courseId}/chapters`)
}

// 获取章节详情
export const getChapterDetail = (chapterId: number): Promise<ApiResponse<Chapter>> => {
  return axios.get(`/api/chapters/${chapterId}`)
}

// 获取课程资源
export const getCourseResources = (courseId: number): Promise<ApiResponse<Resource[]>> => {
  return axios.get(`/api/courses/${courseId}/resources`)
}

// 学生课程相关API
export interface JoinCourseRequest {
  inviteCode?: string
  courseId?: number
}

export interface CourseProgress {
  courseId: number
  progress: number
  completedLessons: number
  totalLessons: number
  lastAccessTime: string
}

// 加入课程
export const joinCourse = (data: JoinCourseRequest): Promise<ApiResponse<void>> => {
  return axios.post('/api/student/courses/join', data)
}

// 退出课程
export const leaveCourse = (courseId: number): Promise<ApiResponse<void>> => {
  return axios.delete(`/api/student/courses/${courseId}/leave`)
}

// 获取我的课程
export const getMyCourses = (params: CourseListParams & { status?: string }): Promise<ApiResponse<CourseListResponse>> => {
  return axios.get('/api/student/courses', { params })
}

// 获取课程学习进度
export const getCourseProgress = (courseId: number): Promise<ApiResponse<CourseProgress>> => {
  return axios.get(`/api/student/courses/${courseId}/progress`)
}

// 更新学习进度
export const updateLearningProgress = (courseId: number, lessonId: number): Promise<ApiResponse<void>> => {
  return axios.post(`/api/student/courses/${courseId}/lessons/${lessonId}/complete`)
}

// 收藏课程
export const favoriteCourse = (courseId: number): Promise<ApiResponse<void>> => {
  return axios.post(`/api/student/courses/${courseId}/favorite`)
}

// 取消收藏
export const unfavoriteCourse = (courseId: number): Promise<ApiResponse<void>> => {
  return axios.delete(`/api/student/courses/${courseId}/favorite`)
}

// 获取收藏的课程
export const getFavoriteCourses = (params?: CourseListParams): Promise<ApiResponse<CourseListResponse>> => {
  return axios.get('/api/student/courses/favorites', { params })
}

// 评价课程
export const rateCourse = (courseId: number, data: { rating: number; comment?: string }): Promise<ApiResponse<void>> => {
  return axios.post(`/api/student/courses/${courseId}/rate`, data)
}

// 获取课程评价
export const getCourseReviews = (courseId: number, params?: { page?: number; size?: number }): Promise<ApiResponse<any>> => {
  return axios.get(`/api/courses/${courseId}/reviews`, { params })
}

// 报告课程问题
export const reportCourse = (courseId: number, data: { reason: string; description?: string }): Promise<ApiResponse<void>> => {
  return axios.post(`/api/courses/${courseId}/report`, data)
}

// 评论相关接口
export function getSectionComments(sectionId: number, page = 1, size = 10) {
  return axios.get(`/api/sections/${sectionId}/comments`, {
    params: { page, size }
  })
}

export function getSectionCommentReplies(sectionId: number, commentId: number, page = 1, size = 50) {
  return axios.get(`/api/sections/${sectionId}/comments/${commentId}/replies`, {
    params: { page, size }
  })
}

export function getCourseComments(courseId: number, page = 1, size = 10) {
  return axios.get(`/api/courses/${courseId}/comments`, {
    params: { page, size }
  })
}

export function createSectionComment(sectionId: number, data: {
  content: string
  parentId?: number
}) {
  return axios.post(`/api/sections/${sectionId}/comments`, data)
}

export function updateSectionComment(sectionId: number, commentId: number, data: {
  content: string
}) {
  return axios.put(`/api/sections/${sectionId}/comments/${commentId}`, data)
}

export function deleteSectionComment(sectionId: number, commentId: number) {
  return axios.delete(`/api/sections/${sectionId}/comments/${commentId}`)
}

// 课程资源相关接口
export interface CourseResource {
  id: number
  courseId: number
  name: string
  fileType: string
  fileSize: number
  formattedSize: string
  fileUrl: string
  description?: string
  downloadCount: number
  uploadUserId: number
  uploadUserName: string
  createTime: string
}

// 获取课程资源列表
export function getTeacherCourseResources(courseId: number) {
  return axios.get(`/api/teacher/courses/${courseId}/resources`)
}

// 分页获取课程资源
export function getTeacherCourseResourcesPage(courseId: number, page = 1, size = 10) {
  return axios.get(`/api/teacher/courses/${courseId}/resources/page`, {
    params: { page, size }
  })
}

// 上传课程资源
export function uploadCourseResource(courseId: number, file: File, name?: string, description?: string) {
  const formData = new FormData()
  formData.append('file', file)
  if (name) formData.append('name', name)
  if (description) formData.append('description', description)
  
  return axios.post(`/api/teacher/courses/${courseId}/resources/upload`, formData, {
    headers: {
      'Content-Type': 'multipart/form-data'
    }
  })
}

// 删除课程资源
export function deleteCourseResource(resourceId: number) {
  return axios.delete(`/api/teacher/courses/resources/${resourceId}`)
}

// 获取资源详情
export function getResourceDetail(resourceId: number) {
  return axios.get(`/api/teacher/courses/resources/${resourceId}`)
}

// 获取资源下载链接
export function getResourceDownloadUrl(resourceId: number) {
  return `/api/teacher/courses/resources/${resourceId}/download`
}

// 获取资源预览链接
export function getResourcePreviewUrl(resourceId: number) {
  return `/api/teacher/courses/resources/${resourceId}/preview`
}