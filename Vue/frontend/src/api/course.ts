import request from './request'
import type { ApiResponse } from './types'

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

export interface InstructorInfo {
  name: string
  title: string
  university: string
  bio: string
  avatar: string
  courses: number
  students: number
}

export interface Course {
  id: number
  title: string
  description: string
  longDescription?: string
  instructor: string
  university: string
  category: string
  level: string
  students: number
  rating: number
  reviewCount: number
  duration: string
  effort: string
  language?: string
  price: number
  originalPrice?: number
  image: string
  tags: string[]
  startDate?: string
  endDate?: string
  enrolled?: boolean
  certificate?: boolean
  prerequisites?: string[]
  skills?: string[]
  instructorInfo?: InstructorInfo
  status?: string
  createTime?: string
  updateTime?: string
}

export interface CourseListResponse {
  total: number
  page: number
  size: number
  totalPages: number
  courses: Course[]
}

export interface CourseSearchResponse {
  keyword: string
  total: number
  page: number
  size: number
  totalPages: number
  searchTime?: number
  courses: Course[]
  suggestions?: string[]
  relatedCategories?: string[]
}

// 获取课程列表
export const getCourseList = (params: CourseListParams): Promise<ApiResponse<CourseListResponse>> => {
  return request.get('/api/courses', { params })
}

// 获取课程详情
export const getCourseDetail = (courseId: number): Promise<ApiResponse<Course>> => {
  return request.get(`/api/courses/${courseId}`)
}

// 搜索课程
export const searchCourses = (params: CourseListParams & { q: string; type?: string }): Promise<ApiResponse<CourseSearchResponse>> => {
  return request.get('/api/courses/search', { params })
}

// 获取热门课程
export const getPopularCourses = (limit: number = 8): Promise<ApiResponse<Course[]>> => {
  return request.get('/api/courses/popular', { params: { limit } })
}

// 获取课程分类
export const getCourseCategories = (): Promise<ApiResponse<string[]>> => {
  return request.get('/api/courses/categories')
}

// 获取相关课程
export const getRelatedCourses = (courseId: number, limit: number = 4): Promise<ApiResponse<Course[]>> => {
  return request.get(`/api/courses/${courseId}/related`, { params: { limit } })
}

// 学生课程相关API
export interface JoinCourseRequest {
  inviteCode?: string
  courseId?: number
}

export interface LearningProgressResponse {
  courseId: number
  totalProgress: number
  chapterProgress: Array<{
    chapterId: number
    chapterName: string
    progress: number
    completed: boolean
  }>
  studyTime: number
  lastStudyTime: string
}

// 学生加入课程
export const joinCourse = (data: JoinCourseRequest): Promise<ApiResponse<any>> => {
  return request.post('/api/student/courses/join', data)
}

// 获取我的课程
export const getMyCourses = (params: CourseListParams & { status?: string }): Promise<ApiResponse<CourseListResponse>> => {
  return request.get('/api/student/courses', { params })
}

// 退出课程
export const quitCourse = (courseId: number): Promise<ApiResponse<void>> => {
  return request.delete(`/api/student/courses/${courseId}/quit`)
}

// 获取学习进度
export const getLearningProgress = (courseId: number): Promise<ApiResponse<LearningProgressResponse>> => {
  return request.get(`/api/student/courses/${courseId}/progress`)
}

// 收藏课程
export const favoriteCourse = (courseId: number): Promise<ApiResponse<any>> => {
  return request.post(`/api/student/courses/${courseId}/favorite`)
}

// 获取收藏的课程
export const getFavoriteCourses = (params: { page?: number; size?: number }): Promise<ApiResponse<CourseListResponse>> => {
  return request.get('/api/student/courses/favorites', { params })
}

// 评价课程
export interface CourseEvaluationRequest {
  rating: number
  comment: string
}

export const evaluateCourse = (courseId: number, data: CourseEvaluationRequest): Promise<ApiResponse<any>> => {
  return request.post(`/api/student/courses/${courseId}/evaluate`, data)
}