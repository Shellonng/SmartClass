import axios from 'axios'

// 章节类型定义
export interface Chapter {
  id: number
  courseId: number
  title: string
  description?: string
  sortOrder?: number
  sections?: Section[]
  createTime?: string
  updateTime?: string
}

// 小节类型定义
export interface Section {
  id: number
  chapterId: number
  title: string
  description?: string
  videoUrl?: string
  duration?: number
  sortOrder?: number
  createTime?: string
  updateTime?: string
}

// 章节API
export const chapterAPI = {
  /**
   * 获取课程的章节列表
   */
  getChaptersByCourseId(courseId: number) {
    return axios.get(`/api/teacher/chapters/course/${courseId}`)
  },

  /**
   * 获取章节详情
   */
  getChapterDetail(chapterId: number) {
    return axios.get(`/api/teacher/chapters/${chapterId}`)
  },

  /**
   * 创建章节
   */
  createChapter(chapter: Chapter) {
    return axios.post('/api/teacher/chapters', chapter)
  },

  /**
   * 更新章节
   */
  updateChapter(chapterId: number, chapter: Chapter) {
    return axios.put(`/api/teacher/chapters/${chapterId}`, chapter)
  },

  /**
   * 删除章节
   */
  deleteChapter(chapterId: number) {
    return axios.delete(`/api/teacher/chapters/${chapterId}`)
  },

  /**
   * 获取章节的小节列表
   */
  getSectionsByChapterId(chapterId: number) {
    return axios.get(`/api/teacher/chapters/${chapterId}/sections`)
  },

  /**
   * 获取小节详情
   */
  getSectionDetail(sectionId: number) {
    return axios.get(`/api/teacher/sections/${sectionId}`)
  },

  /**
   * 创建小节
   */
  createSection(section: Section) {
    return axios.post('/api/teacher/sections', section)
  },

  /**
   * 更新小节
   */
  updateSection(sectionId: number, section: Section) {
    return axios.put(`/api/teacher/sections/${sectionId}`, section)
  },

  /**
   * 删除小节
   */
  deleteSection(sectionId: number) {
    return axios.delete(`/api/teacher/sections/${sectionId}`)
  }
} 