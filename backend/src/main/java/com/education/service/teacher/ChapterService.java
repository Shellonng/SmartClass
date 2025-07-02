package com.education.service.teacher;

import com.education.entity.Chapter;

import java.util.List;

/**
 * 章节服务接口
 */
public interface ChapterService {

    /**
     * 根据课程ID获取章节列表
     * 
     * @param courseId 课程ID
     * @return 章节列表（包含小节信息）
     */
    List<Chapter> getChaptersByCourseId(Long courseId);
    
    /**
     * 创建章节
     * 
     * @param chapter 章节信息
     * @return 创建后的章节
     */
    Chapter createChapter(Chapter chapter);
    
    /**
     * 更新章节
     * 
     * @param chapter 章节信息
     * @return 更新后的章节
     */
    Chapter updateChapter(Chapter chapter);
    
    /**
     * 删除章节
     * 
     * @param chapterId 章节ID
     * @return 是否删除成功
     */
    boolean deleteChapter(Long chapterId);
} 