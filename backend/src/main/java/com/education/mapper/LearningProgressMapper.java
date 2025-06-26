package com.education.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.education.entity.LearningProgress;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;
import org.apache.ibatis.annotations.Select;

import java.util.List;

/**
 * 学习进度数据访问层
 * 
 * @author Education Platform Team
 * @version 1.0.0
 * @since 2024
 */
@Mapper
public interface LearningProgressMapper extends BaseMapper<LearningProgress> {

    /**
     * 根据学生ID和课程ID查询学习进度
     * 
     * @param studentId 学生ID
     * @param courseId 课程ID
     * @return 学习进度列表
     */
    @Select("SELECT * FROM learning_progress WHERE student_id = #{studentId} AND course_id = #{courseId}")
    List<LearningProgress> selectByStudentAndCourse(@Param("studentId") Long studentId, @Param("courseId") Long courseId);

    /**
     * 根据学生ID、课程ID和章节ID查询学习进度
     * 
     * @param studentId 学生ID
     * @param courseId 课程ID
     * @param chapterId 章节ID
     * @return 学习进度
     */
    @Select("SELECT * FROM learning_progress WHERE student_id = #{studentId} AND course_id = #{courseId} AND chapter_id = #{chapterId}")
    LearningProgress selectByStudentCourseChapter(@Param("studentId") Long studentId, @Param("courseId") Long courseId, @Param("chapterId") Long chapterId);

    /**
     * 统计学生在课程中已完成的章节数
     * 
     * @param studentId 学生ID
     * @param courseId 课程ID
     * @return 已完成章节数
     */
    @Select("SELECT COUNT(*) FROM learning_progress WHERE student_id = #{studentId} AND course_id = #{courseId} AND status = 'COMPLETED'")
    Integer countCompletedChapters(@Param("studentId") Long studentId, @Param("courseId") Long courseId);

    /**
     * 计算学生在课程中的平均进度
     * 
     * @param studentId 学生ID
     * @param courseId 课程ID
     * @return 平均进度百分比
     */
    @Select("SELECT COALESCE(AVG(progress_percentage), 0) FROM learning_progress WHERE student_id = #{studentId} AND course_id = #{courseId}")
    Double calculateAverageProgress(@Param("studentId") Long studentId, @Param("courseId") Long courseId);

    /**
     * 统计学生在课程中的总学习时长
     * 
     * @param studentId 学生ID
     * @param courseId 课程ID
     * @return 总学习时长（分钟）
     */
    @Select("SELECT COALESCE(SUM(study_duration), 0) FROM learning_progress WHERE student_id = #{studentId} AND course_id = #{courseId}")
    Integer calculateTotalStudyDuration(@Param("studentId") Long studentId, @Param("courseId") Long courseId);

    /**
     * 统计学生的总学习时长（所有课程）
     * 
     * @param studentId 学生ID
     * @return 总学习时长（分钟）
     */
    @Select("SELECT COALESCE(SUM(study_duration), 0) FROM learning_progress WHERE student_id = #{studentId}")
    Integer calculateTotalStudyDurationForStudent(@Param("studentId") Long studentId);
}