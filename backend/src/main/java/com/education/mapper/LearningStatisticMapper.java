package com.education.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.education.entity.LearningStatistic;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;
import org.apache.ibatis.annotations.Select;

import java.time.LocalDate;
import java.util.List;
import java.util.Map;

@Mapper
public interface LearningStatisticMapper extends BaseMapper<LearningStatistic> {
    
    /**
     * 获取学生某课程在指定日期范围内的每日学习统计
     */
    @Select("SELECT date, total_duration, sections_completed, resources_viewed " +
           "FROM learning_statistics " +
           "WHERE student_id = #{studentId} " +
           "AND course_id = #{courseId} " +
           "AND date >= #{startDate} " +
           "AND date <= #{endDate} " +
           "ORDER BY date")
    List<Map<String, Object>> getStudentCourseStatistics(@Param("studentId") Long studentId, 
                                                        @Param("courseId") Long courseId,
                                                        @Param("startDate") LocalDate startDate,
                                                        @Param("endDate") LocalDate endDate);
    
    /**
     * 获取学生所有课程的总学习时长
     */
    @Select("SELECT course_id, SUM(total_duration) as total_duration " +
           "FROM learning_statistics " +
           "WHERE student_id = #{studentId} " +
           "GROUP BY course_id")
    List<Map<String, Object>> getCourseTotalDuration(@Param("studentId") Long studentId);
} 