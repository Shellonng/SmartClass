package com.education.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.education.entity.LearningRecord;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;
import org.apache.ibatis.annotations.Select;

import java.time.LocalDate;
import java.util.List;
import java.util.Map;

@Mapper
public interface LearningRecordMapper extends BaseMapper<LearningRecord> {
    
    /**
     * 获取学生某课程在指定日期范围内的每日学习时长
     */
    @Select("SELECT DATE(start_time) as date, SUM(duration) as duration " +
           "FROM learning_records " +
           "WHERE student_id = #{studentId} " +
           "AND course_id = #{courseId} " +
           "AND start_time >= #{startDate} " +
           "AND start_time <= #{endDate} " +
           "GROUP BY DATE(start_time)")
    List<Map<String, Object>> getDailyLearningDuration(@Param("studentId") Long studentId, 
                                                      @Param("courseId") Long courseId,
                                                      @Param("startDate") LocalDate startDate,
                                                      @Param("endDate") LocalDate endDate);
    
    /**
     * 获取学生某课程各章节的学习时长分布
     */
    @Select("SELECT s.id as section_id, s.title as section_title, COALESCE(SUM(lr.duration), 0) as duration " +
           "FROM section s " +
           "LEFT JOIN learning_records lr ON s.id = lr.section_id AND lr.student_id = #{studentId} " +
           "WHERE s.course_id = #{courseId} " +
           "GROUP BY s.id, s.title")
    List<Map<String, Object>> getSectionLearningDistribution(@Param("studentId") Long studentId, 
                                                            @Param("courseId") Long courseId);
    
    /**
     * 获取学生某课程各资源类型的学习时长分布
     */
    @Select("SELECT resource_type, SUM(duration) as duration " +
           "FROM learning_records " +
           "WHERE student_id = #{studentId} " +
           "AND course_id = #{courseId} " +
           "GROUP BY resource_type")
    List<Map<String, Object>> getResourceTypeLearningDistribution(@Param("studentId") Long studentId, 
                                                                 @Param("courseId") Long courseId);
} 