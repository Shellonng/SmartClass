package com.education.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.education.entity.Grade;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;

import java.util.List;
import java.util.Map;

/**
 * 成绩数据访问层
 * 
 * @author Education Platform Team
 * @version 1.0.0
 * @since 2024
 */
@Mapper
public interface GradeMapper extends BaseMapper<Grade> {
    
    /**
     * 根据学生ID和课程ID查询成绩
     */
    List<Grade> selectByStudentAndCourse(@Param("studentId") Long studentId, @Param("courseId") Long courseId);
    
    /**
     * 根据任务ID查询所有学生成绩
     */
    List<Grade> selectByTaskId(@Param("taskId") Long taskId);
    
    /**
     * 查询学生的平均成绩
     */
    Double selectAverageScoreByStudent(@Param("studentId") Long studentId);
    
    /**
     * 查询课程的平均成绩
     */
    Double selectAverageScoreByCourse(@Param("courseId") Long courseId);
    
    /**
     * 查询班级的平均成绩
     */
    Double selectAverageScoreByClass(@Param("classId") Long classId);
    
    /**
     * 查询学生在班级中的排名
     */
    Integer selectStudentRankInClass(@Param("studentId") Long studentId, @Param("classId") Long classId);
    
    /**
     * 查询学生在课程中的排名
     */
    Integer selectStudentRankInCourse(@Param("studentId") Long studentId, @Param("courseId") Long courseId);
    
    /**
     * 查询成绩分布统计
     */
    List<Map<String, Object>> selectGradeDistribution(@Param("courseId") Long courseId, @Param("classId") Long classId);
    
    /**
     * 查询学生成绩趋势
     */
    List<Map<String, Object>> selectGradeTrend(@Param("studentId") Long studentId, @Param("courseId") Long courseId);
    
    /**
     * 查询优秀学生列表
     */
    List<Map<String, Object>> selectTopStudents(@Param("courseId") Long courseId, @Param("classId") Long classId, @Param("limit") Integer limit);
    
    /**
     * 查询需要关注的学生列表（成绩较低）
     */
    List<Map<String, Object>> selectLowPerformanceStudents(@Param("courseId") Long courseId, @Param("classId") Long classId, @Param("threshold") Double threshold);
    
    /**
     * 批量插入成绩
     */
    int batchInsert(@Param("grades") List<Grade> grades);
    
    /**
     * 批量更新成绩
     */
    int batchUpdate(@Param("grades") List<Grade> grades);
}