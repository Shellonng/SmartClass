package com.education.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.education.entity.ExamQuestion;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;
import org.apache.ibatis.annotations.Select;

import java.util.List;
import java.util.Map;

/**
 * 考试题目关联Mapper接口
 */
@Mapper
public interface ExamQuestionMapper extends BaseMapper<ExamQuestion> {
    
    /**
     * 获取考试的所有题目
     * @param examId 考试ID
     * @return 题目列表
     */
    @Select("SELECT q.*, eq.score, eq.sequence, " +
            "CASE q.question_type " +
            "  WHEN 'single' THEN '单选题' " +
            "  WHEN 'multiple' THEN '多选题' " +
            "  WHEN 'true_false' THEN '判断题' " +
            "  WHEN 'blank' THEN '填空题' " +
            "  WHEN 'short' THEN '简答题' " +
            "  WHEN 'code' THEN '编程题' " +
            "  ELSE '未知类型' " +
            "END as question_type_desc " +
            "FROM assignment_question eq " +
            "JOIN question q ON eq.question_id = q.id " +
            "WHERE eq.assignment_id = #{examId} " +
            "ORDER BY eq.sequence")
    List<Map<String, Object>> getExamQuestions(@Param("examId") Long examId);
    
    /**
     * 根据题目类型和条件随机获取指定数量的题目
     * @param courseId 课程ID
     * @param questionType 题目类型
     * @param count 数量
     * @param difficulty 难度等级
     * @param knowledgePoint 知识点
     * @param createdBy 创建者ID
     * @return 题目列表
     */
    @Select("<script>" +
            "SELECT * FROM question " +
            "WHERE course_id = #{courseId} " +
            "AND question_type = #{questionType} " +
            "<if test='difficulty != null'> AND difficulty = #{difficulty} </if>" +
            "<if test='knowledgePoint != null and knowledgePoint != \"\"'> AND knowledge_point = #{knowledgePoint} </if>" +
            "<if test='createdBy != null'> AND created_by = #{createdBy} </if>" +
            "ORDER BY RAND() " +
            "LIMIT #{count}" +
            "</script>")
    List<Map<String, Object>> getRandomQuestions(@Param("courseId") Long courseId,
                                               @Param("questionType") String questionType,
                                               @Param("count") Integer count,
                                               @Param("difficulty") Integer difficulty,
                                               @Param("knowledgePoint") String knowledgePoint,
                                               @Param("createdBy") Long createdBy);
} 