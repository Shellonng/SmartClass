package com.education.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.education.entity.QuestionBank;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;

import java.util.List;
import java.util.Map;

/**
 * 题库数据访问层
 * 
 * @author Education Platform Team
 * @version 1.0.0
 * @since 2024
 */
@Mapper
public interface QuestionBankMapper extends BaseMapper<QuestionBank> {
    
    /**
     * 根据课程ID查询题目
     */
    List<QuestionBank> selectByCourseId(@Param("courseId") Long courseId);
    
    /**
     * 根据题目类型查询题目
     */
    List<QuestionBank> selectByQuestionType(@Param("questionType") String questionType);
    
    /**
     * 根据难度等级查询题目
     */
    List<QuestionBank> selectByDifficulty(@Param("difficulty") Integer difficulty);
    
    /**
     * 根据知识点ID查询题目
     */
    List<QuestionBank> selectByKnowledgePointId(@Param("knowledgePointId") Long knowledgePointId);
    
    /**
     * 根据创建者ID查询题目
     */
    List<QuestionBank> selectByCreatedBy(@Param("createdBy") Long createdBy);
    
    /**
     * 搜索题目
     */
    List<QuestionBank> searchQuestions(@Param("keyword") String keyword, @Param("questionType") String questionType, @Param("difficulty") Integer difficulty);
    
    /**
     * 根据标签查询题目
     */
    List<QuestionBank> selectByTags(@Param("tags") List<String> tags);
    
    /**
     * 查询公开题目
     */
    List<QuestionBank> selectPublicQuestions();
    
    /**
     * 随机选择题目
     */
    List<QuestionBank> selectRandomQuestions(@Param("courseId") Long courseId, @Param("questionType") String questionType, @Param("difficulty") Integer difficulty, @Param("count") Integer count);
    
    /**
     * 查询热门题目
     */
    List<QuestionBank> selectPopularQuestions(@Param("limit") Integer limit);
    
    /**
     * 查询题目统计信息
     */
    Map<String, Object> selectQuestionStats(@Param("questionId") Long questionId);
    
    /**
     * 查询题目使用统计
     */
    List<Map<String, Object>> selectQuestionUsageStats(@Param("createdBy") Long createdBy);
    
    /**
     * 查询题目正确率统计
     */
    List<Map<String, Object>> selectAccuracyStats(@Param("courseId") Long courseId);
    
    /**
     * 更新题目使用次数
     */
    int updateUsageCount(@Param("questionId") Long questionId);
    
    /**
     * 更新题目正确率
     */
    int updateAccuracyRate(@Param("questionId") Long questionId, @Param("accuracyRate") Double accuracyRate);
    
    /**
     * 批量插入题目
     */
    int batchInsert(@Param("questions") List<QuestionBank> questions);
    
    /**
     * 批量更新题目状态
     */
    int batchUpdateStatus(@Param("questionIds") List<Long> questionIds, @Param("status") String status);
    
    /**
     * 批量删除题目
     */
    int batchDelete(@Param("questionIds") List<Long> questionIds);
    
    /**
     * 复制题目到其他课程
     */
    int copyQuestionsToCourse(@Param("questionIds") List<Long> questionIds, @Param("targetCourseId") Long targetCourseId, @Param("createdBy") Long createdBy);
}