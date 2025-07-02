package com.education.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.baomidou.mybatisplus.core.metadata.IPage;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.education.entity.Question;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;

import java.util.List;

/**
 * 题目Mapper接口
 */
@Mapper
public interface QuestionMapper extends BaseMapper<Question> {

    /**
     * 分页查询题目列表
     *
     * @param page 分页参数
     * @param courseId 课程ID
     * @param chapterId 章节ID
     * @param questionType 题目类型
     * @param difficulty 难度等级
     * @param knowledgePoint 知识点
     * @param keyword 关键词
     * @return 分页结果
     */
    IPage<Question> selectQuestionPage(
            Page<Question> page,
            @Param("courseId") Long courseId,
            @Param("chapterId") Long chapterId,
            @Param("questionType") String questionType,
            @Param("difficulty") Integer difficulty,
            @Param("knowledgePoint") String knowledgePoint,
            @Param("keyword") String keyword
    );

    /**
     * 根据课程ID查询题目列表
     *
     * @param courseId 课程ID
     * @return 题目列表
     */
    List<Question> selectByCourseId(@Param("courseId") Long courseId);

    /**
     * 根据章节ID查询题目列表
     *
     * @param chapterId 章节ID
     * @return 题目列表
     */
    List<Question> selectByChapterId(@Param("chapterId") Long chapterId);
} 