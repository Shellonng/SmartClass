package com.education.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.education.entity.QuestionOption;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;

import java.util.List;

/**
 * 题目选项Mapper接口
 */
@Mapper
public interface QuestionOptionMapper extends BaseMapper<QuestionOption> {

    /**
     * 根据题目ID查询选项列表
     *
     * @param questionId 题目ID
     * @return 选项列表
     */
    List<QuestionOption> selectByQuestionId(@Param("questionId") Long questionId);

    /**
     * 批量插入题目选项
     *
     * @param options 选项列表
     * @return 影响行数
     */
    int batchInsert(@Param("options") List<QuestionOption> options);

    /**
     * 根据题目ID删除选项
     *
     * @param questionId 题目ID
     * @return 影响行数
     */
    int deleteByQuestionId(@Param("questionId") Long questionId);
} 