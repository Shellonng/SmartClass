package com.education.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.education.entity.QuestionImage;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;

import java.util.List;

/**
 * 题目图片Mapper接口
 */
@Mapper
public interface QuestionImageMapper extends BaseMapper<QuestionImage> {

    /**
     * 根据题目ID查询图片列表
     *
     * @param questionId 题目ID
     * @return 图片列表
     */
    List<QuestionImage> selectByQuestionId(@Param("questionId") Long questionId);

    /**
     * 批量插入题目图片
     *
     * @param images 图片列表
     * @return 影响行数
     */
    int batchInsert(@Param("images") List<QuestionImage> images);

    /**
     * 根据题目ID删除图片
     *
     * @param questionId 题目ID
     * @return 影响行数
     */
    int deleteByQuestionId(@Param("questionId") Long questionId);
} 