package com.education.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.education.entity.AssignmentQuestion;
import org.apache.ibatis.annotations.Mapper;

/**
 * 作业题目关联Mapper接口
 */
@Mapper
public interface AssignmentQuestionMapper extends BaseMapper<AssignmentQuestion> {
} 