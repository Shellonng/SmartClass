package com.education.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.education.entity.StudentAnswer;
import org.apache.ibatis.annotations.Mapper;

/**
 * 学生答题记录Mapper接口
 */
@Mapper
public interface StudentAnswerMapper extends BaseMapper<StudentAnswer> {
} 