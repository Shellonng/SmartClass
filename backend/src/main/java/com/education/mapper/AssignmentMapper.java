package com.education.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.education.entity.Assignment;
import org.apache.ibatis.annotations.Mapper;

/**
 * 作业或考试Mapper接口
 */
@Mapper
public interface AssignmentMapper extends BaseMapper<Assignment> {
} 