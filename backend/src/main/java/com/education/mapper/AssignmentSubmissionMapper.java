package com.education.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.education.entity.AssignmentSubmission;
import org.apache.ibatis.annotations.Mapper;

/**
 * 作业提交记录Mapper接口
 */
@Mapper
public interface AssignmentSubmissionMapper extends BaseMapper<AssignmentSubmission> {
} 