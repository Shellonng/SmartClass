package com.education.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.education.entity.AssignmentSubmissionAnswer;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;
import org.apache.ibatis.annotations.Select;

import java.util.List;

/**
 * 作业题目答案Mapper接口
 */
@Mapper
public interface AssignmentSubmissionAnswerMapper extends BaseMapper<AssignmentSubmissionAnswer> {

    /**
     * 根据题目ID列表查询所有的学生答题记录
     * @param questionIds 题目ID列表
     * @return 答题记录列表
     */
    @Select("<script>"
            + "SELECT * FROM assignment_submission_answer "
            + "WHERE question_id IN "
            + "<foreach collection='questionIds' item='id' open='(' separator=',' close=')'>"
            + "#{id}"
            + "</foreach>"
            + "</script>")
    List<AssignmentSubmissionAnswer> selectByQuestionIds(@Param("questionIds") List<Long> questionIds);
} 