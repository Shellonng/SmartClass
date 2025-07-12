package com.education.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.baomidou.mybatisplus.core.metadata.IPage;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.education.entity.Question;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;
import org.apache.ibatis.annotations.Select;

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
    
    /**
     * 根据教师ID获取该教师创建的所有题目
     * @param teacherId 教师ID
     * @return 题目列表
     */
    @Select("SELECT * FROM question WHERE created_by = #{teacherId}")
    List<Question> selectByTeacher(@Param("teacherId") Long teacherId);
    
    /**
     * 根据教师ID和课程ID获取该教师在指定课程中创建的所有题目
     * @param teacherId 教师ID
     * @param courseId 课程ID
     * @return 题目列表
     */
    @Select("SELECT * FROM question WHERE created_by = #{teacherId} AND course_id = #{courseId}")
    List<Question> selectByTeacherAndCourse(@Param("teacherId") Long teacherId, @Param("courseId") Long courseId);
} 