package com.education.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.baomidou.mybatisplus.core.metadata.IPage;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.education.entity.Exam;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;
import org.apache.ibatis.annotations.Select;

/**
 * 考试Mapper接口
 */
@Mapper
public interface ExamMapper extends BaseMapper<Exam> {
    
    /**
     * 分页查询课程的考试列表
     * @param page 分页参数
     * @param courseId 课程ID
     * @param keyword 关键词
     * @param status 状态
     * @return 分页结果
     */
    @Select("<script>" +
            "SELECT a.*, c.name as course_name, u.real_name as user_name " +
            "FROM assignment a " +
            "LEFT JOIN course c ON a.course_id = c.id " +
            "LEFT JOIN user u ON a.user_id = u.id " +
            "WHERE a.type = 'exam' " +
            "<if test='courseId != null'> AND a.course_id = #{courseId} </if>" +
            "<if test='keyword != null and keyword != \"\"'> AND a.title LIKE CONCAT('%', #{keyword}, '%') </if>" +
            "<if test='status != null'> AND a.status = #{status} </if>" +
            "ORDER BY a.create_time DESC" +
            "</script>")
    IPage<Exam> pageExams(Page<Exam> page, @Param("courseId") Long courseId, 
                          @Param("keyword") String keyword, @Param("status") Integer status);
} 