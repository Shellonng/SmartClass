package com.education.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.baomidou.mybatisplus.core.metadata.IPage;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.education.entity.CourseEnrollmentRequest;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;
import org.apache.ibatis.annotations.Select;

/**
 * 课程选课申请Mapper接口
 */
@Mapper
public interface CourseEnrollmentRequestMapper extends BaseMapper<CourseEnrollmentRequest> {

    /**
     * 根据教师ID分页查询选课申请
     *
     * @param page 分页参数
     * @param teacherId 教师ID
     * @return 分页选课申请列表
     */
    @Select("SELECT r.* FROM course_enrollment_request r " +
            "JOIN course c ON r.course_id = c.id " +
            "WHERE c.teacher_id = #{teacherId} AND r.status = 0 " +
            "ORDER BY r.submit_time DESC")
    IPage<CourseEnrollmentRequest> selectPageByTeacherId(Page<CourseEnrollmentRequest> page, @Param("teacherId") Long teacherId);
} 