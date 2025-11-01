package com.education.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.education.entity.CourseResource;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;
import org.apache.ibatis.annotations.Select;
import org.apache.ibatis.annotations.Update;

import java.util.List;

/**
 * 课程资源Mapper接口
 */
@Mapper
public interface CourseResourceMapper extends BaseMapper<CourseResource> {

    /**
     * 根据课程ID查询资源列表
     *
     * @param courseId 课程ID
     * @return 资源列表
     */
    @Select("SELECT r.*, u.username as upload_user_name FROM course_resource r " +
            "LEFT JOIN user u ON r.upload_user_id = u.id " +
            "WHERE r.course_id = #{courseId} " +
            "ORDER BY r.create_time DESC")
    List<CourseResource> selectByCourseId(@Param("courseId") Long courseId);
    
    /**
     * 更新资源下载次数
     *
     * @param resourceId 资源ID
     * @return 影响行数
     */
    @Update("UPDATE course_resource SET download_count = download_count + 1 WHERE id = #{resourceId}")
    int incrementDownloadCount(@Param("resourceId") Long resourceId);
} 