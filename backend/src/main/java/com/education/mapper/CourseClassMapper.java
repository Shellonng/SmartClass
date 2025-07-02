package com.education.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.baomidou.mybatisplus.core.metadata.IPage;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.education.entity.CourseClass;
import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;
import org.apache.ibatis.annotations.Select;
import org.apache.ibatis.annotations.Insert;
import org.apache.ibatis.annotations.Update;
import org.apache.ibatis.annotations.Options;

import java.util.List;

/**
 * 课程班级数据访问层
 */
@Mapper
public interface CourseClassMapper extends BaseMapper<CourseClass> {

    /**
     * 分页查询教师的班级列表
     * 
     * @param page 分页参数
     * @param teacherId 教师ID
     * @param keyword 班级名称关键词
     * @param courseId 课程ID
     * @return 班级列表
     */
    @Select("<script>" +
            "SELECT cc.*, " +
            "(SELECT COUNT(1) FROM class_student cs WHERE cs.class_id = cc.id) AS student_count " +
            "FROM course_class cc " +
            "WHERE cc.teacher_id = #{teacherId} " +
            "<if test='keyword != null and keyword != \"\"'>" +
            "  AND cc.name LIKE CONCAT('%', #{keyword}, '%') " +
            "</if>" +
            "<if test='courseId != null'>" +
            "  AND cc.course_id = #{courseId} " +
            "</if>" +
            "ORDER BY cc.create_time DESC" +
            "</script>")
    IPage<CourseClass> selectPageByTeacherId(Page<CourseClass> page,
                                           @Param("teacherId") Long teacherId,
                                           @Param("keyword") String keyword,
                                           @Param("courseId") Long courseId);
    
    /**
     * 查询教师的班级列表（不分页）
     * 
     * @param teacherId 教师ID
     * @return 班级列表
     */
    @Select("SELECT cc.*, " +
            "(SELECT COUNT(1) FROM class_student cs WHERE cs.class_id = cc.id) AS student_count " +
            "FROM course_class cc " +
            "WHERE cc.teacher_id = #{teacherId} " +
            "ORDER BY cc.create_time DESC")
    List<CourseClass> selectListByTeacherId(@Param("teacherId") Long teacherId);
    
    /**
     * 根据课程ID查询默认班级
     * 
     * @param courseId 课程ID
     * @return 默认班级
     */
    @Select("SELECT * FROM course_class WHERE course_id = #{courseId} AND is_default = TRUE LIMIT 1")
    CourseClass selectDefaultClassByCourseId(@Param("courseId") Long courseId);
    
    /**
     * 根据课程ID统计班级数
     * 
     * @param courseId 课程ID
     * @return 班级数量
     */
    @Select("SELECT COUNT(1) FROM course_class WHERE course_id = #{courseId}")
    int countByCourseId(@Param("courseId") Long courseId);
    
    /**
     * 插入班级记录，允许course_id为NULL
     * 
     * @param courseClass 班级信息
     * @return 影响行数
     */
    @Insert("INSERT INTO course_class(name, description, course_id, teacher_id, is_default, create_time) " +
            "VALUES(#{name}, #{description}, #{courseId}, #{teacherId}, #{isDefault}, #{createTime})")
    int insertWithNullCourseId(CourseClass courseClass);
    
    /**
     * 插入班级记录，确保course_id正确保存
     * 
     * @param courseClass 班级信息
     * @return 影响行数
     */
    @Insert("INSERT INTO course_class(name, description, course_id, teacher_id, is_default, create_time) " +
            "VALUES(#{name}, #{description}, #{courseId}, #{teacherId}, #{isDefault}, NOW())")
    @Options(useGeneratedKeys = true, keyProperty = "id")
    int insertWithCourseId(CourseClass courseClass);
    
    /**
     * 更新班级的课程ID
     * 
     * @param classId 班级ID
     * @param courseId 课程ID
     * @return 影响行数
     */
    @Update("UPDATE course_class SET course_id = #{courseId} WHERE id = #{classId}")
    int updateCourseId(@Param("classId") Long classId, @Param("courseId") Long courseId);
} 