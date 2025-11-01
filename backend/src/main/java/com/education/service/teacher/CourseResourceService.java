package com.education.service.teacher;

import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.education.dto.CourseResourceDTO;
import com.education.dto.common.PageRequest;
import com.education.dto.common.PageResponse;
import com.education.dto.common.Result;
import com.education.entity.CourseResource;
import org.springframework.web.multipart.MultipartFile;

import java.util.List;

/**
 * 课程资源服务接口
 */
public interface CourseResourceService {

    /**
     * 上传课程资源
     *
     * @param userId 用户ID
     * @param courseId 课程ID
     * @param file 文件
     * @param name 资源名称
     * @param description 资源描述
     * @return 上传的资源信息
     */
    CourseResourceDTO uploadResource(Long userId, Long courseId, MultipartFile file, String name, String description);

    /**
     * 获取课程资源列表
     *
     * @param courseId 课程ID
     * @return 资源列表
     */
    List<CourseResourceDTO> listResources(Long courseId);

    /**
     * 分页获取课程资源
     *
     * @param courseId 课程ID
     * @param pageRequest 分页请求
     * @return 分页资源列表
     */
    PageResponse<CourseResourceDTO> listByCourse(Long courseId, PageRequest pageRequest);

    /**
     * 删除课程资源
     *
     * @param resourceId 资源ID
     * @param userId 用户ID
     * @return 是否删除成功
     */
    boolean deleteCourseResource(Long resourceId, Long userId);

    /**
     * 获取资源详情
     *
     * @param resourceId 资源ID
     * @param userId 用户ID
     * @return 资源详情
     */
    CourseResourceDTO getResourceInfo(Long resourceId, Long userId);

    /**
     * 更新资源下载次数
     *
     * @param resourceId 资源ID
     * @return 是否更新成功
     */
    boolean incrementDownloadCount(Long resourceId);

    /**
     * 获取所有资源列表
     * @param pageRequest 分页请求
     * @return 资源列表
     */
    Result<Page<CourseResource>> getAllResources(PageRequest pageRequest);
} 