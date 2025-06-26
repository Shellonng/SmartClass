package com.education.controller.teacher;

import com.education.dto.ResourceUploadRequest;
import com.education.dto.ResourceShareRequest;
import com.education.dto.ResourceBatchDeleteRequest;
import com.education.dto.ResourceBatchUploadRequest;
import com.education.dto.ResourceCopyRequest;
import com.education.dto.ResourceQueryParams;
import com.education.dto.ResourceMoveRequest;
import com.education.dto.FolderCreateRequest;
import com.education.dto.ResourceResponse;
import com.education.dto.ResourceDetailResponse;
import com.education.dto.ResourceUpdateRequest;
import com.education.dto.ResourceShareRecordResponse;
import com.education.dto.ResourceStatisticsResponse;
import com.education.dto.BatchUploadResponse;
import com.education.dto.FolderResponse;
import com.education.dto.common.PageRequest;
import com.education.dto.common.PageResponse;
import com.education.dto.common.Result;
import com.education.service.teacher.ResourceService;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;
import java.util.List;
import java.util.ArrayList;

/**
 * 教师端资源管理控制器
 * 
 * @author Education Platform Team
 * @version 1.0.0
 * @since 2024
 */
@Tag(name = "教师端-资源管理", description = "教师资源上传、管理、分享等接口")
@RestController("teacherResourceController")
@RequestMapping("/api/teacher/resources")
public class ResourceController {

    @Autowired
    private ResourceService resourceService;

    @Operation(summary = "上传资源", description = "教师上传教学资源")
    @PostMapping("/upload")
    public Result<Object> uploadResource(
            @RequestParam("file") MultipartFile file,
            @RequestParam(required = false) String title,
            @RequestParam(required = false) String description,
            @RequestParam(required = false) Long courseId,
            @RequestParam(required = false) String category) {
        try {
            Long teacherId = getCurrentTeacherId();
            ResourceUploadRequest uploadParams = buildUploadParams(file, title, description, courseId, category);
            ResourceResponse resource = resourceService.uploadResource(uploadParams, teacherId);
            return Result.success(resource);
        } catch (Exception e) {
            return Result.error("上传资源失败: " + e.getMessage());
        }
    }

    @Operation(summary = "获取资源列表", description = "获取教师的资源列表")
    @GetMapping
    public Result<Object> getResources(
            @RequestParam(defaultValue = "1") Integer page,
            @RequestParam(defaultValue = "10") Integer size,
            @RequestParam(required = false) String keyword,
            @RequestParam(required = false) Long courseId,
            @RequestParam(required = false) String category,
            @RequestParam(required = false) String fileType) {
        try {
            Long teacherId = getCurrentTeacherId();
            PageRequest pageRequest = new PageRequest(page, size);
            PageResponse<ResourceResponse> resources = resourceService.getResourceList(teacherId, pageRequest);
            return Result.success(resources);
        } catch (Exception e) {
            return Result.error("获取资源列表失败: " + e.getMessage());
        }
    }

    @Operation(summary = "获取资源详情", description = "获取指定资源的详细信息")
    @GetMapping("/{resourceId}")
    public Result<Object> getResourceDetail(@PathVariable Long resourceId) {
        try {
            Long teacherId = getCurrentTeacherId();
            ResourceDetailResponse resource = resourceService.getResourceDetail(resourceId, teacherId);
            return Result.success(resource);
        } catch (Exception e) {
            return Result.error("获取资源详情失败: " + e.getMessage());
        }
    }

    @Operation(summary = "更新资源信息", description = "更新资源的基本信息")
    @PutMapping("/{resourceId}")
    public Result<ResourceResponse> updateResource(@PathVariable Long resourceId, @RequestBody ResourceUpdateRequest updateRequest) {
        try {
            Long teacherId = getCurrentTeacherId();
            ResourceResponse resource = resourceService.updateResource(resourceId, updateRequest, teacherId);
            return Result.success(resource);
        } catch (Exception e) {
            return Result.error("更新资源失败: " + e.getMessage());
        }
    }

    @Operation(summary = "删除资源", description = "删除指定资源")
    @DeleteMapping("/{resourceId}")
    public Result<Void> deleteResource(@PathVariable Long resourceId) {
        try {
            Long teacherId = getCurrentTeacherId();
            resourceService.deleteResource(resourceId, teacherId);
            return Result.success();
        } catch (Exception e) {
            return Result.error("删除资源失败: " + e.getMessage());
        }
    }

    @Operation(summary = "批量删除资源", description = "批量删除多个资源")
    @DeleteMapping("/batch")
    public Result<Void> batchDeleteResources(@RequestBody ResourceBatchDeleteRequest deleteRequest) {
        try {
            Long teacherId = getCurrentTeacherId();
            resourceService.batchDeleteResources(deleteRequest.getResourceIds(), teacherId);
            return Result.success();
        } catch (Exception e) {
            return Result.error("批量删除资源失败: " + e.getMessage());
        }
    }

    @Operation(summary = "分享资源", description = "将资源分享给指定班级或学生")
    @PostMapping("/{resourceId}/share")
    public Result<Void> shareResource(@PathVariable Long resourceId, @RequestBody ResourceShareRequest shareRequest) {
        try {
            Long teacherId = getCurrentTeacherId();
            resourceService.shareResource(resourceId, shareRequest, teacherId);
            return Result.success();
        } catch (Exception e) {
            return Result.error("分享资源失败: " + e.getMessage());
        }
    }

    @Operation(summary = "取消分享", description = "取消资源的分享")
    @DeleteMapping("/{resourceId}/share")
    public Result<Void> unshareResource(@PathVariable Long resourceId) {
        try {
            Long teacherId = getCurrentTeacherId();
            resourceService.unshareResource(resourceId, teacherId);
            return Result.success();
        } catch (Exception e) {
            return Result.error("取消分享失败: " + e.getMessage());
        }
    }

    @Operation(summary = "获取资源分享记录", description = "获取资源的分享历史")
    @GetMapping("/{resourceId}/shares")
    public Result<Object> getResourceShares(@PathVariable Long resourceId,
            @RequestParam(defaultValue = "1") Integer page,
            @RequestParam(defaultValue = "10") Integer size) {
        try {
            Long teacherId = getCurrentTeacherId();
            PageRequest pageRequest = new PageRequest(page, size);
            PageResponse<ResourceShareRecordResponse> shares = resourceService.getResourceShareRecords(resourceId, teacherId, pageRequest);
            return Result.success(shares);
        } catch (Exception e) {
            return Result.error("获取分享记录失败: " + e.getMessage());
        }
    }

    @Operation(summary = "下载资源", description = "下载指定资源文件")
    @GetMapping("/{resourceId}/download")
    public Result<Object> downloadResource(@PathVariable Long resourceId) {
        try {
            Long teacherId = getCurrentTeacherId();
            Object downloadInfo = resourceService.downloadResource(resourceId, teacherId);
            return Result.success(downloadInfo);
        } catch (Exception e) {
            return Result.error("下载资源失败: " + e.getMessage());
        }
    }

    @Operation(summary = "获取资源统计", description = "获取教师的资源统计")
    @GetMapping("/statistics")
    public Result<Object> getResourceStatistics() {
        try {
            Long teacherId = getCurrentTeacherId();
            ResourceStatisticsResponse statistics = resourceService.getResourceStatistics(teacherId);
            return Result.success(statistics);
        } catch (Exception e) {
            return Result.error("获取资源统计失败: " + e.getMessage());
        }
    }

    @Operation(summary = "批量上传资源", description = "批量上传多个资源文件")
    @PostMapping("/batch-upload")
    public Result<Object> batchUploadResources(
            @RequestParam("files") MultipartFile[] files,
            @RequestParam(required = false) Long courseId,
            @RequestParam(required = false) String category) {
        try {
            Long teacherId = getCurrentTeacherId();
            ResourceBatchUploadRequest uploadParams = buildBatchUploadParams(files, courseId, category);
            BatchUploadResponse results = resourceService.batchUploadResources(uploadParams.getUploadRequests(), teacherId);
            return Result.success(results);
        } catch (Exception e) {
            return Result.error("批量上传资源失败: " + e.getMessage());
        }
    }

    @Operation(summary = "创建资源文件夹", description = "创建资源分类文件夹")
    @PostMapping("/folders")
    public Result<FolderResponse> createFolder(@RequestBody FolderCreateRequest createRequest) {
        try {
            Long teacherId = getCurrentTeacherId();
            FolderResponse folder = resourceService.createFolder(createRequest, teacherId);
            return Result.success(folder);
        } catch (Exception e) {
            return Result.error("创建文件夹失败: " + e.getMessage());
        }
    }

    @Operation(summary = "移动资源", description = "将资源移动到指定文件夹")
    @PostMapping("/{resourceId}/move")
    public Result<Void> moveResource(@PathVariable Long resourceId, @RequestBody ResourceMoveRequest moveRequest) {
        try {
            Long teacherId = getCurrentTeacherId();
            Boolean result = resourceService.moveResource(resourceId, moveRequest.getTargetFolderId(), teacherId);
            return Result.success();
        } catch (Exception e) {
            return Result.error("移动资源失败: " + e.getMessage());
        }
    }

    // 辅助方法
    private Long getCurrentTeacherId() {
        // TODO: 从JWT token或session中获取当前教师ID
        return 1L; // 临时返回值
    }

    private ResourceUploadRequest buildUploadParams(MultipartFile file, String title, String description, Long courseId, String category) {
        ResourceUploadRequest uploadRequest = new ResourceUploadRequest();
        uploadRequest.setFile(file);
        uploadRequest.setResourceName(title != null ? title : file.getOriginalFilename());
        uploadRequest.setDescription(description);
        uploadRequest.setCourseId(courseId);
        uploadRequest.setCategory(category);
        return uploadRequest;
    }

    private ResourceQueryParams buildResourceQueryParams(int page, int size, String keyword, Long courseId, String category, String fileType) {
        ResourceQueryParams queryParams = new ResourceQueryParams(page, size, keyword, courseId, category, fileType);
        return queryParams;
    }

    private ResourceBatchUploadRequest buildBatchUploadParams(MultipartFile[] files, Long courseId, String category) {
        ResourceBatchUploadRequest batchRequest = new ResourceBatchUploadRequest();
        List<ResourceUploadRequest> uploadRequests = new ArrayList<>();
        
        for (MultipartFile file : files) {
            ResourceUploadRequest uploadRequest = buildUploadParams(file, null, null, courseId, category);
            uploadRequests.add(uploadRequest);
        }
        
        batchRequest.setUploadRequests(uploadRequests);
        batchRequest.setCourseId(courseId);
        batchRequest.setCategory(category);
        return batchRequest;
    }
    @Operation(summary = "复制资源", description = "复制资源到指定位置")
    @PostMapping("/{resourceId}/copy")
    public Result<ResourceResponse> copyResource(@PathVariable Long resourceId, @RequestBody ResourceCopyRequest copyRequest) {
        try {
            Long teacherId = getCurrentTeacherId();
            ResourceResponse newResource = resourceService.copyResource(resourceId, copyRequest.getTargetCourseId(), teacherId);
            return Result.success(newResource);
        } catch (Exception e) {
            return Result.error("复制资源失败: " + e.getMessage());
        }
    }
}