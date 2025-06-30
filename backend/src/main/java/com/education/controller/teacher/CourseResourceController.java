package com.education.controller.teacher;

import com.education.dto.common.PageRequest;
import com.education.dto.common.PageResponse;
import com.education.dto.common.Result;
import com.education.entity.CourseResource;
import com.education.service.teacher.CourseResourceService;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.Parameter;
import io.swagger.v3.oas.annotations.tags.Tag;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.core.io.Resource;
import org.springframework.core.io.UrlResource;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.security.core.Authentication;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;
import java.net.MalformedURLException;
import java.net.URLDecoder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * 课程资源控制器
 */
@Tag(name = "教师-课程资源管理", description = "课程资源管理相关接口")
@RestController
@RequestMapping("/api/teacher/courses")
public class CourseResourceController {

    private static final Logger logger = LoggerFactory.getLogger(CourseResourceController.class);

    @Autowired
    private CourseResourceService courseResourceService;

    @Operation(summary = "上传课程资源", description = "上传课程资源文件")
    @PostMapping("/{courseId}/resources/upload")
    public Result<CourseResource> uploadResource(
            @PathVariable Long courseId,
            @RequestParam("file") MultipartFile file,
            @RequestParam(value = "name", required = false) String name,
            @RequestParam(value = "description", required = false) String description) {
        
        Authentication authentication = SecurityContextHolder.getContext().getAuthentication();
        String username = authentication.getName();
        
        logger.info("接收到上传课程资源请求 - 课程ID: {}, 用户名: {}, 文件名: {}, 文件大小: {} bytes",
                courseId, username, file.getOriginalFilename(), file.getSize());
        
        CourseResource resource = courseResourceService.uploadResource(username, courseId, file, name, description);
        
        logger.info("课程资源上传成功 - 资源ID: {}, 资源名称: {}, 文件类型: {}", 
                resource.getId(), resource.getName(), resource.getFileType());
        
        return Result.success(resource);
    }

    @Operation(summary = "获取课程资源列表", description = "获取指定课程的资源列表")
    @GetMapping("/{courseId}/resources")
    public Result<List<CourseResource>> getCourseResources(@PathVariable Long courseId) {
        Authentication authentication = SecurityContextHolder.getContext().getAuthentication();
        String username = authentication.getName();
        
        List<CourseResource> resources = courseResourceService.getCourseResources(username, courseId);
        return Result.success(resources);
    }

    @Operation(summary = "分页获取课程资源", description = "分页获取指定课程的资源列表")
    @GetMapping("/{courseId}/resources/page")
    public Result<PageResponse<CourseResource>> getCourseResourcesPage(
            @PathVariable Long courseId,
            @Parameter(description = "页码") @RequestParam(defaultValue = "1") Integer page,
            @Parameter(description = "每页大小") @RequestParam(defaultValue = "10") Integer size) {
        
        Authentication authentication = SecurityContextHolder.getContext().getAuthentication();
        String username = authentication.getName();
        
        PageRequest pageRequest = new PageRequest(page, size);
        PageResponse<CourseResource> resources = courseResourceService.getCourseResourcesPage(username, courseId, pageRequest);
        
        return Result.success(resources);
    }

    @Operation(summary = "获取资源详情", description = "获取指定资源的详细信息")
    @GetMapping("/resources/{resourceId}")
    public Result<CourseResource> getResourceDetail(@PathVariable Long resourceId) {
        Authentication authentication = SecurityContextHolder.getContext().getAuthentication();
        String username = authentication.getName();
        
        CourseResource resource = courseResourceService.getResourceDetail(username, resourceId);
        return Result.success(resource);
    }

    @Operation(summary = "删除课程资源", description = "删除指定的课程资源")
    @DeleteMapping("/resources/{resourceId}")
    public Result<Boolean> deleteResource(@PathVariable Long resourceId) {
        Authentication authentication = SecurityContextHolder.getContext().getAuthentication();
        String username = authentication.getName();
        
        logger.info("接收到删除课程资源请求 - 资源ID: {}, 用户名: {}", resourceId, username);
        
        boolean success = courseResourceService.deleteResource(username, resourceId);
        
        logger.info("课程资源删除{} - 资源ID: {}", success ? "成功" : "失败", resourceId);
        
        return Result.success(success);
    }

    @Operation(summary = "下载课程资源", description = "下载指定的课程资源")
    @GetMapping("/resources/{resourceId}/download")
    public ResponseEntity<Resource> downloadResource(@PathVariable Long resourceId) {
        Authentication authentication = SecurityContextHolder.getContext().getAuthentication();
        String username = authentication.getName();
        
        logger.info("接收到下载课程资源请求 - 资源ID: {}, 用户名: {}", resourceId, username);
        
        // 获取资源详情
        CourseResource resource = courseResourceService.getResourceDetail(username, resourceId);
        
        // 更新下载次数
        courseResourceService.incrementDownloadCount(resourceId);
        
        try {
            // 从URL中提取文件路径
            String fileUrl = resource.getFileUrl();
            String filePath = URLDecoder.decode(fileUrl.replace("/files/", ""), StandardCharsets.UTF_8.name());
            
            // 获取文件
            Path path = Paths.get(System.getProperty("user.dir"), "uploads", filePath);
            Resource fileResource = new UrlResource(path.toUri());
            
            if (!fileResource.exists()) {
                logger.error("文件不存在: {}", path);
                return ResponseEntity.notFound().build();
            }
            
            // 设置响应头
            HttpHeaders headers = new HttpHeaders();
            headers.add(HttpHeaders.CONTENT_DISPOSITION, "attachment; filename=\"" + resource.getName() + "." + resource.getFileType() + "\"");
            
            // 根据文件类型设置Content-Type
            MediaType mediaType = MediaType.APPLICATION_OCTET_STREAM;
            if (resource.getFileType().equalsIgnoreCase("pdf")) {
                mediaType = MediaType.APPLICATION_PDF;
            } else if (resource.getFileType().equalsIgnoreCase("jpg") || resource.getFileType().equalsIgnoreCase("jpeg")) {
                mediaType = MediaType.IMAGE_JPEG;
            } else if (resource.getFileType().equalsIgnoreCase("png")) {
                mediaType = MediaType.IMAGE_PNG;
            }
            
            logger.info("文件下载成功 - 资源ID: {}, 文件名: {}", resourceId, resource.getName());
            
            return ResponseEntity.ok()
                    .headers(headers)
                    .contentType(mediaType)
                    .body(fileResource);
            
        } catch (MalformedURLException e) {
            logger.error("文件URL格式错误", e);
            return ResponseEntity.badRequest().build();
        } catch (IOException e) {
            logger.error("文件读取失败", e);
            return ResponseEntity.internalServerError().build();
        }
    }

    @Operation(summary = "预览课程资源", description = "预览支持在线查看的课程资源")
    @GetMapping("/resources/{resourceId}/preview")
    public ResponseEntity<Resource> previewResource(@PathVariable Long resourceId) {
        Authentication authentication = SecurityContextHolder.getContext().getAuthentication();
        String username = authentication.getName();
        
        logger.info("接收到预览课程资源请求 - 资源ID: {}, 用户名: {}", resourceId, username);
        
        // 获取资源详情
        CourseResource resource = courseResourceService.getResourceDetail(username, resourceId);
        
        // 检查是否支持预览
        String fileType = resource.getFileType().toLowerCase();
        boolean supportPreview = fileType.equals("pdf") || 
                                 fileType.equals("jpg") || 
                                 fileType.equals("jpeg") || 
                                 fileType.equals("png") || 
                                 fileType.equals("gif") ||
                                 fileType.equals("mp4") ||
                                 fileType.equals("mp3");
        
        if (!supportPreview) {
            logger.warn("不支持预览的文件类型: {}", fileType);
            return ResponseEntity.badRequest().build();
        }
        
        try {
            // 从URL中提取文件路径
            String fileUrl = resource.getFileUrl();
            String filePath = URLDecoder.decode(fileUrl.replace("/files/", ""), StandardCharsets.UTF_8.name());
            
            // 获取文件
            Path path = Paths.get(System.getProperty("user.dir"), "uploads", filePath);
            Resource fileResource = new UrlResource(path.toUri());
            
            if (!fileResource.exists()) {
                logger.error("文件不存在: {}", path);
                return ResponseEntity.notFound().build();
            }
            
            // 设置响应头
            HttpHeaders headers = new HttpHeaders();
            
            // 根据文件类型设置Content-Type
            MediaType mediaType = MediaType.APPLICATION_OCTET_STREAM;
            if (fileType.equals("pdf")) {
                mediaType = MediaType.APPLICATION_PDF;
            } else if (fileType.equals("jpg") || fileType.equals("jpeg")) {
                mediaType = MediaType.IMAGE_JPEG;
            } else if (fileType.equals("png")) {
                mediaType = MediaType.IMAGE_PNG;
            } else if (fileType.equals("gif")) {
                mediaType = MediaType.parseMediaType("image/gif");
            } else if (fileType.equals("mp4")) {
                mediaType = MediaType.parseMediaType("video/mp4");
            } else if (fileType.equals("mp3")) {
                mediaType = MediaType.parseMediaType("audio/mpeg");
            }
            
            logger.info("文件预览成功 - 资源ID: {}, 文件名: {}", resourceId, resource.getName());
            
            return ResponseEntity.ok()
                    .headers(headers)
                    .contentType(mediaType)
                    .body(fileResource);
            
        } catch (MalformedURLException e) {
            logger.error("文件URL格式错误", e);
            return ResponseEntity.badRequest().build();
        } catch (IOException e) {
            logger.error("文件读取失败", e);
            return ResponseEntity.internalServerError().build();
        }
    }
} 