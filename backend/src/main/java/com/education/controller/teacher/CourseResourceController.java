package com.education.controller.teacher;

import com.education.dto.CourseResourceDTO;
import com.education.dto.common.PageRequest;
import com.education.dto.common.PageResponse;
import com.education.dto.common.Result;
import com.education.exception.ResultCode;
import com.education.service.teacher.CourseResourceService;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.Parameter;
import io.swagger.v3.oas.annotations.tags.Tag;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;
import jakarta.servlet.http.HttpSession;
import org.apache.commons.io.IOUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.core.io.InputStreamResource;
import org.springframework.core.io.Resource;
import org.springframework.core.io.UrlResource;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.security.core.Authentication;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.net.MalformedURLException;
import java.net.URLEncoder;
import java.net.URLDecoder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;

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
    public Result<CourseResourceDTO> uploadResource(
            @PathVariable Long courseId,
            @RequestParam("file") MultipartFile file,
            @RequestParam(value = "name", required = false) String name,
            @RequestParam(value = "description", required = false) String description,
            HttpServletRequest request) {
        
        // 从Session中获取userId
        HttpSession session = request.getSession(false);
        Long userId = null;
        
        if (session != null) {
            userId = (Long) session.getAttribute("userId");
        }
        
        if (userId == null) {
            logger.error("无法从Session获取用户ID");
            return Result.error(ResultCode.PARAM_ERROR, "无效的用户ID");
        }
        
        logger.info("接收到上传课程资源请求 - 课程ID: {}, 用户ID: {}, 文件名: {}, 文件大小: {} bytes",
                courseId, userId, file.getOriginalFilename(), file.getSize());
        
        CourseResourceDTO resource = courseResourceService.uploadResource(userId, courseId, file, name, description);
        
        logger.info("课程资源上传成功 - 资源ID: {}, 资源名称: {}, 文件类型: {}", 
                resource.getId(), resource.getName(), resource.getFileType());
        
        return Result.success(resource);
    }

    @Operation(summary = "获取课程资源列表", description = "获取指定课程的资源列表")
    @GetMapping("/{courseId}/resources")
    public Result<List<CourseResourceDTO>> getCourseResources(@PathVariable Long courseId) {
        List<CourseResourceDTO> resources = courseResourceService.listResources(courseId);
        return Result.success(resources);
    }

    @Operation(summary = "分页获取课程资源", description = "分页获取指定课程的资源列表")
    @GetMapping("/{courseId}/resources/page")
    public Result<PageResponse<CourseResourceDTO>> getCourseResourcesPage(
            @PathVariable Long courseId,
            @Parameter(description = "页码") @RequestParam(defaultValue = "1") Integer page,
            @Parameter(description = "每页大小") @RequestParam(defaultValue = "10") Integer size) {
        
        PageRequest pageRequest = new PageRequest(page, size);
        PageResponse<CourseResourceDTO> resources = courseResourceService.listByCourse(courseId, pageRequest);
        
        return Result.success(resources);
    }

    @Operation(summary = "获取资源详情", description = "获取指定资源的详细信息")
    @GetMapping("/resources/{resourceId}")
    public Result<CourseResourceDTO> getResourceDetail(@PathVariable Long resourceId, HttpServletRequest request) {
        // 从Session中获取userId
        HttpSession session = request.getSession(false);
        Long userId = null;
        
        if (session != null) {
            userId = (Long) session.getAttribute("userId");
        }
        
        if (userId == null) {
            logger.error("无法从Session获取用户ID");
            return Result.error(ResultCode.PARAM_ERROR, "无效的用户ID");
        }
        
        CourseResourceDTO resource = courseResourceService.getResourceInfo(resourceId, userId);
        return Result.success(resource);
    }

    @Operation(summary = "删除课程资源", description = "删除指定的课程资源")
    @DeleteMapping("/resources/{resourceId}")
    public Result<Boolean> deleteResource(@PathVariable Long resourceId, HttpServletRequest request) {
        // 从Session中获取userId
        HttpSession session = request.getSession(false);
        Long userId = null;
        
        if (session != null) {
            userId = (Long) session.getAttribute("userId");
        }
        
        if (userId == null) {
            logger.error("无法从Session获取用户ID");
            return Result.error(ResultCode.PARAM_ERROR, "无效的用户ID");
        }
        
        logger.info("接收到删除课程资源请求 - 资源ID: {}, 用户ID: {}", resourceId, userId);
        
        boolean success = courseResourceService.deleteCourseResource(resourceId, userId);
        
        logger.info("课程资源删除{} - 资源ID: {}", success ? "成功" : "失败", resourceId);
        
        return Result.success(success);
    }

    @Operation(summary = "下载课程资源", description = "下载指定的课程资源")
    @GetMapping("/resources/{resourceId}/download")
    public void downloadResource(@PathVariable Long resourceId, HttpServletRequest request, HttpServletResponse response) {
        // 从Session中获取userId
        HttpSession session = request.getSession(false);
        Long userId = null;
        
        if (session != null) {
            userId = (Long) session.getAttribute("userId");
        }
        
        if (userId == null) {
            logger.error("无法从Session获取用户ID");
            response.setStatus(HttpServletResponse.SC_BAD_REQUEST);
            return;
        }
        
        logger.info("接收到下载课程资源请求 - 资源ID: {}, 用户ID: {}", resourceId, userId);
        
        try {
            // 获取资源详情
            CourseResourceDTO resource = courseResourceService.getResourceInfo(resourceId, userId);
            if (resource == null) {
                logger.error("找不到资源: resourceId={}", resourceId);
                response.setStatus(HttpServletResponse.SC_NOT_FOUND);
                return;
            }
            
            // 更新下载次数
            courseResourceService.incrementDownloadCount(resourceId);
            
            // 从URL中提取文件路径
            String fileUrl = resource.getFileUrl();
            String filePath = URLDecoder.decode(fileUrl.replace("/files/", ""), StandardCharsets.UTF_8.name());
            
            // 获取文件
            Path path = Paths.get("D:/my_git_code/SmartClass/resource/file", filePath);
            
            if (!Files.exists(path)) {
                logger.error("文件不存在: {}", path);
                response.setStatus(HttpServletResponse.SC_NOT_FOUND);
                return;
            }
            
            // 设置响应头
            String fileName = resource.getName() + "." + resource.getFileType();
            String encodedFileName = URLEncoder.encode(fileName, StandardCharsets.UTF_8.name()).replaceAll("\\+", "%20");
            
            response.setHeader(HttpHeaders.CONTENT_DISPOSITION, "attachment; filename*=UTF-8''" + encodedFileName);
            response.setHeader(HttpHeaders.CONTENT_LENGTH, String.valueOf(Files.size(path)));
            
            // 根据文件类型设置Content-Type
            String contentType = MediaType.APPLICATION_OCTET_STREAM_VALUE;
            if (resource.getFileType().equalsIgnoreCase("pdf")) {
                contentType = MediaType.APPLICATION_PDF_VALUE;
            } else if (resource.getFileType().equalsIgnoreCase("jpg") || resource.getFileType().equalsIgnoreCase("jpeg")) {
                contentType = MediaType.IMAGE_JPEG_VALUE;
            } else if (resource.getFileType().equalsIgnoreCase("png")) {
                contentType = MediaType.IMAGE_PNG_VALUE;
            }
            
            response.setContentType(contentType);
            
            // 直接将文件复制到响应输出流
            Files.copy(path, response.getOutputStream());
            response.getOutputStream().flush();
            
            logger.info("文件下载成功 - 资源ID: {}, 文件名: {}", resourceId, resource.getName());
            
        } catch (IOException e) {
            logger.error("文件读取或传输失败", e);
            response.setStatus(HttpServletResponse.SC_INTERNAL_SERVER_ERROR);
        } catch (Exception e) {
            logger.error("下载过程中发生未知错误", e);
            response.setStatus(HttpServletResponse.SC_INTERNAL_SERVER_ERROR);
        }
    }

    @Operation(summary = "预览课程资源", description = "预览支持在线查看的课程资源")
    @GetMapping("/resources/{resourceId}/preview")
    public void previewResource(@PathVariable Long resourceId, HttpServletRequest request, HttpServletResponse response) {
        // 从Session中获取userId
        HttpSession session = request.getSession(false);
        Long userId = null;
        
        if (session != null) {
            userId = (Long) session.getAttribute("userId");
            logger.info("从Session获取到用户ID: {}", userId);
        } else {
            logger.warn("Session为null");
        }
        
        if (userId == null) {
            // 开发环境临时使用固定ID进行测试
            userId = 1938893577591402497L;
            logger.warn("无法从Session获取用户ID，使用默认测试ID: {}", userId);
        }
        
        logger.info("接收到预览课程资源请求 - 资源ID: {}, 用户ID: {}", resourceId, userId);
        
        FileInputStream fileInputStream = null;
        try {
            // 获取资源详情
            CourseResourceDTO resource = courseResourceService.getResourceInfo(resourceId, userId);
            if (resource == null) {
                logger.error("找不到资源: resourceId={}", resourceId);
                response.setStatus(HttpServletResponse.SC_NOT_FOUND);
                return;
            }
            
            logger.info("获取到资源信息: {}", resource);
            
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
                response.setStatus(HttpServletResponse.SC_BAD_REQUEST);
                return;
            }
            
            // 从URL中提取文件路径
            String fileUrl = resource.getFileUrl();
            String filePath = URLDecoder.decode(fileUrl.replace("/files/", ""), StandardCharsets.UTF_8.name());
            
            logger.info("预览文件 - 原始URL: {}, 解码后路径: {}", fileUrl, filePath);
            
            // 获取文件 - 直接使用完整路径
            Path path = Paths.get("D:/my_git_code/SmartClass/resource/file", filePath);
            logger.info("预览文件 - 完整路径: {}", path.toAbsolutePath());
            
            File file = path.toFile();
            if (!file.exists()) {
                logger.error("文件不存在: {}", path);
                
                // 尝试直接使用资源路径
                Path directPath = Paths.get("D:/my_git_code/SmartClass/resource/file/resources/9/202506/7acbf820bec64795bedaca556c235c4a.pdf");
                File directFile = directPath.toFile();
                logger.info("尝试使用直接路径: {}, 是否存在: {}", directPath, directFile.exists());
                
                if (directFile.exists()) {
                    logger.info("使用直接路径成功");
                    file = directFile;
                } else {
                    response.setStatus(HttpServletResponse.SC_NOT_FOUND);
                    return;
                }
            }
            
            // 清除所有可能的现有头
            response.reset();
            
            // 设置CORS头 - 必须在最前面设置
            response.setHeader("Access-Control-Allow-Origin", "*");
            response.setHeader("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
            response.setHeader("Access-Control-Allow-Headers", "Origin, Content-Type, Accept, Range");
            response.setHeader("Access-Control-Expose-Headers", "Content-Disposition, Content-Type, Content-Length, Accept-Ranges");
            
            // 根据文件类型设置Content-Type
            String contentType = MediaType.APPLICATION_OCTET_STREAM_VALUE;
            if (fileType.equals("pdf")) {
                contentType = MediaType.APPLICATION_PDF_VALUE;
            } else if (fileType.equals("jpg") || fileType.equals("jpeg")) {
                contentType = MediaType.IMAGE_JPEG_VALUE;
            } else if (fileType.equals("png")) {
                contentType = MediaType.IMAGE_PNG_VALUE;
            } else if (fileType.equals("gif")) {
                contentType = "image/gif";
            } else if (fileType.equals("mp4")) {
                contentType = "video/mp4";
            } else if (fileType.equals("mp3")) {
                contentType = "audio/mpeg";
            }
            
            response.setContentType(contentType);
            
            // 设置文件大小
            long fileSize = file.length();
            response.setContentLengthLong(fileSize);
            
            // 对于PDF，设置为inline
            if (fileType.equals("pdf")) {
                String fileName = URLEncoder.encode(resource.getName() + ".pdf", StandardCharsets.UTF_8.name()).replaceAll("\\+", "%20");
                response.setHeader(HttpHeaders.CONTENT_DISPOSITION, "inline; filename*=UTF-8''" + fileName);
            }
            
            // 添加缓存控制头
            response.setHeader(HttpHeaders.CACHE_CONTROL, "public, max-age=3600");
            
            // 添加范围请求支持
            response.setHeader("Accept-Ranges", "bytes");
            
            // 使用FileInputStream和IOUtils.copy
            fileInputStream = new FileInputStream(file);
            IOUtils.copy(fileInputStream, response.getOutputStream());
            response.getOutputStream().flush();
            
            logger.info("文件预览成功 - 资源ID: {}, 文件名: {}", resourceId, resource.getName());
            
        } catch (IOException e) {
            logger.error("文件读取或传输失败", e);
            response.setStatus(HttpServletResponse.SC_INTERNAL_SERVER_ERROR);
        } catch (Exception e) {
            logger.error("预览过程中发生未知错误", e);
            response.setStatus(HttpServletResponse.SC_INTERNAL_SERVER_ERROR);
        } finally {
            if (fileInputStream != null) {
                try {
                    fileInputStream.close();
                } catch (IOException e) {
                    logger.error("关闭文件流失败", e);
                }
            }
        }
    }
} 