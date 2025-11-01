package com.education.controller.common;

import com.education.dto.common.Result;
import com.education.service.common.FileService;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;
import org.springframework.http.MediaType;
import org.springframework.core.io.FileSystemResource;
import org.springframework.core.io.Resource;
import org.springframework.http.ResponseEntity;
import org.springframework.web.context.request.RequestAttributes;
import org.springframework.web.context.request.RequestContextHolder;
import org.springframework.web.context.request.ServletRequestAttributes;

import jakarta.servlet.http.HttpServletResponse;
import java.io.File;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.Map;
import java.util.UUID;

/**
 * 文件管理控制器
 * 
 * @author Education Platform Team
 * @version 1.0.0
 * @since 2024
 */
@Tag(name = "公共-文件管理", description = "文件上传下载相关接口")
@RestController
@RequestMapping("/api/common/files")
public class FileController {

    private static final Logger logger = LoggerFactory.getLogger(FileController.class);

    @Autowired
    private FileService fileService;
    
    @Value("${file.upload.path:D:/my_git_code/SmartClass/resource/file}")
    private String fileUploadPath;
    
    @Value("${photo.upload.path:D:/my_git_code/SmartClass/resource/photo}")
    private String photoUploadPath;

    @Operation(summary = "上传文件", description = "上传文件并返回访问URL")
    @PostMapping("/upload")
    public Result<Map<String, String>> uploadFile(@RequestParam("file") MultipartFile file, 
                                                 @RequestParam(value = "type", required = false, defaultValue = "common") String type) {
        logger.info("接收到文件上传请求，类型: {}", type);
        logger.info("文件信息 - 名称: {}, 大小: {} bytes, 类型: {}", 
                file.getOriginalFilename(), file.getSize(), file.getContentType());
                
        try {
            String fileUrl = fileService.uploadFile(file, type);
            logger.info("文件上传成功，URL: {}", fileUrl);
            
            Map<String, String> result = new HashMap<>();
            result.put("url", fileUrl);
            return Result.success(result);
        } catch (Exception e) {
            logger.error("文件上传失败", e);
            return Result.error("文件上传失败: " + e.getMessage());
        }
    }

    @Operation(summary = "上传图片", description = "上传图片文件并返回访问URL")
    @PostMapping("/upload/image")
    public Result<Map<String, String>> uploadImage(@RequestParam("file") MultipartFile file) {
        logger.info("接收到图片上传请求");
        logger.info("图片信息 - 名称: {}, 大小: {} bytes, 类型: {}", 
                file.getOriginalFilename(), file.getSize(), file.getContentType());
                
        try {
            String fileUrl = fileService.uploadFile(file, "images");
            logger.info("图片上传成功，URL: {}", fileUrl);
            
            Map<String, String> result = new HashMap<>();
            result.put("url", fileUrl);
            return Result.success(result);
        } catch (Exception e) {
            logger.error("图片上传失败", e);
            return Result.error("图片上传失败: " + e.getMessage());
        }
    }

    @Operation(summary = "上传课程封面(旧方法)", description = "上传课程封面图片并返回访问URL")
    @PostMapping("/upload/course-cover")
    public Result<Map<String, String>> uploadCourseCover(@RequestParam("file") MultipartFile file) {
        logger.info("接收到课程封面上传请求");
        logger.info("封面信息 - 名称: {}, 大小: {} bytes, 类型: {}", 
                file.getOriginalFilename(), file.getSize(), file.getContentType());
                
        try {
            logger.info("准备调用fileService.uploadFile方法上传课程封面");
            String fileUrl = fileService.uploadFile(file, "courses/covers");
            logger.info("课程封面上传成功，URL: {}", fileUrl);
            
            Map<String, String> result = new HashMap<>();
            result.put("url", fileUrl);
            return Result.success(result);
        } catch (Exception e) {
            logger.error("课程封面上传失败: {}", e.getMessage(), e);
            return Result.error("课程封面上传失败: " + e.getMessage());
        }
    }
    
    /**
     * 新方法: 直接将课程封面上传到photo目录
     */
    @Operation(summary = "上传课程封面到photo目录", description = "将课程封面直接上传到photo目录并返回访问URL")
    @PostMapping("/upload/course-photo")
    public Result<Map<String, String>> uploadCoursePhoto(@RequestParam("file") MultipartFile file) {
        logger.info("接收到课程封面上传请求 (photo目录)");
        logger.info("封面信息 - 名称: {}, 大小: {} bytes, 类型: {}", 
                file.getOriginalFilename(), file.getSize(), file.getContentType());
                
        try {
            // 检查文件类型
            String originalFilename = file.getOriginalFilename();
            String extension = getFileExtension(originalFilename);
            logger.info("文件扩展名: {}", extension);
            
            if (!isImageExtension(extension)) {
                logger.error("不支持的图片类型: {}", extension);
                return Result.error("不支持的图片类型，仅支持: jpg, jpeg, png, gif");
            }
            
            // 创建保存路径
            String yearMonth = getYearMonth();
            Path dirPath = Paths.get(photoUploadPath, yearMonth);
            Files.createDirectories(dirPath);
            logger.info("创建目录: {}", dirPath);
            
            // 生成唯一文件名
            String newFilename = UUID.randomUUID().toString().replace("-", "") + "." + extension;
            Path filePath = dirPath.resolve(newFilename);
            logger.info("保存文件到: {}", filePath);
            
            // 保存文件
            file.transferTo(filePath.toFile());
            
            // 生成访问URL (使用新的photo路径格式)
            String fileUrl = "/api/photo/" + yearMonth + "/" + newFilename;
            logger.info("图片上传成功，URL: {}", fileUrl);
            
            Map<String, String> result = new HashMap<>();
            result.put("url", fileUrl);
            return Result.success(result);
            
        } catch (Exception e) {
            logger.error("课程封面上传失败: {}", e.getMessage(), e);
            return Result.error("课程封面上传失败: " + e.getMessage());
        }
    }

    @Operation(summary = "删除文件", description = "根据文件URL删除文件")
    @DeleteMapping
    public Result<Void> deleteFile(@RequestParam("url") String fileUrl) {
        logger.info("接收到文件删除请求，URL: {}", fileUrl);
        
        try {
            boolean success = fileService.deleteFile(fileUrl);
            if (success) {
                logger.info("文件删除成功");
            } else {
                logger.warn("文件删除失败，可能文件不存在");
            }
            return Result.success();
        } catch (Exception e) {
            logger.error("文件删除失败", e);
            return Result.error("文件删除失败: " + e.getMessage());
        }
    }
    
    /**
     * 获取文件，用于前端直接访问本地文件
     * 
     * @param filepath 文件相对路径，相对于上传根目录，例如 courses/covers/202507/xxx.png
     * @param response HTTP响应
     * @return 文件响应
     */
    @GetMapping("/get/**")
    public ResponseEntity<Resource> getFile(HttpServletResponse response) {
        try {
            // 获取请求路径，提取文件路径部分
            ServletRequestAttributes attributes = (ServletRequestAttributes) RequestContextHolder.getRequestAttributes();
            String requestPath = attributes.getRequest().getRequestURI();
            String filepath = requestPath.substring("/api/common/files/get/".length());
            
            logger.info("接收到文件获取请求，相对路径: {}", filepath);
            
            // 构建文件的实际路径
            Path filePath = Paths.get(fileUploadPath, filepath);
            File file = filePath.toFile();
            
            if (!file.exists() || !file.isFile()) {
                logger.error("文件不存在: {}", filePath);
                return ResponseEntity.notFound().build();
            }
            
            // 确定文件类型
            String contentType = determineContentType(file);
            
            // 创建文件资源
            FileSystemResource resource = new FileSystemResource(file);
            
            // 返回文件资源
            return ResponseEntity.ok()
                    .contentType(MediaType.parseMediaType(contentType))
                    .body(resource);
            
        } catch (Exception e) {
            logger.error("获取文件失败", e);
            return ResponseEntity.internalServerError().build();
        }
    }
    
    /**
     * 新方法: 获取photo目录下的图片
     * 注意这个方法的路径映射，使用了单独的/api/photo/**路径
     */
    @GetMapping(value = "/photo/{yearMonth}/{filename:.+}", produces = {MediaType.IMAGE_JPEG_VALUE, MediaType.IMAGE_PNG_VALUE, MediaType.IMAGE_GIF_VALUE})
    public ResponseEntity<Resource> getPhoto(
            @PathVariable("yearMonth") String yearMonth,
            @PathVariable("filename") String filename) {
        try {
            logger.info("接收到图片获取请求，年月: {}, 文件名: {}", yearMonth, filename);
            
            // 构建文件的实际路径
            Path filePath = Paths.get(photoUploadPath, yearMonth, filename);
            File file = filePath.toFile();
            
            if (!file.exists() || !file.isFile()) {
                logger.error("图片不存在: {}", filePath);
                return ResponseEntity.notFound().build();
            }
            
            // 确定文件类型
            String contentType = determineContentType(file);
            
            // 创建文件资源
            FileSystemResource resource = new FileSystemResource(file);
            
            // 返回文件资源
            return ResponseEntity.ok()
                    .contentType(MediaType.parseMediaType(contentType))
                    .header("Cache-Control", "max-age=86400") // 1天缓存
                    .body(resource);
            
        } catch (Exception e) {
            logger.error("获取图片失败: {}", e.getMessage(), e);
            return ResponseEntity.internalServerError().build();
        }
    }
    
    /**
     * 根据文件名确定Content-Type
     */
    private String determineContentType(File file) {
        try {
            return Files.probeContentType(file.toPath());
        } catch (IOException e) {
            logger.warn("无法确定文件类型，将使用通用二进制类型", e);
            
            // 根据文件扩展名判断
            String fileName = file.getName().toLowerCase();
            if (fileName.endsWith(".jpg") || fileName.endsWith(".jpeg")) {
                return "image/jpeg";
            } else if (fileName.endsWith(".png")) {
                return "image/png";
            } else if (fileName.endsWith(".gif")) {
                return "image/gif";
            } else if (fileName.endsWith(".pdf")) {
                return "application/pdf";
            } else {
                return "application/octet-stream";
            }
        }
    }
    
    /**
     * 获取文件扩展名
     */
    private String getFileExtension(String filename) {
        if (filename == null || filename.lastIndexOf(".") == -1) {
            return "";
        }
        return filename.substring(filename.lastIndexOf(".") + 1).toLowerCase();
    }
    
    /**
     * 判断是否为图片扩展名
     */
    private boolean isImageExtension(String extension) {
        return "jpg".equals(extension) || 
               "jpeg".equals(extension) || 
               "png".equals(extension) || 
               "gif".equals(extension);
    }
    
    /**
     * 获取当前年月，用于文件存储路径
     * 
     * @return 格式如：202507
     */
    private String getYearMonth() {
        java.time.LocalDate now = java.time.LocalDate.now();
        return String.format("%d%02d", now.getYear(), now.getMonthValue());
    }
} 