package com.education.controller.common;

import com.education.dto.common.Result;
import com.education.service.common.FileService;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import jakarta.servlet.http.HttpServletResponse;
import java.util.HashMap;
import java.util.Map;

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

    @Operation(summary = "上传课程封面", description = "上传课程封面图片并返回访问URL")
    @PostMapping("/upload/course-cover")
    public Result<Map<String, String>> uploadCourseCover(@RequestParam("file") MultipartFile file) {
        logger.info("接收到课程封面上传请求");
        logger.info("封面信息 - 名称: {}, 大小: {} bytes, 类型: {}", 
                file.getOriginalFilename(), file.getSize(), file.getContentType());
                
        try {
            String fileUrl = fileService.uploadFile(file, "courses/covers");
            logger.info("课程封面上传成功，URL: {}", fileUrl);
            
            Map<String, String> result = new HashMap<>();
            result.put("url", fileUrl);
            return Result.success(result);
        } catch (Exception e) {
            logger.error("课程封面上传失败", e);
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
} 