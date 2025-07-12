package com.education.controller.common;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.core.io.FileSystemResource;
import org.springframework.core.io.Resource;
import org.springframework.http.HttpHeaders;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

@RestController
@RequestMapping("/resource")
public class MediaResourceController {

    private static final Logger logger = LoggerFactory.getLogger(MediaResourceController.class);
    
    @Value("${video.upload.path}")
    private String videoUploadPath;
    
    @Value("${file.upload.path}")
    private String fileUploadPath;
    
    /**
     * 获取视频资源
     * @param yearMonth 年月目录
     * @param filename 文件名
     * @return 视频资源
     */
    @GetMapping("/video/{yearMonth}/{filename}")
    public ResponseEntity<Resource> getVideo(
            @PathVariable String yearMonth,
            @PathVariable String filename) {
        
        logger.info("请求视频资源: {}/{}", yearMonth, filename);
        
        try {
            // 构建完整的文件路径
            Path filePath = Paths.get(videoUploadPath, yearMonth, filename);
            Resource resource = new FileSystemResource(filePath.toFile());
            
            // 检查文件是否存在
            if (!resource.exists()) {
                logger.error("视频文件不存在: {}", filePath);
                return ResponseEntity.notFound().build();
            }
            
            // 获取文件类型
            String contentType = Files.probeContentType(filePath);
            if (contentType == null) {
                contentType = "application/octet-stream";
            }
            
            logger.info("返回视频文件: {}, 类型: {}, 大小: {} bytes", 
                    filePath, contentType, resource.contentLength());
            
            return ResponseEntity.ok()
                    .contentType(MediaType.parseMediaType(contentType))
                    .header(HttpHeaders.CONTENT_DISPOSITION, "inline; filename=\"" + resource.getFilename() + "\"")
                    .body(resource);
            
        } catch (IOException e) {
            logger.error("获取视频文件失败", e);
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).build();
        }
    }
    
    /**
     * 获取文件资源
     * @param path 文件路径
     * @return 文件资源
     */
    @GetMapping("/file/**")
    public ResponseEntity<Resource> getFile() {
        String path = extractPathFromUrl();
        logger.info("请求文件资源: {}", path);
        
        try {
            // 构建完整的文件路径
            Path filePath = Paths.get(fileUploadPath, path);
            Resource resource = new FileSystemResource(filePath.toFile());
            
            // 检查文件是否存在
            if (!resource.exists()) {
                logger.error("文件不存在: {}", filePath);
                return ResponseEntity.notFound().build();
            }
            
            // 获取文件类型
            String contentType = Files.probeContentType(filePath);
            if (contentType == null) {
                contentType = "application/octet-stream";
            }
            
            logger.info("返回文件: {}, 类型: {}, 大小: {} bytes", 
                    filePath, contentType, resource.contentLength());
            
            return ResponseEntity.ok()
                    .contentType(MediaType.parseMediaType(contentType))
                    .header(HttpHeaders.CONTENT_DISPOSITION, "inline; filename=\"" + resource.getFilename() + "\"")
                    .body(resource);
            
        } catch (IOException e) {
            logger.error("获取文件失败", e);
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).build();
        }
    }
    
    /**
     * 从请求URL中提取文件路径
     * @return 文件路径
     */
    private String extractPathFromUrl() {
        String requestUri = org.springframework.web.context.request.RequestContextHolder
                .currentRequestAttributes()
                .getAttribute("javax.servlet.forward.request_uri", 0)
                .toString();
        
        // 移除前缀 "/resource/file/"
        return requestUri.substring("/resource/file/".length());
    }
} 