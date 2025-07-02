package com.education.service.common.impl;

import com.education.exception.BusinessException;
import com.education.exception.ResultCode;
import com.education.service.common.FileService;
import com.education.utils.FileUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.List;
import java.util.UUID;

/**
 * 文件服务实现类
 * 
 * @author Education Platform Team
 * @version 1.0.0
 * @since 2024
 */
@Service
public class FileServiceImpl implements FileService {
    
    private static final Logger logger = LoggerFactory.getLogger(FileServiceImpl.class);
    
    private static final List<String> ALLOWED_IMAGE_EXTENSIONS = Arrays.asList("jpg", "jpeg", "png", "gif");
    
    @Value("${file.upload.path:D:/my_git_code/SmartClass/resource/file}")
    private String uploadBasePath;
    
    @Value("${file.access.url.prefix:/files}")
    private String accessUrlPrefix;
    
    @Override
    public String uploadFile(MultipartFile file, String type) {
        if (file == null || file.isEmpty()) {
            throw new BusinessException(ResultCode.FILE_UPLOAD_ERROR, "上传文件不能为空");
        }
        
        // 检查文件类型
        String originalFilename = file.getOriginalFilename();
        String extension = FileUtils.getFileExtension(originalFilename);
        
        logger.info("开始处理文件上传 - 原始文件名: {}, 扩展名: {}, 类型: {}", originalFilename, extension, type);
        
        // 如果是图片类型，额外检查
        if ("images".equals(type) || "courses/covers".equals(type)) {
            if (!ALLOWED_IMAGE_EXTENSIONS.contains(extension.toLowerCase())) {
                logger.error("不支持的图片类型: {}", extension);
                throw new BusinessException(ResultCode.FILE_TYPE_ERROR, 
                        "不支持的图片类型，仅支持: " + String.join(", ", ALLOWED_IMAGE_EXTENSIONS));
            }
        }
        
        // 创建目录
        String relativePath = type + "/" + getYearMonth();
        Path uploadPath = Paths.get(uploadBasePath, relativePath);
        try {
            logger.info("创建上传目录: {}", uploadPath);
            // 确保基础目录存在
            Path baseDir = Paths.get(uploadBasePath);
            if (!Files.exists(baseDir)) {
                Files.createDirectories(baseDir);
                logger.info("基础目录创建成功: {}", baseDir);
            }
            
            // 确保子目录存在
            if (!Files.exists(uploadPath)) {
                Files.createDirectories(uploadPath);
                logger.info("子目录创建成功: {}", uploadPath);
            } else {
                logger.info("目录已存在");
            }
            
            // 检查目录权限
            if (!Files.isWritable(uploadPath)) {
                logger.error("目录没有写入权限: {}", uploadPath);
                throw new BusinessException(ResultCode.FILE_UPLOAD_ERROR, "上传目录没有写入权限: " + uploadPath);
            }
            
            // 生成新的文件名
            String newFilename = UUID.randomUUID().toString().replace("-", "") + "." + extension;
            Path filePath = uploadPath.resolve(newFilename);
            
            logger.info("准备保存文件到: {}", filePath);
            
            // 保存文件
            file.transferTo(filePath.toFile());
            logger.info("文件保存成功");
            
            // 返回访问URL
            String fileUrl = accessUrlPrefix + "/" + relativePath + "/" + newFilename;
            logger.info("生成文件访问URL: {}", fileUrl);
            
            return fileUrl;
            
        } catch (IOException e) {
            logger.error("文件上传失败", e);
            throw new BusinessException(ResultCode.FILE_UPLOAD_ERROR, "文件上传失败: " + e.getMessage());
        }
    }
    
    @Override
    public boolean deleteFile(String fileUrl) {
        if (fileUrl == null || fileUrl.isEmpty()) {
            return false;
        }
        
        // 从URL中提取相对路径
        String relativePath = fileUrl.replace(accessUrlPrefix + "/", "");
        Path filePath = Paths.get(uploadBasePath, relativePath);
        
        try {
            return Files.deleteIfExists(filePath);
        } catch (IOException e) {
            logger.error("文件删除失败", e);
            throw new BusinessException(ResultCode.FILE_DELETE_ERROR, "文件删除失败: " + e.getMessage());
        }
    }
    
    /**
     * 获取当前年月，用于文件存储路径
     * 
     * @return 格式如：202406
     */
    private String getYearMonth() {
        java.time.LocalDate now = java.time.LocalDate.now();
        return String.format("%d%02d", now.getYear(), now.getMonthValue());
    }
} 