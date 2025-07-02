package com.education.service.common;

import org.springframework.web.multipart.MultipartFile;

/**
 * 文件服务接口
 * 
 * @author Education Platform Team
 * @version 1.0.0
 * @since 2024
 */
public interface FileService {
    
    /**
     * 上传文件
     * 
     * @param file 文件
     * @param type 文件类型
     * @return 文件访问URL
     */
    String uploadFile(MultipartFile file, String type);
    
    /**
     * 删除文件
     * 
     * @param fileUrl 文件URL
     * @return 是否删除成功
     */
    boolean deleteFile(String fileUrl);
} 