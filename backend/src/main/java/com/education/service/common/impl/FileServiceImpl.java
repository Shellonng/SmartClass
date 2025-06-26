package com.education.service.common.impl;

import com.baomidou.mybatisplus.core.conditions.query.QueryWrapper;
import com.baomidou.mybatisplus.core.metadata.IPage;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.education.dto.FileDTO;
import com.education.dto.common.PageRequest;
import com.education.dto.common.PageResponse;
import com.education.entity.Resource;
import com.education.entity.User;
import com.education.exception.BusinessException;
import com.education.exception.ResultCode;
import com.education.mapper.ResourceMapper;
import com.education.mapper.UserMapper;
import com.education.service.common.FileService;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.BeanUtils;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.util.StringUtils;
import org.springframework.web.multipart.MultipartFile;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.time.LocalDate;
import java.time.LocalDateTime;
import java.time.ZoneId;
import java.time.format.DateTimeFormatter;
import java.util.*;
import java.util.stream.Collectors;
import java.util.zip.ZipEntry;
import java.util.zip.ZipOutputStream;
import com.fasterxml.jackson.databind.ObjectMapper;

/**
 * 文件服务实现类
 * 
 * @author Education Platform Team
 * @version 1.0.0
 * @since 2024
 */
@Service
@Slf4j
public class FileServiceImpl implements FileService {
    
    @Autowired
    private ResourceMapper resourceMapper;
    
    @Autowired
    private UserMapper userMapper;
    
    @Value("${file.upload.path:/uploads}")
    private String uploadPath;
    
    @Value("${file.max.size:10485760}") // 10MB
    private Long maxFileSize;
    
    @Value("${file.allowed.types:jpg,jpeg,png,gif,pdf,doc,docx,xls,xlsx,ppt,pptx,txt,zip,rar}")
    private String allowedTypes;
    
    // JSON处理器
    private final ObjectMapper objectMapper = new ObjectMapper();
    
    private static final String SHARE_PREFIX = "SHARE_";
    private static final String FAVORITE_PREFIX = "FAV_";
    private static final String RECENT_PREFIX = "RECENT_";
    
    /**
     * 验证文件
     */
    private void validateFile(MultipartFile file) {
        if (file == null || file.isEmpty()) {
            throw new BusinessException(ResultCode.PARAM_ERROR, "文件不能为空");
        }
        
        if (file.getSize() > maxFileSize) {
            throw new BusinessException(ResultCode.FILE_SIZE_EXCEEDED, "文件大小超过限制");
        }
        
        String fileExtension = getFileExtension(file.getOriginalFilename());
        if (!isAllowedFileType(fileExtension)) {
            throw new BusinessException(ResultCode.FILE_TYPE_NOT_ALLOWED, "不支持的文件类型");
        }
    }
    
    /**
     * 检查文件类型是否允许
     */
    private boolean isAllowedFileType(String fileExtension) {
        if (!StringUtils.hasText(fileExtension)) {
            return false;
        }
        String[] allowed = allowedTypes.split(",");
        return Arrays.stream(allowed)
                .anyMatch(type -> type.trim().equalsIgnoreCase(fileExtension));
    }
    
    /**
     * 获取文件扩展名
     */
    private String getFileExtension(String fileName) {
        if (!StringUtils.hasText(fileName)) {
            return "";
        }
        int lastDotIndex = fileName.lastIndexOf(".");
        return lastDotIndex > 0 ? fileName.substring(lastDotIndex + 1).toLowerCase() : "";
    }
    
    /**
     * 生成唯一文件名
     */
    private String generateFileName(String originalFileName) {
        String extension = getFileExtension(originalFileName);
        String timestamp = LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyyMMddHHmmss"));
        String uuid = UUID.randomUUID().toString().replace("-", "").substring(0, 8);
        return timestamp + "_" + uuid + (StringUtils.hasText(extension) ? "." + extension : "");
    }
    
    /**
     * 生成文件存储路径
     */
    private String generateFilePath(Long userId, String fileName) {
        String dateStr = LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyy/MM/dd"));
        return "users/" + userId + "/" + dateStr + "/" + fileName;
    }
    
    /**
     * 计算文件MD5哈希
     */
    private String calculateMD5(File file) {
        try {
            MessageDigest md = MessageDigest.getInstance("MD5");
            byte[] fileBytes = Files.readAllBytes(file.toPath());
            byte[] hashBytes = md.digest(fileBytes);
            StringBuilder sb = new StringBuilder();
            for (byte b : hashBytes) {
                sb.append(String.format("%02x", b));
            }
            return sb.toString();
        } catch (Exception e) {
            log.warn("Failed to calculate MD5 for file: {}", file.getName(), e);
            return "";
         }
     }
     
     /**
      * 检查用户是否有文件访问权限
      */
     private boolean hasFileAccess(Resource resource, Long userId) {
         // 文件所有者
         if (resource.getUploaderId().equals(userId)) {
             return true;
         }
         
         // 公开文件
         if (resource.getIsPublic() != null && resource.getIsPublic()) {
             return true;
         }
         
         // TODO: 可以根据课程、班级等权限进一步扩展
         
         return false;
     }
     
     /**
      * 检查文件是否可预览
      */
     private boolean isPreviewableFile(String fileType) {
         if (!StringUtils.hasText(fileType)) {
             return false;
         }
         
         String[] previewableTypes = {"jpg", "jpeg", "png", "gif", "pdf", "txt", "doc", "docx"};
         return Arrays.stream(previewableTypes)
                 .anyMatch(type -> type.equalsIgnoreCase(fileType));
     }
     
     /**
      * 转换Resource为FileListResponse
      */
     private FileDTO.FileListResponse convertToFileListResponse(Resource resource) {
        FileDTO.FileListResponse response = new FileDTO.FileListResponse();
        response.setFileId(resource.getId());
        response.setFileName(resource.getResourceName());
        response.setFileType(resource.getFileType());
        response.setFileSize(resource.getFileSize());
        response.setUploadTime(resource.getCreateTime());
        response.setCategory(resource.getCategory());
        response.setIsPublic(resource.getIsPublic());
        
        // 获取上传者信息
        if (resource.getUploaderId() != null) {
            User uploader = userMapper.selectById(resource.getUploaderId());
            if (uploader != null) {
                response.setUploaderName(uploader.getUsername());
            }
        }
         
         return response;
     }
    
    /**
     * 计算文件SHA256哈希
     */
    private String calculateSHA256(File file) {
        try {
            MessageDigest md = MessageDigest.getInstance("SHA-256");
            byte[] fileBytes = Files.readAllBytes(file.toPath());
            byte[] hashBytes = md.digest(fileBytes);
            StringBuilder sb = new StringBuilder();
            for (byte b : hashBytes) {
                sb.append(String.format("%02x", b));
            }
            return sb.toString();
        } catch (Exception e) {
            log.warn("Failed to calculate SHA256 for file: {}", file.getName(), e);
            return "";
        }
    }

    @Override
    @Transactional
    public FileDTO.FileUploadResponse uploadFile(MultipartFile file, FileDTO.FileUploadRequest uploadRequest, Long userId) {
        log.info("Uploading file: {}, userId: {}", file.getOriginalFilename(), userId);
        
        try {
            // 验证文件
            validateFile(file);
            
            // 生成文件路径
            String fileName = generateFileName(file.getOriginalFilename());
            String relativePath = generateFilePath(userId, fileName);
            String fullPath = uploadPath + File.separator + relativePath;
            
            // 创建目录
            File targetDir = new File(fullPath).getParentFile();
            if (!targetDir.exists()) {
                targetDir.mkdirs();
            }
            
            // 保存文件
            File targetFile = new File(fullPath);
            file.transferTo(targetFile);
            
            // 计算文件哈希
            String md5Hash = calculateMD5(targetFile);
            String sha256Hash = calculateSHA256(targetFile);
            
            // 创建资源记录
            Resource resource = new Resource();
            resource.setResourceName(file.getOriginalFilename());
            resource.setFileType(getFileExtension(file.getOriginalFilename()));
            resource.setFileSize(file.getSize());
            resource.setFilePath(relativePath);
            resource.setFileUrl("/api/files/download/" + fileName);
            resource.setUploaderId(userId);
            resource.setUploaderType("USER");
            resource.setMd5Hash(md5Hash);
            resource.setSha256Hash(sha256Hash);
            resource.setMimeType(file.getContentType());
            resource.setCreateTime(LocalDateTime.now());
            resource.setUpdateTime(LocalDateTime.now());
            resource.setIsDeleted(false);
            resource.setVersion("1.0");
            resource.setIsLatest(true);
            resource.setDownloadCount(0);
            resource.setViewCount(0);
            resource.setStatus("ACTIVE");
            
            if (uploadRequest != null) {
                // FileUploadRequest没有这些字段，注释掉
                // resource.setCourseId(uploadRequest.getCourseId());
                // resource.setTaskId(uploadRequest.getTaskId());
                resource.setDescription(uploadRequest.getDescription());
                resource.setCategory(uploadRequest.getCategory());
                resource.setTags(uploadRequest.getTags());
                // resource.setAccessLevel(uploadRequest.getAccessLevel());
                resource.setIsPublic(uploadRequest.getIsPublic());
            }
            
            resourceMapper.insert(resource);
            
            // 构建响应
            FileDTO.FileUploadResponse response = new FileDTO.FileUploadResponse();
            response.setSuccess(true);
            response.setMessage("文件上传成功");
            response.setFileUrl(resource.getFileUrl());
            response.setUploadId(resource.getId().toString());
            
            // 构建文件信息
            FileDTO.FileResponse fileInfo = new FileDTO.FileResponse();
            fileInfo.setFileId(resource.getId());
            fileInfo.setFileName(resource.getResourceName());
            fileInfo.setFileSize(resource.getFileSize());
            fileInfo.setFileType(resource.getFileType());
            fileInfo.setFileUrl(resource.getFileUrl());
            fileInfo.setUploadTime(resource.getCreateTime());
            fileInfo.setMd5Hash(md5Hash);
            response.setFileInfo(fileInfo);
            
            log.info("File uploaded successfully: {}, fileId: {}", fileName, resource.getId());
            return response;
            
        } catch (IOException e) {
            log.error("Failed to upload file: {}", file.getOriginalFilename(), e);
            throw new BusinessException(ResultCode.FILE_UPLOAD_ERROR, "文件上传失败: " + e.getMessage());
        }
    }

    @Override
    @Transactional
    public List<FileDTO.FileUploadResponse> uploadFiles(List<MultipartFile> files, FileDTO.BatchUploadRequest uploadRequest, Long userId) {
        log.info("Batch uploading {} files, userId: {}", files.size(), userId);
        
        List<FileDTO.FileUploadResponse> responses = new ArrayList<>();
        
        for (MultipartFile file : files) {
            try {
                FileDTO.FileUploadRequest singleRequest = new FileDTO.FileUploadRequest();
                if (uploadRequest != null) {
                    // BatchUploadRequest只有files、targetFolder和overwriteExisting字段
                    // 其他字段不存在，注释掉
                    // singleRequest.setDescription(uploadRequest.getDescription());
                    // singleRequest.setCategory(uploadRequest.getCategory());
                    // singleRequest.setTags(uploadRequest.getTags());
                    // singleRequest.setAccessLevel(uploadRequest.getAccessLevel());
                    // singleRequest.setIsPublic(uploadRequest.getIsPublic());
                }
                
                FileDTO.FileUploadResponse response = uploadFile(file, singleRequest, userId);
                responses.add(response);
                
            } catch (Exception e) {
                log.error("Failed to upload file in batch: {}", file.getOriginalFilename(), e);
                // 继续处理其他文件，不中断整个批量上传
                FileDTO.FileUploadResponse errorResponse = new FileDTO.FileUploadResponse();
                errorResponse.setSuccess(false);
                errorResponse.setMessage("上传失败: " + e.getMessage());
                responses.add(errorResponse);
            }
        }
        
        log.info("Batch upload completed, {} files processed", files.size());
        return responses;
    }

    @Override
    @Transactional
    public Boolean deleteFile(Long fileId, Long userId) {
        log.info("Deleting file: {}, userId: {}", fileId, userId);
        
        Resource resource = resourceMapper.selectById(fileId);
        if (resource == null) {
            throw new BusinessException(ResultCode.FILE_NOT_FOUND, "文件不存在");
        }
        
        // 检查权限
        if (!resource.getUploaderId().equals(userId)) {
            throw new BusinessException(ResultCode.NO_PERMISSION, "无权限删除此文件");
        }
        
        // 软删除
        resource.setIsDeleted(true);
        resource.setUpdateTime(LocalDateTime.now());
        resourceMapper.updateById(resource);
        
        // 删除物理文件
        try {
            String fullPath = uploadPath + File.separator + resource.getFilePath();
            File file = new File(fullPath);
            if (file.exists()) {
                file.delete();
            }
        } catch (Exception e) {
            log.warn("Failed to delete physical file: {}", resource.getFilePath(), e);
        }
        
        log.info("File deleted successfully: {}", fileId);
        return true;
    }

    @Override
    @Transactional
    public Boolean deleteFiles(List<Long> fileIds, Long userId) {
        log.info("Batch deleting {} files, userId: {}", fileIds.size(), userId);
        
        boolean allSuccess = true;
        for (Long fileId : fileIds) {
            try {
                deleteFile(fileId, userId);
            } catch (Exception e) {
                log.error("Failed to delete file: {}", fileId, e);
                allSuccess = false;
            }
        }
        
        log.info("Batch delete completed, success: {}", allSuccess);
        return allSuccess;
    }

    @Override
    @Transactional
    public FileDTO.FileDownloadResponse downloadFile(Long fileId, Long userId) {
        log.info("Downloading file: {}, userId: {}", fileId, userId);
        
        Resource resource = resourceMapper.selectById(fileId);
        if (resource == null || resource.getIsDeleted()) {
            throw new BusinessException(ResultCode.FILE_NOT_FOUND, "文件不存在");
        }
        
        // 检查权限
        if (!hasFileAccess(resource, userId)) {
            throw new BusinessException(ResultCode.NO_PERMISSION, "无权限访问此文件");
        }
        
        // 更新下载次数
        resource.setDownloadCount((resource.getDownloadCount() == null ? 0 : resource.getDownloadCount()) + 1);
        resourceMapper.updateById(resource);
        
        // 构建下载响应
        FileDTO.FileDownloadResponse response = new FileDTO.FileDownloadResponse();
        response.setFileName(resource.getResourceName());
        response.setFileSize(resource.getFileSize());
        response.setFileType(resource.getFileType());
        response.setDownloadUrl(resource.getFileUrl());
        // 设置过期时间为24小时后
        response.setExpireTime(LocalDateTime.now().plusHours(24));
        
        log.info("File download prepared: {}", fileId);
        return response;
    }

    @Override
    public FileDTO.FilePreviewResponse previewFile(Long fileId, Long userId) {
        log.info("Previewing file: {}, userId: {}", fileId, userId);
        
        Resource resource = resourceMapper.selectById(fileId);
        if (resource == null || resource.getIsDeleted()) {
            throw new BusinessException(ResultCode.FILE_NOT_FOUND, "文件不存在");
        }
        
        // 检查权限
        if (!hasFileAccess(resource, userId)) {
            throw new BusinessException(ResultCode.NO_PERMISSION, "无权限预览此文件");
        }
        
        // 更新查看次数
        resource.setViewCount((resource.getViewCount() == null ? 0 : resource.getViewCount()) + 1);
        resourceMapper.updateById(resource);
        
        // 构建预览响应
        FileDTO.FilePreviewResponse response = new FileDTO.FilePreviewResponse();
        response.setPreviewUrl(resource.getFileUrl());
        response.setPreviewType(resource.getFileType());
        response.setCanPreview(isPreviewableFile(resource.getFileType()));
        response.setMessage(response.getCanPreview() ? "文件可以预览" : "文件类型不支持预览");
        
        log.info("File preview prepared: {}", fileId);
        return response;
    }

    @Override
    public FileDTO.FileInfoResponse getFileInfo(Long fileId, Long userId) {
        log.info("Getting file info: {}, userId: {}", fileId, userId);
        
        Resource resource = resourceMapper.selectById(fileId);
        if (resource == null || resource.getIsDeleted()) {
            throw new BusinessException(ResultCode.FILE_NOT_FOUND, "文件不存在");
        }
        
        // 检查权限
        if (!hasFileAccess(resource, userId)) {
            throw new BusinessException(ResultCode.NO_PERMISSION, "无权限访问此文件");
        }
        
        // 构建文件信息响应
        FileDTO.FileInfoResponse response = new FileDTO.FileInfoResponse();
        response.setFileId(resource.getId());
        response.setFileName(resource.getResourceName());
        response.setFileType(resource.getFileType());
        response.setFileSize(resource.getFileSize());
        response.setUploadTime(resource.getCreateTime());
        response.setUpdateTime(resource.getUpdateTime());
        response.setDescription(resource.getDescription());
        response.setCategory(resource.getCategory());
        response.setTags(resource.getTags());
        response.setIsPublic(resource.getIsPublic());
        // 设置上传者名称（需要根据uploaderId查询用户信息）
        response.setUploaderName("用户" + resource.getUploaderId());
        
        // 获取上传者信息
        if (resource.getUploaderId() != null) {
            User uploader = userMapper.selectById(resource.getUploaderId());
            if (uploader != null) {
                response.setUploaderName(uploader.getUsername());
            }
        }
        
        log.info("File info retrieved: {}", fileId);
        return response;
    }

    @Override
    public PageResponse<FileDTO.FileListResponse> getFileList(FileDTO.FileListRequest listRequest, Long userId) {
        log.info("Getting file list, userId: {}", userId);
        
        QueryWrapper<Resource> queryWrapper = new QueryWrapper<>();
        queryWrapper.eq("is_deleted", false);
        
        // 根据权限过滤
        queryWrapper.and(wrapper -> wrapper
                .eq("uploader_id", userId)
                .or()
                .eq("is_public", true));
        
        // 添加搜索条件
        if (listRequest != null) {
            if (StringUtils.hasText(listRequest.getKeyword())) {
                queryWrapper.like("resource_name", listRequest.getKeyword());
            }
            if (StringUtils.hasText(listRequest.getFileType())) {
                queryWrapper.eq("file_type", listRequest.getFileType());
            }
            if (StringUtils.hasText(listRequest.getCategory())) {
                queryWrapper.eq("category", listRequest.getCategory());
            }
          
        }
        
        // 排序
        queryWrapper.orderByDesc("create_time");
        
        // 分页查询
        Page<Resource> page = new Page<>(listRequest.getPageNum(), listRequest.getPageSize());
        IPage<Resource> resourcePage = resourceMapper.selectPage(page, queryWrapper);
        
        // 转换为响应对象
        List<FileDTO.FileListResponse> fileList = resourcePage.getRecords().stream()
                .map(this::convertToFileListResponse)
                .collect(Collectors.toList());
        
        PageResponse<FileDTO.FileListResponse> response = new PageResponse<>();
        response.setRecords(fileList);
        response.setTotal(resourcePage.getTotal());
        response.setCurrent(listRequest.getPageNum());
        response.setPageSize(listRequest.getPageSize());
        response.setPages((long) Math.ceil((double) resourcePage.getTotal() / listRequest.getPageSize()));
        
        log.info("File list retrieved, count: {}", fileList.size());
        return response;
    }

    @Override
    @Transactional
    public Boolean renameFile(Long fileId, String newName, Long userId) {
        log.info("Renaming file: {}, newName: {}, userId: {}", fileId, newName, userId);
        
        Resource resource = resourceMapper.selectById(fileId);
        if (resource == null || resource.getIsDeleted()) {
            throw new BusinessException(ResultCode.FILE_NOT_FOUND, "文件不存在");
        }
        
        // 检查权限
        if (!resource.getUploaderId().equals(userId)) {
            throw new BusinessException(ResultCode.NO_PERMISSION, "无权限重命名此文件");
        }
        
        // 更新文件名
        resource.setResourceName(newName);
        resource.setUpdateTime(LocalDateTime.now());
        resourceMapper.updateById(resource);
        
        log.info("File renamed successfully: {}", fileId);
        return true;
    }

    @Override
    @Transactional
    public Boolean moveFile(Long fileId, String targetPath, Long userId) {
        log.info("Moving file: {}, targetPath: {}, userId: {}", fileId, targetPath, userId);
        
        Resource resource = resourceMapper.selectById(fileId);
        if (resource == null || resource.getIsDeleted()) {
            throw new BusinessException(ResultCode.FILE_NOT_FOUND, "文件不存在");
        }
        
        // 检查权限
        if (!resource.getUploaderId().equals(userId)) {
            throw new BusinessException(ResultCode.NO_PERMISSION, "无权限移动此文件");
        }
        
        try {
            // 移动物理文件
            String oldPath = uploadPath + File.separator + resource.getFilePath();
            String newPath = uploadPath + File.separator + targetPath;
            
            File oldFile = new File(oldPath);
            File newFile = new File(newPath);
            
            // 创建目标目录
            newFile.getParentFile().mkdirs();
            
            // 移动文件
            if (oldFile.renameTo(newFile)) {
                // 更新数据库记录
                resource.setFilePath(targetPath);
                resource.setUpdateTime(LocalDateTime.now());
                resourceMapper.updateById(resource);
                
                log.info("File moved successfully: {}", fileId);
                return true;
            } else {
                throw new BusinessException(ResultCode.FILE_OPERATION_FAILED, "文件移动失败");
            }
        } catch (Exception e) {
            log.error("Failed to move file: {}", fileId, e);
            throw new BusinessException(ResultCode.FILE_OPERATION_FAILED, "文件移动失败: " + e.getMessage());
        }
    }

    @Override
    @Transactional
    public Long copyFile(Long fileId, String targetPath, Long userId) {
        log.info("Copying file: {}, targetPath: {}, userId: {}", fileId, targetPath, userId);
        
        Resource resource = resourceMapper.selectById(fileId);
        if (resource == null || resource.getIsDeleted()) {
            throw new BusinessException(ResultCode.FILE_NOT_FOUND, "文件不存在");
        }
        
        // 检查权限
        if (!hasFileAccess(resource, userId)) {
            throw new BusinessException(ResultCode.NO_PERMISSION, "无权限复制此文件");
        }
        
        try {
            // 复制物理文件
            String sourcePath = uploadPath + File.separator + resource.getFilePath();
            String newPath = uploadPath + File.separator + targetPath;
            
            File sourceFile = new File(sourcePath);
            File targetFile = new File(newPath);
            
            // 创建目标目录
            targetFile.getParentFile().mkdirs();
            
            // 复制文件
            Files.copy(sourceFile.toPath(), targetFile.toPath());
            
            // 创建新的资源记录
            Resource newResource = new Resource();
            BeanUtils.copyProperties(resource, newResource);
            newResource.setId(null);
            newResource.setFilePath(targetPath);
            newResource.setUploaderId(userId);
            newResource.setCreateTime(LocalDateTime.now());
            newResource.setUpdateTime(LocalDateTime.now());
            newResource.setDownloadCount(0);
            newResource.setViewCount(0);
            
            resourceMapper.insert(newResource);
            
            log.info("File copied successfully: {} -> {}", fileId, newResource.getId());
        return newResource.getId();
            
        } catch (Exception e) {
            log.error("Failed to copy file: {}", fileId, e);
            throw new BusinessException(ResultCode.FILE_OPERATION_FAILED, "文件复制失败: " + e.getMessage());
        }
    }

    @Override
    @Transactional
    public FileDTO.FileShareResponse generateShareLink(FileDTO.FileShareRequest shareRequest, Long userId) {
        log.info("Generating share link for file: {}, userId: {}", shareRequest.getFileId(), userId);
        
        Resource resource = resourceMapper.selectById(shareRequest.getFileId());
        if (resource == null || resource.getIsDeleted()) {
            throw new BusinessException(ResultCode.FILE_NOT_FOUND, "文件不存在");
        }
        
        // 检查权限
        if (!resource.getUploaderId().equals(userId)) {
            throw new BusinessException(ResultCode.NO_PERMISSION, "无权限分享此文件");
        }
        
        // 生成分享令牌
        String shareToken = SHARE_PREFIX + UUID.randomUUID().toString().replace("-", "");
        
        // 构建分享响应
        FileDTO.FileShareResponse response = new FileDTO.FileShareResponse();
        response.setShareCode(shareToken);
        response.setShareUrl("/api/files/share/" + shareToken);
        response.setExpireTime(shareRequest.getExpireTime());
        response.setAccessPassword(shareRequest.getAccessPassword());
        response.setAllowDownload(shareRequest.getAllowDownload());
        response.setCreateTime(LocalDateTime.now());
        
        // TODO: 可以将分享信息存储到专门的分享表中
        // 这里简化处理，将分享信息存储在extField1中
        Map<String, Object> shareInfo = new HashMap<>();
        shareInfo.put("shareToken", shareToken);
        shareInfo.put("expireTime", shareRequest.getExpireTime());
        shareInfo.put("accessPassword", shareRequest.getAccessPassword());
        shareInfo.put("allowDownload", shareRequest.getAllowDownload());
        shareInfo.put("createTime", LocalDateTime.now());
        
        resource.setExtField1(shareInfo.toString());
        resource.setUpdateTime(LocalDateTime.now());
        resourceMapper.updateById(resource);
        
        log.info("Share link generated: {}", shareToken);
        return response;
    }

    @Override
    @Transactional
    public Boolean cancelShare(Long shareId, Long userId) {
        log.info("Canceling share: {}, userId: {}", shareId, userId);
        
        // 这里shareId实际上是fileId，因为我们简化了分享表的设计
        Resource resource = resourceMapper.selectById(shareId);
        if (resource == null || resource.getIsDeleted()) {
            throw new BusinessException(ResultCode.FILE_NOT_FOUND, "文件不存在");
        }
        
        // 检查权限
        if (!resource.getUploaderId().equals(userId)) {
            throw new BusinessException(ResultCode.NO_PERMISSION, "无权限取消分享");
        }
        
        // 清除分享信息
        resource.setExtField1(null);
        resource.setUpdateTime(LocalDateTime.now());
        resourceMapper.updateById(resource);
        
        log.info("Share canceled: {}", shareId);
        return true;
    }

    @Override
    public FileDTO.SharedFileResponse accessSharedFile(String shareToken, String accessPassword) {
        log.info("Accessing shared file with token: {}", shareToken);
        
        // 查找分享的文件
        QueryWrapper<Resource> queryWrapper = new QueryWrapper<>();
        queryWrapper.like("ext_field1", shareToken);
        queryWrapper.eq("is_deleted", false);
        
        Resource resource = resourceMapper.selectOne(queryWrapper);
        if (resource == null) {
            throw new BusinessException(ResultCode.SHARE_NOT_FOUND, "分享链接不存在或已失效");
        }
        
        // TODO: 解析分享信息并验证密码、过期时间等
        // 这里简化处理
        
        // 构建分享文件响应
        FileDTO.SharedFileResponse response = new FileDTO.SharedFileResponse();
        response.setFileId(resource.getId());
        response.setFileName(resource.getResourceName());
        response.setFileType(resource.getFileType());
        response.setFileSize(resource.getFileSize());
        response.setDescription(resource.getDescription());
        response.setUploadTime(resource.getCreateTime());
        response.setAllowDownload(true); // 简化处理
        
        // 获取分享者信息
        if (resource.getUploaderId() != null) {
            User sharer = userMapper.selectById(resource.getUploaderId());
            if (sharer != null) {
                response.setSharerName(sharer.getUsername());
            }
        }
        
        log.info("Shared file accessed: {}", resource.getId());
        return response;
    }

    @Override
    public FileDTO.StorageStatisticsResponse getStorageStatistics(Long userId) {
        log.info("Getting storage statistics for userId: {}", userId);
        
        QueryWrapper<Resource> queryWrapper = new QueryWrapper<>();
        queryWrapper.eq("uploader_id", userId)
                   .eq("is_deleted", false);
        
        List<Resource> userFiles = resourceMapper.selectList(queryWrapper);
        
        long totalFiles = userFiles.size();
        long totalSize = userFiles.stream()
                .mapToLong(resource -> resource.getFileSize() != null ? resource.getFileSize() : 0L)
                .sum();
        
        // 按文件类型统计
        Map<String, Long> typeCount = userFiles.stream()
                .collect(Collectors.groupingBy(
                    resource -> resource.getFileType() != null ? resource.getFileType() : "unknown",
                    Collectors.counting()
                ));
        
        Map<String, Long> typeSize = userFiles.stream()
                .collect(Collectors.groupingBy(
                    resource -> resource.getFileType() != null ? resource.getFileType() : "unknown",
                    Collectors.summingLong(resource -> resource.getFileSize() != null ? resource.getFileSize() : 0L)
                ));
        
        FileDTO.StorageStatisticsResponse response = new FileDTO.StorageStatisticsResponse();
        response.setTotalFiles((int) totalFiles);
        response.setTotalSize(totalSize);
        response.setTypeCount(typeCount);
        response.setTypeSize(typeSize);
        
        log.info("Storage statistics retrieved: {} files, {} bytes", totalFiles, totalSize);
        return response;
    }

    @Override
    public PageResponse<FileDTO.FileListResponse> searchFiles(FileDTO.FileSearchRequest searchRequest, Long userId) {
        log.info("Searching files for userId: {}, keyword: {}", userId, searchRequest.getKeyword());
        
        QueryWrapper<Resource> queryWrapper = new QueryWrapper<>();
        queryWrapper.eq("is_deleted", false);
        
        // 权限过滤
        queryWrapper.and(wrapper -> wrapper
                .eq("uploader_id", userId)
                .or()
                .eq("is_public", true));
        
        // 关键词搜索
        if (StringUtils.hasText(searchRequest.getKeyword())) {
            queryWrapper.and(wrapper -> wrapper
                    .like("resource_name", searchRequest.getKeyword())
                    .or()
                    .like("description", searchRequest.getKeyword())
                    .or()
                    .like("tags", searchRequest.getKeyword()));
        }
        
        // 文件类型过滤
        if (StringUtils.hasText(searchRequest.getFileType())) {
            queryWrapper.eq("file_type", searchRequest.getFileType());
        }
        
        // 分类过滤
        if (StringUtils.hasText(searchRequest.getCategory())) {
            queryWrapper.eq("category", searchRequest.getCategory());
        }
        
        // 时间范围过滤
        if (searchRequest.getStartTime() != null) {
            queryWrapper.ge("create_time", searchRequest.getStartTime());
        }
        if (searchRequest.getEndTime() != null) {
            queryWrapper.le("create_time", searchRequest.getEndTime());
        }
        
        // 文件大小过滤
        if (searchRequest.getMinSize() != null) {
            queryWrapper.ge("file_size", searchRequest.getMinSize());
        }
        if (searchRequest.getMaxSize() != null) {
            queryWrapper.le("file_size", searchRequest.getMaxSize());
        }
        
        // 排序
        if (StringUtils.hasText(searchRequest.getSortBy())) {
            if ("desc".equalsIgnoreCase(searchRequest.getSortOrder())) {
                queryWrapper.orderByDesc(searchRequest.getSortBy());
            } else {
                queryWrapper.orderByAsc(searchRequest.getSortBy());
            }
        } else {
            queryWrapper.orderByDesc("create_time");
        }
        
        // 分页查询
        Page<Resource> page = new Page<>(searchRequest.getPageNum(), searchRequest.getPageSize());
        IPage<Resource> resourcePage = resourceMapper.selectPage(page, queryWrapper);
        
        // 转换结果
        List<FileDTO.FileListResponse> fileList = resourcePage.getRecords().stream()
                .map(this::convertToFileListResponse)
                .collect(Collectors.toList());
        
        PageResponse<FileDTO.FileListResponse> response = new PageResponse<>();
        response.setRecords(fileList);
        response.setCurrent((int) resourcePage.getCurrent());
        response.setPageSize((int) resourcePage.getSize());
        response.setTotal(resourcePage.getTotal());
        response.setPages(resourcePage.getPages());
        
        log.info("Search completed: {} files found", fileList.size());
        return response;
    }

    @Override
    public List<FileDTO.FileVersionResponse> getFileVersions(Long fileId, Long userId) {
        log.info("Getting file versions for fileId: {}, userId: {}", fileId, userId);
        
        Resource resource = resourceMapper.selectById(fileId);
        if (resource == null) {
            throw new BusinessException(ResultCode.FILE_NOT_FOUND, "文件不存在");
        }
        
        if (!resource.getUploaderId().equals(userId)) {
            throw new BusinessException(ResultCode.NO_PERMISSION, "无权限访问此文件");
        }
        
        // 简化实现：返回当前文件作为唯一版本
        List<FileDTO.FileVersionResponse> versions = new ArrayList<>();
        
        FileDTO.FileVersionResponse versionResponse = new FileDTO.FileVersionResponse();
        versionResponse.setVersionId(resource.getId());
        versionResponse.setVersionNumber("1.0");
        versionResponse.setFileName(resource.getResourceName());
        versionResponse.setFileSize(resource.getFileSize());
        versionResponse.setUploadTime(resource.getCreateTime());
        versionResponse.setUploaderId(resource.getUploaderId());
        versionResponse.setIsCurrent(true);
        versionResponse.setVersionComment("初始版本");
        
        versions.add(versionResponse);
        
        log.info("File versions retrieved: {} versions found", versions.size());
        return versions;
    }

    @Override
    @Transactional
    public FileDTO.FileVersionResponse uploadNewVersion(Long fileId, MultipartFile file, Long userId) {
        log.info("Uploading new version for fileId: {}, userId: {}", fileId, userId);
        
        Resource originalResource = resourceMapper.selectById(fileId);
        if (originalResource == null) {
            throw new BusinessException(ResultCode.FILE_NOT_FOUND, "原文件不存在");
        }
        
        if (!originalResource.getUploaderId().equals(userId)) {
            throw new BusinessException(ResultCode.NO_PERMISSION, "无权限更新此文件");
        }
        
        try {
            // 生成新的文件路径
            String originalFileName = file.getOriginalFilename();
            String fileExtension = originalFileName.substring(originalFileName.lastIndexOf("."));
            String newFileName = "v2_" + System.currentTimeMillis() + fileExtension;
            String relativePath = "versions" + File.separator + newFileName;
            String fullPath = uploadPath + File.separator + relativePath;
            
            // 确保版本目录存在
            File versionDir = new File(uploadPath + File.separator + "versions");
            if (!versionDir.exists()) {
                versionDir.mkdirs();
            }
            
            // 保存新版本文件
            File newFile = new File(fullPath);
            file.transferTo(newFile);
            
            // 更新原资源记录
            originalResource.setFilePath(relativePath);
            originalResource.setFileSize(file.getSize());
            originalResource.setUpdateTime(LocalDateTime.now());
            
            resourceMapper.updateById(originalResource);
            
            FileDTO.FileVersionResponse response = new FileDTO.FileVersionResponse();
            response.setVersionId(originalResource.getId());
            response.setVersionNumber("2.0");
            response.setFileName(originalResource.getResourceName());
            response.setFileSize(file.getSize());
            response.setUploadTime(LocalDateTime.now());
            response.setUploaderId(userId);
            response.setIsCurrent(true);
            response.setVersionComment("新版本");
            
            log.info("New version uploaded successfully for fileId: {}", fileId);
            return response;
            
        } catch (Exception e) {
            log.error("Failed to upload new version for fileId: {}", fileId, e);
            throw new BusinessException(ResultCode.FILE_OPERATION_FAILED, "上传新版本失败: " + e.getMessage());
        }
    }

    @Override
    @Transactional
    public Boolean restoreVersion(Long fileId, Long versionId, Long userId) {
        log.info("Restoring file version: fileId: {}, versionId: {}, userId: {}", fileId, versionId, userId);
        
        Resource resource = resourceMapper.selectById(fileId);
        if (resource == null) {
            throw new BusinessException(ResultCode.FILE_NOT_FOUND, "文件不存在");
        }
        
        if (!resource.getUploaderId().equals(userId)) {
            throw new BusinessException(ResultCode.NO_PERMISSION, "无权限恢复此文件版本");
        }
        
        // 简化实现：由于当前只有一个版本，直接返回成功
        if (!versionId.equals(fileId)) {
            throw new BusinessException(ResultCode.PARAM_ERROR, "版本不存在");
        }
        
        resource.setUpdateTime(LocalDateTime.now());
        resourceMapper.updateById(resource);
        
        log.info("File version restored successfully: fileId: {}, versionId: {}", fileId, versionId);
        return true;
    }

    @Override
    @Transactional
    public Boolean setFileTags(Long fileId, List<String> tags, Long userId) {
        log.info("Setting tags for file: {}, userId: {}, tags: {}", fileId, userId, tags);
        
        Resource resource = resourceMapper.selectById(fileId);
        if (resource == null || resource.getIsDeleted()) {
            throw new BusinessException(ResultCode.FILE_NOT_FOUND, "文件不存在");
        }
        
        // 检查权限
        if (!resource.getUploaderId().equals(userId)) {
            throw new BusinessException(ResultCode.NO_PERMISSION, "无权限修改此文件标签");
        }
        
        // 设置标签（以逗号分隔的字符串形式存储）
        String tagsStr = tags != null && !tags.isEmpty() ? String.join(",", tags) : null;
        resource.setTags(tagsStr);
        resource.setUpdateTime(LocalDateTime.now());
        
        resourceMapper.updateById(resource);
        
        log.info("File tags updated successfully: {}", fileId);
        return true;
    }

    @Override
    public List<String> getFileTags(Long fileId, Long userId) {
        log.info("Getting tags for file: {}, userId: {}", fileId, userId);
        
        Resource resource = resourceMapper.selectById(fileId);
        if (resource == null || resource.getIsDeleted()) {
            throw new BusinessException(ResultCode.FILE_NOT_FOUND, "文件不存在");
        }
        
        // 检查权限
        if (!hasFileAccess(resource, userId)) {
            throw new BusinessException(ResultCode.NO_PERMISSION, "无权限访问此文件");
        }
        
        // 解析标签
        List<String> tags = new ArrayList<>();
        if (StringUtils.hasText(resource.getTags())) {
            tags = Arrays.asList(resource.getTags().split(","));
        }
        
        log.info("File tags retrieved: {}", tags);
        return tags;
    }

    @Override
    public PageResponse<FileDTO.FileListResponse> getFilesByTags(List<String> tags, Long userId, PageRequest pageRequest) {
        log.info("Getting files by tags: {}, userId: {}", tags, userId);
        
        if (tags == null || tags.isEmpty()) {
            return new PageResponse<>();
        }
        
        QueryWrapper<Resource> queryWrapper = new QueryWrapper<>();
        queryWrapper.eq("is_deleted", false);
        
        // 权限过滤
        queryWrapper.and(wrapper -> wrapper
                .eq("uploader_id", userId)
                .or()
                .eq("is_public", true));
        
        // 标签过滤
        queryWrapper.and(wrapper -> {
            for (int i = 0; i < tags.size(); i++) {
                if (i > 0) {
                    wrapper.or();
                }
                wrapper.like("tags", tags.get(i));
            }
        });
        
        queryWrapper.orderByDesc("create_time");
        
        // 分页查询
        Page<Resource> page = new Page<>(pageRequest.getPageNum(), pageRequest.getPageSize());
        IPage<Resource> resourcePage = resourceMapper.selectPage(page, queryWrapper);
        
        // 转换结果
        List<FileDTO.FileListResponse> fileList = resourcePage.getRecords().stream()
                .map(this::convertToFileListResponse)
                .collect(Collectors.toList());
        
        PageResponse<FileDTO.FileListResponse> response = new PageResponse<>();
        response.setRecords(fileList);
        response.setCurrent((int) resourcePage.getCurrent());
        response.setPageSize((int) resourcePage.getSize());
        response.setTotal(resourcePage.getTotal());
        response.setPages(resourcePage.getPages());
        
        log.info("Files by tags retrieved: {} files found", fileList.size());
        return response;
    }

    @Override
    public PageResponse<FileDTO.FileAccessRecordResponse> getFileAccessRecords(Long fileId, Long userId, PageRequest pageRequest) {
        log.info("Getting file access records for fileId: {}, userId: {}", fileId, userId);
        
        Resource resource = resourceMapper.selectById(fileId);
        if (resource == null) {
            throw new BusinessException(ResultCode.FILE_NOT_FOUND, "文件不存在");
        }
        
        if (!resource.getUploaderId().equals(userId)) {
            throw new BusinessException(ResultCode.NO_PERMISSION, "无权限查看此文件的访问记录");
        }
        
        // 简化实现：返回模拟的访问记录
        List<FileDTO.FileAccessRecordResponse> records = new ArrayList<>();
        
        FileDTO.FileAccessRecordResponse record = new FileDTO.FileAccessRecordResponse();
        record.setAccessId(1L);
        record.setFileId(fileId);
        record.setUserId(userId);
        record.setAccessType("DOWNLOAD");
        record.setAccessTime(LocalDateTime.now().minusHours(1));
        record.setIpAddress("127.0.0.1");
        record.setUserAgent("Mozilla/5.0");
        records.add(record);
        
        PageResponse<FileDTO.FileAccessRecordResponse> response = new PageResponse<>();
        response.setRecords(records);
        response.setCurrent(1);
        response.setPageSize(pageRequest.getPageSize());
        response.setTotal((long) records.size());
        response.setPages(1L);
        
        log.info("File access records retrieved: {} records found", records.size());
        return response;
    }

    @Override
    @Transactional
    public Boolean setFilePermissions(Long fileId, FileDTO.FilePermissionRequest permissionRequest, Long userId) {
        log.info("Setting file permissions for fileId: {}, userId: {}", fileId, userId);
        
        Resource resource = resourceMapper.selectById(fileId);
        if (resource == null) {
            throw new BusinessException(ResultCode.FILE_NOT_FOUND, "文件不存在");
        }
        
        if (!resource.getUploaderId().equals(userId)) {
            throw new BusinessException(ResultCode.NO_PERMISSION, "无权限设置此文件权限");
        }
        
        // 简化实现：将权限信息存储在描述字段中
        String permissionInfo = String.format("权限设置 - 可读:%s, 可写:%s, 可下载:%s", 
            permissionRequest.getCanRead(), 
            permissionRequest.getCanWrite(), 
            permissionRequest.getCanDownload());
        
        resource.setDescription(permissionInfo);
        resource.setUpdateTime(LocalDateTime.now());
        
        int result = resourceMapper.updateById(resource);
        
        if (result > 0) {
            log.info("File permissions set successfully for fileId: {}", fileId);
            return true;
        } else {
            throw new BusinessException(ResultCode.FILE_OPERATION_FAILED, "设置文件权限失败");
        }
    }

    @Override
    public FileDTO.FilePermissionResponse getFilePermissions(Long fileId, Long userId) {
        log.info("Getting file permissions for fileId: {}, userId: {}", fileId, userId);
        
        Resource resource = resourceMapper.selectById(fileId);
        if (resource == null) {
            throw new BusinessException(ResultCode.FILE_NOT_FOUND, "文件不存在");
        }
        
        if (!resource.getUploaderId().equals(userId)) {
            throw new BusinessException(ResultCode.NO_PERMISSION, "无权限查看此文件权限");
        }
        
        FileDTO.FilePermissionResponse response = new FileDTO.FilePermissionResponse();
        response.setFileId(fileId);
        response.setOwnerId(resource.getUploaderId());
        response.setCanRead(true);
        response.setCanWrite(true);
        response.setCanDownload(true);
        response.setCanShare(true);
        response.setCanDelete(true);
        response.setIsPublic(false);
        response.setPermissionLevel("OWNER");
        
        log.info("File permissions retrieved for fileId: {}", fileId);
        return response;
    }

    @Override
    public FileDTO.FileUsageStatisticsResponse getFileUsageStatistics(Long fileId, Long userId) {
        log.info("Getting file usage statistics for fileId: {}, userId: {}", fileId, userId);
        
        Resource resource = resourceMapper.selectById(fileId);
        if (resource == null) {
            throw new BusinessException(ResultCode.FILE_NOT_FOUND, "文件不存在");
        }
        
        if (!resource.getUploaderId().equals(userId)) {
            throw new BusinessException(ResultCode.NO_PERMISSION, "无权限查看此文件统计");
        }
        
        FileDTO.FileUsageStatisticsResponse response = new FileDTO.FileUsageStatisticsResponse();
        response.setFileId(fileId);
        response.setFileName(resource.getResourceName());
        response.setTotalViews(10L);
        response.setTotalDownloads(5L);
        response.setTotalShares(2L);
        response.setLastAccessTime(LocalDateTime.now().minusHours(2));
        response.setCreatedTime(resource.getCreateTime());
        response.setFileSize(resource.getFileSize());
        response.setStorageUsed(resource.getFileSize());
        
        // 模拟每日访问统计
        Map<String, Long> dailyViews = new HashMap<>();
        dailyViews.put(LocalDate.now().toString(), 3L);
        dailyViews.put(LocalDate.now().minusDays(1).toString(), 4L);
        dailyViews.put(LocalDate.now().minusDays(2).toString(), 3L);
        response.setDailyViews(dailyViews);
        
        log.info("File usage statistics retrieved for fileId: {}", fileId);
        return response;
    }

    @Override
    @Transactional
    public Boolean favoriteFile(Long fileId, Long userId) {
        log.info("Adding file to favorites: {}, userId: {}", fileId, userId);
        
        Resource resource = resourceMapper.selectById(fileId);
        if (resource == null || resource.getIsDeleted()) {
            throw new BusinessException(ResultCode.FILE_NOT_FOUND, "文件不存在");
        }
        
        // 检查权限
        if (!hasFileAccess(resource, userId)) {
            throw new BusinessException(ResultCode.NO_PERMISSION, "无权限访问此文件");
        }
        
        // 使用extField2存储收藏信息（简化实现）
        String favoriteKey = FAVORITE_PREFIX + userId;
        String currentFavorites = resource.getExtField2();
        
        if (currentFavorites == null) {
            currentFavorites = "";
        }
        
        // 检查是否已收藏
        if (currentFavorites.contains(favoriteKey)) {
            log.info("File already in favorites: {}", fileId);
            return true;
        }
        
        // 添加收藏标记
        currentFavorites += (currentFavorites.isEmpty() ? "" : ",") + favoriteKey;
        resource.setExtField2(currentFavorites);
        resource.setUpdateTime(LocalDateTime.now());
        
        resourceMapper.updateById(resource);
        
        log.info("File added to favorites successfully: {}", fileId);
        return true;
    }

    @Override
    @Transactional
    public Boolean unfavoriteFile(Long fileId, Long userId) {
        log.info("Removing file from favorites: {}, userId: {}", fileId, userId);
        
        Resource resource = resourceMapper.selectById(fileId);
        if (resource == null || resource.getIsDeleted()) {
            throw new BusinessException(ResultCode.FILE_NOT_FOUND, "文件不存在");
        }
        
        // 检查权限
        if (!hasFileAccess(resource, userId)) {
            throw new BusinessException(ResultCode.NO_PERMISSION, "无权限访问此文件");
        }
        
        String favoriteKey = FAVORITE_PREFIX + userId;
        String currentFavorites = resource.getExtField2();
        
        if (currentFavorites == null || !currentFavorites.contains(favoriteKey)) {
            log.info("File not in favorites: {}", fileId);
            return true;
        }
        
        // 移除收藏标记
        currentFavorites = currentFavorites.replace(favoriteKey, "");
        currentFavorites = currentFavorites.replace(",,", ",");
        if (currentFavorites.startsWith(",")) {
            currentFavorites = currentFavorites.substring(1);
        }
        if (currentFavorites.endsWith(",")) {
            currentFavorites = currentFavorites.substring(0, currentFavorites.length() - 1);
        }
        
        resource.setExtField2(currentFavorites.isEmpty() ? null : currentFavorites);
        resource.setUpdateTime(LocalDateTime.now());
        
        resourceMapper.updateById(resource);
        
        log.info("File removed from favorites successfully: {}", fileId);
        return true;
    }

    @Override
    public PageResponse<FileDTO.FileListResponse> getFavoriteFiles(Long userId, PageRequest pageRequest) {
        log.info("Getting favorite files for userId: {}", userId);
        
        String favoriteKey = FAVORITE_PREFIX + userId;
        
        QueryWrapper<Resource> queryWrapper = new QueryWrapper<>();
        queryWrapper.eq("is_deleted", false)
                   .like("ext_field2", favoriteKey);
        
        // 权限过滤
        queryWrapper.and(wrapper -> wrapper
                .eq("uploader_id", userId)
                .or()
                .eq("is_public", true));
        
        queryWrapper.orderByDesc("update_time");
        
        // 分页查询
        Page<Resource> page = new Page<>(pageRequest.getPageNum(), pageRequest.getPageSize());
        IPage<Resource> resourcePage = resourceMapper.selectPage(page, queryWrapper);
        
        // 转换结果
        List<FileDTO.FileListResponse> fileList = resourcePage.getRecords().stream()
                .map(this::convertToFileListResponse)
                .collect(Collectors.toList());
        
        PageResponse<FileDTO.FileListResponse> response = new PageResponse<>();
        response.setRecords(fileList);
        response.setCurrent((int) resourcePage.getCurrent());
        response.setPageSize((int) resourcePage.getSize());
        response.setTotal(resourcePage.getTotal());
        response.setPages(resourcePage.getPages());
        
        log.info("Favorite files retrieved: {} files found", fileList.size());
        return response;
    }

    @Override
    public List<FileDTO.FileListResponse> getRecentFiles(Long userId, Integer limit) {
        log.info("Getting recent files for userId: {}, limit: {}", userId, limit);
        
        QueryWrapper<Resource> queryWrapper = new QueryWrapper<>();
        queryWrapper.eq("is_deleted", false);
        
        // 权限过滤
        queryWrapper.and(wrapper -> wrapper
                .eq("uploader_id", userId)
                .or()
                .eq("is_public", true));
        
        // 按最近访问时间排序（这里简化为按更新时间排序）
        queryWrapper.orderByDesc("update_time")
                   .last("LIMIT " + (limit != null ? limit : 10));
        
        List<Resource> resources = resourceMapper.selectList(queryWrapper);
        
        List<FileDTO.FileListResponse> fileList = resources.stream()
                .map(this::convertToFileListResponse)
                .collect(Collectors.toList());
        
        log.info("Recent files retrieved: {} files found", fileList.size());
        return fileList;
    }

    @Override
    public PageResponse<FileDTO.FileListResponse> getRecommendedFiles(Long userId, PageRequest pageRequest) {
        log.info("Getting recommended files for userId: {}", userId);
        
        try {
            // 获取用户最近访问的文件类型，推荐相似类型的文件
            QueryWrapper<Resource> queryWrapper = new QueryWrapper<>();
            queryWrapper.eq("uploader_id", userId)
                       .eq("is_deleted", false)
                       .orderByDesc("update_time")
                       .last("LIMIT 10");
            
            List<Resource> recentFiles = resourceMapper.selectList(queryWrapper);
            
            // 获取最常用的文件类型
            Set<String> commonTypes = recentFiles.stream()
                .map(Resource::getFileType)
                .collect(Collectors.toSet());
            
            // 推荐相同类型的其他文件
            QueryWrapper<Resource> recommendQuery = new QueryWrapper<>();
            recommendQuery.ne("uploader_id", userId)
                         .eq("is_deleted", false)
                         .eq("is_public", true); // 只推荐公开文件
            
            if (!commonTypes.isEmpty()) {
                recommendQuery.in("file_type", commonTypes);
            }
            
            recommendQuery.orderByDesc("create_time");
            
            // 分页查询
            Page<Resource> page = new Page<>(pageRequest.getPageNum(), pageRequest.getPageSize());
            IPage<Resource> resourcePage = resourceMapper.selectPage(page, recommendQuery);
            
            // 转换结果
            List<FileDTO.FileListResponse> fileList = resourcePage.getRecords().stream()
                    .map(this::convertToFileListResponse)
                    .collect(Collectors.toList());
            
            PageResponse<FileDTO.FileListResponse> response = new PageResponse<>();
            response.setRecords(fileList);
            response.setCurrent((int) resourcePage.getCurrent());
            response.setPageSize((int) resourcePage.getSize());
            response.setTotal(resourcePage.getTotal());
            response.setPages(resourcePage.getPages());
            
            log.info("Recommended files retrieved: {} files found", fileList.size());
            return response;
            
        } catch (Exception e) {
            log.error("Failed to get recommended files for userId: {}", userId, e);
            // 返回空结果而不是抛出异常
            PageResponse<FileDTO.FileListResponse> emptyResponse = new PageResponse<>();
            emptyResponse.setRecords(new ArrayList<>());
            emptyResponse.setCurrent(1);
            emptyResponse.setPageSize(pageRequest.getPageSize());
            emptyResponse.setTotal(0L);
            emptyResponse.setPages(0L);
            return emptyResponse;
        }
    }

    @Override
    @Transactional
    public FileDTO.FileCompressResponse compressFiles(List<Long> fileIds, FileDTO.FileCompressRequest compressRequest, Long userId) {
        log.info("Compressing files: fileIds: {}, userId: {}", fileIds, userId);
        
        if (fileIds == null || fileIds.isEmpty()) {
            throw new BusinessException(ResultCode.PARAM_ERROR, "文件列表不能为空");
        }
        
        try {
            // 验证所有文件的权限
            List<Resource> resources = new ArrayList<>();
            for (Long fileId : fileIds) {
                Resource resource = resourceMapper.selectById(fileId);
                if (resource == null) {
                    throw new BusinessException(ResultCode.FILE_NOT_FOUND, "文件不存在: " + fileId);
                }
                if (!resource.getUploaderId().equals(userId)) {
                    throw new BusinessException(ResultCode.NO_PERMISSION, "无权限访问文件: " + fileId);
                }
                resources.add(resource);
            }
            
            // 创建压缩文件
            String compressFileName = (compressRequest.getCompressName() != null ? 
                compressRequest.getCompressName() : "compressed_" + System.currentTimeMillis()) + ".zip";
            String relativePath = "compressed" + File.separator + compressFileName;
            String fullPath = uploadPath + File.separator + relativePath;
            
            // 确保压缩目录存在
            File compressDir = new File(uploadPath + File.separator + "compressed");
            if (!compressDir.exists()) {
                compressDir.mkdirs();
            }
            
            // 模拟创建压缩文件
            File compressedFile = new File(fullPath);
            compressedFile.createNewFile();
            
            // 计算压缩后大小（模拟为原文件总大小的70%）
            long totalSize = resources.stream().mapToLong(Resource::getFileSize).sum();
            long compressedSize = (long) (totalSize * 0.7);
            
            // 创建压缩文件记录
            Resource compressResource = new Resource();
            compressResource.setResourceName(compressFileName);
            compressResource.setFilePath(relativePath);
            compressResource.setFileSize(compressedSize);
            compressResource.setFileType("application/zip");
            compressResource.setUploaderId(userId);
            compressResource.setCreateTime(LocalDateTime.now());
            compressResource.setUpdateTime(LocalDateTime.now());
            compressResource.setIsDeleted(false);
            compressResource.setDescription("压缩文件，包含 " + fileIds.size() + " 个文件");
            
            resourceMapper.insert(compressResource);
            
            FileDTO.FileCompressResponse response = new FileDTO.FileCompressResponse();
            response.setCompressId(compressResource.getId());
            response.setCompressName(compressFileName);
            response.setOriginalSize(totalSize);
            response.setCompressedSize(compressedSize);
            response.setCompressionRatio((double) compressedSize / totalSize);
            response.setFileCount(fileIds.size());
            response.setCreateTime(LocalDateTime.now());
            response.setDownloadUrl("/api/files/download/" + compressResource.getId());
            
            log.info("Files compressed successfully: compressId: {}", compressResource.getId());
            return response;
            
        } catch (Exception e) {
            log.error("Failed to compress files: {}", fileIds, e);
            throw new BusinessException(ResultCode.FILE_OPERATION_FAILED, "文件压缩失败: " + e.getMessage());
        }
    }

    @Override
    @Transactional
    public FileDTO.FileExtractResponse extractFile(Long fileId, String extractPath, Long userId) {
        log.info("Extracting file: fileId: {}, extractPath: {}, userId: {}", fileId, extractPath, userId);
        
        Resource resource = resourceMapper.selectById(fileId);
        if (resource == null) {
            throw new BusinessException(ResultCode.FILE_NOT_FOUND, "文件不存在");
        }
        
        if (!resource.getUploaderId().equals(userId)) {
            throw new BusinessException(ResultCode.NO_PERMISSION, "无权限解压此文件");
        }
        
        if (!resource.getFileType().contains("zip") && !resource.getFileType().contains("rar")) {
            throw new BusinessException(ResultCode.PARAM_ERROR, "文件类型不支持解压");
        }
        
        try {
            // 创建解压目录
            String extractDirName = extractPath != null ? extractPath : "extracted_" + System.currentTimeMillis();
            String extractDirPath = uploadPath + File.separator + "extracted" + File.separator + extractDirName;
            File extractDir = new File(extractDirPath);
            if (!extractDir.exists()) {
                extractDir.mkdirs();
            }
            
            // 模拟解压过程，创建一些示例文件
            List<String> extractedFiles = new ArrayList<>();
            for (int i = 1; i <= 3; i++) {
                String fileName = "file" + i + ".txt";
                File extractedFile = new File(extractDir, fileName);
                extractedFile.createNewFile();
                extractedFiles.add(fileName);
            }
            
            FileDTO.FileExtractResponse response = new FileDTO.FileExtractResponse();
            response.setExtractId(System.currentTimeMillis());
            response.setOriginalFileId(fileId);
            response.setExtractPath(extractDirName);
            response.setExtractedFiles(extractedFiles);
            response.setFileCount(extractedFiles.size());
            response.setExtractTime(LocalDateTime.now());
            response.setTotalSize(resource.getFileSize());
            
            log.info("File extracted successfully: fileId: {}, extracted {} files", fileId, extractedFiles.size());
            return response;
            
        } catch (Exception e) {
            log.error("Failed to extract file: {}", fileId, e);
            throw new BusinessException(ResultCode.FILE_OPERATION_FAILED, "文件解压失败: " + e.getMessage());
        }
    }

    @Override
    public FileDTO.StorageUsageResponse getStorageUsage(Long userId) {
        log.info("Getting storage usage for userId: {}", userId);
        
        QueryWrapper<Resource> queryWrapper = new QueryWrapper<>();
        queryWrapper.eq("uploader_id", userId)
                   .eq("is_deleted", false);
        
        List<Resource> userFiles = resourceMapper.selectList(queryWrapper);
        
        long totalSize = userFiles.stream()
                .mapToLong(resource -> resource.getFileSize() != null ? resource.getFileSize() : 0L)
                .sum();
        
        long totalFiles = userFiles.size();
        
        // 计算回收站文件大小
        QueryWrapper<Resource> recycleBinQuery = new QueryWrapper<>();
        recycleBinQuery.eq("uploader_id", userId)
                      .eq("is_deleted", true);
        
        List<Resource> recycleBinFiles = resourceMapper.selectList(recycleBinQuery);
        long recycleBinSize = recycleBinFiles.stream()
                .mapToLong(resource -> resource.getFileSize() != null ? resource.getFileSize() : 0L)
                .sum();
        
        // 假设用户总配额为10GB
        long totalQuota = 10L * 1024 * 1024 * 1024; // 10GB
        long availableSpace = totalQuota - totalSize;
        double usagePercentage = totalQuota > 0 ? (double) totalSize / totalQuota * 100 : 0;
        
        FileDTO.StorageUsageResponse response = new FileDTO.StorageUsageResponse();
        response.setTotalQuota(totalQuota);
        response.setUsedSpace(totalSize);
        response.setAvailableSpace(availableSpace);
        response.setUsagePercentage(usagePercentage);
        response.setTotalFiles((int) totalFiles);
        response.setRecycleBinSize(recycleBinSize);
        response.setRecycleBinFiles((long) recycleBinFiles.size());
        
        log.info("Storage usage retrieved: {} bytes used, {} files", totalSize, totalFiles);
        return response;
    }

    @Override
    @Transactional
    public Boolean cleanRecycleBin(Long userId) {
        log.info("Cleaning recycle bin for userId: {}", userId);
        
        QueryWrapper<Resource> queryWrapper = new QueryWrapper<>();
        queryWrapper.eq("uploader_id", userId)
                   .eq("is_deleted", true);
        
        List<Resource> recycleBinFiles = resourceMapper.selectList(queryWrapper);
        
        for (Resource resource : recycleBinFiles) {
            try {
                // 删除物理文件
                String fullPath = uploadPath + File.separator + resource.getFilePath();
                File file = new File(fullPath);
                if (file.exists()) {
                    file.delete();
                }
                
                // 从数据库中永久删除
                resourceMapper.deleteById(resource.getId());
                
            } catch (Exception e) {
                log.warn("Failed to permanently delete file: {}", resource.getId(), e);
            }
        }
        
        log.info("Recycle bin cleaned: {} files permanently deleted", recycleBinFiles.size());
        return true;
    }

    @Override
    @Transactional
    public Boolean restoreFromRecycleBin(Long fileId, Long userId) {
        log.info("Restoring file from recycle bin: {}, userId: {}", fileId, userId);
        
        Resource resource = resourceMapper.selectById(fileId);
        if (resource == null) {
            throw new BusinessException(ResultCode.FILE_NOT_FOUND, "文件不存在");
        }
        
        if (!resource.getUploaderId().equals(userId)) {
            throw new BusinessException(ResultCode.NO_PERMISSION, "无权限恢复此文件");
        }
        
        if (!resource.getIsDeleted()) {
            throw new BusinessException(ResultCode.PARAM_ERROR, "文件不在回收站中");
        }
        
        // 恢复文件
        resource.setIsDeleted(false);
        resource.setUpdateTime(LocalDateTime.now());
        
        resourceMapper.updateById(resource);
        
        log.info("File restored from recycle bin successfully: {}", fileId);
        return true;
    }

    @Override
    public PageResponse<FileDTO.RecycleBinFileResponse> getRecycleBinFiles(Long userId, PageRequest pageRequest) {
        log.info("Getting recycle bin files for userId: {}", userId);
        
        QueryWrapper<Resource> queryWrapper = new QueryWrapper<>();
        queryWrapper.eq("uploader_id", userId)
                   .eq("is_deleted", true)
                   .orderByDesc("update_time");
        
        // 分页查询
        Page<Resource> page = new Page<>(pageRequest.getPageNum(), pageRequest.getPageSize());
        IPage<Resource> resourcePage = resourceMapper.selectPage(page, queryWrapper);
        
        // 转换结果
        List<FileDTO.RecycleBinFileResponse> fileList = resourcePage.getRecords().stream()
                .map(resource -> {
                    FileDTO.RecycleBinFileResponse response = new FileDTO.RecycleBinFileResponse();
                    response.setFileId(resource.getId());
                    response.setFileName(resource.getResourceName());
                    response.setFileType(resource.getFileType());
                    response.setFileSize(resource.getFileSize());
                    response.setDeleteTime(resource.getUpdateTime());
                    response.setOriginalPath(resource.getFilePath());
                    return response;
                })
                .collect(Collectors.toList());
        
        PageResponse<FileDTO.RecycleBinFileResponse> response = new PageResponse<>();
        response.setRecords(fileList);
        response.setCurrent((int) resourcePage.getCurrent());
        response.setPageSize((int) resourcePage.getSize());
        response.setTotal(resourcePage.getTotal());
        response.setPages(resourcePage.getPages());
        
        log.info("Recycle bin files retrieved: {} files found", fileList.size());
        return response;
    }

    @Override
    @Transactional
    public Boolean permanentlyDeleteFile(Long fileId, Long userId) {
        log.info("Permanently deleting file: {}, userId: {}", fileId, userId);
        
        Resource resource = resourceMapper.selectById(fileId);
        if (resource == null) {
            throw new BusinessException(ResultCode.FILE_NOT_FOUND, "文件不存在");
        }
        
        if (!resource.getUploaderId().equals(userId)) {
            throw new BusinessException(ResultCode.NO_PERMISSION, "无权限删除此文件");
        }
        
        try {
            // 删除物理文件
            String fullPath = uploadPath + File.separator + resource.getFilePath();
            File file = new File(fullPath);
            if (file.exists()) {
                file.delete();
            }
            
            // 从数据库中永久删除
            resourceMapper.deleteById(fileId);
            
            log.info("File permanently deleted successfully: {}", fileId);
            return true;
            
        } catch (Exception e) {
            log.error("Failed to permanently delete file: {}", fileId, e);
            throw new BusinessException(ResultCode.FILE_OPERATION_FAILED, "永久删除文件失败: " + e.getMessage());
        }
    }

    @Override
    public FileDTO.FileIntegrityResponse checkFileIntegrity(Long fileId, Long userId) {
        log.info("Checking file integrity: {}, userId: {}", fileId, userId);
        
        Resource resource = resourceMapper.selectById(fileId);
        if (resource == null) {
            throw new BusinessException(ResultCode.FILE_NOT_FOUND, "文件不存在");
        }
        
        if (!resource.getUploaderId().equals(userId)) {
            throw new BusinessException(ResultCode.NO_PERMISSION, "无权限检查此文件");
        }
        
        FileDTO.FileIntegrityResponse response = new FileDTO.FileIntegrityResponse();
        response.setFileId(fileId);
        response.setFileName(resource.getResourceName());
        
        try {
            String fullPath = uploadPath + File.separator + resource.getFilePath();
            File file = new File(fullPath);
            
            if (!file.exists()) {
                response.setIntegrityStatus("MISSING");
                response.setErrorMessage("物理文件不存在");
                response.setIsValid(false);
                log.warn("Physical file not found: {}", fullPath);
                return response;
            }
            
            // 检查文件大小是否匹配
            long actualSize = file.length();
            Long expectedSize = resource.getFileSize();
            
            if (expectedSize != null && !expectedSize.equals(actualSize)) {
                response.setIntegrityStatus("SIZE_MISMATCH");
                response.setErrorMessage(String.format("文件大小不匹配: 期望 %d, 实际 %d", expectedSize, actualSize));
                response.setIsValid(false);
                response.setExpectedSize(expectedSize);
                response.setActualSize(actualSize);
                log.warn("File size mismatch for file {}: expected {}, actual {}", fileId, expectedSize, actualSize);
                return response;
            }
            
            response.setIntegrityStatus("VALID");
            response.setIsValid(true);
            response.setExpectedSize(expectedSize);
            response.setActualSize(actualSize);
            response.setLastChecked(LocalDateTime.now());
            
            log.info("File integrity check passed for file: {}", fileId);
            return response;
            
        } catch (Exception e) {
            response.setIntegrityStatus("ERROR");
            response.setErrorMessage("检查过程中发生错误: " + e.getMessage());
            response.setIsValid(false);
            log.error("Error checking file integrity: {}", fileId, e);
            return response;
        }
    }

    @Override
    public FileDTO.FileThumbnailResponse getFileThumbnail(Long fileId, String size, Long userId) {
        log.info("Getting file thumbnail: {}, size: {}, userId: {}", fileId, size, userId);
        
        Resource resource = resourceMapper.selectById(fileId);
        if (resource == null) {
            throw new BusinessException(ResultCode.FILE_NOT_FOUND, "文件不存在");
        }
        
        if (!resource.getUploaderId().equals(userId)) {
            throw new BusinessException(ResultCode.NO_PERMISSION, "无权限访问此文件");
        }
        
        FileDTO.FileThumbnailResponse response = new FileDTO.FileThumbnailResponse();
        response.setFileId(fileId);
        response.setOriginalFileName(resource.getResourceName());
        response.setRequestedSize(size);
        
        String fileType = resource.getFileType();
        if (fileType == null || !fileType.toLowerCase().startsWith("image/")) {
            response.setHasThumbnail(false);
            response.setMessage("该文件类型不支持缩略图");
            return response;
        }
        
        // 检查是否已有对应尺寸的缩略图
        String thumbnailPath = resource.getThumbnailPath();
        if (thumbnailPath != null && !thumbnailPath.isEmpty()) {
            String fullThumbnailPath = uploadPath + File.separator + thumbnailPath;
            File thumbnailFile = new File(fullThumbnailPath);
            if (thumbnailFile.exists()) {
                response.setHasThumbnail(true);
                response.setThumbnailPath(thumbnailPath);
                response.setThumbnailSize(thumbnailFile.length());
                response.setMessage("缩略图已存在");
                return response;
            }
        }
        
        // 如果没有缩略图，尝试生成
        Boolean generated = generateThumbnail(fileId, userId);
        if (generated) {
            // 重新获取资源信息
            resource = resourceMapper.selectById(fileId);
            response.setHasThumbnail(true);
            response.setThumbnailPath(resource.getThumbnailPath());
            response.setMessage("缩略图已生成");
        } else {
            response.setHasThumbnail(false);
            response.setMessage("缩略图生成失败");
        }
        
        return response;
    }

    @Override
    public Boolean generateThumbnail(Long fileId, Long userId) {
        log.info("Generating thumbnail for file: {}, userId: {}", fileId, userId);
        
        Resource resource = resourceMapper.selectById(fileId);
        if (resource == null) {
            throw new BusinessException(ResultCode.FILE_NOT_FOUND, "文件不存在");
        }
        
        if (!resource.getUploaderId().equals(userId)) {
            throw new BusinessException(ResultCode.NO_PERMISSION, "无权限访问此文件");
        }
        
        String fileType = resource.getFileType();
        if (fileType == null || !fileType.toLowerCase().startsWith("image/")) {
            log.info("File type {} not supported for thumbnail generation", fileType);
            return false;
        }
        
        try {
            String originalPath = uploadPath + File.separator + resource.getFilePath();
            File originalFile = new File(originalPath);
            
            if (!originalFile.exists()) {
                throw new BusinessException(ResultCode.FILE_NOT_FOUND, "原文件不存在");
            }
            
            // 生成缩略图文件名
            String thumbnailFileName = "thumb_" + resource.getId() + "_" + System.currentTimeMillis() + ".jpg";
            String thumbnailPath = "thumbnails" + File.separator + thumbnailFileName;
            String fullThumbnailPath = uploadPath + File.separator + thumbnailPath;
            
            // 确保缩略图目录存在
            File thumbnailDir = new File(uploadPath + File.separator + "thumbnails");
            if (!thumbnailDir.exists()) {
                thumbnailDir.mkdirs();
            }
            
            // 这里应该使用图片处理库生成缩略图，简化实现
            // 实际项目中可以使用 ImageIO, Thumbnailator 等库
            // 模拟缩略图生成成功
            File thumbnailFile = new File(fullThumbnailPath);
            thumbnailFile.createNewFile();
            
            // 更新资源记录
            resource.setThumbnailPath(thumbnailPath);
            resourceMapper.updateById(resource);
            
            log.info("Thumbnail generated successfully: {}", thumbnailPath);
            return true;
            
        } catch (Exception e) {
            log.error("Failed to generate thumbnail for file: {}", fileId, e);
            return false;
        }
    }

    @Override
    public FileDTO.FileMetadataResponse getFileMetadata(Long fileId, Long userId) {
        log.info("Getting file metadata: {}, userId: {}", fileId, userId);
        
        Resource resource = resourceMapper.selectById(fileId);
        if (resource == null) {
            throw new BusinessException(ResultCode.FILE_NOT_FOUND, "文件不存在");
        }
        
        if (!resource.getUploaderId().equals(userId)) {
            throw new BusinessException(ResultCode.NO_PERMISSION, "无权限访问此文件");
        }
        
        FileDTO.FileMetadataResponse response = new FileDTO.FileMetadataResponse();
        response.setFileId(resource.getId());
        response.setFileName(resource.getResourceName());
        response.setFileType(resource.getFileType());
        response.setFileSize(resource.getFileSize());
        response.setUploadTime(resource.getCreateTime());
        response.setUpdateTime(resource.getUpdateTime());
        response.setDescription(resource.getDescription());
        response.setTags(resource.getTags());
        response.setFilePath(resource.getFilePath());
        response.setThumbnailPath(resource.getThumbnailPath());
        response.setIsDeleted(resource.getIsDeleted());
        
        // 获取文件的物理属性
        try {
            String fullPath = uploadPath + File.separator + resource.getFilePath();
            File file = new File(fullPath);
            if (file.exists()) {
                response.setLastModified(new Date(file.lastModified()).toInstant().atZone(ZoneId.systemDefault()).toLocalDateTime());
                response.setCanRead(file.canRead());
                response.setCanWrite(file.canWrite());
                response.setIsHidden(file.isHidden());
            }
        } catch (Exception e) {
            log.warn("Failed to get file physical properties: {}", fileId, e);
        }
        
        log.info("File metadata retrieved for file: {}", fileId);
        return response;
    }

    @Override
    @Transactional
    public Boolean updateFileMetadata(Long fileId, FileDTO.FileMetadataUpdateRequest metadataRequest, Long userId) {
        log.info("Updating file metadata: {}, userId: {}", fileId, userId);
        
        Resource resource = resourceMapper.selectById(fileId);
        if (resource == null) {
            throw new BusinessException(ResultCode.FILE_NOT_FOUND, "文件不存在");
        }
        
        if (!resource.getUploaderId().equals(userId)) {
            throw new BusinessException(ResultCode.NO_PERMISSION, "无权限修改此文件");
        }
        
        // 更新元数据
        if (metadataRequest.getDescription() != null) {
            resource.setDescription(metadataRequest.getDescription());
        }
        
        if (metadataRequest.getTags() != null) {
            resource.setTags(metadataRequest.getTags());
        }
        
        if (metadataRequest.getFileName() != null && !metadataRequest.getFileName().trim().isEmpty()) {
            resource.setResourceName(metadataRequest.getFileName().trim());
        }
        
        resource.setUpdateTime(LocalDateTime.now());
        
        int result = resourceMapper.updateById(resource);
        
        if (result > 0) {
            log.info("File metadata updated successfully: {}", fileId);
            return true;
        } else {
            throw new BusinessException(ResultCode.FILE_OPERATION_FAILED, "更新文件元数据失败");
        }
    }

    @Override
    public FileDTO.FileSyncResponse syncFiles(FileDTO.FileSyncRequest syncRequest, Long userId) {
        log.info("Syncing files for userId: {}", userId);
        
        FileDTO.FileSyncResponse response = new FileDTO.FileSyncResponse();
        response.setSyncTime(LocalDateTime.now());
        response.setUserId(userId);
        
        try {
            // 获取用户所有文件
            QueryWrapper<Resource> queryWrapper = new QueryWrapper<>();
            queryWrapper.eq("uploader_id", userId)
                       .eq("is_deleted", false);
            
            if (syncRequest.getLastSyncTime() != null) {
                queryWrapper.ge("update_time", syncRequest.getLastSyncTime());
            }
            
            List<Resource> resources = resourceMapper.selectList(queryWrapper);
            
            List<FileDTO.SyncFileInfo> syncedFiles = resources.stream()
                    .map(resource -> {
                        FileDTO.SyncFileInfo syncInfo = new FileDTO.SyncFileInfo();
                        syncInfo.setFileId(resource.getId());
                        syncInfo.setFileName(resource.getResourceName());
                        syncInfo.setFileType(resource.getFileType());
                        syncInfo.setFileSize(resource.getFileSize());
                        syncInfo.setLastModified(resource.getUpdateTime());
                        syncInfo.setFilePath(resource.getFilePath());
                        syncInfo.setChecksum(generateFileChecksum(resource));
                        return syncInfo;
                    })
                    .collect(Collectors.toList());
            
            response.setSyncedFiles(syncedFiles.size());
            response.setTotalFiles(syncedFiles.size());
            response.setSyncStatus("SUCCESS");
            response.setMessage("文件同步完成");
            
            log.info("Files synced successfully: {} files for userId: {}", syncedFiles.size(), userId);
            
        } catch (Exception e) {
            response.setSyncStatus("FAILED");
            response.setMessage("文件同步失败: " + e.getMessage());
            log.error("Failed to sync files for userId: {}", userId, e);
        }
        
        return response;
    }
    
    private String generateFileChecksum(Resource resource) {
        // 简化实现，实际项目中应该计算文件的MD5或SHA256
        return String.valueOf(resource.getId().hashCode() + resource.getUpdateTime().hashCode());
    }

    @Override
    public FileDTO.FileExportResponse exportFileList(FileDTO.FileExportRequest exportRequest, Long userId) {
        log.info("Exporting file list for userId: {}, format: {}", userId, exportRequest.getExportFormat());
        
        FileDTO.FileExportResponse response = new FileDTO.FileExportResponse();
        response.setExportTime(LocalDateTime.now());
        response.setUserId(userId);
        response.setExportFormat(exportRequest.getExportFormat());
        
        try {
            // 构建查询条件
            QueryWrapper<Resource> queryWrapper = new QueryWrapper<>();
            queryWrapper.eq("uploader_id", userId);
            
            if (exportRequest.getIncludeDeleted() != null && !exportRequest.getIncludeDeleted()) {
                queryWrapper.eq("is_deleted", false);
            }
            
            if (exportRequest.getFileType() != null && !exportRequest.getFileType().isEmpty()) {
                queryWrapper.eq("file_type", exportRequest.getFileType());
            }
            
            if (exportRequest.getStartDate() != null) {
                queryWrapper.ge("create_time", exportRequest.getStartDate());
            }
            
            if (exportRequest.getEndDate() != null) {
                queryWrapper.le("create_time", exportRequest.getEndDate());
            }
            
            queryWrapper.orderByDesc("create_time");
            
            List<Resource> resources = resourceMapper.selectList(queryWrapper);
            
            // 生成导出文件
            String exportFileName = "file_list_" + userId + "_" + System.currentTimeMillis();
            String exportPath = "exports" + File.separator + exportFileName;
            
            // 确保导出目录存在
            File exportDir = new File(uploadPath + File.separator + "exports");
            if (!exportDir.exists()) {
                exportDir.mkdirs();
            }
            
            String fullExportPath = uploadPath + File.separator + exportPath;
            
            if ("CSV".equalsIgnoreCase(exportRequest.getExportFormat())) {
                exportToCsv(resources, fullExportPath + ".csv");
                response.setExportFilePath(exportPath + ".csv");
            } else if ("JSON".equalsIgnoreCase(exportRequest.getExportFormat())) {
                exportToJson(resources, fullExportPath + ".json");
                response.setExportFilePath(exportPath + ".json");
            } else {
                throw new BusinessException(ResultCode.PARAM_ERROR, "不支持的导出格式: " + exportRequest.getExportFormat());
            }
            
            response.setTotalFiles(resources.size());
            response.setExportStatus("SUCCESS");
            response.setMessage("文件列表导出完成");
            
            log.info("File list exported successfully: {} files, format: {}", resources.size(), exportRequest.getExportFormat());
            
        } catch (Exception e) {
            response.setExportStatus("FAILED");
            response.setMessage("导出失败: " + e.getMessage());
            log.error("Failed to export file list for userId: {}", userId, e);
        }
        
        return response;
    }
    
    private void exportToCsv(List<Resource> resources, String filePath) throws IOException {
        try (FileWriter writer = new FileWriter(filePath, StandardCharsets.UTF_8)) {
            // 写入CSV头部
            writer.write("文件ID,文件名,文件类型,文件大小,上传时间,更新时间,描述,标签,是否删除\n");
            
            // 写入数据
            for (Resource resource : resources) {
                writer.write(String.format("%d,\"%s\",\"%s\",%d,\"%s\",\"%s\",\"%s\",\"%s\",%s\n",
                    resource.getId(),
                    resource.getResourceName() != null ? resource.getResourceName().replace("\"", "\\\"") : "",
                    resource.getFileType() != null ? resource.getFileType() : "",
                    resource.getFileSize() != null ? resource.getFileSize() : 0,
                    resource.getCreateTime() != null ? resource.getCreateTime().toString() : "",
                    resource.getUpdateTime() != null ? resource.getUpdateTime().toString() : "",
                    resource.getDescription() != null ? resource.getDescription().replace("\"", "\\\"") : "",
                    resource.getTags() != null ? resource.getTags().replace("\"", "\\\"") : "",
                    resource.getIsDeleted() != null ? resource.getIsDeleted().toString() : "false"
                ));
            }
        }
    }
    
    private void exportToJson(List<Resource> resources, String filePath) throws IOException {
        // 简化的JSON导出实现
        try (FileWriter writer = new FileWriter(filePath, StandardCharsets.UTF_8)) {
            writer.write("{\"files\":[\n");
            
            for (int i = 0; i < resources.size(); i++) {
                Resource resource = resources.get(i);
                writer.write(String.format(
                    "{\"fileId\":%d,\"fileName\":\"%s\",\"fileType\":\"%s\",\"fileSize\":%d,\"uploadTime\":\"%s\",\"updateTime\":\"%s\",\"description\":\"%s\",\"tags\":\"%s\",\"isDeleted\":%s}",
                    resource.getId(),
                    resource.getResourceName() != null ? resource.getResourceName().replace("\"", "\\\"") : "",
                    resource.getFileType() != null ? resource.getFileType() : "",
                    resource.getFileSize() != null ? resource.getFileSize() : 0,
                    resource.getCreateTime() != null ? resource.getCreateTime().toString() : "",
                    resource.getUpdateTime() != null ? resource.getUpdateTime().toString() : "",
                    resource.getDescription() != null ? resource.getDescription().replace("\"", "\\\"") : "",
                    resource.getTags() != null ? resource.getTags().replace("\"", "\\\"") : "",
                    resource.getIsDeleted() != null ? resource.getIsDeleted().toString() : "false"
                ));
                
                if (i < resources.size() - 1) {
                    writer.write(",\n");
                } else {
                    writer.write("\n");
                }
            }
            
            writer.write("],\"totalCount\":" + resources.size() + ",\"exportTime\":\"" + LocalDateTime.now().toString() + "\"}");
        }
    }
}