package com.education.utils;

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
 * 文件工具类
 * 
 * @author Education Platform Team
 * @version 1.0.0
 * @since 2024
 */
public class FileUtils {

    /**
     * 允许的文件类型
     */
    private static final List<String> ALLOWED_EXTENSIONS = Arrays.asList(
            "jpg", "jpeg", "png", "gif", "pdf", "doc", "docx", 
            "xls", "xlsx", "ppt", "pptx", "txt", "zip", "rar"
    );

    /**
     * 保存文件到本地
     * 
     * @param file 文件
     * @param uploadPath 上传路径
     * @return 文件保存路径
     * @throws IOException IO异常
     */
    public static String saveFile(MultipartFile file, String uploadPath) throws IOException {
        // 检查文件是否为空
        if (file.isEmpty()) {
            throw new IOException("文件为空");
        }

        // 检查文件类型
        String originalFilename = file.getOriginalFilename();
        String extension = getFileExtension(originalFilename);
        if (!isAllowedExtension(extension)) {
            throw new IOException("不支持的文件类型: " + extension);
        }

        // 创建目录
        Path uploadDir = Paths.get(uploadPath);
        if (!Files.exists(uploadDir)) {
            Files.createDirectories(uploadDir);
        }

        // 生成新的文件名
        String newFilename = generateUniqueFilename(extension);
        Path filePath = uploadDir.resolve(newFilename);

        // 保存文件
        file.transferTo(filePath.toFile());

        return newFilename;
    }

    /**
     * 删除文件
     * 
     * @param filePath 文件路径
     * @return 是否删除成功
     */
    public static boolean deleteFile(String filePath) {
        File file = new File(filePath);
        return file.exists() && file.delete();
    }

    /**
     * 获取文件扩展名
     * 
     * @param filename 文件名
     * @return 扩展名
     */
    public static String getFileExtension(String filename) {
        if (filename == null || filename.lastIndexOf(".") == -1) {
            return "";
        }
        return filename.substring(filename.lastIndexOf(".") + 1).toLowerCase();
    }

    /**
     * 检查文件类型是否允许
     * 
     * @param extension 扩展名
     * @return 是否允许
     */
    public static boolean isAllowedExtension(String extension) {
        return ALLOWED_EXTENSIONS.contains(extension.toLowerCase());
    }

    /**
     * 生成唯一的文件名
     * 
     * @param extension 扩展名
     * @return 唯一文件名
     */
    public static String generateUniqueFilename(String extension) {
        return UUID.randomUUID().toString().replace("-", "") + "." + extension;
    }

    /**
     * 获取文件大小的可读形式
     * 
     * @param size 文件大小（字节）
     * @return 可读的文件大小
     */
    public static String getReadableFileSize(long size) {
        if (size <= 0) {
            return "0 B";
        }
        final String[] units = new String[]{"B", "KB", "MB", "GB", "TB"};
        int digitGroups = (int) (Math.log10(size) / Math.log10(1024));
        return String.format("%.2f %s", size / Math.pow(1024, digitGroups), units[digitGroups]);
    }
}