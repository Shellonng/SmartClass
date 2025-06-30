package com.education.controller.teacher;

import com.education.dto.common.Result;
import com.education.entity.Section;
import com.education.service.teacher.SectionService;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.io.File;
import java.io.IOException;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.List;
import java.util.UUID;

/**
 * 小节管理控制器
 */
@RestController
@RequestMapping("/teacher/sections")
@Tag(name = "教师小节管理", description = "教师小节管理相关接口")
public class SectionController {

    private static final Logger logger = LoggerFactory.getLogger(SectionController.class);
    
    @Value("${video.upload.path}")
    private String uploadPath;
    
    private final SectionService sectionService;

    public SectionController(SectionService sectionService) {
        this.sectionService = sectionService;
    }
    
    @Operation(summary = "获取章节小节列表", description = "根据章节ID获取小节列表")
    @GetMapping("/chapter/{chapterId}")
    public Result<List<Section>> getSectionsByChapterId(@PathVariable Long chapterId) {
        logger.info("获取章节小节列表 - 章节ID: {}", chapterId);
        
        try {
            List<Section> sections = sectionService.getSectionsByChapterId(chapterId);
            logger.info("获取到小节数量: {}", sections.size());
            return Result.success(sections);
        } catch (Exception e) {
            logger.error("获取章节小节列表失败: {}", e.getMessage(), e);
            return Result.error("获取章节小节列表失败: " + e.getMessage());
        }
    }
    
    @Operation(summary = "创建小节", description = "创建新的小节")
    @PostMapping
    public Result<Section> createSection(@RequestBody Section section) {
        logger.info("创建小节 - 章节ID: {}, 小节标题: {}", section.getChapterId(), section.getTitle());
        
        try {
            Section createdSection = sectionService.createSection(section);
            logger.info("小节创建成功 - ID: {}", createdSection.getId());
            return Result.success(createdSection);
        } catch (Exception e) {
            logger.error("创建小节失败: {}", e.getMessage(), e);
            return Result.error("创建小节失败: " + e.getMessage());
        }
    }
    
    @Operation(summary = "更新小节", description = "更新小节信息")
    @PutMapping("/{id}")
    public Result<Section> updateSection(@PathVariable Long id, @RequestBody Section section) {
        logger.info("更新小节 - ID: {}, 标题: {}", id, section.getTitle());
        
        try {
            section.setId(id);
            Section updatedSection = sectionService.updateSection(section);
            logger.info("小节更新成功 - ID: {}", updatedSection.getId());
            return Result.success(updatedSection);
        } catch (Exception e) {
            logger.error("更新小节失败: {}", e.getMessage(), e);
            return Result.error("更新小节失败: " + e.getMessage());
        }
    }
    
    @Operation(summary = "删除小节", description = "删除小节")
    @DeleteMapping("/{id}")
    public Result<Boolean> deleteSection(@PathVariable Long id) {
        logger.info("删除小节 - ID: {}", id);
        
        try {
            boolean result = sectionService.deleteSection(id);
            logger.info("小节删除结果: {}", result);
            return Result.success(result);
        } catch (Exception e) {
            logger.error("删除小节失败: {}", e.getMessage(), e);
            return Result.error("删除小节失败: " + e.getMessage());
        }
    }

    @PostMapping("/{id}/upload-video")
    public Result<String> uploadVideo(@PathVariable Long id, @RequestParam("file") MultipartFile file) {
        try {
            // 检查小节是否存在
            Section section = sectionService.getById(id);
            if (section == null) {
                return Result.error("小节不存在");
            }

            // 检查文件是否为空
            if (file.isEmpty()) {
                return Result.error("请选择要上传的视频文件");
            }

            // 检查文件类型
            String contentType = file.getContentType();
            if (contentType == null || !contentType.startsWith("video/")) {
                return Result.error("只能上传视频文件");
            }

            // 生成文件保存路径
            String yearMonth = LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyyMM"));
            String savePath = uploadPath + File.separator + yearMonth;
            File saveDir = new File(savePath);
            if (!saveDir.exists() && !saveDir.mkdirs()) {
                return Result.error("创建目录失败");
            }

            // 生成唯一文件名
            String originalFilename = file.getOriginalFilename();
            if (originalFilename == null) {
                return Result.error("文件名不能为空");
            }
            String extension = originalFilename.substring(originalFilename.lastIndexOf("."));
            String newFilename = UUID.randomUUID().toString() + extension;
            String fullPath = savePath + File.separator + newFilename;

            // 保存文件
            file.transferTo(new File(fullPath));

            // 更新数据库中的视频URL
            String videoUrl = yearMonth + "/" + newFilename;
            section.setVideoUrl(videoUrl);
            if (!sectionService.updateById(section)) {
                // 如果更新失败，删除已上传的文件
                new File(fullPath).delete();
                return Result.error("更新数据库失败");
            }

            return Result.success(videoUrl);
        } catch (IOException e) {
            logger.error("视频上传失败", e);
            return Result.error("视频上传失败：" + e.getMessage());
        } catch (Exception e) {
            logger.error("视频上传过程中发生错误", e);
            return Result.error("系统错误：" + e.getMessage());
        }
    }

    @Operation(summary = "获取小节详情", description = "根据ID获取小节详情")
    @GetMapping("/{id}")
    public Result<Section> getSectionById(@PathVariable Long id) {
        logger.info("获取小节详情 - ID: {}", id);
        
        try {
            Section section = sectionService.getById(id);
            if (section == null) {
                logger.warn("小节不存在 - ID: {}", id);
                return Result.error("小节不存在");
            }
            logger.info("获取小节详情成功 - ID: {}", id);
            return Result.success(section);
        } catch (Exception e) {
            logger.error("获取小节详情失败: {}", e.getMessage(), e);
            return Result.error("获取小节详情失败: " + e.getMessage());
        }
    }
} 