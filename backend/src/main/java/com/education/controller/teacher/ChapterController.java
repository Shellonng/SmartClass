package com.education.controller.teacher;

import com.education.dto.common.Result;
import com.education.entity.Chapter;
import com.education.service.teacher.ChapterService;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.security.core.Authentication;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.web.bind.annotation.*;

import java.util.List;

/**
 * 章节管理控制器
 */
@RestController
@RequestMapping("/api/teacher/chapters")
@Tag(name = "教师章节管理", description = "教师章节管理相关接口")
public class ChapterController {

    private static final Logger logger = LoggerFactory.getLogger(ChapterController.class);
    
    @Autowired
    private ChapterService chapterService;
    
    @Operation(summary = "获取课程章节列表", description = "根据课程ID获取章节列表")
    @GetMapping("/course/{courseId}")
    public Result<List<Chapter>> getChaptersByCourseId(@PathVariable Long courseId) {
        logger.info("获取课程章节列表 - 课程ID: {}", courseId);
        
        try {
            // 不需要验证用户身份，直接获取章节列表
            List<Chapter> chapters = chapterService.getChaptersByCourseId(courseId);
            logger.info("获取到章节数量: {}", chapters.size());
            return Result.success(chapters);
        } catch (Exception e) {
            logger.error("获取课程章节列表失败: {}", e.getMessage(), e);
            return Result.error("获取课程章节列表失败: " + e.getMessage());
        }
    }
    
    @Operation(summary = "创建章节", description = "创建新的章节")
    @PostMapping
    public Result<Chapter> createChapter(@RequestBody Chapter chapter) {
        logger.info("创建章节 - 课程ID: {}, 章节标题: {}", chapter.getCourseId(), chapter.getTitle());
        
        try {
            Chapter createdChapter = chapterService.createChapter(chapter);
            logger.info("章节创建成功 - ID: {}", createdChapter.getId());
            return Result.success(createdChapter);
        } catch (Exception e) {
            logger.error("创建章节失败: {}", e.getMessage(), e);
            return Result.error("创建章节失败: " + e.getMessage());
        }
    }
    
    @Operation(summary = "更新章节", description = "更新章节信息")
    @PutMapping("/{id}")
    public Result<Chapter> updateChapter(@PathVariable Long id, @RequestBody Chapter chapter) {
        logger.info("更新章节 - ID: {}, 标题: {}", id, chapter.getTitle());
        
        try {
            chapter.setId(id);
            Chapter updatedChapter = chapterService.updateChapter(chapter);
            logger.info("章节更新成功 - ID: {}", updatedChapter.getId());
            return Result.success(updatedChapter);
        } catch (Exception e) {
            logger.error("更新章节失败: {}", e.getMessage(), e);
            return Result.error("更新章节失败: " + e.getMessage());
        }
    }
    
    @Operation(summary = "删除章节", description = "删除章节及其下的所有小节")
    @DeleteMapping("/{id}")
    public Result<Boolean> deleteChapter(@PathVariable Long id) {
        logger.info("删除章节 - ID: {}", id);
        
        try {
            boolean result = chapterService.deleteChapter(id);
            logger.info("章节删除结果: {}", result);
            return Result.success(result);
        } catch (Exception e) {
            logger.error("删除章节失败: {}", e.getMessage(), e);
            return Result.error("删除章节失败: " + e.getMessage());
        }
    }
} 