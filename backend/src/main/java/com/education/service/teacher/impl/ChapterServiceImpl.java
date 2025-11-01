package com.education.service.teacher.impl;

import com.education.entity.Chapter;
import com.education.entity.Section;
import com.education.exception.BusinessException;
import com.education.exception.ResultCode;
import com.education.mapper.ChapterMapper;
import com.education.mapper.SectionMapper;
import com.education.service.teacher.ChapterService;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.time.LocalDateTime;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

/**
 * 章节服务实现类
 */
@Service
public class ChapterServiceImpl implements ChapterService {

    private static final Logger logger = LoggerFactory.getLogger(ChapterServiceImpl.class);
    
    @Autowired
    private ChapterMapper chapterMapper;
    
    @Autowired
    private SectionMapper sectionMapper;

    @Override
    public List<Chapter> getChaptersByCourseId(Long courseId) {
        logger.info("获取课程章节列表 - 课程ID: {}", courseId);
        
        try {
            // 获取章节列表
            List<Chapter> chapters = chapterMapper.selectByCourseId(courseId);
            logger.info("找到章节数量: {}", chapters.size());
            
            if (!chapters.isEmpty()) {
                // 获取所有小节
                List<Section> allSections = sectionMapper.selectByCourseId(courseId);
                logger.info("找到小节数量: {}", allSections.size());
                
                // 按章节ID分组
                Map<Long, List<Section>> sectionMap = allSections.stream()
                        .collect(Collectors.groupingBy(Section::getChapterId));
                
                // 将小节设置到对应章节中
                chapters.forEach(chapter -> {
                    List<Section> sections = sectionMap.getOrDefault(chapter.getId(), List.of());
                    chapter.setSections(sections);
                });
            }
            
            return chapters;
        } catch (Exception e) {
            logger.error("获取课程章节列表异常", e);
            throw new BusinessException(ResultCode.INTERNAL_SERVER_ERROR, "获取课程章节列表失败: " + e.getMessage());
        }
    }

    @Override
    @Transactional
    public Chapter createChapter(Chapter chapter) {
        logger.info("创建章节 - 课程ID: {}, 章节标题: {}", chapter.getCourseId(), chapter.getTitle());
        
        try {
            // 设置排序顺序
            if (chapter.getSortOrder() == null) {
                // 获取当前最大排序值
                Integer maxOrder = chapterMapper.selectByCourseId(chapter.getCourseId()).stream()
                        .map(Chapter::getSortOrder)
                        .max(Integer::compareTo)
                        .orElse(0);
                chapter.setSortOrder(maxOrder + 1);
                logger.info("设置章节排序值: {}", chapter.getSortOrder());
            }
            
            // 设置创建时间和更新时间
            LocalDateTime now = LocalDateTime.now();
            chapter.setCreateTime(now);
            chapter.setUpdateTime(now);
            
            // 插入章节
            chapterMapper.insert(chapter);
            logger.info("章节创建成功 - ID: {}", chapter.getId());
            
            return chapter;
        } catch (Exception e) {
            logger.error("创建章节异常", e);
            throw new BusinessException(ResultCode.INTERNAL_SERVER_ERROR, "创建章节失败: " + e.getMessage());
        }
    }

    @Override
    @Transactional
    public Chapter updateChapter(Chapter chapter) {
        logger.info("更新章节 - ID: {}, 标题: {}", chapter.getId(), chapter.getTitle());
        
        try {
            // 检查章节是否存在
            Chapter existingChapter = chapterMapper.selectById(chapter.getId());
            if (existingChapter == null) {
                logger.error("章节不存在 - ID: {}", chapter.getId());
                throw new BusinessException(ResultCode.RESOURCE_NOT_FOUND, "章节不存在");
            }
            
            // 更新章节
            chapterMapper.updateById(chapter);
            logger.info("章节更新成功 - ID: {}", chapter.getId());
            
            return chapterMapper.selectById(chapter.getId());
        } catch (BusinessException e) {
            throw e;
        } catch (Exception e) {
            logger.error("更新章节异常", e);
            throw new BusinessException(ResultCode.INTERNAL_SERVER_ERROR, "更新章节失败: " + e.getMessage());
        }
    }

    @Override
    @Transactional
    public boolean deleteChapter(Long chapterId) {
        logger.info("删除章节 - ID: {}", chapterId);
        
        try {
            // 检查章节是否存在
            Chapter chapter = chapterMapper.selectById(chapterId);
            if (chapter == null) {
                logger.error("章节不存在 - ID: {}", chapterId);
                throw new BusinessException(ResultCode.RESOURCE_NOT_FOUND, "章节不存在");
            }
            
            // 删除章节下的所有小节
            List<Section> sections = sectionMapper.selectByChapterId(chapterId);
            if (!sections.isEmpty()) {
                logger.info("删除章节下的小节 - 数量: {}", sections.size());
                for (Section section : sections) {
                    sectionMapper.deleteById(section.getId());
                }
            }
            
            // 删除章节
            int result = chapterMapper.deleteById(chapterId);
            logger.info("章节删除结果: {}", result > 0);
            
            return result > 0;
        } catch (BusinessException e) {
            throw e;
        } catch (Exception e) {
            logger.error("删除章节异常", e);
            throw new BusinessException(ResultCode.INTERNAL_SERVER_ERROR, "删除章节失败: " + e.getMessage());
        }
    }
} 