package com.education.service.teacher.impl;

import com.education.entity.Section;
import com.education.exception.BusinessException;
import com.education.exception.ResultCode;
import com.education.mapper.SectionMapper;
import com.education.service.teacher.SectionService;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.List;

/**
 * 小节服务实现类
 */
@Service
public class SectionServiceImpl implements SectionService {

    private static final Logger logger = LoggerFactory.getLogger(SectionServiceImpl.class);
    
    private final SectionMapper sectionMapper;

    public SectionServiceImpl(SectionMapper sectionMapper) {
        this.sectionMapper = sectionMapper;
    }

    @Override
    public List<Section> getSectionsByChapterId(Long chapterId) {
        logger.info("获取章节小节列表 - 章节ID: {}", chapterId);
        
        try {
            List<Section> sections = sectionMapper.selectByChapterId(chapterId);
            logger.info("找到小节数量: {}", sections.size());
            return sections;
        } catch (Exception e) {
            logger.error("获取章节小节列表异常", e);
            throw new BusinessException(ResultCode.INTERNAL_SERVER_ERROR, "获取章节小节列表失败: " + e.getMessage());
        }
    }

    @Override
    @Transactional
    public Section createSection(Section section) {
        logger.info("创建小节 - 章节ID: {}, 小节标题: {}", section.getChapterId(), section.getTitle());
        
        try {
            // 设置排序顺序
            if (section.getSortOrder() == null) {
                // 获取当前最大排序值
                Integer maxOrder = sectionMapper.selectByChapterId(section.getChapterId()).stream()
                        .map(Section::getSortOrder)
                        .max(Integer::compareTo)
                        .orElse(0);
                section.setSortOrder(maxOrder + 1);
                logger.info("设置小节排序值: {}", section.getSortOrder());
            }
            
            // 插入小节
            sectionMapper.insert(section);
            logger.info("小节创建成功 - ID: {}", section.getId());
            
            return section;
        } catch (Exception e) {
            logger.error("创建小节异常", e);
            throw new BusinessException(ResultCode.INTERNAL_SERVER_ERROR, "创建小节失败: " + e.getMessage());
        }
    }

    @Override
    @Transactional
    public Section updateSection(Section section) {
        logger.info("更新小节 - ID: {}, 标题: {}", section.getId(), section.getTitle());
        
        try {
            // 检查小节是否存在
            Section existingSection = sectionMapper.selectById(section.getId());
            if (existingSection == null) {
                logger.error("小节不存在 - ID: {}", section.getId());
                throw new BusinessException(ResultCode.RESOURCE_NOT_FOUND, "小节不存在");
            }
            
            // 更新小节
            sectionMapper.updateById(section);
            logger.info("小节更新成功 - ID: {}", section.getId());
            
            return sectionMapper.selectById(section.getId());
        } catch (BusinessException e) {
            throw e;
        } catch (Exception e) {
            logger.error("更新小节异常", e);
            throw new BusinessException(ResultCode.INTERNAL_SERVER_ERROR, "更新小节失败: " + e.getMessage());
        }
    }

    @Override
    @Transactional
    public boolean deleteSection(Long sectionId) {
        logger.info("删除小节 - ID: {}", sectionId);
        
        try {
            // 检查小节是否存在
            Section section = sectionMapper.selectById(sectionId);
            if (section == null) {
                logger.error("小节不存在 - ID: {}", sectionId);
                throw new BusinessException(ResultCode.RESOURCE_NOT_FOUND, "小节不存在");
            }
            
            // 删除小节
            int result = sectionMapper.deleteById(sectionId);
            logger.info("小节删除结果: {}", result > 0);
            
            return result > 0;
        } catch (BusinessException e) {
            throw e;
        } catch (Exception e) {
            logger.error("删除小节异常", e);
            throw new BusinessException(ResultCode.INTERNAL_SERVER_ERROR, "删除小节失败: " + e.getMessage());
        }
    }

    @Override
    public Section getById(Long id) {
        return sectionMapper.selectById(id);
    }

    @Override
    @Transactional
    public boolean updateById(Section section) {
        return sectionMapper.updateById(section) > 0;
    }
} 