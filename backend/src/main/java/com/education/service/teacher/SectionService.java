package com.education.service.teacher;

import com.education.entity.Section;

import java.util.List;

/**
 * 小节服务接口
 */
public interface SectionService {

    /**
     * 根据章节ID获取小节列表
     * 
     * @param chapterId 章节ID
     * @return 小节列表
     */
    List<Section> getSectionsByChapterId(Long chapterId);
    
    /**
     * 创建小节
     * 
     * @param section 小节信息
     * @return 创建后的小节
     */
    Section createSection(Section section);
    
    /**
     * 更新小节
     * 
     * @param section 小节信息
     * @return 更新后的小节
     */
    Section updateSection(Section section);
    
    /**
     * 删除小节
     * 
     * @param sectionId 小节ID
     * @return 是否删除成功
     */
    boolean deleteSection(Long sectionId);

    /**
     * 根据ID获取小节信息
     * @param id 小节ID
     * @return 小节信息
     */
    Section getById(Long id);

    /**
     * 更新小节信息
     * @param section 小节信息
     * @return 是否更新成功
     */
    boolean updateById(Section section);
} 