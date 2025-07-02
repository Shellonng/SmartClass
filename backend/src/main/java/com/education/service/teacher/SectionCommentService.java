package com.education.service.teacher;

import com.education.dto.SectionCommentDTO;
import com.education.dto.common.PageRequest;
import com.education.dto.common.PageResponse;

public interface SectionCommentService {
    PageResponse<SectionCommentDTO> getComments(Long sectionId, PageRequest pageRequest);
    
    PageResponse<SectionCommentDTO> getCourseComments(Long courseId, PageRequest pageRequest);
    
    SectionCommentDTO createComment(SectionCommentDTO.CreateRequest request);
    
    void updateComment(SectionCommentDTO.UpdateRequest request);
    
    void deleteComment(Long id);

    /**
     * 获取评论回复
     *
     * @param sectionId 小节ID
     * @param commentId 评论ID
     * @param pageRequest 分页请求
     * @return 评论回复列表
     */
    PageResponse<SectionCommentDTO> getCommentReplies(Long sectionId, Long commentId, PageRequest pageRequest);
} 