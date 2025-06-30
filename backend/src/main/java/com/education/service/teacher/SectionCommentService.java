package com.education.service.teacher;

import com.education.dto.SectionCommentDTO;
import com.education.dto.common.PageRequest;
import com.education.dto.common.PageResponse;

public interface SectionCommentService {
    PageResponse<SectionCommentDTO> getComments(Long sectionId, PageRequest pageRequest);
    
    SectionCommentDTO createComment(SectionCommentDTO.CreateRequest request);
    
    void updateComment(SectionCommentDTO.UpdateRequest request);
    
    void deleteComment(Long id);
} 