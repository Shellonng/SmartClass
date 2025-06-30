package com.education.controller.teacher;

import com.education.dto.SectionCommentDTO;
import com.education.dto.common.PageRequest;
import com.education.dto.common.PageResponse;
import com.education.dto.common.Result;
import com.education.service.teacher.SectionCommentService;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/api/sections/{sectionId}/comments")
@RequiredArgsConstructor
public class SectionCommentController {
    
    private final SectionCommentService commentService;
    
    @GetMapping
    public Result<PageResponse<SectionCommentDTO>> getComments(
            @PathVariable Long sectionId,
            @RequestParam(defaultValue = "1") Integer page,
            @RequestParam(defaultValue = "10") Integer size
    ) {
        PageRequest pageRequest = new PageRequest(page, size);
        return Result.success(commentService.getComments(sectionId, pageRequest));
    }
    
    @PostMapping
    public Result<SectionCommentDTO> createComment(
            @PathVariable Long sectionId,
            @RequestBody SectionCommentDTO.CreateRequest request
    ) {
        request.setSectionId(sectionId);
        return Result.success(commentService.createComment(request));
    }
    
    @PutMapping("/{id}")
    public Result<Void> updateComment(
            @PathVariable Long sectionId,
            @PathVariable Long id,
            @RequestBody SectionCommentDTO.UpdateRequest request
    ) {
        request.setId(id);
        commentService.updateComment(request);
        return Result.success();
    }
    
    @DeleteMapping("/{id}")
    public Result<Void> deleteComment(
            @PathVariable Long sectionId,
            @PathVariable Long id
    ) {
        commentService.deleteComment(id);
        return Result.success();
    }
} 