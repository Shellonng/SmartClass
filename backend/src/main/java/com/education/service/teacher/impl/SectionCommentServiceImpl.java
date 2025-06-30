package com.education.service.teacher.impl;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.education.dto.SectionCommentDTO;
import com.education.dto.common.PageRequest;
import com.education.dto.common.PageResponse;
import com.education.entity.SectionComment;
import com.education.entity.User;
import com.education.mapper.SectionCommentMapper;
import com.education.mapper.UserMapper;
import com.education.service.teacher.SectionCommentService;
import lombok.RequiredArgsConstructor;
import org.springframework.beans.BeanUtils;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.time.LocalDateTime;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

@Service
@RequiredArgsConstructor
public class SectionCommentServiceImpl implements SectionCommentService {
    
    private final SectionCommentMapper commentMapper;
    private final UserMapper userMapper;
    
    @Override
    public PageResponse<SectionCommentDTO> getComments(Long sectionId, PageRequest pageRequest) {
        if (sectionId == null) {
            return PageResponse.of(pageRequest.getPage(), pageRequest.getSize(), 0L, Collections.emptyList());
        }

        // 查询评论
        Page<SectionComment> page = new Page<>(pageRequest.getPage(), pageRequest.getSize());
        LambdaQueryWrapper<SectionComment> wrapper = new LambdaQueryWrapper<SectionComment>()
                .eq(SectionComment::getSectionId, sectionId)
                .isNull(SectionComment::getParentId)  // 只查询主评论
                .orderByDesc(SectionComment::getCreateTime);
        
        Page<SectionComment> commentPage = commentMapper.selectPage(page, wrapper);
        
        if (commentPage.getRecords().isEmpty()) {
            return PageResponse.of(pageRequest.getPage(), pageRequest.getSize(), 0L, Collections.emptyList());
        }
        
        // 获取用户信息
        List<Long> userIds = commentPage.getRecords().stream()
                .map(SectionComment::getUserId)
                .collect(Collectors.toList());
        
        List<User> users = userMapper.selectBatchIds(userIds);
        Map<Long, User> userMap = users.stream()
                .collect(Collectors.toMap(User::getId, user -> user));
        
        // 转换为DTO
        List<SectionCommentDTO> dtos = commentPage.getRecords().stream()
                .map(comment -> {
                    SectionCommentDTO dto = new SectionCommentDTO();
                    BeanUtils.copyProperties(comment, dto);
                    
                    User user = userMap.get(comment.getUserId());
                    if (user != null) {
                        dto.setUserName(user.getRealName());
                        dto.setUserAvatar(user.getAvatar());
                        dto.setUserRole(user.getRole());
                    }
                    
                    // 获取回复数
                    Long replyCount = commentMapper.selectCount(
                            new LambdaQueryWrapper<SectionComment>()
                                    .eq(SectionComment::getParentId, comment.getId())
                    );
                    dto.setReplyCount(replyCount.intValue());
                    
                    return dto;
                })
                .collect(Collectors.toList());
        
        return PageResponse.of(pageRequest.getPage(), pageRequest.getSize(), commentPage.getTotal(), dtos);
    }
    
    @Override
    @Transactional
    public SectionCommentDTO createComment(SectionCommentDTO.CreateRequest request) {
        if (request.getSectionId() == null) {
            throw new IllegalArgumentException("小节ID不能为空");
        }

        // 获取当前用户ID
        String username = SecurityContextHolder.getContext().getAuthentication().getName();
        User user = userMapper.selectOne(
            new LambdaQueryWrapper<User>()
                .eq(User::getUsername, username)
        );
        
        SectionComment comment = new SectionComment();
        comment.setSectionId(request.getSectionId());
        comment.setUserId(user.getId());
        comment.setContent(request.getContent());
        comment.setParentId(request.getParentId());
        comment.setCreateTime(LocalDateTime.now());
        comment.setUpdateTime(LocalDateTime.now());
        
        commentMapper.insert(comment);
        
        SectionCommentDTO dto = new SectionCommentDTO();
        BeanUtils.copyProperties(comment, dto);
        dto.setUserName(user.getRealName());
        dto.setUserAvatar(user.getAvatar());
        dto.setUserRole(user.getRole());
        return dto;
    }
    
    @Override
    @Transactional
    public void updateComment(SectionCommentDTO.UpdateRequest request) {
        if (request.getId() == null) {
            throw new IllegalArgumentException("评论ID不能为空");
        }

        // 获取当前用户ID
        String username = SecurityContextHolder.getContext().getAuthentication().getName();
        User user = userMapper.selectOne(
            new LambdaQueryWrapper<User>()
                .eq(User::getUsername, username)
        );

        SectionComment comment = commentMapper.selectById(request.getId());
        if (comment != null && comment.getUserId().equals(user.getId())) {
            comment.setContent(request.getContent());
            comment.setUpdateTime(LocalDateTime.now());
            commentMapper.updateById(comment);
        } else {
            throw new IllegalArgumentException("无权修改此评论");
        }
    }
    
    @Override
    @Transactional
    public void deleteComment(Long id) {
        if (id == null) {
            throw new IllegalArgumentException("评论ID不能为空");
        }

        // 获取当前用户ID
        String username = SecurityContextHolder.getContext().getAuthentication().getName();
        User user = userMapper.selectOne(
            new LambdaQueryWrapper<User>()
                .eq(User::getUsername, username)
        );

        SectionComment comment = commentMapper.selectById(id);
        if (comment != null && comment.getUserId().equals(user.getId())) {
            // 删除评论及其回复
            commentMapper.deleteById(id);
            commentMapper.delete(
                    new LambdaQueryWrapper<SectionComment>()
                            .eq(SectionComment::getParentId, id)
            );
        } else {
            throw new IllegalArgumentException("无权删除此评论");
        }
    }
} 