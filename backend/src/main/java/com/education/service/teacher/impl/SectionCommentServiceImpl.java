package com.education.service.teacher.impl;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.education.dto.SectionCommentDTO;
import com.education.dto.common.PageRequest;
import com.education.dto.common.PageResponse;
import com.education.entity.Chapter;
import com.education.entity.Section;
import com.education.entity.SectionComment;
import com.education.entity.User;
import com.education.mapper.ChapterMapper;
import com.education.mapper.SectionCommentMapper;
import com.education.mapper.SectionMapper;
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
    private final ChapterMapper chapterMapper;
    private final SectionMapper sectionMapper;
    
    @Override
    public PageResponse<SectionCommentDTO> getComments(Long sectionId, PageRequest pageRequest) {
        if (sectionId == null) {
            return PageResponse.of(pageRequest.getPage(), pageRequest.getSize(), 0L, Collections.emptyList());
        }

        // 查询主评论
        Page<SectionComment> page = new Page<>(pageRequest.getPage(), pageRequest.getSize());
        LambdaQueryWrapper<SectionComment> wrapper = new LambdaQueryWrapper<SectionComment>()
                .eq(SectionComment::getSectionId, sectionId)
                .isNull(SectionComment::getParentId)  // 只查询主评论
                .orderByDesc(SectionComment::getCreateTime);
        
        Page<SectionComment> commentPage = commentMapper.selectPage(page, wrapper);
        
        if (commentPage.getRecords().isEmpty()) {
            return PageResponse.of(pageRequest.getPage(), pageRequest.getSize(), 0L, Collections.emptyList());
        }
        
        // 获取所有主评论ID
        List<Long> parentCommentIds = commentPage.getRecords().stream()
                .map(SectionComment::getId)
                .collect(Collectors.toList());
        
        // 查询所有子评论
        List<SectionComment> replies = commentMapper.selectList(
                new LambdaQueryWrapper<SectionComment>()
                        .eq(SectionComment::getSectionId, sectionId)
                        .in(SectionComment::getParentId, parentCommentIds)
                        .orderByAsc(SectionComment::getCreateTime)
        );
        
        // 构建父评论ID到子评论列表的映射
        Map<Long, List<SectionComment>> replyMap = replies.stream()
                .collect(Collectors.groupingBy(SectionComment::getParentId));
        
        // 获取所有用户ID（包括主评论和子评论的用户）
        List<Long> userIds = new java.util.ArrayList<>();
        userIds.addAll(commentPage.getRecords().stream()
                .map(SectionComment::getUserId)
                .collect(Collectors.toList()));
        userIds.addAll(replies.stream()
                .map(SectionComment::getUserId)
                .collect(Collectors.toList()));
        
        // 查询用户信息
        List<User> users = userMapper.selectBatchIds(userIds);
        Map<Long, User> userMap = users.stream()
                .collect(Collectors.toMap(User::getId, user -> user, (u1, u2) -> u1));
        
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
                    List<SectionComment> commentReplies = replyMap.getOrDefault(comment.getId(), Collections.emptyList());
                    dto.setReplyCount(commentReplies.size());
                    
                    // 添加子评论
                    List<SectionCommentDTO> replyDtos = commentReplies.stream()
                            .map(reply -> {
                                SectionCommentDTO replyDto = new SectionCommentDTO();
                                BeanUtils.copyProperties(reply, replyDto);
                                
                                User replyUser = userMap.get(reply.getUserId());
                                if (replyUser != null) {
                                    replyDto.setUserName(replyUser.getRealName());
                                    replyDto.setUserAvatar(replyUser.getAvatar());
                                    replyDto.setUserRole(replyUser.getRole());
                                }
                                
                                return replyDto;
                            })
                            .collect(Collectors.toList());
                    
                    dto.setReplies(replyDtos);
                    
                    return dto;
                })
                .collect(Collectors.toList());
        
        return PageResponse.of(pageRequest.getPage(), pageRequest.getSize(), commentPage.getTotal(), dtos);
    }
    
    @Override
    public PageResponse<SectionCommentDTO> getCourseComments(Long courseId, PageRequest pageRequest) {
        if (courseId == null) {
            return PageResponse.of(pageRequest.getPage(), pageRequest.getSize(), 0L, Collections.emptyList());
        }
        
        // 1. 查询课程下的所有章节
        List<Chapter> chapters = chapterMapper.selectList(
                new LambdaQueryWrapper<Chapter>()
                        .eq(Chapter::getCourseId, courseId)
        );
        
        if (chapters.isEmpty()) {
            return PageResponse.of(pageRequest.getPage(), pageRequest.getSize(), 0L, Collections.emptyList());
        }
        
        // 2. 查询章节下的所有小节
        List<Long> chapterIds = chapters.stream()
                .map(Chapter::getId)
                .collect(Collectors.toList());
        
        List<Section> sections = sectionMapper.selectList(
                new LambdaQueryWrapper<Section>()
                        .in(Section::getChapterId, chapterIds)
        );
        
        if (sections.isEmpty()) {
            return PageResponse.of(pageRequest.getPage(), pageRequest.getSize(), 0L, Collections.emptyList());
        }
        
        // 3. 查询小节下的所有评论
        List<Long> sectionIds = sections.stream()
                .map(Section::getId)
                .collect(Collectors.toList());
        
        // 创建分页对象
        Page<SectionComment> page = new Page<>(pageRequest.getPage(), pageRequest.getSize());
        
        // 查询评论
        LambdaQueryWrapper<SectionComment> wrapper = new LambdaQueryWrapper<SectionComment>()
                .in(SectionComment::getSectionId, sectionIds)
                .isNull(SectionComment::getParentId)  // 只查询主评论
                .orderByDesc(SectionComment::getCreateTime);
        
        Page<SectionComment> commentPage = commentMapper.selectPage(page, wrapper);
        
        if (commentPage.getRecords().isEmpty()) {
            return PageResponse.of(pageRequest.getPage(), pageRequest.getSize(), 0L, Collections.emptyList());
        }
        
        // 4. 获取用户信息
        List<Long> userIds = commentPage.getRecords().stream()
                .map(SectionComment::getUserId)
                .collect(Collectors.toList());
        
        List<User> users = userMapper.selectBatchIds(userIds);
        Map<Long, User> userMap = users.stream()
                .collect(Collectors.toMap(User::getId, user -> user));
        
        // 5. 创建小节信息映射
        Map<Long, Section> sectionMap = sections.stream()
                .collect(Collectors.toMap(Section::getId, section -> section));
        
        // 6. 转换为DTO
        List<SectionCommentDTO> dtos = commentPage.getRecords().stream()
                .map(comment -> {
                    SectionCommentDTO dto = new SectionCommentDTO();
                    BeanUtils.copyProperties(comment, dto);
                    
                    // 设置用户信息
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
                    
                    // 设置小节信息
                    Section section = sectionMap.get(comment.getSectionId());
                    if (section != null) {
                        dto.setSectionTitle(section.getTitle());
                    }
                    
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

        // 获取当前用户ID和角色
        String username = SecurityContextHolder.getContext().getAuthentication().getName();
        User user = userMapper.selectOne(
            new LambdaQueryWrapper<User>()
                .eq(User::getUsername, username)
        );
        
        if (user == null) {
            throw new IllegalArgumentException("用户不存在");
        }
        
        boolean isTeacher = "TEACHER".equalsIgnoreCase(user.getRole());

        SectionComment comment = commentMapper.selectById(id);
        if (comment == null) {
            throw new IllegalArgumentException("评论不存在");
        }
        
        // 检查权限：如果是教师可以删除任何评论，否则只能删除自己的评论
        if (!isTeacher && !comment.getUserId().equals(user.getId())) {
            throw new IllegalArgumentException("无权删除此评论");
        }
        
        // 先删除子评论，确保级联删除成功
        int deletedReplies = commentMapper.delete(
                new LambdaQueryWrapper<SectionComment>()
                        .eq(SectionComment::getParentId, id)
        );
        System.out.println("已删除" + deletedReplies + "条子评论，父评论ID: " + id);
        
        // 再删除父评论
        commentMapper.deleteById(id);
        System.out.println("已删除父评论，ID: " + id);
    }
    
    @Override
    public PageResponse<SectionCommentDTO> getCommentReplies(Long sectionId, Long commentId, PageRequest pageRequest) {
        if (sectionId == null || commentId == null) {
            return PageResponse.of(pageRequest.getPage(), pageRequest.getSize(), 0L, Collections.emptyList());
        }

        // 查询子评论
        Page<SectionComment> page = new Page<>(pageRequest.getPage(), pageRequest.getSize());
        LambdaQueryWrapper<SectionComment> wrapper = new LambdaQueryWrapper<SectionComment>()
                .eq(SectionComment::getSectionId, sectionId)
                .eq(SectionComment::getParentId, commentId)
                .orderByAsc(SectionComment::getCreateTime);
        
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
                .collect(Collectors.toMap(User::getId, user -> user, (u1, u2) -> u1));
        
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
                    
                    return dto;
                })
                .collect(Collectors.toList());
        
        return PageResponse.of(pageRequest.getPage(), pageRequest.getSize(), commentPage.getTotal(), dtos);
    }
} 