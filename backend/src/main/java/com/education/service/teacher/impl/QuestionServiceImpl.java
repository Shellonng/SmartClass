package com.education.service.teacher.impl;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import com.education.dto.QuestionDTO;
import com.education.dto.common.PageRequest;
import com.education.dto.common.PageResponse;
import com.education.entity.Question;
import com.education.entity.QuestionImage;
import com.education.entity.QuestionOption;
import com.education.exception.BusinessException;
import com.education.mapper.QuestionImageMapper;
import com.education.mapper.QuestionMapper;
import com.education.mapper.QuestionOptionMapper;
import com.education.service.teacher.QuestionService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.BeanUtils;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.util.StringUtils;

import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.stream.Collectors;

@Service
@RequiredArgsConstructor
@Slf4j
public class QuestionServiceImpl extends ServiceImpl<QuestionMapper, Question> implements QuestionService {

    private final QuestionMapper questionMapper;
    private final QuestionOptionMapper questionOptionMapper;
    private final QuestionImageMapper questionImageMapper;

    @Override
    @Transactional
    public QuestionDTO createQuestion(QuestionDTO.AddRequest request) {
        // 验证请求参数
        validateAddRequest(request);

        // 创建题目
        Question question = new Question();
        BeanUtils.copyProperties(request, question);
        question.setCreateTime(LocalDateTime.now());
        question.setUpdateTime(LocalDateTime.now());
        
        // 确保createdBy字段被设置
        if (question.getCreatedBy() == null) {
            log.warn("题目创建者ID未设置，使用默认值");
            question.setCreatedBy(1L); // 使用默认ID
        }
        
        log.info("准备插入题目到数据库: {}, 创建者ID(用户ID): {}, 课程ID: {}", 
                question.getTitle(), question.getCreatedBy(), question.getCourseId());
        try {
            int insertResult = questionMapper.insert(question);
            log.info("题目插入结果: {}, 获取到的题目ID: {}", insertResult, question.getId());
            
            if (question.getId() == null || question.getId() <= 0) {
                log.error("题目插入失败，未获取到有效ID");
                throw new BusinessException("题目保存失败，请检查数据库配置");
            }
        } catch (Exception e) {
            log.error("题目插入数据库时发生异常: {}", e.getMessage(), e);
            throw new BusinessException("题目保存失败: " + e.getMessage());
        }

        // 保存选项
        if (request.getOptions() != null && !request.getOptions().isEmpty()) {
            log.info("准备插入题目选项，数量: {}", request.getOptions().size());
            try {
                for (QuestionOption option : request.getOptions()) {
                    option.setQuestionId(question.getId());
                    questionOptionMapper.insert(option);
                }
                log.info("题目选项插入成功");
            } catch (Exception e) {
                log.error("题目选项插入失败: {}", e.getMessage(), e);
                throw new BusinessException("题目选项保存失败: " + e.getMessage());
            }
        }

        // 保存图片
        if (request.getImages() != null && !request.getImages().isEmpty()) {
            log.info("准备插入题目图片，数量: {}", request.getImages().size());
            try {
                for (QuestionImage image : request.getImages()) {
                    image.setQuestionId(question.getId());
                    image.setUploadTime(LocalDateTime.now());
                    questionImageMapper.insert(image);
                }
                log.info("题目图片插入成功");
            } catch (Exception e) {
                log.error("题目图片插入失败: {}", e.getMessage(), e);
                throw new BusinessException("题目图片保存失败: " + e.getMessage());
            }
        }

        log.info("题目创建完成，ID: {}", question.getId());
        return getQuestion(question.getId());
    }

    @Override
    @Transactional
    public QuestionDTO updateQuestion(QuestionDTO.UpdateRequest request) {
        // 验证题目是否存在
        Question question = questionMapper.selectById(request.getId());
        if (question == null) {
            throw new BusinessException("题目不存在");
        }

        // 更新题目基本信息
        BeanUtils.copyProperties(request, question);
        question.setUpdateTime(LocalDateTime.now());
        questionMapper.updateById(question);

        // 更新选项
        if (request.getOptions() != null) {
            // 删除旧选项
            questionOptionMapper.delete(
                new LambdaQueryWrapper<QuestionOption>()
                    .eq(QuestionOption::getQuestionId, request.getId())
            );
            
            // 添加新选项
            for (QuestionOption option : request.getOptions()) {
                option.setQuestionId(request.getId());
                questionOptionMapper.insert(option);
            }
        }

        // 更新图片
        if (request.getImages() != null) {
            // 删除旧图片
            questionImageMapper.delete(
                new LambdaQueryWrapper<QuestionImage>()
                    .eq(QuestionImage::getQuestionId, request.getId())
            );
            
            // 添加新图片
            for (QuestionImage image : request.getImages()) {
                image.setQuestionId(request.getId());
                image.setUploadTime(LocalDateTime.now());
                questionImageMapper.insert(image);
            }
        }

        return getQuestion(request.getId());
    }

    @Override
    @Transactional
    public void deleteQuestion(Long id) {
        // 验证题目是否存在
        Question question = questionMapper.selectById(id);
        if (question == null) {
            throw new BusinessException("题目不存在");
        }

        // 删除选项（由于外键级联删除，实际上可以不需要这一步）
        questionOptionMapper.delete(
            new LambdaQueryWrapper<QuestionOption>()
                .eq(QuestionOption::getQuestionId, id)
        );

        // 删除图片（由于外键级联删除，实际上可以不需要这一步）
        questionImageMapper.delete(
            new LambdaQueryWrapper<QuestionImage>()
                .eq(QuestionImage::getQuestionId, id)
        );

        // 删除题目
        questionMapper.deleteById(id);
    }

    @Override
    public QuestionDTO getQuestion(Long id) {
        // 获取题目基本信息
        Question question = questionMapper.selectById(id);
        if (question == null) {
            throw new BusinessException("题目不存在");
        }

        // 转换为DTO
        QuestionDTO dto = new QuestionDTO();
        BeanUtils.copyProperties(question, dto);

        // 获取选项
        List<QuestionOption> options = questionOptionMapper.selectList(
            new LambdaQueryWrapper<QuestionOption>()
                .eq(QuestionOption::getQuestionId, id)
                .orderByAsc(QuestionOption::getOptionLabel)
        );
        dto.setOptions(options);

        // 获取图片
        List<QuestionImage> images = questionImageMapper.selectList(
            new LambdaQueryWrapper<QuestionImage>()
                .eq(QuestionImage::getQuestionId, id)
                .orderByAsc(QuestionImage::getSequence)
        );
        dto.setImages(images);

        return dto;
    }

    @Override
    public PageResponse<QuestionDTO> listQuestions(QuestionDTO.QueryRequest request) {
        try {
            log.info("查询题目列表，参数: {}", request);
            
            // 构建查询条件
            LambdaQueryWrapper<Question> wrapper = new LambdaQueryWrapper<>();
            
            if (request.getCourseId() != null) {
                wrapper.eq(Question::getCourseId, request.getCourseId());
            }
            if (request.getChapterId() != null) {
                wrapper.eq(Question::getChapterId, request.getChapterId());
            }
            if (StringUtils.hasText(request.getQuestionType())) {
                wrapper.eq(Question::getQuestionType, request.getQuestionType());
            }
            if (request.getDifficulty() != null) {
                wrapper.eq(Question::getDifficulty, request.getDifficulty());
            }
            if (StringUtils.hasText(request.getKnowledgePoint())) {
                wrapper.eq(Question::getKnowledgePoint, request.getKnowledgePoint());
            }
            if (StringUtils.hasText(request.getKeyword())) {
                wrapper.like(Question::getTitle, request.getKeyword())
                      .or()
                      .like(Question::getExplanation, request.getKeyword());
            }

            // 分页查询
            Page<Question> page = new Page<>(request.getPageNum(), request.getPageSize());
            Page<Question> questionPage = questionMapper.selectPage(page, wrapper);

            // 转换结果
            List<QuestionDTO> dtoList = questionPage.getRecords().stream()
                .map(this::convertToDTO)
                .collect(Collectors.toList());

            // 构建分页响应
            PageResponse<QuestionDTO> response = new PageResponse<>();
            response.setList(dtoList);
            response.setTotal(questionPage.getTotal());
            response.setPageNum((int)questionPage.getCurrent());
            response.setPageSize((int)questionPage.getSize());
            
            log.info("查询题目列表成功，共 {} 条记录", response.getTotal());
            return response;
        } catch (Exception e) {
            log.error("查询题目列表失败", e);
            // 发生异常时返回空结果，避免前端出错
            PageResponse<QuestionDTO> emptyResponse = new PageResponse<>();
            emptyResponse.setList(Collections.emptyList());
            emptyResponse.setTotal(0L);
            emptyResponse.setPageNum(request.getPageNum());
            emptyResponse.setPageSize(request.getPageSize());
            return emptyResponse;
        }
    }

    @Override
    public PageResponse<QuestionDTO> searchQuestions(String keyword, String type, String difficulty, PageRequest pageRequest) {
        // 构建查询条件
        LambdaQueryWrapper<Question> wrapper = new LambdaQueryWrapper<>();
        
        if (StringUtils.hasText(keyword)) {
            wrapper.like(Question::getTitle, keyword)
                  .or()
                  .like(Question::getExplanation, keyword);
        }
        
        if (StringUtils.hasText(type)) {
            wrapper.eq(Question::getQuestionType, type);
        }
        
        if (StringUtils.hasText(difficulty)) {
            try {
                Integer difficultyValue = Integer.parseInt(difficulty);
                wrapper.eq(Question::getDifficulty, difficultyValue);
            } catch (NumberFormatException e) {
                // 忽略转换错误
            }
        }

        // 分页查询
        Page<Question> page = new Page<>(pageRequest.getPageNum(), pageRequest.getPageSize());
        Page<Question> questionPage = questionMapper.selectPage(page, wrapper);

        // 转换结果
        List<QuestionDTO> dtoList = questionPage.getRecords().stream()
            .map(this::convertToDTO)
            .collect(Collectors.toList());

        // 构建分页响应
        PageResponse<QuestionDTO> response = new PageResponse<>();
        response.setList(dtoList);
        response.setTotal(questionPage.getTotal());
        response.setPageNum((int)questionPage.getCurrent());
        response.setPageSize((int)questionPage.getSize());

        return response;
    }

    @Override
    public List<QuestionDTO> getQuestionsByChapter(Long chapterId) {
        // 根据章节ID查询题目
        LambdaQueryWrapper<Question> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(Question::getChapterId, chapterId);
        
        List<Question> questions = questionMapper.selectList(wrapper);
        
        // 转换为DTO
        return questions.stream()
            .map(this::convertToDTO)
            .collect(Collectors.toList());
    }

    /**
     * 验证添加题目请求
     */
    private void validateAddRequest(QuestionDTO.AddRequest request) {
        if (!StringUtils.hasText(request.getTitle())) {
            throw new BusinessException("题目标题不能为空");
        }
        if (!StringUtils.hasText(request.getQuestionType())) {
            throw new BusinessException("题目类型不能为空");
        }
        if (request.getDifficulty() == null || request.getDifficulty() < 1 || request.getDifficulty() > 5) {
            throw new BusinessException("题目难度必须在1-5之间");
        }
        if (!StringUtils.hasText(request.getCorrectAnswer())) {
            throw new BusinessException("题目答案不能为空");
        }
        if (request.getCourseId() == null) {
            throw new BusinessException("所属课程不能为空");
        }
        if (request.getChapterId() == null) {
            throw new BusinessException("所属章节不能为空");
        }
    }

    /**
     * 将实体转换为DTO
     */
    private QuestionDTO convertToDTO(Question question) {
        QuestionDTO dto = new QuestionDTO();
        BeanUtils.copyProperties(question, dto);
        
        // 设置题目类型描述
        String questionType = question.getQuestionType();
        if (questionType != null) {
            switch (questionType) {
                case "single":
                    dto.setQuestionTypeDesc("单选题");
                    break;
                case "multiple":
                    dto.setQuestionTypeDesc("多选题");
                    break;
                case "true_false":
                    dto.setQuestionTypeDesc("判断题");
                    break;
                case "blank":
                    dto.setQuestionTypeDesc("填空题");
                    break;
                case "short":
                    dto.setQuestionTypeDesc("简答题");
                    break;
                case "code":
                    dto.setQuestionTypeDesc("编程题");
                    break;
                default:
                    dto.setQuestionTypeDesc("其他");
            }
        } else {
            dto.setQuestionTypeDesc("未知");
        }
        
        // 获取选项
        if (question.getId() != null) {
            List<QuestionOption> options = questionOptionMapper.selectList(
                new LambdaQueryWrapper<QuestionOption>()
                    .eq(QuestionOption::getQuestionId, question.getId())
                    .orderByAsc(QuestionOption::getOptionLabel)
            );
            dto.setOptions(options);
            
            // 获取图片
            List<QuestionImage> images = questionImageMapper.selectList(
                new LambdaQueryWrapper<QuestionImage>()
                    .eq(QuestionImage::getQuestionId, question.getId())
                    .orderByAsc(QuestionImage::getSequence)
            );
            dto.setImages(images);
        }
        
        return dto;
    }
} 