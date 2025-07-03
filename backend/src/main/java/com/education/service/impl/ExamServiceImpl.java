package com.education.service.impl;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.core.metadata.IPage;
import com.baomidou.mybatisplus.core.toolkit.CollectionUtils;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import com.education.dto.ExamDTO;
import com.education.dto.common.PageRequest;
import com.education.dto.common.PageResponse;
import com.education.entity.Exam;
import com.education.entity.ExamQuestion;
import com.education.entity.Question;
import com.education.entity.QuestionOption;
import com.education.exception.BusinessException;
import com.education.exception.ResultCode;
import com.education.mapper.ExamMapper;
import com.education.mapper.ExamQuestionMapper;
import com.education.mapper.QuestionMapper;
import com.education.mapper.QuestionOptionMapper;
import com.education.service.ExamService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.BeanUtils;
import org.springframework.dao.EmptyResultDataAccessException;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.util.StringUtils;

import java.time.Duration;
import java.time.LocalDateTime;
import java.util.*;
import java.util.stream.Collectors;

/**
 * 考试服务实现类
 */
@Service
@RequiredArgsConstructor
@Slf4j
public class ExamServiceImpl extends ServiceImpl<ExamMapper, Exam> implements ExamService {
    
    private final ExamMapper examMapper;
    private final ExamQuestionMapper examQuestionMapper;
    private final QuestionMapper questionMapper;
    private final QuestionOptionMapper questionOptionMapper;
    private final JdbcTemplate jdbcTemplate;
    
    @Override
    public PageResponse<ExamDTO> pageExams(PageRequest pageRequest, Long courseId, Long userId, String keyword, Integer status) {
        // 构建查询条件
        LambdaQueryWrapper<Exam> queryWrapper = new LambdaQueryWrapper<>();
        
        // 固定查询type为exam的记录
        queryWrapper.eq(Exam::getType, "exam");
        
        // 根据课程ID和用户ID过滤
        if (courseId != null) {
            queryWrapper.eq(Exam::getCourseId, courseId);
        }
        
        if (userId != null) {
            queryWrapper.eq(Exam::getUserId, userId);
        }
        
        // 根据课程ID或用户ID过滤（OR条件）
        if (courseId != null && userId != null) {
            queryWrapper.and(wrapper -> 
                wrapper.eq(Exam::getCourseId, courseId)
                    .or()
                    .eq(Exam::getUserId, userId)
            );
        }
        
        // 关键词搜索
        if (StringUtils.hasText(keyword)) {
            queryWrapper.like(Exam::getTitle, keyword);
        }
        
        // 状态过滤
        if (status != null) {
            queryWrapper.eq(Exam::getStatus, status);
        }
        
        // 排序
        queryWrapper.orderByDesc(Exam::getCreateTime);
        
        // 分页查询
        IPage<Exam> page = new Page<>(pageRequest.getCurrent(), pageRequest.getPageSize());
        IPage<Exam> result = examMapper.selectPage(page, queryWrapper);
        
        // 构建返回结果
        List<ExamDTO> records = result.getRecords().stream().map(exam -> {
            ExamDTO dto = new ExamDTO();
            BeanUtils.copyProperties(exam, dto);
            
            // 为每个考试添加提交率
            try {
                double submissionRate = getSubmissionRate(exam.getId());
                dto.setSubmissionRate(submissionRate);
            } catch (Exception e) {
                dto.setSubmissionRate(0.0);
            }
            
            return dto;
        }).collect(Collectors.toList());
        
        return new PageResponse<ExamDTO>(
                (int)result.getCurrent(),
                (int)result.getSize(),
                result.getTotal(),
                records
        );
    }
    
    @Override
    public PageResponse<ExamDTO> pageExamsByType(PageRequest pageRequest, Long courseId, Long userId, String keyword, Integer status, String type) {
        // 构建查询条件
        LambdaQueryWrapper<Exam> queryWrapper = new LambdaQueryWrapper<>();
        
        // 根据类型过滤（exam 或 homework）
        if (StringUtils.hasText(type)) {
            queryWrapper.eq(Exam::getType, type);
        }
        
        // 根据课程ID和用户ID过滤
        if (courseId != null) {
            queryWrapper.eq(Exam::getCourseId, courseId);
        }
        
        if (userId != null) {
            queryWrapper.eq(Exam::getUserId, userId);
        }
        
        // 根据课程ID或用户ID过滤（OR条件）
        if (courseId != null && userId != null) {
            queryWrapper.and(wrapper -> 
                wrapper.eq(Exam::getCourseId, courseId)
                    .or()
                    .eq(Exam::getUserId, userId)
            );
        }
        
        // 关键词搜索
        if (StringUtils.hasText(keyword)) {
            queryWrapper.like(Exam::getTitle, keyword);
        }
        
        // 状态过滤
        if (status != null) {
            queryWrapper.eq(Exam::getStatus, status);
        }
        
        // 排序
        queryWrapper.orderByDesc(Exam::getCreateTime);
        
        // 分页查询
        IPage<Exam> page = new Page<>(pageRequest.getCurrent(), pageRequest.getPageSize());
        IPage<Exam> result = examMapper.selectPage(page, queryWrapper);
        
        // 获取所有涉及的课程ID
        List<Long> courseIds = result.getRecords().stream()
                .map(Exam::getCourseId)
                .filter(Objects::nonNull)
                .distinct()
                .collect(Collectors.toList());
        
        // 批量查询课程信息
        Map<Long, String> courseNameMap = new HashMap<>();
        if (!courseIds.isEmpty()) {
            try {
                List<Map<String, Object>> courseInfos = jdbcTemplate.queryForList(
                    "SELECT id, title FROM course WHERE id IN (" + 
                    String.join(",", courseIds.stream().map(String::valueOf).collect(Collectors.toList())) + 
                    ")"
                );
                
                for (Map<String, Object> courseInfo : courseInfos) {
                    Long id = ((Number) courseInfo.get("id")).longValue();
                    String title = (String) courseInfo.get("title");
                    courseNameMap.put(id, title);
                }
            } catch (Exception e) {
                log.error("批量查询课程信息失败: {}", e.getMessage(), e);
            }
        }
        
        // 构建返回结果
        List<ExamDTO> records = result.getRecords().stream().map(exam -> {
            ExamDTO dto = new ExamDTO();
            BeanUtils.copyProperties(exam, dto);
            
            // 设置课程名称
            if (exam.getCourseId() != null) {
                dto.setCourseName(courseNameMap.getOrDefault(exam.getCourseId(), "未知课程"));
            }
            
            // 为每个考试/作业添加提交率
            try {
                double submissionRate = getSubmissionRate(exam.getId());
                dto.setSubmissionRate(submissionRate);
            } catch (Exception e) {
                dto.setSubmissionRate(0.0);
            }
            
            return dto;
        }).collect(Collectors.toList());
        
        return new PageResponse<ExamDTO>(
                (int)result.getCurrent(),
                (int)result.getSize(),
                result.getTotal(),
                records
        );
    }
    
    @Override
    public ExamDTO getExamDetail(Long id) {
        Exam exam = getById(id);
        if (exam == null) {
            throw new BusinessException(ResultCode.DATA_NOT_FOUND, "考试不存在");
        }
        
        ExamDTO examDTO = new ExamDTO();
        BeanUtils.copyProperties(exam, examDTO);
        
        // 查询课程名称
        if (exam.getCourseId() != null) {
            try {
                log.info("开始查询课程名称，课程ID: {}", exam.getCourseId());
                String courseName = jdbcTemplate.queryForObject(
                    "SELECT name FROM course WHERE id = ?", 
                    String.class, 
                    exam.getCourseId()
                );
                log.info("查询到课程名称: {}", courseName);
                examDTO.setCourseName(courseName);
            } catch (Exception e) {
                log.error("查询课程名称失败: {}", e.getMessage(), e);
            }
        } else {
            log.warn("课程ID为空，无法查询课程名称");
        }
        
        // 计算考试时长
        if (exam.getStartTime() != null && exam.getEndTime() != null) {
            Duration duration = Duration.between(exam.getStartTime(), exam.getEndTime());
            examDTO.setDuration((int) duration.toMinutes());
        }
        
        // 设置状态描述
        if (exam.getStatus() != null) {
            examDTO.setStatusDesc(exam.getStatus() == 0 ? "未发布" : "已发布");
        }
        
        // 获取提交率
        try {
            double submissionRate = getSubmissionRate(id);
            examDTO.setSubmissionRate(submissionRate);
        } catch (Exception e) {
            examDTO.setSubmissionRate(0.0);
        }
        
        // 获取考试题目
        List<Map<String, Object>> questionMaps = examQuestionMapper.getExamQuestions(id);
        if (!CollectionUtils.isEmpty(questionMaps)) {
            List<ExamDTO.ExamQuestionDTO> questions = new ArrayList<>();
            
            for (Map<String, Object> map : questionMaps) {
                ExamDTO.ExamQuestionDTO questionDTO = new ExamDTO.ExamQuestionDTO();
                
                // 设置题目基本信息
                questionDTO.setId(Long.valueOf(map.get("id").toString()));
                questionDTO.setTitle(map.get("title").toString());
                questionDTO.setQuestionType(map.get("question_type").toString());
                questionDTO.setQuestionTypeDesc(map.get("question_type_desc").toString());
                questionDTO.setDifficulty(Integer.valueOf(map.get("difficulty").toString()));
                questionDTO.setScore(Integer.valueOf(map.get("score").toString()));
                questionDTO.setSequence(Integer.valueOf(map.get("sequence").toString()));
                
                if (map.get("correct_answer") != null) {
                    questionDTO.setCorrectAnswer(map.get("correct_answer").toString());
                }
                
                if (map.get("explanation") != null) {
                    questionDTO.setExplanation(map.get("explanation").toString());
                }
                
                if (map.get("knowledge_point") != null) {
                    questionDTO.setKnowledgePoint(map.get("knowledge_point").toString());
                }
                
                // 获取选项（如果是选择题或判断题）
                if ("single".equals(questionDTO.getQuestionType()) || 
                    "multiple".equals(questionDTO.getQuestionType()) || 
                    "true_false".equals(questionDTO.getQuestionType())) {
                    
                    LambdaQueryWrapper<QuestionOption> wrapper = new LambdaQueryWrapper<>();
                    wrapper.eq(QuestionOption::getQuestionId, questionDTO.getId());
                    wrapper.orderByAsc(QuestionOption::getOptionLabel);
                    
                    List<QuestionOption> options = questionOptionMapper.selectList(wrapper);
                    if (!CollectionUtils.isEmpty(options)) {
                        List<ExamDTO.QuestionOptionDTO> optionDTOs = options.stream().map(option -> {
                            ExamDTO.QuestionOptionDTO optionDTO = new ExamDTO.QuestionOptionDTO();
                            optionDTO.setId(option.getId());
                            optionDTO.setOptionLabel(option.getOptionLabel());
                            optionDTO.setOptionText(option.getOptionText());
                            return optionDTO;
                        }).collect(Collectors.toList());
                        
                        questionDTO.setOptions(optionDTOs);
                    }
                }
                
                questions.add(questionDTO);
            }
            
            // 按序号排序
            questions.sort(Comparator.comparing(ExamDTO.ExamQuestionDTO::getSequence));
            examDTO.setQuestions(questions);
            
            // 计算总分
            int totalScore = questions.stream().mapToInt(ExamDTO.ExamQuestionDTO::getScore).sum();
            examDTO.setTotalScore(totalScore);
        }
        
        return examDTO;
    }
    
    @Override
    @Transactional
    public Long createExam(ExamDTO examDTO) {
        // 基本验证
        if (examDTO == null) {
            throw new BusinessException(ResultCode.PARAM_ERROR, "考试信息不能为空");
        }
        
        if (!StringUtils.hasText(examDTO.getTitle())) {
            throw new BusinessException(ResultCode.PARAM_ERROR, "考试标题不能为空");
        }
        
        if (examDTO.getCourseId() == null) {
            throw new BusinessException(ResultCode.PARAM_ERROR, "所属课程不能为空");
        }
        
        if (examDTO.getUserId() == null) {
            throw new BusinessException(ResultCode.PARAM_ERROR, "用户ID不能为空");
        }
        
        if (examDTO.getStartTime() == null || examDTO.getEndTime() == null) {
            throw new BusinessException(ResultCode.PARAM_ERROR, "考试时间不能为空");
        }
        
        if (examDTO.getStartTime().isAfter(examDTO.getEndTime())) {
            throw new BusinessException(ResultCode.PARAM_ERROR, "开始时间不能晚于结束时间");
        }
        
        // 创建考试
        Exam exam = new Exam();
        BeanUtils.copyProperties(examDTO, exam);
        // 如果没有设置类型，默认为exam
        if (exam.getType() == null) {
            exam.setType("exam");
        }
        exam.setStatus(0); // 默认未发布
        exam.setCreateTime(LocalDateTime.now());
        exam.setUpdateTime(LocalDateTime.now());
        
        save(exam);
        return exam.getId();
    }
    
    @Override
    @Transactional
    public boolean updateExam(ExamDTO examDTO) {
        // 基本验证
        if (examDTO == null || examDTO.getId() == null) {
            throw new BusinessException(ResultCode.PARAM_ERROR, "考试信息不完整");
        }
        
        Exam exam = getById(examDTO.getId());
        if (exam == null) {
            throw new BusinessException(ResultCode.DATA_NOT_FOUND, "考试不存在");
        }
        
        // 检查是否为取消发布操作（从已发布状态变为未发布状态）
        boolean isUnpublishOperation = exam.getStatus() == 1 && examDTO.getStatus() != null && examDTO.getStatus() == 0;
        
        // 如果不是取消发布操作，且考试已发布，则禁止修改
        if (exam.getStatus() == 1 && !isUnpublishOperation) {
            throw new BusinessException(ResultCode.BUSINESS_ERROR, "已发布的考试不能修改");
        }
        
        // 更新考试基本信息
        BeanUtils.copyProperties(examDTO, exam);
        exam.setUpdateTime(LocalDateTime.now());
        
        return updateById(exam);
    }
    
    @Override
    @Transactional
    public boolean deleteExam(Long id) {
        Exam exam = getById(id);
        if (exam == null) {
            throw new BusinessException(ResultCode.DATA_NOT_FOUND, "考试不存在");
        }
        
        if (exam.getStatus() == 1) {
            throw new BusinessException(ResultCode.BUSINESS_ERROR, "已发布的考试不能删除");
        }
        
        // 删除考试题目关联
        LambdaQueryWrapper<ExamQuestion> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(ExamQuestion::getAssignmentId, id);
        examQuestionMapper.delete(wrapper);
        
        // 删除考试
        return removeById(id);
    }
    
    @Override
    @Transactional
    public boolean publishExam(Long id) {
        Exam exam = getById(id);
        if (exam == null) {
            throw new BusinessException(ResultCode.DATA_NOT_FOUND, "考试不存在");
        }
        
        // 检查是否有题目
        LambdaQueryWrapper<ExamQuestion> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(ExamQuestion::getAssignmentId, id);
        int count = examQuestionMapper.selectCount(wrapper).intValue();
        
        if (count == 0) {
            throw new BusinessException(ResultCode.BUSINESS_ERROR, "考试没有题目，不能发布");
        }
        
        // 发布考试
        exam.setStatus(1);
        exam.setUpdateTime(LocalDateTime.now());
        
        return updateById(exam);
    }
    
    @Override
    @Transactional
    public ExamDTO generateExamPaper(ExamDTO examDTO) {
        if (examDTO == null || examDTO.getId() == null || examDTO.getPaperConfig() == null) {
            throw new BusinessException(ResultCode.PARAM_ERROR, "考试信息或组卷配置不完整");
        }
        
        Exam exam = getById(examDTO.getId());
        if (exam == null) {
            throw new BusinessException(ResultCode.DATA_NOT_FOUND, "考试不存在");
        }
        
        if (exam.getStatus() == 1) {
            throw new BusinessException(ResultCode.BUSINESS_ERROR, "已发布的考试不能重新组卷");
        }
        
        // 清除原有题目
        LambdaQueryWrapper<ExamQuestion> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(ExamQuestion::getAssignmentId, examDTO.getId());
        examQuestionMapper.delete(wrapper);
        
        ExamDTO.ExamPaperConfig config = examDTO.getPaperConfig();
        Long courseId = exam.getCourseId();
        Integer difficulty = config.getDifficulty();
        String knowledgePoint = config.getKnowledgePoint();
        Long createdBy = exam.getUserId(); // 使用考试创建者ID
        
        int sequence = 1;
        int totalScore = 0;
        List<ExamDTO.ExamQuestionDTO> questions = new ArrayList<>();
        
        // 添加单选题
        if (config.getSingleCount() != null && config.getSingleCount() > 0) {
            List<Map<String, Object>> singleQuestions = examQuestionMapper.getRandomQuestions(
                courseId, "single", config.getSingleCount(), difficulty, knowledgePoint, createdBy);
            
            for (Map<String, Object> questionMap : singleQuestions) {
                Long questionId = Long.valueOf(questionMap.get("id").toString());
                
                ExamQuestion examQuestion = new ExamQuestion();
                examQuestion.setAssignmentId(examDTO.getId());
                examQuestion.setQuestionId(questionId);
                examQuestion.setScore(config.getSingleScore());
                examQuestion.setSequence(sequence++);
                examQuestionMapper.insert(examQuestion);
                
                totalScore += config.getSingleScore();
                
                // 构建题目DTO
                ExamDTO.ExamQuestionDTO questionDTO = buildQuestionDTO(questionMap, config.getSingleScore(), examQuestion.getSequence());
                questions.add(questionDTO);
            }
        }
        
        // 添加多选题
        if (config.getMultipleCount() != null && config.getMultipleCount() > 0) {
            List<Map<String, Object>> multipleQuestions = examQuestionMapper.getRandomQuestions(
                courseId, "multiple", config.getMultipleCount(), difficulty, knowledgePoint, createdBy);
            
            for (Map<String, Object> questionMap : multipleQuestions) {
                Long questionId = Long.valueOf(questionMap.get("id").toString());
                
                ExamQuestion examQuestion = new ExamQuestion();
                examQuestion.setAssignmentId(examDTO.getId());
                examQuestion.setQuestionId(questionId);
                examQuestion.setScore(config.getMultipleScore());
                examQuestion.setSequence(sequence++);
                examQuestionMapper.insert(examQuestion);
                
                totalScore += config.getMultipleScore();
                
                // 构建题目DTO
                ExamDTO.ExamQuestionDTO questionDTO = buildQuestionDTO(questionMap, config.getMultipleScore(), examQuestion.getSequence());
                questions.add(questionDTO);
            }
        }
        
        // 添加判断题
        if (config.getTrueFalseCount() != null && config.getTrueFalseCount() > 0) {
            List<Map<String, Object>> trueFalseQuestions = examQuestionMapper.getRandomQuestions(
                courseId, "true_false", config.getTrueFalseCount(), difficulty, knowledgePoint, createdBy);
            
            for (Map<String, Object> questionMap : trueFalseQuestions) {
                Long questionId = Long.valueOf(questionMap.get("id").toString());
                
                ExamQuestion examQuestion = new ExamQuestion();
                examQuestion.setAssignmentId(examDTO.getId());
                examQuestion.setQuestionId(questionId);
                examQuestion.setScore(config.getTrueFalseScore());
                examQuestion.setSequence(sequence++);
                examQuestionMapper.insert(examQuestion);
                
                totalScore += config.getTrueFalseScore();
                
                // 构建题目DTO
                ExamDTO.ExamQuestionDTO questionDTO = buildQuestionDTO(questionMap, config.getTrueFalseScore(), examQuestion.getSequence());
                questions.add(questionDTO);
            }
        }
        
        // 添加填空题d
        if (config.getShortCount() != null && config.getShortCount() > 0) {
            List<Map<String, Object>> shortQuestions = examQuestionMapper.getRandomQuestions(
                courseId, "short", config.getShortCount(), difficulty, knowledgePoint, createdBy);
            
            for (Map<String, Object> questionMap : shortQuestions) {
                Long questionId = Long.valueOf(questionMap.get("id").toString());
                
                ExamQuestion examQuestion = new ExamQuestion();
                examQuestion.setAssignmentId(examDTO.getId());
                examQuestion.setQuestionId(questionId);
                examQuestion.setScore(config.getShortScore());
                examQuestion.setSequence(sequence++);
                examQuestionMapper.insert(examQuestion);
                
                totalScore += config.getShortScore();
                
                // 构建题目DTO
                ExamDTO.ExamQuestionDTO questionDTO = buildQuestionDTO(questionMap, config.getShortScore(), examQuestion.getSequence());
                questions.add(questionDTO);
            }
        }
        
        // 添加编程题
        if (config.getCodeCount() != null && config.getCodeCount() > 0) {
            List<Map<String, Object>> codeQuestions = examQuestionMapper.getRandomQuestions(
                courseId, "code", config.getCodeCount(), difficulty, knowledgePoint, createdBy);
            
            for (Map<String, Object> questionMap : codeQuestions) {
                Long questionId = Long.valueOf(questionMap.get("id").toString());
                
                ExamQuestion examQuestion = new ExamQuestion();
                examQuestion.setAssignmentId(examDTO.getId());
                examQuestion.setQuestionId(questionId);
                examQuestion.setScore(config.getCodeScore());
                examQuestion.setSequence(sequence++);
                examQuestionMapper.insert(examQuestion);
                
                totalScore += config.getCodeScore();
                
                // 构建题目DTO
                ExamDTO.ExamQuestionDTO questionDTO = buildQuestionDTO(questionMap, config.getCodeScore(), examQuestion.getSequence());
                questions.add(questionDTO);
            }
        }
        
        // 更新考试总分
        exam.setUpdateTime(LocalDateTime.now());
        updateById(exam);
        
        // 构建返回结果
        ExamDTO result = new ExamDTO();
        BeanUtils.copyProperties(exam, result);
        result.setQuestions(questions);
        result.setTotalScore(totalScore);
        
        // 计算考试时长
        if (exam.getStartTime() != null && exam.getEndTime() != null) {
            Duration duration = Duration.between(exam.getStartTime(), exam.getEndTime());
            result.setDuration((int) duration.toMinutes());
        }
        
        // 设置状态描述
        if (exam.getStatus() != null) {
            result.setStatusDesc(exam.getStatus() == 0 ? "未发布" : "已发布");
        }
        
        return result;
    }
    
    @Override
    @Transactional
    public boolean selectQuestions(Long examId, List<Long> questionIds, List<Integer> scores) {
        if (examId == null || CollectionUtils.isEmpty(questionIds)) {
            throw new BusinessException(ResultCode.PARAM_ERROR, "考试ID或题目ID不能为空");
        }
        
        if (questionIds.size() != scores.size()) {
            throw new BusinessException(ResultCode.PARAM_ERROR, "题目数量与分值数量不匹配");
        }
        
        Exam exam = getById(examId);
        if (exam == null) {
            throw new BusinessException(ResultCode.DATA_NOT_FOUND, "考试不存在");
        }
        
        if (exam.getStatus() == 1) {
            throw new BusinessException(ResultCode.BUSINESS_ERROR, "已发布的考试不能修改题目");
        }
        
        try {
            // 清除原有题目关联
            LambdaQueryWrapper<ExamQuestion> wrapper = new LambdaQueryWrapper<>();
            wrapper.eq(ExamQuestion::getAssignmentId, examId);
            examQuestionMapper.delete(wrapper);
            
            // 添加新题目关联 - 将题目ID和分值保存到assignment_question表中
            for (int i = 0; i < questionIds.size(); i++) {
                ExamQuestion examQuestion = new ExamQuestion();
                examQuestion.setAssignmentId(examId);
                examQuestion.setQuestionId(questionIds.get(i));
                examQuestion.setScore(scores.get(i));
                examQuestion.setSequence(i + 1); // 设置题目顺序，从1开始递增
                examQuestionMapper.insert(examQuestion);
            }
            
            // 更新考试
            exam.setUpdateTime(LocalDateTime.now());
            return updateById(exam);
        } catch (Exception e) {
            log.error("保存考试题目关联失败", e);
            throw new BusinessException(ResultCode.SYSTEM_ERROR, "保存考试题目关联失败：" + e.getMessage());
        }
    }
    
    /**
     * 构建题目DTO
     */
    private ExamDTO.ExamQuestionDTO buildQuestionDTO(Map<String, Object> questionMap, Integer score, Integer sequence) {
        ExamDTO.ExamQuestionDTO questionDTO = new ExamDTO.ExamQuestionDTO();
        
        questionDTO.setId(Long.valueOf(questionMap.get("id").toString()));
        questionDTO.setTitle(questionMap.get("title").toString());
        questionDTO.setQuestionType(questionMap.get("question_type").toString());
        
        // 设置题型描述
        String questionType = questionMap.get("question_type").toString();
        switch (questionType) {
            case "single":
                questionDTO.setQuestionTypeDesc("单选题");
                break;
            case "multiple":
                questionDTO.setQuestionTypeDesc("多选题");
                break;
            case "true_false":
                questionDTO.setQuestionTypeDesc("判断题");
                break;
            case "blank":
                questionDTO.setQuestionTypeDesc("填空题");
                break;
            case "short":
                questionDTO.setQuestionTypeDesc("简答题");
                break;
            case "code":
                questionDTO.setQuestionTypeDesc("编程题");
                break;
            default:
                questionDTO.setQuestionTypeDesc("");
        }
        
        questionDTO.setDifficulty(Integer.valueOf(questionMap.get("difficulty").toString()));
        questionDTO.setScore(score);
        questionDTO.setSequence(sequence);
        
        if (questionMap.get("correct_answer") != null) {
            questionDTO.setCorrectAnswer(questionMap.get("correct_answer").toString());
        }
        
        if (questionMap.get("explanation") != null) {
            questionDTO.setExplanation(questionMap.get("explanation").toString());
        }
        
        if (questionMap.get("knowledge_point") != null) {
            questionDTO.setKnowledgePoint(questionMap.get("knowledge_point").toString());
        }
        
        // 获取选项（如果是选择题或判断题）
        if ("single".equals(questionType) || "multiple".equals(questionType) || "true_false".equals(questionType)) {
            LambdaQueryWrapper<QuestionOption> wrapper = new LambdaQueryWrapper<>();
            wrapper.eq(QuestionOption::getQuestionId, questionDTO.getId());
            wrapper.orderByAsc(QuestionOption::getOptionLabel);
            
            List<QuestionOption> options = questionOptionMapper.selectList(wrapper);
            if (!CollectionUtils.isEmpty(options)) {
                List<ExamDTO.QuestionOptionDTO> optionDTOs = options.stream().map(option -> {
                    ExamDTO.QuestionOptionDTO optionDTO = new ExamDTO.QuestionOptionDTO();
                    optionDTO.setId(option.getId());
                    optionDTO.setOptionLabel(option.getOptionLabel());
                    optionDTO.setOptionText(option.getOptionText());
                    return optionDTO;
                }).collect(Collectors.toList());
                
                questionDTO.setOptions(optionDTOs);
            }
        }
        
        return questionDTO;
    }
    
    /**
     * 获取知识点列表
     * @param courseId 课程ID
     * @param createdBy 创建者ID
     * @return 知识点列表
     */
    @Override
    public List<String> getKnowledgePoints(Long courseId, Long createdBy) {
        // 构建查询条件
        LambdaQueryWrapper<Question> queryWrapper = new LambdaQueryWrapper<>();
        queryWrapper.select(Question::getKnowledgePoint);
        
        if (courseId != null) {
            queryWrapper.eq(Question::getCourseId, courseId);
        }
        
        if (createdBy != null) {
            queryWrapper.eq(Question::getCreatedBy, createdBy);
        }
        
        // 只查询有知识点的题目
        queryWrapper.isNotNull(Question::getKnowledgePoint);
        queryWrapper.ne(Question::getKnowledgePoint, "");
        
        // 获取所有不同的知识点
        List<Question> questions = questionMapper.selectList(queryWrapper);
        return questions.stream()
                .map(Question::getKnowledgePoint)
                .distinct()
                .filter(StringUtils::hasText)
                .collect(Collectors.toList());
    }
    
    /**
     * 获取题目列表（按题型分类）
     * @param courseId 课程ID
     * @param questionType 题目类型
     * @param difficulty 难度
     * @param knowledgePoint 知识点
     * @param createdBy 创建者ID
     * @param keyword 关键词
     * @return 题目列表（按题型分类）
     */
    @Override
    public Map<String, List<Map<String, Object>>> getQuestionsByType(Long courseId, String questionType, Integer difficulty, String knowledgePoint, Long createdBy, String keyword) {
        Map<String, List<Map<String, Object>>> result = new HashMap<>();
        
        // 如果指定了题型，只查询该题型的题目
        if (StringUtils.hasText(questionType)) {
            List<Map<String, Object>> questions = getQuestionsByTypeInternal(courseId, questionType, difficulty, knowledgePoint, createdBy, keyword);
            result.put(questionType, questions);
            return result;
        }
        
        // 否则查询所有题型的题目
        List<Map<String, Object>> singleQuestions = getQuestionsByTypeInternal(courseId, "single", difficulty, knowledgePoint, createdBy, keyword);
        List<Map<String, Object>> multipleQuestions = getQuestionsByTypeInternal(courseId, "multiple", difficulty, knowledgePoint, createdBy, keyword);
        List<Map<String, Object>> trueFalseQuestions = getQuestionsByTypeInternal(courseId, "true_false", difficulty, knowledgePoint, createdBy, keyword);
        List<Map<String, Object>> blankQuestions = getQuestionsByTypeInternal(courseId, "blank", difficulty, knowledgePoint, createdBy, keyword);
        List<Map<String, Object>> shortQuestions = getQuestionsByTypeInternal(courseId, "short", difficulty, knowledgePoint, createdBy, keyword);
        List<Map<String, Object>> codeQuestions = getQuestionsByTypeInternal(courseId, "code", difficulty, knowledgePoint, createdBy, keyword);
        
        result.put("single", singleQuestions);
        result.put("multiple", multipleQuestions);
        result.put("true_false", trueFalseQuestions);
        result.put("blank", blankQuestions);
        result.put("short", shortQuestions);
        result.put("code", codeQuestions);
        
        return result;
    }
    
    /**
     * 内部方法：按题型查询题目
     */
    private List<Map<String, Object>> getQuestionsByTypeInternal(Long courseId, String questionType, Integer difficulty, String knowledgePoint, Long createdBy, String keyword) {
        // 构建查询条件
        LambdaQueryWrapper<Question> queryWrapper = new LambdaQueryWrapper<>();
        
        queryWrapper.eq(Question::getCourseId, courseId);
        queryWrapper.eq(Question::getQuestionType, questionType);
        
        if (difficulty != null) {
            queryWrapper.eq(Question::getDifficulty, difficulty);
        }
        
        if (StringUtils.hasText(knowledgePoint)) {
            queryWrapper.eq(Question::getKnowledgePoint, knowledgePoint);
        }
        
        if (createdBy != null) {
            queryWrapper.eq(Question::getCreatedBy, createdBy);
        }
        
        // 添加关键词搜索条件
        if (StringUtils.hasText(keyword)) {
            queryWrapper.and(wrapper -> 
                wrapper.like(Question::getTitle, keyword)
                    .or()
                    .like(Question::getKnowledgePoint, keyword)
            );
        }
        
        // 查询题目
        List<Question> questions = questionMapper.selectList(queryWrapper);
        
        // 转换为Map格式
        return questions.stream().map(question -> {
            Map<String, Object> map = new HashMap<>();
            map.put("id", question.getId());
            map.put("title", question.getTitle());
            map.put("question_type", question.getQuestionType());
            map.put("difficulty", question.getDifficulty());
            map.put("knowledge_point", question.getKnowledgePoint());
            map.put("correct_answer", question.getCorrectAnswer());
            map.put("explanation", question.getExplanation());
            
            // 如果是选择题或判断题，获取选项
            if ("single".equals(question.getQuestionType()) || 
                "multiple".equals(question.getQuestionType()) || 
                "true_false".equals(question.getQuestionType())) {
                
                LambdaQueryWrapper<QuestionOption> optionWrapper = new LambdaQueryWrapper<>();
                optionWrapper.eq(QuestionOption::getQuestionId, question.getId());
                optionWrapper.orderByAsc(QuestionOption::getOptionLabel);
                
                List<QuestionOption> options = questionOptionMapper.selectList(optionWrapper);
                List<Map<String, Object>> optionList = new ArrayList<>();
                
                for (QuestionOption option : options) {
                    Map<String, Object> optionMap = new HashMap<>();
                    optionMap.put("id", option.getId());
                    optionMap.put("option_label", option.getOptionLabel());
                    optionMap.put("option_text", option.getOptionText());
                    optionList.add(optionMap);
                }
                
                map.put("options", optionList);
            }
            
            return map;
        }).collect(Collectors.toList());
    }
    
    @Override
    public double getSubmissionRate(Long assignmentId) {
        if (assignmentId == null) {
            throw new BusinessException(ResultCode.PARAM_ERROR, "作业/考试ID不能为空");
        }
        
        // 查询作业/考试是否存在
        Exam exam = getById(assignmentId);
        if (exam == null) {
            throw new BusinessException(ResultCode.DATA_NOT_FOUND, "作业/考试不存在");
        }
        
        try {
            // 查询assignment_submission表中与该assignmentId关联的总记录数
            String totalCountSql = "SELECT COUNT(*) FROM assignment_submission WHERE assignment_id = ?";
            Integer totalCount = jdbcTemplate.queryForObject(totalCountSql, Integer.class, assignmentId);
            
            // 如果没有记录，说明还没有学生关联此作业/考试
            if (totalCount == null || totalCount == 0) {
                return 0.0;
            }
            
            // 查询status > 0（已提交）的记录数
            String submittedCountSql = "SELECT COUNT(*) FROM assignment_submission WHERE assignment_id = ? AND status > 0";
            Integer submittedCount = jdbcTemplate.queryForObject(submittedCountSql, Integer.class, assignmentId);
            
            // 计算提交率
            if (submittedCount == null) {
                return 0.0;
            }
            
            return (double) submittedCount / totalCount * 100;
        } catch (Exception e) {
            log.error("计算作业/考试提交率出错", e);
            return 0.0;
        }
    }

    /**
     * 获取作业提交记录列表
     * @param assignmentId 作业ID
     * @param pageRequest 分页请求
     * @param status 状态筛选
     * @return 提交记录分页结果
     */
    @Override
    public PageResponse<Map<String, Object>> getAssignmentSubmissions(Long assignmentId, PageRequest pageRequest, Integer status) {
        if (assignmentId == null) {
            throw new BusinessException(ResultCode.PARAM_ERROR, "作业ID不能为空");
        }
        
        // 查询作业是否存在
        Exam exam = getById(assignmentId);
        if (exam == null) {
            throw new BusinessException(ResultCode.DATA_NOT_FOUND, "作业不存在");
        }
        
        log.info("开始查询作业提交记录，作业ID: {}, 状态: {}", assignmentId, status);
        
        try {
            // 检查assignment_submission表是否存在，如果不存在则创建
            try {
                jdbcTemplate.queryForObject("SELECT 1 FROM assignment_submission LIMIT 1", Integer.class);
                log.info("assignment_submission表已存在");
            } catch (Exception e) {
                log.warn("assignment_submission表不存在，正在创建...");
                // 创建表
                String createTableSql = "CREATE TABLE IF NOT EXISTS `assignment_submission` (" +
                        "`id` bigint NOT NULL AUTO_INCREMENT COMMENT '主键ID'," +
                        "`assignment_id` bigint NOT NULL COMMENT '作业ID'," +
                        "`student_id` bigint NOT NULL COMMENT '学生ID'," +
                        "`status` int NOT NULL DEFAULT '0' COMMENT '状态：0-未提交，1-已提交未批改，2-已批改'," +
                        "`score` int DEFAULT NULL COMMENT '得分'," +
                        "`feedback` text COMMENT '教师反馈'," +
                        "`submit_time` datetime DEFAULT NULL COMMENT '提交时间'," +
                        "`grade_time` datetime DEFAULT NULL COMMENT '批改时间'," +
                        "`graded_by` bigint DEFAULT NULL COMMENT '批改人ID'," +
                        "`content` text COMMENT '提交内容'," +
                        "`create_time` datetime DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间'," +
                        "`update_time` datetime DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间'," +
                        "PRIMARY KEY (`id`)," +
                        "KEY `idx_assignment_id` (`assignment_id`)," +
                        "KEY `idx_student_id` (`student_id`)," +
                        "KEY `idx_status` (`status`)" +
                        ") ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci COMMENT='作业提交记录表'";
                jdbcTemplate.execute(createTableSql);
                
                // 尝试添加外键约束（可能会失败，但不影响主要功能）
                try {
                    String addForeignKeysSql = "ALTER TABLE `assignment_submission` " +
                            "ADD CONSTRAINT `fk_submission_assignment` FOREIGN KEY (`assignment_id`) REFERENCES `assignment` (`id`) ON DELETE CASCADE," +
                            "ADD CONSTRAINT `fk_submission_student` FOREIGN KEY (`student_id`) REFERENCES `user` (`id`) ON DELETE CASCADE";
                    jdbcTemplate.execute(addForeignKeysSql);
                } catch (Exception ex) {
                    log.warn("添加外键约束失败，但表已创建: " + ex.getMessage());
                }
                
                // 添加测试数据
                try {
                    // 检查是否有学生用户
                    List<Map<String, Object>> students = jdbcTemplate.queryForList("SELECT id FROM user WHERE role = 'STUDENT' LIMIT 3");
                    
                    if (!students.isEmpty()) {
                        // 为当前作业添加测试数据
                        String insertSql = "INSERT INTO assignment_submission (assignment_id, student_id, status, score, feedback, submit_time, grade_time, graded_by, content) VALUES " +
                                "(?, ?, 1, NULL, NULL, NOW(), NULL, NULL, '这是学生提交的作业内容')," +
                                "(?, ?, 2, 85, '做得不错，但有些地方需要改进', DATE_SUB(NOW(), INTERVAL 1 HOUR), NOW(), 3, '这是另一个学生提交的作业内容')," +
                                "(?, ?, 0, NULL, NULL, NULL, NULL, NULL, NULL)";
                        
                        Long studentId1 = ((Number)students.get(0).get("id")).longValue();
                        Long studentId2 = students.size() > 1 ? ((Number)students.get(1).get("id")).longValue() : studentId1;
                        Long studentId3 = students.size() > 2 ? ((Number)students.get(2).get("id")).longValue() : studentId1;
                        
                        jdbcTemplate.update(insertSql, 
                            assignmentId, studentId1,
                            assignmentId, studentId2,
                            assignmentId, studentId3
                        );
                        
                        log.info("已添加测试数据到assignment_submission表");
                    } else {
                        log.warn("未找到学生用户，无法添加测试数据");
                    }
                } catch (Exception ex) {
                    log.warn("添加测试数据失败: " + ex.getMessage());
                }
                
                log.info("assignment_submission表创建成功");
            }
            
            // 构建基础SQL
            StringBuilder sqlBuilder = new StringBuilder();
            sqlBuilder.append("SELECT ");
            sqlBuilder.append("  s.id, ");
            sqlBuilder.append("  s.assignment_id, ");
            sqlBuilder.append("  s.student_id, ");
            sqlBuilder.append("  u.real_name AS student_name, ");
            sqlBuilder.append("  s.status, ");
            sqlBuilder.append("  s.score, ");
            sqlBuilder.append("  s.feedback, ");
            sqlBuilder.append("  s.submit_time, ");
            sqlBuilder.append("  s.grade_time, ");
            sqlBuilder.append("  s.graded_by ");
            sqlBuilder.append("FROM assignment_submission s ");
            sqlBuilder.append("LEFT JOIN user u ON s.student_id = u.id ");
            sqlBuilder.append("WHERE s.assignment_id = ? ");
            
            // 添加状态过滤条件
            if (status != null) {
                sqlBuilder.append("AND s.status = ? ");
            }
            
            // 添加排序
            sqlBuilder.append("ORDER BY s.submit_time DESC, s.student_id ASC ");
            
            // 构建计数SQL
            String countSql = "SELECT COUNT(*) FROM assignment_submission s WHERE s.assignment_id = ?";
            if (status != null) {
                countSql += " AND s.status = ?";
            }
            
            // 添加分页
            sqlBuilder.append("LIMIT ? OFFSET ? ");
            
            // 准备参数
            List<Object> params = new ArrayList<>();
            params.add(assignmentId);
            if (status != null) {
                params.add(status);
            }
            
            // 计算总记录数
            Integer totalCount;
            try {
                log.info("执行计数SQL: {}", countSql);
                if (status != null) {
                    totalCount = jdbcTemplate.queryForObject(countSql, Integer.class, assignmentId, status);
                } else {
                    totalCount = jdbcTemplate.queryForObject(countSql, Integer.class, assignmentId);
                }
                log.info("查询到记录总数: {}", totalCount);
                
                if (totalCount == null || totalCount == 0) {
                    // 没有记录，返回空结果
                    log.info("没有找到符合条件的记录");
                    return PageResponse.<Map<String, Object>>builder()
                            .records(Collections.emptyList())
                            .total(0L)
                            .current(pageRequest.getCurrent())
                            .pageSize(pageRequest.getPageSize())
                            .build();
                }
            } catch (Exception e) {
                log.error("查询总记录数失败: {}", e.getMessage(), e);
                return PageResponse.<Map<String, Object>>builder()
                        .records(Collections.emptyList())
                        .total(0L)
                        .current(pageRequest.getCurrent())
                        .pageSize(pageRequest.getPageSize())
                        .build();
            }
            
            // 添加分页参数
            params.add(pageRequest.getPageSize());
            params.add((pageRequest.getCurrent() - 1) * pageRequest.getPageSize());
            
            // 执行查询
            log.info("执行查询SQL: {}", sqlBuilder.toString());
            List<Map<String, Object>> records = jdbcTemplate.queryForList(sqlBuilder.toString(), params.toArray());
            log.info("查询到记录数: {}", records.size());
            
            // 处理日期格式等
            for (Map<String, Object> record : records) {
                // 将日期格式化为字符串
                if (record.get("submit_time") != null) {
                    record.put("submitTime", record.get("submit_time").toString());
                    record.remove("submit_time");
                }
                if (record.get("grade_time") != null) {
                    record.put("gradeTime", record.get("grade_time").toString());
                    record.remove("grade_time");
                }
                
                // 处理其他字段名称，使其符合前端驼峰命名
                record.put("assignmentId", record.get("assignment_id"));
                record.remove("assignment_id");
                
                record.put("studentId", record.get("student_id"));
                record.remove("student_id");
                
                record.put("studentName", record.get("student_name"));
                record.remove("student_name");
                
                // 添加状态文本描述
                int statusValue = ((Number) record.get("status")).intValue();
                switch (statusValue) {
                    case 0:
                        record.put("statusText", "未提交");
                        break;
                    case 1:
                        record.put("statusText", "已提交未批改");
                        break;
                    case 2:
                        record.put("statusText", "已批改");
                        break;
                    default:
                        record.put("statusText", "未知状态");
                }
            }
            
            // 构建分页响应
            return PageResponse.<Map<String, Object>>builder()
                    .records(records)
                    .total(Long.valueOf(totalCount))
                    .current(pageRequest.getCurrent())
                    .pageSize(pageRequest.getPageSize())
                    .build();
            
        } catch (Exception e) {
            log.error("查询作业提交记录失败: {}", e.getMessage(), e);
            throw new BusinessException(ResultCode.SYSTEM_ERROR, "查询作业提交记录失败");
        }
    }

    /**
     * 批改作业提交
     * @param submissionId 提交记录ID
     * @param score 分数
     * @param feedback 评语
     * @return 是否成功
     */
    @Override
    @Transactional
    public boolean gradeAssignmentSubmission(Long submissionId, int score, String feedback) {
        if (submissionId == null) {
            throw new BusinessException(ResultCode.PARAM_ERROR, "提交记录ID不能为空");
        }
        
        try {
            // 检查表是否存在
            try {
                jdbcTemplate.queryForObject("SELECT 1 FROM assignment_submission LIMIT 1", Integer.class);
            } catch (Exception e) {
                throw new BusinessException(ResultCode.SYSTEM_ERROR, "作业提交记录表不存在，请先访问作业详情页面");
            }
            
            // 获取当前用户ID
            Long currentUserId = 1L; // TODO: 从当前用户上下文中获取
            
            // 查询提交记录是否存在
            String checkSql = "SELECT id, status FROM assignment_submission WHERE id = ?";
            Map<String, Object> submission = null;
            try {
                submission = jdbcTemplate.queryForMap(checkSql, submissionId);
            } catch (EmptyResultDataAccessException e) {
                throw new BusinessException(ResultCode.DATA_NOT_FOUND, "提交记录不存在");
            }
            
            // 检查状态是否为已提交未批改
            Integer status = (Integer) submission.get("status");
            if (status == null || status == 0) {
                throw new BusinessException(ResultCode.PARAM_ERROR, "该作业尚未提交，无法批改");
            }
            
            // 更新提交记录
            String updateSql = "UPDATE assignment_submission SET status = 2, score = ?, feedback = ?, grade_time = NOW(), graded_by = ? WHERE id = ?";
            int rows = jdbcTemplate.update(updateSql, score, feedback, currentUserId, submissionId);
            
            return rows > 0;
        } catch (BusinessException e) {
            throw e;
        } catch (Exception e) {
            log.error("批改作业提交出错", e);
            throw new BusinessException(ResultCode.SYSTEM_ERROR, "批改作业提交失败: " + e.getMessage());
        }
    }

    /**
     * 删除提交记录
     * @param submissionId 提交记录ID
     * @return 是否成功
     */
    @Override
    @Transactional
    public boolean deleteAssignmentSubmission(Long submissionId) {
        if (submissionId == null) {
            throw new BusinessException(ResultCode.PARAM_ERROR, "提交记录ID不能为空");
        }
        
        try {
            // 检查表是否存在
            try {
                jdbcTemplate.queryForObject("SELECT 1 FROM assignment_submission LIMIT 1", Integer.class);
            } catch (Exception e) {
                throw new BusinessException(ResultCode.SYSTEM_ERROR, "作业提交记录表不存在，请先访问作业详情页面");
            }
            
            // 查询提交记录是否存在
            String checkSql = "SELECT id FROM assignment_submission WHERE id = ?";
            try {
                jdbcTemplate.queryForMap(checkSql, submissionId);
            } catch (EmptyResultDataAccessException e) {
                throw new BusinessException(ResultCode.DATA_NOT_FOUND, "提交记录不存在");
            }
            
            // 删除提交记录
            String deleteSql = "DELETE FROM assignment_submission WHERE id = ?";
            int rows = jdbcTemplate.update(deleteSql, submissionId);
            
            return rows > 0;
        } catch (BusinessException e) {
            throw e;
        } catch (Exception e) {
            log.error("删除提交记录出错", e);
            throw new BusinessException(ResultCode.SYSTEM_ERROR, "删除提交记录失败: " + e.getMessage());
        }
    }
} 