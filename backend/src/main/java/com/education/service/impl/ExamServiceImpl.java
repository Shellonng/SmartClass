package com.education.service.impl;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.core.metadata.IPage;
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
import org.springframework.beans.BeanUtils;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.util.CollectionUtils;
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
public class ExamServiceImpl extends ServiceImpl<ExamMapper, Exam> implements ExamService {
    
    private final ExamMapper examMapper;
    private final ExamQuestionMapper examQuestionMapper;
    private final QuestionMapper questionMapper;
    private final QuestionOptionMapper questionOptionMapper;
    
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
        
        // 计算考试时长
        if (exam.getStartTime() != null && exam.getEndTime() != null) {
            Duration duration = Duration.between(exam.getStartTime(), exam.getEndTime());
            examDTO.setDuration((int) duration.toMinutes());
        }
        
        // 设置状态描述
        if (exam.getStatus() != null) {
            examDTO.setStatusDesc(exam.getStatus() == 0 ? "未发布" : "已发布");
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
        
        if (examDTO.getStartTime() == null || examDTO.getEndTime() == null) {
            throw new BusinessException(ResultCode.PARAM_ERROR, "考试时间不能为空");
        }
        
        if (examDTO.getStartTime().isAfter(examDTO.getEndTime())) {
            throw new BusinessException(ResultCode.PARAM_ERROR, "开始时间不能晚于结束时间");
        }
        
        // 创建考试
        Exam exam = new Exam();
        BeanUtils.copyProperties(examDTO, exam);
        exam.setType("exam");
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
        
        if (exam.getStatus() == 1) {
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
        
        // 添加填空题
        if (config.getBlankCount() != null && config.getBlankCount() > 0) {
            List<Map<String, Object>> blankQuestions = examQuestionMapper.getRandomQuestions(
                courseId, "blank", config.getBlankCount(), difficulty, knowledgePoint, createdBy);
            
            for (Map<String, Object> questionMap : blankQuestions) {
                Long questionId = Long.valueOf(questionMap.get("id").toString());
                
                ExamQuestion examQuestion = new ExamQuestion();
                examQuestion.setAssignmentId(examDTO.getId());
                examQuestion.setQuestionId(questionId);
                examQuestion.setScore(config.getBlankScore());
                examQuestion.setSequence(sequence++);
                examQuestionMapper.insert(examQuestion);
                
                totalScore += config.getBlankScore();
                
                // 构建题目DTO
                ExamDTO.ExamQuestionDTO questionDTO = buildQuestionDTO(questionMap, config.getBlankScore(), examQuestion.getSequence());
                questions.add(questionDTO);
            }
        }
        
        // 添加简答题
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
        
        // 清除原有题目
        LambdaQueryWrapper<ExamQuestion> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(ExamQuestion::getAssignmentId, examId);
        examQuestionMapper.delete(wrapper);
        
        // 添加新题目
        for (int i = 0; i < questionIds.size(); i++) {
            ExamQuestion examQuestion = new ExamQuestion();
            examQuestion.setAssignmentId(examId);
            examQuestion.setQuestionId(questionIds.get(i));
            examQuestion.setScore(scores.get(i));
            examQuestion.setSequence(i + 1);
            examQuestionMapper.insert(examQuestion);
        }
        
        // 更新考试
        exam.setUpdateTime(LocalDateTime.now());
        return updateById(exam);
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
                questionDTO.setQuestionTypeDesc("未知类型");
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
     * @return 题目列表（按题型分类）
     */
    @Override
    public Map<String, List<Map<String, Object>>> getQuestionsByType(Long courseId, String questionType, Integer difficulty, String knowledgePoint, Long createdBy) {
        Map<String, List<Map<String, Object>>> result = new HashMap<>();
        
        // 如果指定了题型，只查询该题型的题目
        if (StringUtils.hasText(questionType)) {
            List<Map<String, Object>> questions = getQuestionsByTypeInternal(courseId, questionType, difficulty, knowledgePoint, createdBy);
            result.put(questionType, questions);
            return result;
        }
        
        // 否则查询所有题型的题目
        List<Map<String, Object>> singleQuestions = getQuestionsByTypeInternal(courseId, "single", difficulty, knowledgePoint, createdBy);
        List<Map<String, Object>> multipleQuestions = getQuestionsByTypeInternal(courseId, "multiple", difficulty, knowledgePoint, createdBy);
        List<Map<String, Object>> trueFalseQuestions = getQuestionsByTypeInternal(courseId, "true_false", difficulty, knowledgePoint, createdBy);
        List<Map<String, Object>> blankQuestions = getQuestionsByTypeInternal(courseId, "blank", difficulty, knowledgePoint, createdBy);
        List<Map<String, Object>> shortQuestions = getQuestionsByTypeInternal(courseId, "short", difficulty, knowledgePoint, createdBy);
        List<Map<String, Object>> codeQuestions = getQuestionsByTypeInternal(courseId, "code", difficulty, knowledgePoint, createdBy);
        
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
    private List<Map<String, Object>> getQuestionsByTypeInternal(Long courseId, String questionType, Integer difficulty, String knowledgePoint, Long createdBy) {
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
} 