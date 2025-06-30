package com.education.service;

import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.baomidou.mybatisplus.extension.service.IService;
import com.education.dto.ExamDTO;
import com.education.dto.common.PageRequest;
import com.education.dto.common.PageResponse;
import com.education.entity.Exam;

import java.util.List;
import java.util.Map;

/**
 * 考试服务接口
 */
public interface ExamService extends IService<Exam> {
    
    /**
     * 分页查询考试列表
     * @param pageRequest 分页请求
     * @param courseId 课程ID
     * @param userId 用户ID
     * @param keyword 关键词
     * @param status 状态
     * @return 分页结果
     */
    PageResponse<ExamDTO> pageExams(PageRequest pageRequest, Long courseId, Long userId, String keyword, Integer status);
    
    /**
     * 获取考试详情
     * @param id 考试ID
     * @return 考试详情
     */
    ExamDTO getExamDetail(Long id);
    
    /**
     * 创建考试
     * @param examDTO 考试信息
     * @return 创建的考试ID
     */
    Long createExam(ExamDTO examDTO);
    
    /**
     * 更新考试
     * @param examDTO 考试信息
     * @return 是否成功
     */
    boolean updateExam(ExamDTO examDTO);
    
    /**
     * 删除考试
     * @param id 考试ID
     * @return 是否成功
     */
    boolean deleteExam(Long id);
    
    /**
     * 发布考试
     * @param id 考试ID
     * @return 是否成功
     */
    boolean publishExam(Long id);
    
    /**
     * 组卷（自动生成试卷）
     * @param examDTO 考试信息（包含组卷配置）
     * @return 组卷后的考试信息
     */
    ExamDTO generateExamPaper(ExamDTO examDTO);
    
    /**
     * 手动选题
     * @param examId 考试ID
     * @param questionIds 题目ID列表
     * @param scores 分值列表
     * @return 是否成功
     */
    boolean selectQuestions(Long examId, List<Long> questionIds, List<Integer> scores);
    
    /**
     * 获取知识点列表
     * @param courseId 课程ID
     * @param createdBy 创建者ID
     * @return 知识点列表
     */
    List<String> getKnowledgePoints(Long courseId, Long createdBy);
    
    /**
     * 获取题目列表（按题型分类）
     * @param courseId 课程ID
     * @param questionType 题目类型
     * @param difficulty 难度
     * @param knowledgePoint 知识点
     * @param createdBy 创建者ID
     * @return 题目列表（按题型分类）
     */
    Map<String, List<Map<String, Object>>> getQuestionsByType(Long courseId, String questionType, Integer difficulty, String knowledgePoint, Long createdBy);
} 