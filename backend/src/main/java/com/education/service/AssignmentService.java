package com.education.service;

import com.baomidou.mybatisplus.core.metadata.IPage;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.education.dto.AssignmentDTO;
import com.education.entity.Assignment;
import com.baomidou.mybatisplus.extension.service.IService;

import java.util.List;
import java.util.Map;

/**
 * 作业服务接口
 */
public interface AssignmentService extends IService<Assignment> {
    
    /**
     * 分页查询作业列表
     * @param page 分页参数
     * @param courseId 课程ID（可选）
     * @param userId 用户ID（可选）
     * @param keyword 关键词（可选）
     * @param status 状态（可选）
     * @return 分页结果
     */
    IPage<AssignmentDTO> pageAssignments(Page<Assignment> page, Long courseId, Long userId, String keyword, Integer status);
    
    /**
     * 根据ID获取作业详情
     * @param id 作业ID
     * @return 作业详情
     */
    AssignmentDTO getAssignmentById(Long id);
    
    /**
     * 创建作业
     * @param assignmentDTO 作业数据
     * @return 作业ID
     */
    Long createAssignment(AssignmentDTO assignmentDTO);
    
    /**
     * 更新作业
     * @param id 作业ID
     * @param assignmentDTO 作业数据
     * @return 是否成功
     */
    Boolean updateAssignment(Long id, AssignmentDTO assignmentDTO);
    
    /**
     * 删除作业
     * @param id 作业ID
     * @return 是否成功
     */
    Boolean deleteAssignment(Long id);
    
    /**
     * 发布作业
     * @param id 作业ID
     * @return 是否成功
     */
    Boolean publishAssignment(Long id);
    
    /**
     * 取消发布作业
     * @param id 作业ID
     * @return 是否成功
     */
    Boolean unpublishAssignment(Long id);
    
    /**
     * 智能组卷
     * @param assignmentDTO 作业配置
     * @return 生成的题目列表
     */
    List<AssignmentDTO.AssignmentQuestionDTO> generatePaper(AssignmentDTO assignmentDTO);
    
    /**
     * 手动选题
     * @param assignmentId 作业ID
     * @param questionIds 题目ID列表
     * @return 是否成功
     */
    Boolean selectQuestions(Long assignmentId, List<Long> questionIds);
    
    /**
     * 获取作业提交率
     * @param assignmentId 作业ID
     * @return 提交率
     */
    Double getSubmissionRate(Long assignmentId);
    
    /**
     * 获取作业提交记录
     * @param assignmentId 作业ID
     * @param page 分页参数
     * @param status 状态筛选（可选）
     * @return 提交记录列表
     */
    IPage<Map<String, Object>> getAssignmentSubmissions(Long assignmentId, Page<Object> page, Integer status);
    
    /**
     * 设置参考答案
     * @param assignmentId 作业ID
     * @param referenceAnswer 参考答案
     * @return 是否成功
     */
    Boolean setReferenceAnswer(Long assignmentId, String referenceAnswer);
    
    /**
     * 批量智能批改
     * @param assignmentId 作业ID
     * @return 批改任务ID
     */
    String aiGradeBatch(Long assignmentId);
    
    /**
     * 获取批改状态
     * @param taskId 任务ID
     * @return 批改状态
     */
    Map<String, Object> getGradingStatus(String taskId);
    
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
     * @param keyword 关键词
     * @return 题目列表（按题型分类）
     */
    Map<String, List<Map<String, Object>>> getQuestionsByType(Long courseId, String questionType, Integer difficulty, 
                                                             String knowledgePoint, Long createdBy, String keyword);
}