package com.education.controller.teacher;

import com.education.dto.common.Result;
import com.education.service.ExamService;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.*;

import java.util.Map;

/**
 * 作业提交记录控制器
 */
@RestController
@RequestMapping("/api/teacher/submissions")
@RequiredArgsConstructor
public class SubmissionController {

    private final ExamService examService;
    
    /**
     * 批改作业提交
     * @param submissionId 提交记录ID
     * @param requestBody 批改信息
     * @return 是否成功
     */
    @PutMapping("/{submissionId}/grade")
    public Result<Boolean> gradeSubmission(
            @PathVariable Long submissionId,
            @RequestBody Map<String, Object> requestBody) {
        
        // 修复类型转换问题，处理多种可能的数字类型
        Object scoreObj = requestBody.get("score");
        String feedback = (String) requestBody.get("feedback");
        
        int score = 0;
        if (scoreObj != null) {
            if (scoreObj instanceof Integer) {
                score = (Integer) scoreObj;
            } else if (scoreObj instanceof Double) {
                score = ((Double) scoreObj).intValue();
            } else if (scoreObj instanceof String) {
                try {
                    score = Integer.parseInt((String) scoreObj);
                } catch (NumberFormatException e) {
                    try {
                        score = (int) Double.parseDouble((String) scoreObj);
                    } catch (NumberFormatException ex) {
                        // 如果无法解析，使用默认值0
                    }
                }
            } else if (scoreObj instanceof Number) {
                score = ((Number) scoreObj).intValue();
            }
        }
        
        boolean success = examService.gradeAssignmentSubmission(submissionId, score, feedback);
        return Result.success(success);
    }
    
    /**
     * 删除提交记录
     * @param submissionId 提交记录ID
     * @return 是否成功
     */
    @DeleteMapping("/{submissionId}")
    public Result<Boolean> deleteSubmission(@PathVariable Long submissionId) {
        boolean success = examService.deleteAssignmentSubmission(submissionId);
        return Result.success(success);
    }
} 