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
        
        Double scoreObj = (Double) requestBody.get("score");
        String feedback = (String) requestBody.get("feedback");
        
        int score = scoreObj != null ? scoreObj.intValue() : 0;
        
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