package com.education.controller.teacher;

import com.education.dto.QuestionDTO;
import com.education.dto.common.Result;
import com.education.service.teacher.QuestionService;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpSession;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.web.bind.annotation.*;

import java.util.ArrayList;
import java.util.List;

@RestController
@RequestMapping("/api/teacher/questions")
@RequiredArgsConstructor
@Tag(name = "题目管理", description = "教师题目管理接口")
@Slf4j
public class QuestionController {

    private final QuestionService questionService;

    @PostMapping
    @Operation(summary = "添加题目")
    public Result<QuestionDTO> addQuestion(@RequestBody QuestionDTO.AddRequest request, HttpServletRequest httpRequest) {
        try {
            log.info("接收到添加题目请求: {}", request.getTitle());
            
            // 从Session中获取当前用户ID
            HttpSession session = httpRequest.getSession(false);
            if (session != null) {
                Long userId = (Long) session.getAttribute("userId");
                if (userId != null) {
                    log.info("从Session中获取到用户ID: {}", userId);
                    request.setCreatedBy(userId);
                    // 检查是否有对应的教师记录
                    log.info("设置创建者ID为用户ID: {}，注意：此ID应该存在于user表中，而不是teacher表中", userId);
                } else {
                    log.warn("Session中未找到用户ID，将使用默认值");
                    // 设置默认值，防止数据库错误
                    request.setCreatedBy(1L); // 使用默认ID，如系统管理员ID
                }
            } else {
                log.warn("未找到有效的Session，将使用默认值");
                request.setCreatedBy(1L); // 使用默认ID
            }
            
            log.info("开始调用服务创建题目，标题: {}, 创建者ID: {}, 课程ID: {}", 
                    request.getTitle(), request.getCreatedBy(), request.getCourseId());
            QuestionDTO questionDTO = questionService.createQuestion(request);
            log.info("题目创建成功，ID: {}", questionDTO.getId());
            
            return Result.success(questionDTO);
        } catch (Exception e) {
            log.error("添加题目失败: {}", e.getMessage(), e);
            return Result.error(500, "添加题目失败: " + e.getMessage());
        }
    }

    @PutMapping
    @Operation(summary = "更新题目")
    public Result<QuestionDTO> updateQuestion(@RequestBody QuestionDTO.UpdateRequest request) {
        QuestionDTO questionDTO = questionService.updateQuestion(request);
        return Result.success(questionDTO);
    }

    @DeleteMapping("/{id}")
    @Operation(summary = "删除题目")
    public Result<Void> deleteQuestion(@PathVariable Long id) {
        questionService.deleteQuestion(id);
        return Result.success();
    }

    @GetMapping("/{id}")
    @Operation(summary = "获取题目详情")
    public Result<QuestionDTO> getQuestion(@PathVariable Long id) {
        QuestionDTO questionDTO = questionService.getQuestion(id);
        return Result.success(questionDTO);
    }

    @GetMapping("/list")
    @Operation(summary = "分页查询题目")
    public Result<Object> listQuestions(QuestionDTO.QueryRequest request) {
        return Result.success(questionService.listQuestions(request));
    }
    
    @GetMapping("/course/{courseId}")
    @Operation(summary = "获取课程下的题目")
    public Result<List<QuestionDTO>> getQuestionsByCourse(@PathVariable Long courseId) {
        try {
            log.info("获取课程ID为 {} 的题目列表", courseId);
            QuestionDTO.QueryRequest request = new QuestionDTO.QueryRequest();
            request.setCourseId(courseId);
            request.setPageNum(1);
            request.setPageSize(1000); // 设置一个较大的值，获取所有题目
            List<QuestionDTO> questions = questionService.listQuestions(request).getList();
            log.info("成功获取课程ID为 {} 的题目列表，共 {} 条记录", courseId, questions.size());
            return Result.success(questions);
        } catch (Exception e) {
            log.error("获取课程ID为 {} 的题目列表时发生错误: {}", courseId, e.getMessage(), e);
            // 发生错误时返回空列表，避免前端出错
            return Result.success(new ArrayList<>());
        }
    }

    @GetMapping("/chapter/{chapterId}")
    @Operation(summary = "获取章节下的题目")
    public Result<Object> getQuestionsByChapter(@PathVariable Long chapterId) {
        return Result.success(questionService.getQuestionsByChapter(chapterId));
    }
} 