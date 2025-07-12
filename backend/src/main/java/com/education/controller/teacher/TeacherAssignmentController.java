package com.education.controller.teacher;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.education.entity.Assignment;
import com.education.entity.AssignmentSubmission;
import com.education.entity.Student;
import com.education.entity.User;
import com.education.mapper.AssignmentMapper;
import com.education.mapper.AssignmentSubmissionMapper;
import com.education.mapper.StudentMapper;
import com.education.mapper.UserMapper;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.Parameter;
import io.swagger.v3.oas.annotations.tags.Tag;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.web.bind.annotation.*;

import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;
import jakarta.servlet.http.HttpSession;
import java.io.File;
import java.io.IOException;
import java.net.URLEncoder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;

@RestController
@RequestMapping("/api/teacher/assignments")
@Tag(name = "教师作业管理接口", description = "提供教师管理作业的相关接口")
public class TeacherAssignmentController {
    
    private static final Logger logger = LoggerFactory.getLogger(TeacherAssignmentController.class);
    
    @Autowired
    private AssignmentMapper assignmentMapper;
    
    @Autowired
    private AssignmentSubmissionMapper assignmentSubmissionMapper;
    
    @Autowired
    private StudentMapper studentMapper;
    
    @Autowired
    private UserMapper userMapper;
    
    /**
     * 获取学生姓名
     * @param studentId 学生ID
     * @return 学生姓名
     */
    private String getStudentName(Long studentId) {
        String studentName = "未知学生";
        
        if (studentId != null) {
            // 获取学生信息
            Student student = studentMapper.selectById(studentId);
            
            if (student != null) {
                // 如果学生对象存在，尝试获取关联的用户信息
                if (student.getUser() != null && student.getUser().getRealName() != null) {
                    // 如果用户对象存在且有真实姓名，则使用真实姓名
                    studentName = student.getUser().getRealName();
                } else {
                    // 否则尝试从数据库加载用户信息
                    User user = userMapper.selectById(student.getUserId());
                    if (user != null && user.getRealName() != null && !user.getRealName().isEmpty()) {
                        studentName = user.getRealName();
                    } else if (user != null) {
                        // 如果没有真实姓名，则使用用户名
                        studentName = user.getUsername();
                    } else {
                        // 如果无法获取用户信息，则使用学生ID
                        studentName = "学生" + student.getId();
                    }
                }
            }
        }
        
        return studentName;
    }

    /**
     * 下载学生提交的作业文件
     * @param submissionId 提交记录ID
     * @param request HTTP请求
     * @param response HTTP响应
     */
    @Operation(summary = "下载学生提交的作业文件", description = "教师下载学生提交的作业文件")
    @GetMapping("/submissions/{submissionId}/download")
    public void downloadSubmissionFile(
            @Parameter(description = "提交记录ID") @PathVariable Long submissionId,
            HttpServletRequest request, 
            HttpServletResponse response) {
        
        // 从Session中获取userId
        HttpSession session = request.getSession(false);
        Long userId = null;
        
        if (session != null) {
            userId = (Long) session.getAttribute("userId");
        }
        
        if (userId == null) {
            logger.error("无法从Session获取用户ID");
            response.setStatus(HttpServletResponse.SC_BAD_REQUEST);
            return;
        }
        
        logger.info("接收到下载学生提交文件请求 - 提交ID: {}, 教师ID: {}", submissionId, userId);
        
        try {
            // 获取提交记录
            AssignmentSubmission submission = assignmentSubmissionMapper.selectById(submissionId);
            if (submission == null) {
                logger.error("找不到提交记录: submissionId={}", submissionId);
                response.setStatus(HttpServletResponse.SC_NOT_FOUND);
                return;
            }
            
            // 获取作业信息
            Assignment assignment = assignmentMapper.selectById(submission.getAssignmentId());
            if (assignment == null) {
                logger.error("找不到作业: assignmentId={}", submission.getAssignmentId());
                response.setStatus(HttpServletResponse.SC_NOT_FOUND);
                return;
            }
            
            // 验证教师权限（确保是该课程的教师）
            if (!assignment.getUserId().equals(userId)) {
                logger.error("权限不足，非作业创建者: assignmentId={}, teacherId={}", assignment.getId(), userId);
                response.setStatus(HttpServletResponse.SC_FORBIDDEN);
                return;
            }
            
            // 查找学生提交的文件
            String uploadDir = "D:/my_git_code/SmartClass/resource/file/assignments/" + submission.getAssignmentId() + "/" + submission.getStudentId() + "/";
            File dir = new File(uploadDir);
            
            if (!dir.exists() || !dir.isDirectory()) {
                logger.error("提交文件目录不存在: {}", uploadDir);
                response.setStatus(HttpServletResponse.SC_NOT_FOUND);
                return;
            }
            
            // 获取目录中的第一个文件（假设每次提交只有一个文件）
            File[] files = dir.listFiles();
            if (files == null || files.length == 0) {
                logger.error("提交目录中没有文件: {}", uploadDir);
                response.setStatus(HttpServletResponse.SC_NOT_FOUND);
                return;
            }
            
            File submittedFile = files[0]; // 获取第一个文件
            
            // 获取学生信息
            String studentName = getStudentName(submission.getStudentId());
            
            // 获取原始文件名和扩展名
            String fileName = submittedFile.getName();
            String fileExtension = "";
            int dotIndex = fileName.lastIndexOf('.');
            if (dotIndex > 0) {
                fileExtension = fileName.substring(dotIndex);
            }
            
            // 构建下载文件名（学生姓名_作业标题.扩展名）
            String downloadFileName = studentName + "_" + assignment.getTitle() + fileExtension;
            String encodedFileName = URLEncoder.encode(downloadFileName, StandardCharsets.UTF_8.name()).replaceAll("\\+", "%20");
            
            // 设置响应头
            response.setHeader(HttpHeaders.CONTENT_DISPOSITION, "attachment; filename*=UTF-8''" + encodedFileName);
            response.setHeader(HttpHeaders.CONTENT_LENGTH, String.valueOf(submittedFile.length()));
            
            // 根据文件扩展名设置Content-Type
            String contentType = MediaType.APPLICATION_OCTET_STREAM_VALUE;
            if (fileExtension.equalsIgnoreCase(".pdf")) {
                contentType = MediaType.APPLICATION_PDF_VALUE;
            } else if (fileExtension.equalsIgnoreCase(".jpg") || fileExtension.equalsIgnoreCase(".jpeg")) {
                contentType = MediaType.IMAGE_JPEG_VALUE;
            } else if (fileExtension.equalsIgnoreCase(".png")) {
                contentType = MediaType.IMAGE_PNG_VALUE;
            }
            
            response.setContentType(contentType);
            
            // 将文件复制到响应输出流
            Files.copy(submittedFile.toPath(), response.getOutputStream());
            response.getOutputStream().flush();
            
            logger.info("文件下载成功 - 提交ID: {}, 学生: {}, 文件名: {}", submissionId, studentName, downloadFileName);
            
        } catch (IOException e) {
            logger.error("文件读取或传输失败", e);
            response.setStatus(HttpServletResponse.SC_INTERNAL_SERVER_ERROR);
        } catch (Exception e) {
            logger.error("下载过程中发生未知错误", e);
            response.setStatus(HttpServletResponse.SC_INTERNAL_SERVER_ERROR);
        }
    }
    
    /**
     * 预览学生提交的作业文件
     * @param submissionId 提交记录ID
     * @param request HTTP请求
     * @param response HTTP响应
     */
    @Operation(summary = "预览学生提交的作业文件", description = "教师预览学生提交的作业文件")
    @GetMapping("/submissions/{submissionId}/preview")
    public void previewSubmissionFile(
            @Parameter(description = "提交记录ID") @PathVariable Long submissionId,
            HttpServletRequest request, 
            HttpServletResponse response) {
        
        // 从Session中获取userId
        HttpSession session = request.getSession(false);
        Long userId = null;
        
        if (session != null) {
            userId = (Long) session.getAttribute("userId");
        }
        
        if (userId == null) {
            logger.error("无法从Session获取用户ID");
            response.setStatus(HttpServletResponse.SC_BAD_REQUEST);
            return;
        }
        
        logger.info("接收到预览学生提交文件请求 - 提交ID: {}, 教师ID: {}", submissionId, userId);
        
        try {
            // 获取提交记录
            AssignmentSubmission submission = assignmentSubmissionMapper.selectById(submissionId);
            if (submission == null) {
                logger.error("找不到提交记录: submissionId={}", submissionId);
                response.setStatus(HttpServletResponse.SC_NOT_FOUND);
                return;
            }
            
            // 获取作业信息
            Assignment assignment = assignmentMapper.selectById(submission.getAssignmentId());
            if (assignment == null) {
                logger.error("找不到作业: assignmentId={}", submission.getAssignmentId());
                response.setStatus(HttpServletResponse.SC_NOT_FOUND);
                return;
            }
            
            // 验证教师权限（确保是该课程的教师）
            if (!assignment.getUserId().equals(userId)) {
                logger.error("权限不足，非作业创建者: assignmentId={}, teacherId={}", assignment.getId(), userId);
                response.setStatus(HttpServletResponse.SC_FORBIDDEN);
                return;
            }
            
            // 查找学生提交的文件
            String uploadDir = "D:/my_git_code/SmartClass/resource/file/assignments/" + submission.getAssignmentId() + "/" + submission.getStudentId() + "/";
            File dir = new File(uploadDir);
            
            if (!dir.exists() || !dir.isDirectory()) {
                logger.error("提交文件目录不存在: {}", uploadDir);
                response.setStatus(HttpServletResponse.SC_NOT_FOUND);
                return;
            }
            
            // 获取目录中的第一个文件（假设每次提交只有一个文件）
            File[] files = dir.listFiles();
            if (files == null || files.length == 0) {
                logger.error("提交目录中没有文件: {}", uploadDir);
                response.setStatus(HttpServletResponse.SC_NOT_FOUND);
                return;
            }
            
            File submittedFile = files[0]; // 获取第一个文件
            
            // 获取文件扩展名
            String fileName = submittedFile.getName();
            String fileExtension = "";
            int dotIndex = fileName.lastIndexOf('.');
            if (dotIndex > 0) {
                fileExtension = fileName.substring(dotIndex + 1).toLowerCase();
            }
            
            // 检查是否支持预览
            boolean supportPreview = fileExtension.equals("pdf") || 
                                   fileExtension.equals("jpg") || 
                                   fileExtension.equals("jpeg") || 
                                   fileExtension.equals("png") || 
                                   fileExtension.equals("gif") ||
                                   fileExtension.equals("mp4") ||
                                   fileExtension.equals("mp3");
            
            if (!supportPreview) {
                logger.warn("不支持预览的文件类型: {}", fileExtension);
                response.setStatus(HttpServletResponse.SC_BAD_REQUEST);
                return;
            }
            
            // 根据文件扩展名设置Content-Type
            String contentType = MediaType.APPLICATION_OCTET_STREAM_VALUE;
            if (fileExtension.equals("pdf")) {
                contentType = MediaType.APPLICATION_PDF_VALUE;
            } else if (fileExtension.equals("jpg") || fileExtension.equals("jpeg")) {
                contentType = MediaType.IMAGE_JPEG_VALUE;
            } else if (fileExtension.equals("png")) {
                contentType = MediaType.IMAGE_PNG_VALUE;
            } else if (fileExtension.equals("gif")) {
                contentType = "image/gif";
            } else if (fileExtension.equals("mp4")) {
                contentType = "video/mp4";
            } else if (fileExtension.equals("mp3")) {
                contentType = "audio/mpeg";
            }
            
            response.setContentType(contentType);
            
            // 将文件复制到响应输出流
            Files.copy(submittedFile.toPath(), response.getOutputStream());
            response.getOutputStream().flush();
            
            logger.info("文件预览成功 - 提交ID: {}, 文件类型: {}", submissionId, contentType);
            
        } catch (IOException e) {
            logger.error("文件读取或传输失败", e);
            response.setStatus(HttpServletResponse.SC_INTERNAL_SERVER_ERROR);
        } catch (Exception e) {
            logger.error("预览过程中发生未知错误", e);
            response.setStatus(HttpServletResponse.SC_INTERNAL_SERVER_ERROR);
        }
    }
} 