package com.education.dto;

import lombok.Data;

/**
 * 智能批改请求DTO
 */
@Data
public class GradingRequestDTO {
    
    /**
     * 用户标识
     */
    private String user;
    
    /**
     * 响应模式：streaming（流式）或 blocking（阻塞）
     */
    private String responseMode = "blocking";
    
    /**
     * 输入参数
     */
    private GradingInputs inputs;
    
    /**
     * 批改输入参数
     */
    @Data
    public static class GradingInputs {
        
        /**
         * 参考答案
         */
        private String referenceAnswer;
        
        /**
         * 提交的文件列表
         */
        private java.util.List<FileInput> submittedFiles;
        
        /**
         * 批改标准/要求
         */
        private String gradingCriteria;
        
        /**
         * 满分分值
         */
        private Integer fullScore;
        
        /**
         * 题目类型
         */
        private String questionType;
        
        /**
         * 课程信息
         */
        private String courseInfo;
    }
    
    /**
     * 文件输入参数
     */
    @Data
    public static class FileInput {
        
        /**
         * 文件类型：document, image, audio, video, custom
         */
        private String type;
        
        /**
         * 传递方式：remote_url 或 local_file
         */
        private String transferMethod;
        
        /**
         * 文件URL（当传递方式为 remote_url 时）
         */
        private String url;
        
        /**
         * 上传文件ID（当传递方式为 local_file 时）
         */
        private String uploadFileId;
    }
} 