package com.education.dto;

import jakarta.validation.constraints.NotNull;
import java.time.LocalDateTime;
import java.util.List;
import java.util.Map;

/**
 * 知识点相关DTO扩展类2
 * 
 * @author Education Platform Team
 * @version 1.0.0
 * @since 2024
 */
public class KnowledgeDTOExtension2 {

    /**
     * 学习路径响应DTO
     */
    public static class LearningPathResponse {
        private Long pathId;
        private String pathName;
        private String description;
        private List<PathNode> pathNodes;
        private Integer totalNodes;
        private Integer estimatedHours;
        private String difficulty;
        private Double completionRate;
        private LocalDateTime createTime;
        
        // Getters and Setters
        public Long getPathId() { return pathId; }
        public void setPathId(Long pathId) { this.pathId = pathId; }
        public String getPathName() { return pathName; }
        public void setPathName(String pathName) { this.pathName = pathName; }
        public String getDescription() { return description; }
        public void setDescription(String description) { this.description = description; }
        public List<PathNode> getPathNodes() { return pathNodes; }
        public void setPathNodes(List<PathNode> pathNodes) { this.pathNodes = pathNodes; }
        public Integer getTotalNodes() { return totalNodes; }
        public void setTotalNodes(Integer totalNodes) { this.totalNodes = totalNodes; }
        public Integer getEstimatedHours() { return estimatedHours; }
        public void setEstimatedHours(Integer estimatedHours) { this.estimatedHours = estimatedHours; }
        public String getDifficulty() { return difficulty; }
        public void setDifficulty(String difficulty) { this.difficulty = difficulty; }
        public Double getCompletionRate() { return completionRate; }
        public void setCompletionRate(Double completionRate) { this.completionRate = completionRate; }
        public LocalDateTime getCreateTime() { return createTime; }
        public void setCreateTime(LocalDateTime createTime) { this.createTime = createTime; }
        
        public static class PathNode {
            private Long nodeId;
            private String nodeName;
            private Integer order;
            private Boolean completed;
            private String status;
            
            // Getters and Setters
            public Long getNodeId() { return nodeId; }
            public void setNodeId(Long nodeId) { this.nodeId = nodeId; }
            public String getNodeName() { return nodeName; }
            public void setNodeName(String nodeName) { this.nodeName = nodeName; }
            public Integer getOrder() { return order; }
            public void setOrder(Integer order) { this.order = order; }
            public Boolean getCompleted() { return completed; }
            public void setCompleted(Boolean completed) { this.completed = completed; }
            public String getStatus() { return status; }
            public void setStatus(String status) { this.status = status; }
        }
    }

    /**
     * 知识掌握度响应DTO
     */
    public static class KnowledgeMasteryResponse {
        private Long studentId;
        private String studentName;
        private Long graphId;
        private String graphName;
        private List<NodeMastery> nodeMasteries;
        private Double overallMastery;
        private Integer masteredCount;
        private Integer totalCount;
        private LocalDateTime analysisTime;
        
        // Getters and Setters
        public Long getStudentId() { return studentId; }
        public void setStudentId(Long studentId) { this.studentId = studentId; }
        public String getStudentName() { return studentName; }
        public void setStudentName(String studentName) { this.studentName = studentName; }
        public Long getGraphId() { return graphId; }
        public void setGraphId(Long graphId) { this.graphId = graphId; }
        public String getGraphName() { return graphName; }
        public void setGraphName(String graphName) { this.graphName = graphName; }
        public List<NodeMastery> getNodeMasteries() { return nodeMasteries; }
        public void setNodeMasteries(List<NodeMastery> nodeMasteries) { this.nodeMasteries = nodeMasteries; }
        public Double getOverallMastery() { return overallMastery; }
        public void setOverallMastery(Double overallMastery) { this.overallMastery = overallMastery; }
        public Integer getMasteredCount() { return masteredCount; }
        public void setMasteredCount(Integer masteredCount) { this.masteredCount = masteredCount; }
        public Integer getTotalCount() { return totalCount; }
        public void setTotalCount(Integer totalCount) { this.totalCount = totalCount; }
        public LocalDateTime getAnalysisTime() { return analysisTime; }
        public void setAnalysisTime(LocalDateTime analysisTime) { this.analysisTime = analysisTime; }
        
        public static class NodeMastery {
            private Long nodeId;
            private String nodeName;
            private Double masteryLevel;
            private String masteryStatus; // NOT_STARTED, LEARNING, MASTERED, NEEDS_REVIEW
            private Integer practiceCount;
            private Double averageScore;
            private LocalDateTime lastPracticeTime;
            
            // Getters and Setters
            public Long getNodeId() { return nodeId; }
            public void setNodeId(Long nodeId) { this.nodeId = nodeId; }
            public String getNodeName() { return nodeName; }
            public void setNodeName(String nodeName) { this.nodeName = nodeName; }
            public Double getMasteryLevel() { return masteryLevel; }
            public void setMasteryLevel(Double masteryLevel) { this.masteryLevel = masteryLevel; }
            public String getMasteryStatus() { return masteryStatus; }
            public void setMasteryStatus(String masteryStatus) { this.masteryStatus = masteryStatus; }
            public Integer getPracticeCount() { return practiceCount; }
            public void setPracticeCount(Integer practiceCount) { this.practiceCount = practiceCount; }
            public Double getAverageScore() { return averageScore; }
            public void setAverageScore(Double averageScore) { this.averageScore = averageScore; }
            public LocalDateTime getLastPracticeTime() { return lastPracticeTime; }
            public void setLastPracticeTime(LocalDateTime lastPracticeTime) { this.lastPracticeTime = lastPracticeTime; }
        }
    }

    /**
     * 知识统计响应DTO
     */
    public static class KnowledgeStatisticsResponse {
        private Long graphId;
        private String graphName;
        private Integer totalNodes;
        private Integer totalRelations;
        private Map<String, Integer> nodeTypeDistribution;
        private Map<String, Integer> difficultyDistribution;
        private Map<String, Integer> relationTypeDistribution;
        private Double averageMasteryLevel;
        private Integer activeStudents;
        private Integer completedPaths;
        private LocalDateTime statisticsTime;
        
        // Getters and Setters
        public Long getGraphId() { return graphId; }
        public void setGraphId(Long graphId) { this.graphId = graphId; }
        public String getGraphName() { return graphName; }
        public void setGraphName(String graphName) { this.graphName = graphName; }
        public Integer getTotalNodes() { return totalNodes; }
        public void setTotalNodes(Integer totalNodes) { this.totalNodes = totalNodes; }
        public Integer getTotalRelations() { return totalRelations; }
        public void setTotalRelations(Integer totalRelations) { this.totalRelations = totalRelations; }
        public Map<String, Integer> getNodeTypeDistribution() { return nodeTypeDistribution; }
        public void setNodeTypeDistribution(Map<String, Integer> nodeTypeDistribution) { this.nodeTypeDistribution = nodeTypeDistribution; }
        public Map<String, Integer> getDifficultyDistribution() { return difficultyDistribution; }
        public void setDifficultyDistribution(Map<String, Integer> difficultyDistribution) { this.difficultyDistribution = difficultyDistribution; }
        public Map<String, Integer> getRelationTypeDistribution() { return relationTypeDistribution; }
        public void setRelationTypeDistribution(Map<String, Integer> relationTypeDistribution) { this.relationTypeDistribution = relationTypeDistribution; }
        public Double getAverageMasteryLevel() { return averageMasteryLevel; }
        public void setAverageMasteryLevel(Double averageMasteryLevel) { this.averageMasteryLevel = averageMasteryLevel; }
        public Integer getActiveStudents() { return activeStudents; }
        public void setActiveStudents(Integer activeStudents) { this.activeStudents = activeStudents; }
        public Integer getCompletedPaths() { return completedPaths; }
        public void setCompletedPaths(Integer completedPaths) { this.completedPaths = completedPaths; }
        public LocalDateTime getStatisticsTime() { return statisticsTime; }
        public void setStatisticsTime(LocalDateTime statisticsTime) { this.statisticsTime = statisticsTime; }
    }

    /**
     * 知识依赖响应DTO
     */
    public static class KnowledgeDependencyResponse {
        private Long nodeId;
        private String nodeName;
        private List<DependencyNode> prerequisites;
        private List<DependencyNode> dependents;
        private Integer dependencyDepth;
        private String dependencyPath;
        
        // Getters and Setters
        public Long getNodeId() { return nodeId; }
        public void setNodeId(Long nodeId) { this.nodeId = nodeId; }
        public String getNodeName() { return nodeName; }
        public void setNodeName(String nodeName) { this.nodeName = nodeName; }
        public List<DependencyNode> getPrerequisites() { return prerequisites; }
        public void setPrerequisites(List<DependencyNode> prerequisites) { this.prerequisites = prerequisites; }
        public List<DependencyNode> getDependents() { return dependents; }
        public void setDependents(List<DependencyNode> dependents) { this.dependents = dependents; }
        public Integer getDependencyDepth() { return dependencyDepth; }
        public void setDependencyDepth(Integer dependencyDepth) { this.dependencyDepth = dependencyDepth; }
        public String getDependencyPath() { return dependencyPath; }
        public void setDependencyPath(String dependencyPath) { this.dependencyPath = dependencyPath; }
        
        public static class DependencyNode {
            private Long nodeId;
            private String nodeName;
            private String relationType;
            private Integer distance;
            private Boolean isMastered;
            
            // Getters and Setters
            public Long getNodeId() { return nodeId; }
            public void setNodeId(Long nodeId) { this.nodeId = nodeId; }
            public String getNodeName() { return nodeName; }
            public void setNodeName(String nodeName) { this.nodeName = nodeName; }
            public String getRelationType() { return relationType; }
            public void setRelationType(String relationType) { this.relationType = relationType; }
            public Integer getDistance() { return distance; }
            public void setDistance(Integer distance) { this.distance = distance; }
            public Boolean getIsMastered() { return isMastered; }
            public void setIsMastered(Boolean isMastered) { this.isMastered = isMastered; }
        }
    }

    /**
     * 知识进度响应DTO
     */
    public static class KnowledgeProgressResponse {
        private Long studentId;
        private String studentName;
        private Long graphId;
        private String graphName;
        private Double overallProgress;
        private List<ProgressNode> nodeProgresses;
        private List<String> completedPaths;
        private List<String> recommendedNextNodes;
        private Integer totalStudyTime;
        private LocalDateTime lastUpdateTime;
        
        // Getters and Setters
        public Long getStudentId() { return studentId; }
        public void setStudentId(Long studentId) { this.studentId = studentId; }
        public String getStudentName() { return studentName; }
        public void setStudentName(String studentName) { this.studentName = studentName; }
        public Long getGraphId() { return graphId; }
        public void setGraphId(Long graphId) { this.graphId = graphId; }
        public String getGraphName() { return graphName; }
        public void setGraphName(String graphName) { this.graphName = graphName; }
        public Double getOverallProgress() { return overallProgress; }
        public void setOverallProgress(Double overallProgress) { this.overallProgress = overallProgress; }
        public List<ProgressNode> getNodeProgresses() { return nodeProgresses; }
        public void setNodeProgresses(List<ProgressNode> nodeProgresses) { this.nodeProgresses = nodeProgresses; }
        public List<String> getCompletedPaths() { return completedPaths; }
        public void setCompletedPaths(List<String> completedPaths) { this.completedPaths = completedPaths; }
        public List<String> getRecommendedNextNodes() { return recommendedNextNodes; }
        public void setRecommendedNextNodes(List<String> recommendedNextNodes) { this.recommendedNextNodes = recommendedNextNodes; }
        public Integer getTotalStudyTime() { return totalStudyTime; }
        public void setTotalStudyTime(Integer totalStudyTime) { this.totalStudyTime = totalStudyTime; }
        public LocalDateTime getLastUpdateTime() { return lastUpdateTime; }
        public void setLastUpdateTime(LocalDateTime lastUpdateTime) { this.lastUpdateTime = lastUpdateTime; }
        
        public static class ProgressNode {
            private Long nodeId;
            private String nodeName;
            private Double progress;
            private String status;
            private Integer studyTime;
            private LocalDateTime startTime;
            private LocalDateTime completionTime;
            
            // Getters and Setters
            public Long getNodeId() { return nodeId; }
            public void setNodeId(Long nodeId) { this.nodeId = nodeId; }
            public String getNodeName() { return nodeName; }
            public void setNodeName(String nodeName) { this.nodeName = nodeName; }
            public Double getProgress() { return progress; }
            public void setProgress(Double progress) { this.progress = progress; }
            public String getStatus() { return status; }
            public void setStatus(String status) { this.status = status; }
            public Integer getStudyTime() { return studyTime; }
            public void setStudyTime(Integer studyTime) { this.studyTime = studyTime; }
            public LocalDateTime getStartTime() { return startTime; }
            public void setStartTime(LocalDateTime startTime) { this.startTime = startTime; }
            public LocalDateTime getCompletionTime() { return completionTime; }
            public void setCompletionTime(LocalDateTime completionTime) { this.completionTime = completionTime; }
        }
    }

    /**
     * 知识图谱验证响应DTO
     */
    public static class KnowledgeGraphValidationResponse {
        private Boolean isValid;
        private List<ValidationError> errors;
        private List<ValidationWarning> warnings;
        private ValidationStatistics statistics;
        private List<String> suggestions;
        private LocalDateTime validationTime;
        
        // Getters and Setters
        public Boolean getIsValid() { return isValid; }
        public void setIsValid(Boolean isValid) { this.isValid = isValid; }
        public List<ValidationError> getErrors() { return errors; }
        public void setErrors(List<ValidationError> errors) { this.errors = errors; }
        public List<ValidationWarning> getWarnings() { return warnings; }
        public void setWarnings(List<ValidationWarning> warnings) { this.warnings = warnings; }
        public ValidationStatistics getStatistics() { return statistics; }
        public void setStatistics(ValidationStatistics statistics) { this.statistics = statistics; }
        public List<String> getSuggestions() { return suggestions; }
        public void setSuggestions(List<String> suggestions) { this.suggestions = suggestions; }
        public LocalDateTime getValidationTime() { return validationTime; }
        public void setValidationTime(LocalDateTime validationTime) { this.validationTime = validationTime; }
        
        public static class ValidationError {
            private String errorType;
            private String message;
            private Long nodeId;
            private String nodeName;
            private String severity;
            
            // Getters and Setters
            public String getErrorType() { return errorType; }
            public void setErrorType(String errorType) { this.errorType = errorType; }
            public String getMessage() { return message; }
            public void setMessage(String message) { this.message = message; }
            public Long getNodeId() { return nodeId; }
            public void setNodeId(Long nodeId) { this.nodeId = nodeId; }
            public String getNodeName() { return nodeName; }
            public void setNodeName(String nodeName) { this.nodeName = nodeName; }
            public String getSeverity() { return severity; }
            public void setSeverity(String severity) { this.severity = severity; }
        }
        
        public static class ValidationWarning {
            private String warningType;
            private String message;
            private Long nodeId;
            private String nodeName;
            
            // Getters and Setters
            public String getWarningType() { return warningType; }
            public void setWarningType(String warningType) { this.warningType = warningType; }
            public String getMessage() { return message; }
            public void setMessage(String message) { this.message = message; }
            public Long getNodeId() { return nodeId; }
            public void setNodeId(Long nodeId) { this.nodeId = nodeId; }
            public String getNodeName() { return nodeName; }
            public void setNodeName(String nodeName) { this.nodeName = nodeName; }
        }
        
        public static class ValidationStatistics {
            private Integer totalNodes;
            private Integer totalRelations;
            private Integer isolatedNodes;
            private Integer circularDependencies;
            private Integer missingPrerequisites;
            
            // Getters and Setters
            public Integer getTotalNodes() { return totalNodes; }
            public void setTotalNodes(Integer totalNodes) { this.totalNodes = totalNodes; }
            public Integer getTotalRelations() { return totalRelations; }
            public void setTotalRelations(Integer totalRelations) { this.totalRelations = totalRelations; }
            public Integer getIsolatedNodes() { return isolatedNodes; }
            public void setIsolatedNodes(Integer isolatedNodes) { this.isolatedNodes = isolatedNodes; }
            public Integer getCircularDependencies() { return circularDependencies; }
            public void setCircularDependencies(Integer circularDependencies) { this.circularDependencies = circularDependencies; }
            public Integer getMissingPrerequisites() { return missingPrerequisites; }
            public void setMissingPrerequisites(Integer missingPrerequisites) { this.missingPrerequisites = missingPrerequisites; }
        }
    }
}