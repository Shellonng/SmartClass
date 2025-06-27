package com.education.service.teacher.impl;

import com.education.service.teacher.DashboardService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;

import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.*;

/**
 * 教师端仪表板服务实现
 * 
 * @author Education Platform Team
 * @version 1.0.0
 * @since 2024
 */
@Service("teacherDashboardServiceImpl")
@RequiredArgsConstructor
@Slf4j
public class DashboardServiceImpl implements DashboardService {

    @Override
    public Map<String, Object> getDashboardData(Long teacherId) {
        log.info("获取教师仪表板数据，教师ID：{}", teacherId);
        
        Map<String, Object> dashboardData = new HashMap<>();
        
        // 基本统计信息
        Map<String, Object> statistics = new HashMap<>();
        statistics.put("totalCourses", 3);
        statistics.put("totalStudents", 156);
        statistics.put("totalClasses", 5);
        statistics.put("pendingGrading", 23);
        statistics.put("completedTasks", 45);
        statistics.put("averageClassGrade", 82.3);
        
        // 课程概览
        List<Map<String, Object>> courseOverview = new ArrayList<>();
        Map<String, Object> course1 = new HashMap<>();
        course1.put("courseId", 1L);
        course1.put("courseName", "数据结构");
        course1.put("studentCount", 52);
        course1.put("completionRate", 78.5);
        course1.put("averageGrade", 84.2);
        course1.put("status", "active");
        courseOverview.add(course1);
        
        Map<String, Object> course2 = new HashMap<>();
        course2.put("courseId", 2L);
        course2.put("courseName", "算法设计");
        course2.put("studentCount", 48);
        course2.put("completionRate", 65.8);
        course2.put("averageGrade", 81.7);
        course2.put("status", "active");
        courseOverview.add(course2);
        
        Map<String, Object> course3 = new HashMap<>();
        course3.put("courseId", 3L);
        course3.put("courseName", "软件工程");
        course3.put("studentCount", 56);
        course3.put("completionRate", 88.2);
        course3.put("averageGrade", 85.9);
        course3.put("status", "active");
        courseOverview.add(course3);
        
        // 待处理任务
        List<Map<String, Object>> pendingTasks = new ArrayList<>();
        Map<String, Object> task1 = new HashMap<>();
        task1.put("id", 1L);
        task1.put("type", "grading");
        task1.put("title", "数据结构期中考试批改");
        task1.put("count", 52);
        task1.put("deadline", "2024-01-16");
        task1.put("priority", "high");
        pendingTasks.add(task1);
        
        Map<String, Object> task2 = new HashMap<>();
        task2.put("id", 2L);
        task2.put("type", "review");
        task2.put("title", "算法作业审核");
        task2.put("count", 15);
        task2.put("deadline", "2024-01-18");
        task2.put("priority", "medium");
        pendingTasks.add(task2);
        
        // 最近活动
        List<Map<String, Object>> recentActivities = new ArrayList<>();
        Map<String, Object> activity1 = new HashMap<>();
        activity1.put("id", 1L);
        activity1.put("type", "grading");
        activity1.put("description", "完成了软件工程作业批改");
        activity1.put("timestamp", LocalDateTime.now().minusHours(2).format(DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm")));
        recentActivities.add(activity1);
        
        Map<String, Object> activity2 = new HashMap<>();
        activity2.put("id", 2L);
        activity2.put("type", "course");
        activity2.put("description", "发布了新的数据结构实验");
        activity2.put("timestamp", LocalDateTime.now().minusHours(5).format(DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm")));
        recentActivities.add(activity2);
        
        dashboardData.put("statistics", statistics);
        dashboardData.put("courseOverview", courseOverview);
        dashboardData.put("pendingTasks", pendingTasks);
        dashboardData.put("recentActivities", recentActivities);
        dashboardData.put("lastUpdated", LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss")));
        
        return dashboardData;
    }

    @Override
    public Map<String, Object> getTeachingStatistics(Long teacherId, String timeRange) {
        log.info("获取教学统计数据，教师ID：{}，时间范围：{}", teacherId, timeRange);
        
        Map<String, Object> statistics = new HashMap<>();
        
        // 根据时间范围返回不同的统计数据
        if ("week".equals(timeRange)) {
            statistics.put("gradedAssignments", 25);
            statistics.put("newStudents", 3);
            statistics.put("classHours", 12);
            statistics.put("averageGrade", 83.5);
            statistics.put("studentEngagement", 78.2);
        } else if ("month".equals(timeRange)) {
            statistics.put("gradedAssignments", 156);
            statistics.put("newStudents", 12);
            statistics.put("classHours", 48);
            statistics.put("averageGrade", 82.3);
            statistics.put("studentEngagement", 75.8);
        } else {
            // 默认返回学期统计
            statistics.put("gradedAssignments", 623);
            statistics.put("newStudents", 45);
            statistics.put("classHours", 192);
            statistics.put("averageGrade", 81.7);
            statistics.put("studentEngagement", 76.5);
        }
        
        // 教学趋势数据
        List<Map<String, Object>> trendData = new ArrayList<>();
        for (int i = 0; i < 7; i++) {
            Map<String, Object> dayData = new HashMap<>();
            dayData.put("date", LocalDateTime.now().minusDays(i).format(DateTimeFormatter.ofPattern("MM-dd")));
            dayData.put("graded", 5 + (int)(Math.random() * 10));
            dayData.put("classHours", 2 + (int)(Math.random() * 4));
            dayData.put("studentInteraction", 15 + (int)(Math.random() * 20));
            trendData.add(dayData);
        }
        
        // 课程表现对比
        List<Map<String, Object>> coursePerformance = new ArrayList<>();
        String[] courses = {"数据结构", "算法设计", "软件工程"};
        for (String course : courses) {
            Map<String, Object> performance = new HashMap<>();
            performance.put("courseName", course);
            performance.put("averageGrade", 80 + Math.random() * 10);
            performance.put("completionRate", 70 + Math.random() * 20);
            performance.put("engagement", 75 + Math.random() * 15);
            coursePerformance.add(performance);
        }
        
        statistics.put("trendData", trendData);
        statistics.put("coursePerformance", coursePerformance);
        statistics.put("timeRange", timeRange);
        
        return statistics;
    }

    @Override
    public Map<String, Object> getPendingTasks(Long teacherId) {
        log.info("获取待处理任务，教师ID：{}", teacherId);
        
        Map<String, Object> pendingTasks = new HashMap<>();
        
        // 待批改作业
        List<Map<String, Object>> gradingTasks = new ArrayList<>();
        Map<String, Object> grading1 = new HashMap<>();
        grading1.put("id", 1L);
        grading1.put("courseName", "数据结构");
        grading1.put("taskName", "期中考试");
        grading1.put("submissionCount", 52);
        grading1.put("gradedCount", 35);
        grading1.put("remainingCount", 17);
        grading1.put("deadline", "2024-01-16");
        grading1.put("priority", "high");
        gradingTasks.add(grading1);
        
        Map<String, Object> grading2 = new HashMap<>();
        grading2.put("id", 2L);
        grading2.put("courseName", "算法设计");
        grading2.put("taskName", "编程作业3");
        grading2.put("submissionCount", 48);
        grading2.put("gradedCount", 30);
        grading2.put("remainingCount", 18);
        grading2.put("deadline", "2024-01-18");
        grading2.put("priority", "medium");
        gradingTasks.add(grading2);
        
        // 待审核内容
        List<Map<String, Object>> reviewTasks = new ArrayList<>();
        Map<String, Object> review1 = new HashMap<>();
        review1.put("id", 1L);
        review1.put("type", "course_material");
        review1.put("title", "新增实验指导书");
        review1.put("submittedBy", "助教张三");
        review1.put("submittedAt", "2024-01-14 10:30");
        review1.put("priority", "medium");
        reviewTasks.add(review1);
        
        // 待回复消息
        List<Map<String, Object>> messageTasks = new ArrayList<>();
        Map<String, Object> message1 = new HashMap<>();
        message1.put("id", 1L);
        message1.put("from", "学生李四");
        message1.put("subject", "关于算法作业的疑问");
        message1.put("receivedAt", "2024-01-15 09:15");
        message1.put("priority", "high");
        messageTasks.add(message1);
        
        Map<String, Object> message2 = new HashMap<>();
        message2.put("id", 2L);
        message2.put("from", "学生王五");
        message2.put("subject", "请假申请");
        message2.put("receivedAt", "2024-01-15 14:20");
        message2.put("priority", "low");
        messageTasks.add(message2);
        
        // 统计信息
        Map<String, Object> taskSummary = new HashMap<>();
        taskSummary.put("totalGradingTasks", gradingTasks.size());
        taskSummary.put("totalReviewTasks", reviewTasks.size());
        taskSummary.put("totalMessages", messageTasks.size());
        taskSummary.put("highPriorityTasks", 2);
        taskSummary.put("urgentDeadlines", 1);
        
        pendingTasks.put("gradingTasks", gradingTasks);
        pendingTasks.put("reviewTasks", reviewTasks);
        pendingTasks.put("messageTasks", messageTasks);
        pendingTasks.put("taskSummary", taskSummary);
        
        return pendingTasks;
    }

    @Override
    public Map<String, Object> getCourseOverview(Long teacherId) {
        log.info("获取课程概览，教师ID：{}", teacherId);
        
        Map<String, Object> courseOverview = new HashMap<>();
        
        // 课程列表
        List<Map<String, Object>> courses = new ArrayList<>();
        
        Map<String, Object> course1 = new HashMap<>();
        course1.put("courseId", 1L);
        course1.put("courseName", "数据结构");
        course1.put("courseCode", "CS201");
        course1.put("semester", "2024春季");
        course1.put("studentCount", 52);
        course1.put("classCount", 2);
        course1.put("totalLessons", 32);
        course1.put("completedLessons", 18);
        course1.put("averageGrade", 84.2);
        course1.put("completionRate", 78.5);
        course1.put("status", "active");
        
        // 最近活动
        List<Map<String, Object>> recentActivities1 = new ArrayList<>();
        Map<String, Object> activity1 = new HashMap<>();
        activity1.put("type", "assignment");
        activity1.put("description", "发布了新的编程作业");
        activity1.put("date", "2024-01-14");
        recentActivities1.add(activity1);
        course1.put("recentActivities", recentActivities1);
        
        courses.add(course1);
        
        Map<String, Object> course2 = new HashMap<>();
        course2.put("courseId", 2L);
        course2.put("courseName", "算法设计");
        course2.put("courseCode", "CS301");
        course2.put("semester", "2024春季");
        course2.put("studentCount", 48);
        course2.put("classCount", 2);
        course2.put("totalLessons", 28);
        course2.put("completedLessons", 15);
        course2.put("averageGrade", 81.7);
        course2.put("completionRate", 65.8);
        course2.put("status", "active");
        
        List<Map<String, Object>> recentActivities2 = new ArrayList<>();
        Map<String, Object> activity2 = new HashMap<>();
        activity2.put("type", "exam");
        activity2.put("description", "期中考试已结束");
        activity2.put("date", "2024-01-12");
        recentActivities2.add(activity2);
        course2.put("recentActivities", recentActivities2);
        
        courses.add(course2);
        
        Map<String, Object> course3 = new HashMap<>();
        course3.put("courseId", 3L);
        course3.put("courseName", "软件工程");
        course3.put("courseCode", "CS401");
        course3.put("semester", "2024春季");
        course3.put("studentCount", 56);
        course3.put("classCount", 2);
        course3.put("totalLessons", 30);
        course3.put("completedLessons", 20);
        course3.put("averageGrade", 85.9);
        course3.put("completionRate", 88.2);
        course3.put("status", "active");
        
        List<Map<String, Object>> recentActivities3 = new ArrayList<>();
        Map<String, Object> activity3 = new HashMap<>();
        activity3.put("type", "project");
        activity3.put("description", "项目中期检查完成");
        activity3.put("date", "2024-01-13");
        recentActivities3.add(activity3);
        course3.put("recentActivities", recentActivities3);
        
        courses.add(course3);
        
        // 整体统计
        Map<String, Object> overallStats = new HashMap<>();
        overallStats.put("totalCourses", courses.size());
        overallStats.put("totalStudents", 156);
        overallStats.put("averageGrade", 83.9);
        overallStats.put("averageCompletionRate", 77.5);
        overallStats.put("activeCourses", 3);
        
        // 课程表现排名
        List<Map<String, Object>> performanceRanking = new ArrayList<>();
        Map<String, Object> rank1 = new HashMap<>();
        rank1.put("rank", 1);
        rank1.put("courseName", "软件工程");
        rank1.put("score", 87.1);
        rank1.put("metric", "综合表现");
        performanceRanking.add(rank1);
        
        Map<String, Object> rank2 = new HashMap<>();
        rank2.put("rank", 2);
        rank2.put("courseName", "数据结构");
        rank2.put("score", 81.4);
        rank2.put("metric", "综合表现");
        performanceRanking.add(rank2);
        
        Map<String, Object> rank3 = new HashMap<>();
        rank3.put("rank", 3);
        rank3.put("courseName", "算法设计");
        rank3.put("score", 73.8);
        rank3.put("metric", "综合表现");
        performanceRanking.add(rank3);
        
        courseOverview.put("courses", courses);
        courseOverview.put("overallStats", overallStats);
        courseOverview.put("performanceRanking", performanceRanking);
        
        return courseOverview;
    }

    @Override
    public Map<String, Object> getStudentPerformance(Long teacherId, Long courseId, Long classId) {
        log.info("获取学生表现分析，教师ID：{}，课程ID：{}，班级ID：{}", teacherId, courseId, classId);
        
        Map<String, Object> performance = new HashMap<>();
        
        // 成绩分布
        Map<String, Object> gradeDistribution = new HashMap<>();
        gradeDistribution.put("excellent", 15); // 90-100
        gradeDistribution.put("good", 28);      // 80-89
        gradeDistribution.put("average", 18);   // 70-79
        gradeDistribution.put("poor", 7);       // 60-69
        gradeDistribution.put("fail", 2);       // <60
        
        // 学习活跃度分析
        Map<String, Object> engagementAnalysis = new HashMap<>();
        engagementAnalysis.put("highEngagement", 22);    // 高活跃度学生数
        engagementAnalysis.put("mediumEngagement", 35);  // 中等活跃度学生数
        engagementAnalysis.put("lowEngagement", 13);     // 低活跃度学生数
        engagementAnalysis.put("averageLoginFrequency", 4.2);
        engagementAnalysis.put("averageStudyTime", 125);  // 分钟
        
        // 作业完成情况
        Map<String, Object> assignmentCompletion = new HashMap<>();
        assignmentCompletion.put("onTimeSubmission", 82.5);    // 按时提交率
        assignmentCompletion.put("lateSubmission", 12.3);      // 延迟提交率
        assignmentCompletion.put("noSubmission", 5.2);         // 未提交率
        assignmentCompletion.put("averageScore", 83.7);        // 平均分
        
        // 学习进度分析
        List<Map<String, Object>> progressAnalysis = new ArrayList<>();
        String[] chapters = {"基础概念", "线性表", "栈和队列", "树和二叉树", "图论", "排序算法"};
        for (int i = 0; i < chapters.length; i++) {
            Map<String, Object> chapterProgress = new HashMap<>();
            chapterProgress.put("chapter", chapters[i]);
            chapterProgress.put("completionRate", 70 + Math.random() * 25);
            chapterProgress.put("averageScore", 75 + Math.random() * 20);
            chapterProgress.put("difficulty", i < 3 ? "easy" : i < 5 ? "medium" : "hard");
            progressAnalysis.add(chapterProgress);
        }
        
        // 需要关注的学生
        List<Map<String, Object>> studentsNeedAttention = new ArrayList<>();
        Map<String, Object> student1 = new HashMap<>();
        student1.put("studentId", 1001L);
        student1.put("studentName", "张三");
        student1.put("reason", "连续3次作业未提交");
        student1.put("currentGrade", 65.5);
        student1.put("lastActivity", "2024-01-10");
        student1.put("riskLevel", "high");
        studentsNeedAttention.add(student1);
        
        Map<String, Object> student2 = new HashMap<>();
        student2.put("studentId", 1002L);
        student2.put("studentName", "李四");
        student2.put("reason", "成绩下降趋势明显");
        student2.put("currentGrade", 72.3);
        student2.put("lastActivity", "2024-01-14");
        student2.put("riskLevel", "medium");
        studentsNeedAttention.add(student2);
        
        // 优秀学生
        List<Map<String, Object>> topPerformers = new ArrayList<>();
        Map<String, Object> top1 = new HashMap<>();
        top1.put("studentId", 2001L);
        top1.put("studentName", "王五");
        top1.put("currentGrade", 95.2);
        top1.put("completionRate", 100.0);
        top1.put("rank", 1);
        topPerformers.add(top1);
        
        Map<String, Object> top2 = new HashMap<>();
        top2.put("studentId", 2002L);
        top2.put("studentName", "赵六");
        top2.put("currentGrade", 92.8);
        top2.put("completionRate", 98.5);
        top2.put("rank", 2);
        topPerformers.add(top2);
        
        performance.put("gradeDistribution", gradeDistribution);
        performance.put("engagementAnalysis", engagementAnalysis);
        performance.put("assignmentCompletion", assignmentCompletion);
        performance.put("progressAnalysis", progressAnalysis);
        performance.put("studentsNeedAttention", studentsNeedAttention);
        performance.put("topPerformers", topPerformers);
        performance.put("courseId", courseId);
        performance.put("classId", classId);
        
        return performance;
    }

    @Override
    public Map<String, Object> getTeachingSuggestions(Long teacherId) {
        log.info("获取教学建议，教师ID：{}", teacherId);
        
        Map<String, Object> suggestions = new HashMap<>();
        
        // 教学改进建议
        List<Map<String, Object>> improvementSuggestions = new ArrayList<>();
        
        Map<String, Object> suggestion1 = new HashMap<>();
        suggestion1.put("id", 1L);
        suggestion1.put("type", "teaching_method");
        suggestion1.put("title", "增加互动环节");
        suggestion1.put("content", "建议在课堂中增加更多互动环节，提高学生参与度");
        suggestion1.put("priority", "high");
        suggestion1.put("category", "教学方法");
        suggestion1.put("expectedImpact", "提高学生参与度15-20%");
        improvementSuggestions.add(suggestion1);
        
        Map<String, Object> suggestion2 = new HashMap<>();
        suggestion2.put("id", 2L);
        suggestion2.put("type", "content_difficulty");
        suggestion2.put("title", "调整难度梯度");
        suggestion2.put("content", "图论部分学生反馈较难，建议增加基础练习");
        suggestion2.put("priority", "medium");
        suggestion2.put("category", "内容调整");
        suggestion2.put("expectedImpact", "提高通过率10-15%");
        improvementSuggestions.add(suggestion2);
        
        Map<String, Object> suggestion3 = new HashMap<>();
        suggestion3.put("id", 3L);
        suggestion3.put("type", "assessment");
        suggestion3.put("title", "优化评估方式");
        suggestion3.put("content", "建议采用多元化评估，增加平时表现权重");
        suggestion3.put("priority", "low");
        suggestion3.put("category", "评估方式");
        suggestion3.put("expectedImpact", "更全面反映学生能力");
        improvementSuggestions.add(suggestion3);
        
        // 学生关怀建议
        List<Map<String, Object>> studentCareSuggestions = new ArrayList<>();
        
        Map<String, Object> care1 = new HashMap<>();
        care1.put("studentName", "张三");
        care1.put("issue", "学习积极性下降");
        care1.put("suggestion", "建议私下沟通，了解具体困难");
        care1.put("urgency", "high");
        studentCareSuggestions.add(care1);
        
        Map<String, Object> care2 = new HashMap<>();
        care2.put("studentName", "李四");
        care2.put("issue", "作业质量不稳定");
        care2.put("suggestion", "建议提供额外辅导资源");
        care2.put("urgency", "medium");
        studentCareSuggestions.add(care2);
        
        // 课程优化建议
        List<Map<String, Object>> courseOptimization = new ArrayList<>();
        
        Map<String, Object> opt1 = new HashMap<>();
        opt1.put("courseName", "数据结构");
        opt1.put("issue", "实验环节时间不够");
        opt1.put("suggestion", "建议增加实验课时或提供课后实验时间");
        opt1.put("priority", "high");
        courseOptimization.add(opt1);
        
        Map<String, Object> opt2 = new HashMap<>();
        opt2.put("courseName", "算法设计");
        opt2.put("issue", "理论与实践结合不够");
        opt2.put("suggestion", "建议增加算法可视化演示");
        opt2.put("priority", "medium");
        courseOptimization.add(opt2);
        
        // AI智能建议
        List<Map<String, Object>> aiSuggestions = new ArrayList<>();
        
        Map<String, Object> ai1 = new HashMap<>();
        ai1.put("type", "personalized_learning");
        ai1.put("title", "个性化学习路径");
        ai1.put("content", "基于学生学习数据，为不同能力学生制定个性化学习计划");
        ai1.put("feasibility", "high");
        aiSuggestions.add(ai1);
        
        Map<String, Object> ai2 = new HashMap<>();
        ai2.put("type", "predictive_analytics");
        ai2.put("title", "学习风险预警");
        ai2.put("content", "利用学习行为数据预测学生学习风险，提前干预");
        ai2.put("feasibility", "medium");
        aiSuggestions.add(ai2);
        
        suggestions.put("improvementSuggestions", improvementSuggestions);
        suggestions.put("studentCareSuggestions", studentCareSuggestions);
        suggestions.put("courseOptimization", courseOptimization);
        suggestions.put("aiSuggestions", aiSuggestions);
        suggestions.put("generatedAt", LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss")));
        
        return suggestions;
    }

    @Override
    public Map<String, Object> getRecentActivities(Long teacherId, Integer limit) {
        log.info("获取最近活动，教师ID：{}，限制数量：{}", teacherId, limit);
        
        Map<String, Object> recentActivities = new HashMap<>();
        
        List<Map<String, Object>> activities = new ArrayList<>();
        
        // 模拟最近活动数据
        String[] activityTypes = {"grading", "course_update", "student_interaction", "material_upload", "announcement"};
        String[] descriptions = {
            "完成了数据结构作业批改",
            "更新了算法设计课程内容",
            "回复了学生的课程疑问",
            "上传了新的实验材料",
            "发布了课程公告"
        };
        
        for (int i = 0; i < Math.min(limit, 15); i++) {
            Map<String, Object> activity = new HashMap<>();
            activity.put("id", (long) (i + 1));
            activity.put("type", activityTypes[i % activityTypes.length]);
            activity.put("description", descriptions[i % descriptions.length]);
            activity.put("timestamp", LocalDateTime.now().minusHours(i + 1).format(DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm")));
            activity.put("courseName", i % 3 == 0 ? "数据结构" : i % 3 == 1 ? "算法设计" : "软件工程");
            activity.put("status", "completed");
            activities.add(activity);
        }
        
        // 活动统计
        Map<String, Object> activityStats = new HashMap<>();
        activityStats.put("totalActivities", activities.size());
        activityStats.put("gradingActivities", (int) activities.stream().filter(a -> "grading".equals(a.get("type"))).count());
        activityStats.put("courseUpdates", (int) activities.stream().filter(a -> "course_update".equals(a.get("type"))).count());
        activityStats.put("studentInteractions", (int) activities.stream().filter(a -> "student_interaction".equals(a.get("type"))).count());
        
        // 活动趋势
        List<Map<String, Object>> activityTrend = new ArrayList<>();
        for (int i = 0; i < 7; i++) {
            Map<String, Object> dayTrend = new HashMap<>();
            dayTrend.put("date", LocalDateTime.now().minusDays(i).format(DateTimeFormatter.ofPattern("MM-dd")));
            dayTrend.put("activityCount", 3 + (int)(Math.random() * 8));
            dayTrend.put("gradingCount", 1 + (int)(Math.random() * 4));
            dayTrend.put("interactionCount", 2 + (int)(Math.random() * 6));
            activityTrend.add(dayTrend);
        }
        
        recentActivities.put("activities", activities);
        recentActivities.put("activityStats", activityStats);
        recentActivities.put("activityTrend", activityTrend);
        recentActivities.put("limit", limit);
        
        return recentActivities;
    }
} 