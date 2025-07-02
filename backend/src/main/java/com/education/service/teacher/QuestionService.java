package com.education.service.teacher;

import com.baomidou.mybatisplus.extension.service.IService;
import com.education.dto.QuestionDTO;
import com.education.dto.common.PageRequest;
import com.education.dto.common.PageResponse;
import com.education.entity.Question;

import java.util.List;

/**
 * 题目服务接口
 */
public interface QuestionService extends IService<Question> {

    /**
     * 创建题目
     *
     * @param request 题目创建请求
     * @return 创建后的题目
     */
    QuestionDTO createQuestion(QuestionDTO.AddRequest request);

    /**
     * 更新题目
     *
     * @param request 题目更新请求
     * @return 更新后的题目
     */
    QuestionDTO updateQuestion(QuestionDTO.UpdateRequest request);

    /**
     * 删除题目
     *
     * @param id 题目ID
     */
    void deleteQuestion(Long id);

    /**
     * 获取题目详情
     *
     * @param id 题目ID
     * @return 题目详情
     */
    QuestionDTO getQuestion(Long id);

    /**
     * 分页查询题目
     *
     * @param request 查询请求
     * @return 分页结果
     */
    PageResponse<QuestionDTO> listQuestions(QuestionDTO.QueryRequest request);

    /**
     * 搜索题目
     *
     * @param keyword  关键词
     * @param type     题目类型
     * @param difficulty 难度
     * @param pageRequest 分页请求
     * @return 分页结果
     */
    PageResponse<QuestionDTO> searchQuestions(String keyword, String type, String difficulty, PageRequest pageRequest);

    /**
     * 获取章节下的所有题目
     *
     * @param chapterId 章节ID
     * @return 题目列表
     */
    List<QuestionDTO> getQuestionsByChapter(Long chapterId);
} 