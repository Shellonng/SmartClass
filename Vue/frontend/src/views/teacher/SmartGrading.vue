<template>
  <div class="smart-grading">
    <div class="page-header">
      <h1>🤖 智能批改</h1>
      <p class="description">基于AI技术，自动批改学生作业，提供详细分析</p>
    </div>

    <div class="grading-container">
      <!-- 作业选择和批改配置 -->
      <div class="config-section">
        <a-card title="📝 批改配置" class="config-card">
          <a-form :model="gradingConfig" :label-col="{ span: 6 }" :wrapper-col="{ span: 18 }">
            <!-- 作业选择 -->
            <a-form-item label="选择作业" name="assignmentId" :rules="[{ required: true, message: '请选择作业' }]">
              <a-select 
                v-model:value="gradingConfig.assignmentId"
                placeholder="请选择要批改的作业"
                :loading="assignmentsLoading"
                @change="handleAssignmentChange"
                show-search
                :filter-option="filterOption"
                style="width: 100%"
              >
                <a-select-option v-for="assignment in assignments" :key="assignment.id" :value="assignment.id">
                  {{ assignment.title }} 
                  <span v-if="assignment.submissionCount > 0">
                    (共{{ assignment.submissionCount }}份提交，
                    <span style="color: red">{{ assignment.ungradedCount }}</span> 份未批改)
                  </span>
                  <span v-else>(无提交)</span>
                </a-select-option>
              </a-select>
            </a-form-item>

            <!-- 批改类型 -->
            <a-form-item label="批改类型" name="gradingType">
              <a-radio-group v-model:value="gradingConfig.gradingType">
                <a-radio value="OBJECTIVE">客观题批改</a-radio>
                <a-radio value="SUBJECTIVE">主观题批改</a-radio>
                <a-radio value="MIXED">混合批改</a-radio>
              </a-radio-group>
            </a-form-item>

            <!-- 批改标准 -->
            <a-form-item label="评分标准">
              <a-textarea 
                v-model:value="gradingConfig.gradingCriteria"
                placeholder="请输入评分标准，如：注重步骤完整性、答案准确性等"
                :rows="3"
              />
            </a-form-item>

            <!-- 操作按钮 -->
            <a-form-item :wrapper-col="{ offset: 6, span: 18 }">
              <a-space>
                <a-button type="primary" @click="handleBatchGrade" :loading="batchGradingLoading">
                  <ThunderboltOutlined />
                  批量批改
                </a-button>
                <a-button @click="handleSingleGrade" :loading="singleGradingLoading">
                  <EditOutlined />
                  逐个批改
                </a-button>
                <a-button @click="handleViewStatistics">
                  <BarChartOutlined />
                  查看统计
                </a-button>
              </a-space>
            </a-form-item>
          </a-form>
        </a-card>
      </div>

      <!-- 学生提交列表 -->
      <div class="submissions-section">
        <a-card title="📋 学生提交" class="submissions-card">
          <template #extra>
            <a-space>
              <a-select 
                v-model:value="submissionFilter" 
                style="width: 120px"
                @change="handleFilterChange"
              >
                <a-select-option value="all">全部</a-select-option>
                <a-select-option value="ungraded">未批改</a-select-option>
                <a-select-option value="graded">已批改</a-select-option>
              </a-select>
              <a-button @click="refreshSubmissions" :loading="loadingSubmissions">
                <ReloadOutlined />
                刷新
              </a-button>
            </a-space>
          </template>

          <a-table 
            :columns="submissionColumns" 
            :data-source="filteredSubmissions"
            :loading="loadingSubmissions"
            row-key="id"
            :pagination="{ pageSize: 10 }"
          >
            <template #bodyCell="{ column, record }">
              <template v-if="column.key === 'studentName'">
                <div class="student-info">
                  <a-avatar size="small">{{ record.studentName[0] }}</a-avatar>
                  <span style="margin-left: 8px;">{{ record.studentName }}</span>
                </div>
              </template>

              <template v-if="column.key === 'status'">
                <a-tag 
                  :color="getStatusColor(record.status)"
                  style="margin: 0;"
                >
                  {{ getStatusText(record.status) }}
                </a-tag>
              </template>

              <template v-if="column.key === 'score'">
                <span v-if="record.score !== null" class="score-display">
                  {{ record.score }}/{{ record.totalScore }}
                </span>
                <span v-else class="no-score">未评分</span>
              </template>

              <template v-if="column.key === 'action'">
                <a-space>
                  <a-button 
                    size="small" 
                    @click="handleViewSubmission(record)"
                  >
                    查看
                  </a-button>
                  <a-button 
                    size="small" 
                    type="primary"
                    @click="handleSingleGrade(record)"
                    :loading="record.grading"
                    :disabled="record.status === 'graded'"
                  >
                    {{ record.status === 'graded' ? '已批改' : '批改' }}
                  </a-button>
                </a-space>
              </template>
            </template>
          </a-table>
        </a-card>
      </div>

      <!-- 批改结果展示 -->
      <div v-if="gradingResults.length > 0" class="results-section">
        <a-card title="📊 批改结果" class="results-card">
          <div class="results-overview">
            <div class="overview-stats">
              <div class="stat-item">
                <div class="stat-value">{{ gradingResults.length }}</div>
                <div class="stat-label">已批改</div>
              </div>
              <div class="stat-item">
                <div class="stat-value">{{ averageScore.toFixed(1) }}</div>
                <div class="stat-label">平均分</div>
              </div>
              <div class="stat-item">
                <div class="stat-value">{{ gradingResults.length > 0 ? Math.max(...gradingResults.filter(r => r && r.earnedScore !== undefined).map(r => r.earnedScore)) : 0 }}</div>
                <div class="stat-label">最高分</div>
              </div>
              <div class="stat-item">
                <div class="stat-value">{{ gradingResults.length > 0 ? Math.min(...gradingResults.filter(r => r && r.earnedScore !== undefined).map(r => r.earnedScore)) : 0 }}</div>
                <div class="stat-label">最低分</div>
              </div>
            </div>
          </div>

          <a-divider />

          <div class="results-list">
            <div v-for="result in gradingResults" :key="result.studentId" class="result-item">
              <div class="result-header">
                <div class="student-info">
                  <a-avatar size="small">{{ getStudentName(result.studentId)[0] }}</a-avatar>
                  <span style="margin-left: 8px;">{{ getStudentName(result.studentId) }}</span>
                </div>
                <div class="score-info">
                  <span class="score">{{ result.earnedScore }}/{{ result.totalScore }}</span>
                  <span class="percentage">({{ result.percentage.toFixed(1) }}%)</span>
                </div>
              </div>

              <div class="result-details">
                <div class="overall-comment" v-if="result.overallComment">
                  <strong>总体评价：</strong>{{ result.overallComment }}
                </div>

                <div class="question-results">
                  <div v-for="questionResult in result.results" :key="questionResult.questionId" class="question-result">
                    <div class="question-info">
                      <span class="question-no">题目{{ questionResult.questionId }}</span>
                      <a-tag :color="questionResult.isCorrect ? 'green' : 'red'">
                        {{ questionResult.isCorrect ? '正确' : '错误' }}
                      </a-tag>
                      <span class="question-score">{{ questionResult.score }}/{{ questionResult.totalScore }}分</span>
                    </div>
                    
                    <div v-if="questionResult.comment" class="question-comment">
                      {{ questionResult.comment }}
                    </div>
                    
                    <div v-if="questionResult.suggestion" class="question-suggestion">
                      <strong>建议：</strong>{{ questionResult.suggestion }}
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </a-card>
      </div>
    </div>

    <!-- 单个作业批改弹窗 -->
    <a-modal 
      v-model:open="gradingModalVisible"
      title="智能批改"
      width="900px"
      :footer="null"
    >
      <div v-if="currentSubmission" class="grading-modal">
        <div class="submission-info">
          <h3>{{ getStudentName(currentSubmission.studentId) }} 的作业</h3>
          <p>提交时间：{{ currentSubmission.submitTime }}</p>
        </div>

        <div v-if="gradingInProgress" class="grading-progress">
          <a-spin size="large">
            <div class="progress-text">
              <p>🤖 AI正在分析答案...</p>
              <p>📝 正在评估答题质量...</p>
              <p>⚡ 即将完成批改...</p>
            </div>
          </a-spin>
        </div>

        <div v-else-if="currentGradingResult" class="grading-result">
          <div class="result-summary">
            <h4>批改结果</h4>
            <div class="summary-stats">
              <span>总分：{{ currentGradingResult.totalScore }}</span>
              <span>得分：{{ currentGradingResult.earnedScore }}</span>
              <span>得分率：{{ currentGradingResult.percentage.toFixed(1) }}%</span>
            </div>
          </div>

          <div class="detailed-results">
            <div v-for="result in currentGradingResult.results" :key="result.questionId" class="question-detail">
              <div class="question-header">
                <span>题目 {{ result.questionId }}</span>
                <a-tag :color="result.isCorrect ? 'green' : 'red'">
                  {{ result.isCorrect ? '正确' : '错误' }}
                </a-tag>
                <span>{{ result.score }}/{{ result.totalScore }}分</span>
              </div>
              
              <div class="question-content">
                <div class="student-answer">
                  <strong>学生答案：</strong>{{ getStudentAnswer(result.questionId) }}
                </div>
                
                <div v-if="result.comment" class="ai-comment">
                  <strong>AI评价：</strong>{{ result.comment }}
                </div>
                
                <div v-if="result.suggestion" class="ai-suggestion">
                  <strong>改进建议：</strong>{{ result.suggestion }}
                </div>
              </div>
            </div>
          </div>

          <div class="modal-actions">
            <a-space>
              <a-button @click="gradingModalVisible = false">关闭</a-button>
              <a-button type="primary" @click="handleSaveGrading">保存批改结果</a-button>
            </a-space>
          </div>
        </div>
      </div>
    </a-modal>

    <!-- 统计弹窗 -->
    <a-modal 
      v-model:open="statisticsModalVisible"
      title="批改统计"
      width="800px"
      :footer="null"
    >
      <div class="statistics-content">
        <!-- 统计图表区域 -->
        <div class="statistics-charts">
          <div class="chart-item">
            <h4>成绩分布</h4>
            <div class="score-distribution">
              <div v-for="(count, range) in scoreDistribution" :key="range" class="distribution-item">
                <span class="range">{{ range }}分</span>
                <div class="bar">
                  <div class="bar-fill" :style="{ width: (count / maxCount * 100) + '%' }"></div>
                </div>
                <span class="count">{{ count }}人</span>
              </div>
            </div>
          </div>
        </div>

        <div class="statistics-summary">
          <h4>详细统计</h4>
          <div class="summary-grid">
            <div class="summary-item">
              <div class="item-label">总提交数</div>
              <div class="item-value">{{ statistics.totalSubmissions }}</div>
            </div>
            <div class="summary-item">
              <div class="item-label">已批改数</div>
              <div class="item-value">{{ statistics.gradedSubmissions }}</div>
            </div>
            <div class="summary-item">
              <div class="item-label">平均分</div>
              <div class="item-value">{{ statistics.averageScore?.toFixed(1) }}</div>
            </div>
            <div class="summary-item">
              <div class="item-label">最高分</div>
              <div class="item-value">{{ statistics.highestScore }}</div>
            </div>
            <div class="summary-item">
              <div class="item-label">最低分</div>
              <div class="item-value">{{ statistics.lowestScore }}</div>
            </div>
          </div>
        </div>
      </div>
    </a-modal>

    <!-- 文件预览弹窗 -->
    <a-modal
      v-model:open="documentPreviewVisible"
      title="文件预览"
      width="900px"
      :footer="null"
    >
      <div v-if="currentSubmission" class="document-preview-content">
        <div class="document-info">
          <div v-if="currentSubmission.attachments.length > 1" class="attachment-selector">
            <a-radio-group v-model:value="selectedAttachmentIndex" button-style="solid">
              <a-radio-button 
                v-for="(attachment, index) in currentSubmission.attachments" 
                :key="attachment.id" 
                :value="index"
              >
                {{ attachment.fileName }}
              </a-radio-button>
            </a-radio-group>
          </div>
          
          <h3>文件：{{ currentSubmission.attachments[selectedAttachmentIndex].fileName }}</h3>
          <p>上传时间：{{ currentSubmission.attachments[selectedAttachmentIndex].uploadTime }}</p>
          <p>文件大小：{{ formatBytes(currentSubmission.attachments[selectedAttachmentIndex].fileSize) }}</p>
        </div>
        
        <a-tabs v-model:activeKey="activeTabKey">
          <a-tab-pane key="preview" tab="文件预览">
            <div class="document-preview-frame">
              <!-- Word文档预览 -->
              <div class="word-preview">
                <div class="word-document">
                  <div class="word-page">
                    <div class="word-content">
                      <h1>{{ mockDocumentTitle }}</h1>
                      <div v-for="(paragraph, index) in mockDocumentContent" :key="index" class="word-paragraph">
                        {{ paragraph }}
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </a-tab-pane>
          <a-tab-pane key="answer" tab="作业答案">
            <div class="answer-preview">
              <div v-for="(answer, index) in mockAnswers" :key="index" class="answer-item">
                <div class="answer-question">{{ index + 1 }}. {{ answer.questionText }}</div>
                <div class="answer-content">{{ answer.studentAnswer }}</div>
              </div>
            </div>
          </a-tab-pane>
        </a-tabs>

        <div class="document-actions">
          <a-button type="primary" @click="downloadDocument(getCurrentAttachment())">下载文件</a-button>
          <a-button @click="documentPreviewVisible = false">关闭</a-button>
        </div>
      </div>
    </a-modal>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted } from 'vue'
import { message } from 'ant-design-vue'
import { ThunderboltOutlined, EditOutlined, BarChartOutlined, ReloadOutlined } from '@ant-design/icons-vue'
import teacherGradingApi from '@/api/teacherGrading'
import type { AutoGradingRequest } from '@/api/teacherGrading'
import assignmentApi from '@/api/assignment'

// 响应式数据
const batchGradingLoading = ref(false)
const singleGradingLoading = ref(false)
const loadingSubmissions = ref(false)
const gradingModalVisible = ref(false)
const statisticsModalVisible = ref(false)
const gradingInProgress = ref(false)
const assignmentsLoading = ref(false)
const documentPreviewVisible = ref(false)
const activeTabKey = ref('preview')
const selectedAttachmentIndex = ref(0)
const mockDocumentTitle = ref('作业标题：数据结构与算法分析')
const mockDocumentContent = ref<string[]>([])
const mockAnswers = ref<MockAnswer[]>([])

// 批改配置
const gradingConfig = ref({
  assignmentId: null as number | null,
  gradingType: 'MIXED',
  gradingCriteria: '评分标准：\n1. 答案准确性 (60%)\n2. 解题步骤 (30%)\n3. 表述清晰度 (10%)'
})

const submissionFilter = ref('all')
const gradingResults = ref<any[]>([]) // 假设返回的批改结果结构
const currentSubmission = ref<Submission | null>(null)
const currentGradingResult = ref<any>(null)

// 作业和提交数据
const assignments = ref<any[]>([])
const submissions = ref<Submission[]>([])

// 统计数据
const statistics = ref<Statistics>({
  assignmentId: 1,
  totalSubmissions: 30,
  gradedSubmissions: 28,
  averageScore: 78.5,
  highestScore: 95.0,
  lowestScore: 45.0,
  scoreDistribution: {
    '90-100': 5,
    '80-89': 10,
    '70-79': 8,
    '60-69': 3,
    '0-59': 2
  }
})

// 过滤选项方法（用于搜索过滤）
const filterOption = (input: string, option: any) => {
  return option.children[0].toLowerCase().indexOf(input.toLowerCase()) >= 0
}

/**
 * 获取教师关联的作业及提交情况（使用虚拟数据）
 */
const loadTeacherAssignments = async () => {
  try {
    assignmentsLoading.value = true
    // 使用虚拟数据代替API调用
    assignments.value = generateMockAssignments()
    console.log('加载虚拟作业数据:', assignments.value)
    
    // 如果有作业，自动选择第一个
    if (assignments.value.length > 0) {
      gradingConfig.value.assignmentId = assignments.value[0].id
      handleAssignmentChange(assignments.value[0].id)
    }
  } catch (error) {
    console.error('加载作业数据失败:', error)
    message.error('加载作业数据失败')
  } finally {
    assignmentsLoading.value = false
  }
}

// 答案类型定义
interface MockAnswer {
  questionId: number;
  questionText: string;
  questionType: string;
  correctAnswer: string;
  studentAnswer: string;
  totalScore: number;
}

// 批改结果类型定义
interface GradingResult {
  questionId: number;
  questionType: string;
  isCorrect: boolean;
  score: number;
  maxScore: number;
  comment: string;
}

// 接口定义
interface Statistics {
  assignmentId: number;
  totalSubmissions: number;
  gradedSubmissions: number;
  averageScore: number;
  highestScore: number;
  lowestScore: number;
  scoreDistribution: {
    '90-100': number;
    '80-89': number;
    '70-79': number;
    '60-69': number;
    '0-59': number;
  };
  commonErrors?: string[];
  knowledgePointMastery?: Record<string, number>;
}

interface Attachment {
  id: number;
  fileName: string;
  fileType: string;
  fileSize: number;
  uploadTime: string;
  fileContent?: ArrayBuffer;
  fileTypeName?: string;
}

interface Submission {
  id: number;
  studentId: number;
  studentName: string;
  submitTime: string;
  status: string;
  score: number | null;
  totalScore: number;
  grading: boolean;
  attachments: Attachment[];
}

// 表格列定义
const submissionColumns = [
  {
    title: '学生',
    dataIndex: 'studentName',
    key: 'studentName',
    width: 150
  },
  {
    title: '提交时间',
    dataIndex: 'submitTime',
    key: 'submitTime',
    width: 180
  },
  {
    title: '状态',
    key: 'status',
    width: 100
  },
  {
    title: '得分',
    key: 'score',
    width: 120
  },
  {
    title: '操作',
    key: 'action',
    width: 150
  }
]

// 计算属性
const filteredSubmissions = computed(() => {
  if (gradingConfig.value.assignmentId === null) {
    return []
  }
  if (submissionFilter.value === 'all') {
    return submissions.value
  }
  return submissions.value.filter(s => s.status === submissionFilter.value)
})

const averageScore = computed(() => {
  if (gradingResults.value.length === 0) return 0
  const validResults = gradingResults.value.filter(r => r && r.earnedScore !== undefined)
  if (validResults.length === 0) return 0
  const sum = validResults.reduce((acc, result) => acc + result.earnedScore, 0)
  return sum / validResults.length
})

const scoreDistribution = computed(() => {
  return statistics.value.scoreDistribution
})

const maxCount = computed(() => {
  if (!statistics.value?.scoreDistribution) return 1
  const values = Object.values(statistics.value.scoreDistribution)
  return values.length > 0 ? Math.max(...values) : 1
})

// 方法
const handleAssignmentChange = async (assignmentId: number) => {
  gradingConfig.value.assignmentId = assignmentId
  // 加载对应作业的提交记录
  await loadSubmissions()
  // 加载统计数据
  await loadStatistics()
}

const loadSubmissions = async () => {
  if (gradingConfig.value.assignmentId === null) return
  loadingSubmissions.value = true
  try {
    console.log('加载作业提交记录，作业ID:', gradingConfig.value.assignmentId)
    // 使用虚拟数据代替API调用
    submissions.value = generateMockSubmissions(gradingConfig.value.assignmentId)
    console.log('加载虚拟提交记录:', submissions.value)
  } catch (error) {
    console.error('加载提交记录失败:', error)
    message.error('加载提交记录失败')
  } finally {
    loadingSubmissions.value = false
  }
}

const loadStatistics = async () => {
  if (gradingConfig.value.assignmentId === null) return
  try {
    // 使用虚拟数据代替API调用
    const mockStatistics: Statistics = {
      assignmentId: gradingConfig.value.assignmentId,
      totalSubmissions: 30,
      gradedSubmissions: 18,
      averageScore: 78.5,
      highestScore: 95.0,
      lowestScore: 45.0,
      scoreDistribution: {
        '90-100': 5,
        '80-89': 10,
        '70-79': 8,
        '60-69': 3,
        '0-59': 2
      },
      commonErrors: [
        "概念理解不清晰",
        "计算步骤有误",
        "未正确应用公式"
      ],
      knowledgePointMastery: {
        "数据结构": 85.5,
        "算法分析": 78.0,
        "递归": 65.0
      }
    }
    
    statistics.value = mockStatistics
  } catch (error) {
    console.error('加载统计数据失败:', error)
    message.error('加载统计数据失败')
  }
}

const handleBatchGrade = async () => {
  if (gradingConfig.value.assignmentId === null) {
    message.warning('请先选择作业')
    return
  }

  try {
    batchGradingLoading.value = true
    
    // 使用虚拟提交数据
    const ungradedSubmissions = submissions.value.filter(s => s.status === 'ungraded')
    
    if (ungradedSubmissions.length === 0) {
      message.warning('没有需要批改的作业')
      batchGradingLoading.value = false
      return
    }
    
    const batchRequest = {
      assignmentId: gradingConfig.value.assignmentId,
      submissions: ungradedSubmissions.map(submission => {
        // 为每个提交生成虚拟题目和答案数据
        const mockAnswers = generateMockAnswers(submission.studentId)
        return {
          submissionId: submission.id,
          assignmentId: gradingConfig.value.assignmentId,
          studentId: submission.studentId,
          studentAnswers: mockAnswers.map(answer => ({
            questionId: answer.questionId,
            studentAnswer: answer.studentAnswer
          })),
          questions: mockAnswers.map(answer => ({
            questionId: answer.questionId,
            questionText: answer.questionText,
            questionType: answer.questionType,
            correctAnswer: answer.correctAnswer,
            totalScore: answer.totalScore
          })),
          maxScore: submission.totalScore || 100
        }
      }),
      gradingCriteria: gradingConfig.value.gradingCriteria
    }

    console.log('批量批改请求:', batchRequest)
    
    // 模拟API调用延迟
    await new Promise(resolve => setTimeout(resolve, 2000))
    
    // 生成虚拟的批改结果
    gradingResults.value = ungradedSubmissions.map(submission => {
      const score = 60 + Math.floor(Math.random() * 40)
      const mockAnswers = generateMockAnswers(submission.studentId)
      return {
        submissionId: submission.id,
        studentId: submission.studentId,
        status: 'completed',
        totalScore: 100,
        earnedScore: score,
        percentage: score,
        overallComment: `整体表现${score >= 90 ? '优秀' : (score >= 80 ? '良好' : (score >= 70 ? '中等' : (score >= 60 ? '及格' : '不及格')))}，请继续努力！`,
        results: mockAnswers.map(answer => {
          const isCorrect = Math.random() > 0.3
          return {
            questionId: answer.questionId,
            questionType: answer.questionType,
            isCorrect: isCorrect,
            score: isCorrect ? answer.totalScore : Math.floor(answer.totalScore * 0.6),
            maxScore: answer.totalScore,
            comment: isCorrect ? '答案正确' : '答案有误，请参考正确答案'
          }
        })
      }
    })
    
    // 更新提交状态
    ungradedSubmissions.forEach((submission, index) => {
      submission.status = 'graded'
      submission.score = gradingResults.value[index]?.earnedScore || 0
    })

    message.success(`批量批改完成，共批改 ${gradingResults.value.length} 份作业`)
  } catch (error) {
    console.error('批量批改失败:', error)
    message.error('批量批改失败')
  } finally {
    batchGradingLoading.value = false
  }
}

const handleSingleGrade = async (submission: Submission) => {
  currentSubmission.value = submission
  gradingModalVisible.value = true
  gradingInProgress.value = true
  currentGradingResult.value = null

  try {
    submission.grading = true

    // 生成虚拟题目和答案
    const mockAnswers = generateMockAnswers(submission.studentId)
    const gradingRequest = {
      submissionId: submission.id,
      assignmentId: gradingConfig.value.assignmentId!,
      studentId: submission.studentId,
      studentAnswers: mockAnswers.map(answer => ({
        questionId: answer.questionId,
        studentAnswer: answer.studentAnswer
      })),
      questions: mockAnswers.map(answer => ({
        questionId: answer.questionId,
        questionText: answer.questionText,
        questionType: answer.questionType,
        correctAnswer: answer.correctAnswer,
        totalScore: answer.totalScore
      })),
      gradingCriteria: gradingConfig.value.gradingCriteria,
      maxScore: submission.totalScore || 100
    }

    console.log('单个批改请求:', gradingRequest)
    
    // 模拟API调用延迟
    await new Promise(resolve => setTimeout(resolve, 3000))
    
    // 生成虚拟的批改结果
    const score = 60 + Math.floor(Math.random() * 40)
    currentGradingResult.value = {
      submissionId: submission.id,
      studentId: submission.studentId,
      status: 'completed',
      totalScore: 100,
      earnedScore: score,
      percentage: score,
      overallComment: `整体表现${score >= 90 ? '优秀' : (score >= 80 ? '良好' : (score >= 70 ? '中等' : (score >= 60 ? '及格' : '不及格')))}，请继续努力！`,
      results: mockAnswers.map(answer => {
        const isCorrect = Math.random() > 0.3
        return {
          questionId: answer.questionId,
          questionType: answer.questionType,
          isCorrect: isCorrect,
          score: isCorrect ? answer.totalScore : Math.floor(answer.totalScore * 0.6),
          maxScore: answer.totalScore,
          comment: isCorrect ? '答案正确' : '答案有误，请参考正确答案'
        }
      })
    }
    
    // 更新提交状态
    submission.status = 'graded'
    submission.score = currentGradingResult.value.earnedScore

    message.success('批改完成')
  } catch (error) {
    console.error('批改失败:', error)
    message.error('批改失败')
  } finally {
    submission.grading = false
    gradingInProgress.value = false
  }
}

const handleViewStatistics = async () => {
  if (gradingConfig.value.assignmentId === null) {
    message.warning('请先选择作业')
    return
  }

  await loadStatistics()
  statisticsModalVisible.value = true
}

const handleSaveGrading = () => {
  if (currentGradingResult.value && currentSubmission.value) {
    // 更新提交记录
    currentSubmission.value.status = 'graded'
    currentSubmission.value.score = currentGradingResult.value.earnedScore
    
    // 添加到批改结果列表
    gradingResults.value.push({
      ...currentGradingResult.value,
      studentId: currentSubmission.value.studentId
    })

    message.success('批改结果已保存')
    gradingModalVisible.value = false
  }
}

const refreshSubmissions = async () => {
  if (gradingConfig.value.assignmentId === null) {
    message.warning('请先选择作业')
    return
  }
  loadingSubmissions.value = true
  await loadSubmissions()
  loadingSubmissions.value = false
  message.success('数据已刷新')
}

const handleFilterChange = (value: string) => {
  console.log('筛选条件变更:', value)
}

const getStatusColor = (status: string) => {
  const colorMap: Record<string, string> = {
    'ungraded': 'orange',
    'graded': 'green',
    'grading': 'blue'
  }
  return colorMap[status] || 'default'
}

const getStatusText = (status: string) => {
  const textMap: Record<string, string> = {
    'ungraded': '未批改',
    'graded': '已批改',
    'grading': '批改中'
  }
  return textMap[status] || status
}

const getStudentName = (studentId: number | undefined) => {
  if (!studentId) return '未知学生'
  const submission = submissions.value.find(s => s.studentId === studentId)
  return submission?.studentName || `学生${studentId}`
}

const getStudentAnswer = (questionId: number) => {
  // 模拟获取学生答案
  return `学生对题目${questionId}的答案内容...`
}

// 生成虚拟作业数据
const generateMockAssignments = () => {
  return [
    { 
      id: 1, 
      title: '计算机组成原理期中考试', 
      submissionCount: 30,
      ungradedCount: 15,
      gradedCount: 15,
      status: 1,
      courseId: 9,
      type: 'exam',
      mode: 'question'
    },
    { 
      id: 2, 
      title: '操作系统原理作业1', 
      submissionCount: 25,
      ungradedCount: 10,
      gradedCount: 15,
      status: 1,
      courseId: 9,
      type: 'homework',
      mode: 'question' 
    },
    { 
      id: 3, 
      title: '数据库系统概论实验报告', 
      submissionCount: 28,
      ungradedCount: 28,
      gradedCount: 0,
      status: 1,
      courseId: 9,
      type: 'homework',
      mode: 'question'
    }
  ]
}

// 生成虚拟学生提交数据
const generateMockSubmissions = (assignmentId: number): Submission[] => {
  const studentCount = assignmentId === 1 ? 30 : (assignmentId === 2 ? 25 : 28)
  const gradedCount = assignmentId === 3 ? 0 : (assignmentId === 1 ? 15 : 15)
  
  const submissions: Submission[] = []
  for (let i = 1; i <= studentCount; i++) {
    const isGraded = i <= gradedCount
    
    // 所有文件类型都使用Word文档
    const fileType = {
      type: 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
      ext: 'docx',
      name: 'Word文档'
    };
    
    const attachments = [{
      id: 2000 + i,
      fileName: `学生${i}的作业.${fileType.ext}`,
      fileType: fileType.type,
      fileSize: Math.floor(50000 + Math.random() * 500000),
      uploadTime: new Date(Date.now() - Math.random() * 86400000 * 3).toISOString().replace('T', ' ').substring(0, 19),
      fileTypeName: fileType.name
    }]
    
    // 随机添加第二个附件，也是Word文档
    if (Math.random() > 0.7) {
      attachments.push({
        id: 3000 + i,
        fileName: `学生${i}的补充材料.${fileType.ext}`,
        fileType: fileType.type,
        fileSize: Math.floor(20000 + Math.random() * 200000),
        uploadTime: new Date(Date.now() - Math.random() * 43200000).toISOString().replace('T', ' ').substring(0, 19),
        fileTypeName: fileType.name
      })
    }
    
    submissions.push({
      id: 1000 + i,
      studentId: 100 + i,
      studentName: `学生${i}`,
      submitTime: new Date(Date.now() - Math.random() * 86400000 * 3).toISOString().replace('T', ' ').substring(0, 19),
      status: isGraded ? 'graded' : 'ungraded',
      score: isGraded ? Math.floor(60 + Math.random() * 40) : null,
      totalScore: 100,
      grading: false,
      attachments: attachments
    })
  }
  return submissions
}

// 生成虚拟题目和答案数据
const generateMockAnswers = (studentId: number): MockAnswer[] => {
  const questionTypes = ['single', 'multiple', 'true_false', 'blank', 'short']
  const answers: MockAnswer[] = []
  
  for (let i = 1; i <= 5; i++) {
    const questionType = questionTypes[i - 1]
    let correctAnswer: string, studentAnswer: string
    
    switch (questionType) {
      case 'single':
        correctAnswer = 'A'
        studentAnswer = Math.random() > 0.7 ? 'A' : ['B', 'C', 'D'][Math.floor(Math.random() * 3)]
        break
      case 'multiple':
        correctAnswer = 'A,B,D'
        studentAnswer = Math.random() > 0.7 ? 'A,B,D' : ['A,B', 'B,D', 'A,C,D'][Math.floor(Math.random() * 3)]
        break
      case 'true_false':
        correctAnswer = 'true'
        studentAnswer = Math.random() > 0.7 ? 'true' : 'false'
        break
      case 'blank':
        correctAnswer = '递归算法'
        studentAnswer = Math.random() > 0.7 ? '递归算法' : ['遍历算法', '迭代算法', '分治算法'][Math.floor(Math.random() * 3)]
        break
      case 'short':
        correctAnswer = '算法复杂度分析是评估算法效率的重要手段，通常使用大O表示法来表示时间复杂度和空间复杂度。'
        studentAnswer = Math.random() > 0.7 
          ? '算法复杂度分析是评估算法效率的重要手段，通常使用大O表示法来表示时间复杂度和空间复杂度。' 
          : '算法复杂度是用来衡量算法效率的，包括时间复杂度和空间复杂度两种。'
        break
      default:
        correctAnswer = '暂无答案'
        studentAnswer = '暂无答案'
    }
    
    answers.push({
      questionId: i,
      questionText: `这是第${i}道题目，题型为${questionType}`,
      questionType: questionType,
      correctAnswer: correctAnswer,
      studentAnswer: studentAnswer,
      totalScore: 20
    })
  }
  
  return answers
}

// 生成虚拟文档内容
const generateMockDocumentContent = () => {
  const paragraphs = [
    '摘要：本文分析了数据结构与算法在计算机科学中的重要性，探讨了常见数据结构的应用场景及其复杂度分析。',
    '关键词：数据结构、算法、时间复杂度、空间复杂度',
    '',
    '1. 引言',
    '数据结构是计算机科学中存储和组织数据的方式，它直接影响算法的设计和效率。选择合适的数据结构对于解决特定问题至关重要。',
    '',
    '2. 常见数据结构',
    '2.1 数组',
    '数组是最基本的数据结构，它在内存中连续存储元素。数组的优点是可以在O(1)时间内通过索引访问元素，但插入和删除操作的时间复杂度为O(n)。',
    '',
    '2.2 链表',
    '链表由节点组成，每个节点包含数据和指向下一个节点的指针。链表的插入和删除操作的时间复杂度为O(1)，但查找元素的时间复杂度为O(n)。',
    '',
    '2.3 栈和队列',
    '栈是一种后进先出(LIFO)的数据结构，而队列是一种先进先出(FIFO)的数据结构。这两种数据结构在算法设计和系统实现中有广泛应用。',
    '',
    '3. 算法复杂度分析',
    '算法复杂度分析是评估算法效率的重要手段，通常使用大O表示法来表示时间复杂度和空间复杂度。时间复杂度反映算法执行时间与输入规模的关系，空间复杂度反映算法所需额外空间与输入规模的关系。',
    '',
    '4. 结论',
    '选择合适的数据结构和算法对于解决问题的效率至关重要。在实际应用中，需要根据问题的特点和需求，权衡不同数据结构和算法的优缺点，做出最佳选择。',
    '',
    '参考文献：',
    '1. Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2009). Introduction to Algorithms (3rd ed.). MIT Press.',
    '2. Sedgewick, R., & Wayne, K. (2011). Algorithms (4th ed.). Addison-Wesley Professional.'
  ]
  mockDocumentContent.value = paragraphs
}

// 获取当前选中的附件
const getCurrentAttachment = (): Attachment => {
  if (!currentSubmission.value || !currentSubmission.value.attachments) {
    return {
      id: 0,
      fileName: '无文件',
      fileType: '',
      fileSize: 0,
      uploadTime: ''
    }
  }
  return currentSubmission.value.attachments[selectedAttachmentIndex.value]
}

// 处理查看学生提交详情
const handleViewSubmission = (submission: Submission) => {
  currentSubmission.value = submission
  // 重置选中的附件索引
  selectedAttachmentIndex.value = 0
  // 生成此学生的答案数据
  mockAnswers.value = generateMockAnswers(submission.studentId)
  documentPreviewVisible.value = true
}

// 下载文件
const downloadDocument = (file: Attachment) => {
  try {
    // 生成Word文档内容
    const textEncoder = new TextEncoder();
    const mockContent = textEncoder.encode(mockDocumentContent.value.join('\n'));
    const fileData = new Blob([mockContent], { type: file.fileType });
    
    const url = URL.createObjectURL(fileData);
    const link = document.createElement('a');
    link.href = url;
    link.download = file.fileName;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
    
    message.success('文件下载成功');
  } catch (error) {
    console.error('下载文件失败:', error);
    message.error('下载文件失败');
  }
};

// 格式化文件大小
const formatBytes = (bytes: number, decimals = 2) => {
  if (bytes === 0) return '0 Bytes'
  const k = 1024
  const dm = decimals < 0 ? 0 : decimals
  const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB', 'PB', 'EB', 'ZB', 'YB']
  const i = Math.floor(Math.log(bytes) / Math.log(k))
  return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + ' ' + sizes[i]
}

// 页面加载时执行
onMounted(() => {
  console.log('组件已挂载，开始加载数据...')
  // 生成模拟文档内容
  generateMockDocumentContent()
  // 加载教师关联的作业
  loadTeacherAssignments()
})
</script>

<style scoped>
.smart-grading {
  padding: 24px;
  background: #f5f5f5;
  min-height: 100vh;
}

.page-header {
  text-align: center;
  margin-bottom: 32px;
}

.page-header h1 {
  font-size: 28px;
  margin-bottom: 8px;
  color: #1890ff;
}

.description {
  color: #666;
  font-size: 16px;
}

.grading-container {
  max-width: 1400px;
  margin: 0 auto;
  display: flex;
  flex-direction: column;
  gap: 24px;
}

.config-section {
  display: flex;
  justify-content: center;
}

.config-card {
  width: 100%;
  max-width: 800px;
  border-radius: 12px;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
}

.submissions-card, .results-card {
  border-radius: 12px;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
}

.student-info {
  display: flex;
  align-items: center;
}

.score-display {
  font-weight: 600;
  color: #1890ff;
}

.no-score {
  color: #999;
}

.results-overview {
  margin-bottom: 24px;
}

.overview-stats {
  display: flex;
  justify-content: space-around;
  gap: 24px;
}

.stat-item {
  text-align: center;
  padding: 16px;
  background: #f8f9fa;
  border-radius: 8px;
  flex: 1;
}

.stat-value {
  font-size: 24px;
  font-weight: 600;
  color: #1890ff;
  margin-bottom: 4px;
}

.stat-label {
  color: #666;
  font-size: 14px;
}

.results-list {
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.result-item {
  background: white;
  border: 1px solid #e8e8e8;
  border-radius: 8px;
  padding: 16px;
}

.result-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 12px;
}

.score-info {
  font-weight: 600;
}

.score {
  color: #1890ff;
  font-size: 18px;
}

.percentage {
  color: #666;
  margin-left: 8px;
}

.result-details {
  margin-top: 12px;
}

.overall-comment {
  margin-bottom: 16px;
  padding: 12px;
  background: #f6ffed;
  border-left: 3px solid #52c41a;
  border-radius: 4px;
}

.question-results {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.question-result {
  padding: 12px;
  background: #fafafa;
  border-radius: 6px;
}

.question-info {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 8px;
}

.question-no {
  font-weight: 500;
}

.question-score {
  margin-left: auto;
  font-weight: 600;
}

.question-comment, .question-suggestion {
  margin-top: 8px;
  font-size: 14px;
  color: #666;
  line-height: 1.5;
}

.grading-modal {
  padding: 16px 0;
}

.submission-info {
  margin-bottom: 24px;
  padding-bottom: 16px;
  border-bottom: 1px solid #f0f0f0;
}

.grading-progress {
  text-align: center;
  padding: 60px 20px;
}

.progress-text p {
  margin: 8px 0;
  color: #666;
  font-size: 14px;
}

.grading-result {
  padding: 16px 0;
}

.result-summary {
  margin-bottom: 24px;
  padding: 16px;
  background: #f8f9fa;
  border-radius: 8px;
}

.summary-stats {
  display: flex;
  gap: 24px;
  margin-top: 12px;
}

.summary-stats span {
  color: #666;
}

.detailed-results {
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.question-detail {
  border: 1px solid #e8e8e8;
  border-radius: 8px;
  padding: 16px;
}

.question-header {
  display: flex;
  align-items: center;
  gap: 12px;
  margin-bottom: 12px;
  font-weight: 500;
}

.question-content {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.student-answer, .ai-comment, .ai-suggestion {
  padding: 8px 12px;
  border-radius: 4px;
  font-size: 14px;
  line-height: 1.5;
}

.student-answer {
  background: #f0f8ff;
  border-left: 3px solid #1890ff;
}

.ai-comment {
  background: #f6ffed;
  border-left: 3px solid #52c41a;
}

.ai-suggestion {
  background: #fff7e6;
  border-left: 3px solid #fa8c16;
}

.modal-actions {
  margin-top: 24px;
  text-align: right;
}

.statistics-content {
  padding: 16px 0;
}

.statistics-charts {
  margin-bottom: 24px;
}

.chart-item h4 {
  margin-bottom: 16px;
  color: #333;
}

.score-distribution {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.distribution-item {
  display: flex;
  align-items: center;
  gap: 12px;
}

.range {
  width: 80px;
  font-weight: 500;
}

.bar {
  flex: 1;
  height: 20px;
  background: #f0f0f0;
  border-radius: 10px;
  overflow: hidden;
}

.bar-fill {
  height: 100%;
  background: linear-gradient(90deg, #1890ff, #40a9ff);
  transition: width 0.3s ease;
}

.count {
  width: 40px;
  text-align: right;
  font-weight: 500;
  color: #1890ff;
}

.summary-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: 16px;
}

.summary-item {
  text-align: center;
  padding: 16px;
  background: #f8f9fa;
  border-radius: 8px;
}

.item-label {
  color: #666;
  font-size: 14px;
  margin-bottom: 8px;
}

.item-value {
  font-size: 20px;
  font-weight: 600;
  color: #1890ff;
}

.document-preview-content {
  padding: 20px;
}

.document-info {
  text-align: center;
  margin-bottom: 20px;
}

.document-info h3 {
  margin-bottom: 10px;
  color: #333;
}

.document-info p {
  margin-bottom: 5px;
  color: #666;
  font-size: 14px;
}

.attachment-selector {
  margin-bottom: 15px;
  text-align: center;
}

.document-preview-frame {
  margin: 20px 0;
  border: 1px solid #e8e8e8;
  border-radius: 4px;
  overflow: hidden;
  min-height: 400px;
  background-color: #f5f5f5;
}

.word-preview,
.pdf-preview,
.image-preview {
  padding: 20px;
  background: #fff;
  box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
  height: 500px;
  overflow-y: auto;
}

/* Word文档样式 */
.word-document {
  width: 100%;
  background: #fff;
  min-height: 100%;
}

.word-page {
  margin: 0 auto;
  width: 100%;
  max-width: 800px;
  padding: 40px;
  box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
  background: #fff;
}

.word-content {
  font-family: 'Times New Roman', Times, serif;
  line-height: 1.6;
  color: #333;
}

.word-content h1 {
  font-size: 18px;
  font-weight: bold;
  text-align: center;
  margin-bottom: 20px;
}

.word-paragraph {
  margin-bottom: 10px;
  text-indent: 2em;
}

/* PDF预览样式 */
.pdf-viewer {
  width: 100%;
  height: 100%;
}

.pdf-page {
  margin: 0 auto;
  width: 100%;
  max-width: 800px;
  min-height: 400px;
  padding: 40px;
  box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
  background: #fff;
}

.pdf-page-content {
  text-align: center;
}

.pdf-mock {
  margin-top: 20px;
  width: 100%;
  background: #f9f9f9;
  padding: 15px;
  border-radius: 4px;
}

.pdf-mock-header {
  height: 30px;
  background: #e0e0e0;
  margin-bottom: 15px;
  border-radius: 2px;
}

.pdf-mock-text {
  height: 14px;
  background: #e0e0e0;
  margin-bottom: 10px;
  border-radius: 2px;
  width: 100%;
}

.pdf-mock-text:nth-child(2n) {
  width: 90%;
}

.pdf-mock-image {
  height: 150px;
  background: #d0d0d0;
  margin: 20px 0;
  border-radius: 2px;
  position: relative;
}

.pdf-mock-image:before {
  content: "📊";
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  font-size: 32px;
}

/* 图片预览样式 */
.image-preview {
  display: flex;
  justify-content: center;
  align-items: center;
}

.image-container {
  max-width: 80%;
  text-align: center;
}

.mock-image {
  width: 400px;
  height: 300px;
  background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
  border-radius: 4px;
  margin: 0 auto;
  display: flex;
  justify-content: center;
  align-items: center;
}

.mock-image-content {
  text-align: center;
}

.mock-image-icon {
  font-size: 64px;
  margin-bottom: 10px;
}

/* 不支持文件样式 */
.unsupported-file {
  display: flex;
  justify-content: center;
  align-items: center;
  height: 400px;
  color: #999;
}

.document-actions {
  display: flex;
  justify-content: center;
  gap: 10px;
  margin-top: 20px;
}

/* 作业答案样式 */
.answer-preview {
  padding: 20px;
  background: #fff;
  border: 1px solid #e8e8e8;
  border-radius: 4px;
  height: 500px;
  overflow-y: auto;
}

.answer-item {
  margin-bottom: 20px;
  padding-bottom: 10px;
  border-bottom: 1px solid #e8e8e8;
}

.answer-question {
  font-weight: 600;
  margin-bottom: 10px;
  color: #1890ff;
}

.answer-content {
  padding: 10px;
  background: #f9f9f9;
  border-radius: 4px;
  border-left: 3px solid #1890ff;
}
</style> 