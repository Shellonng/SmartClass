<template>
  <div class="ai-grading-page">
    <div class="page-header">
      <h1>智能批改</h1>
      <p class="description">使用AI技术自动批改学生作业，提高批改效率</p>
    </div>

    <div class="content-container">
      <!-- 批改表单 -->
      <a-row :gutter="24">
        <a-col :span="8">
          <a-card title="批改设置" class="grading-form-card">
            <a-form layout="vertical">
              <!-- 选择作业 -->
              <a-form-item label="选择作业" name="assignmentId">
                <a-select
                  v-model:value="gradingForm.assignmentId"
                  placeholder="请选择要批改的作业"
                  :loading="assignmentsLoading"
                  @change="handleAssignmentChange"
                >
                  <a-select-option v-for="item in assignments" :key="item.id" :value="item.id">
                    {{ item.title }}
                  </a-select-option>
                </a-select>
              </a-form-item>

              <!-- 选择提交记录 -->
              <a-form-item label="选择提交记录" name="submissionId">
                <a-select
                  v-model:value="gradingForm.submissionId"
                  placeholder="请选择要批改的提交记录"
                  :loading="submissionsLoading"
                  :disabled="!gradingForm.assignmentId"
                  @change="handleSubmissionChange"
                >
                  <a-select-option v-for="item in submissions" :key="item.id" :value="item.id">
                    {{ item.studentName }} - {{ formatSubmitTime(item.submitTime) }}
                  </a-select-option>
                </a-select>
              </a-form-item>

              <!-- 参考答案 -->
              <a-form-item label="参考答案" name="referenceAnswer">
                <a-textarea
                  v-model:value="gradingForm.referenceAnswer"
                  placeholder="请输入参考答案或评分标准"
                  :rows="6"
                />
              </a-form-item>

              <!-- 批改标准 -->
              <a-form-item label="批改标准" name="gradingCriteria">
                <a-select
                  v-model:value="gradingForm.gradingCriteria"
                  placeholder="请选择批改标准"
                >
                  <a-select-option value="strict">严格模式</a-select-option>
                  <a-select-option value="normal">标准模式</a-select-option>
                  <a-select-option value="lenient">宽松模式</a-select-option>
                </a-select>
              </a-form-item>

              <!-- 上传学生提交文件 -->
              <a-form-item label="学生提交文件" name="submittedFile">
                <a-upload
                  v-model:file-list="fileList"
                  :before-upload="beforeUpload"
                  :max-count="1"
                >
                  <a-button>
                    <template #icon><upload-outlined /></template>
                    选择文件
                  </a-button>
                </a-upload>
              </a-form-item>

              <!-- 批改按钮 -->
              <a-form-item>
                <a-button
                  type="primary"
                  block
                  @click="handleGrade"
                  :loading="grading"
                  :disabled="!canGrade"
                >
                  开始智能批改
                </a-button>
              </a-form-item>

              <!-- 批量批改 -->
              <a-divider>批量批改</a-divider>
              <a-form-item>
                <a-button
                  type="dashed"
                  block
                  @click="handleBatchGrade"
                  :loading="batchGrading"
                  :disabled="!gradingForm.assignmentId || !gradingForm.referenceAnswer"
                >
                  批量批改所有提交
                </a-button>
              </a-form-item>
            </a-form>
          </a-card>
        </a-col>

        <a-col :span="16">
          <!-- 批改结果 -->
          <a-card title="批改结果" class="grading-result-card">
            <template #extra>
              <a-button type="primary" size="small" @click="saveGradingResult" :disabled="!gradingResult.success">
                保存批改结果
              </a-button>
            </template>

            <div v-if="grading" class="grading-loading">
              <a-spin tip="AI正在批改中，请稍候...">
                <div class="loading-content">
                  <p>正在分析学生提交内容...</p>
                  <p>正在与参考答案进行比对...</p>
                  <p>正在生成评分和反馈...</p>
                </div>
              </a-spin>
            </div>

            <div v-else-if="gradingResult.success" class="grading-success">
              <div class="result-header">
                <div class="score-section">
                  <div class="score">{{ gradingResult.score }}</div>
                  <div class="total-score">/ {{ gradingForm.maxScore || 100 }}</div>
                </div>
                <a-tag :color="getScoreColor(gradingResult.score, gradingForm.maxScore)">
                  {{ getScoreLevel(gradingResult.score, gradingForm.maxScore) }}
                </a-tag>
              </div>

              <a-divider />

              <div class="feedback-section">
                <h3>批改反馈</h3>
                <div class="feedback-content" v-html="formatFeedback(gradingResult.feedback)"></div>
              </div>

              <a-divider />

              <div class="analysis-section">
                <h3>详细分析</h3>
                <div class="analysis-content" v-html="formatAnalysis(gradingResult.analysis)"></div>
              </div>
            </div>

            <div v-else-if="gradingResult.error" class="grading-error">
              <a-result status="error" title="批改失败" :sub-title="gradingResult.error">
                <template #extra>
                  <a-button type="primary" @click="handleGrade">重试</a-button>
                </template>
              </a-result>
            </div>

            <div v-else class="grading-empty">
              <a-empty description="暂无批改结果，请选择作业和提交记录进行批改">
                <template #image>
                  <img src="/images/ai-grading.svg" alt="AI批改" />
                </template>
              </a-empty>
            </div>
          </a-card>

          <!-- 批量批改结果 -->
          <a-card v-if="batchResults.length > 0" title="批量批改结果" class="batch-result-card">
            <a-table
              :columns="batchColumns"
              :data-source="batchResults"
              :pagination="{ pageSize: 5 }"
              size="small"
            >
              <template #bodyCell="{ column, record }">
                <template v-if="column.key === 'status'">
                  <a-tag :color="record.status === 'success' ? 'green' : 'red'">
                    {{ record.status === 'success' ? '成功' : '失败' }}
                  </a-tag>
                </template>
                
                <template v-if="column.key === 'score'">
                  <span v-if="record.status === 'success'">{{ record.score }}</span>
                  <span v-else>-</span>
                </template>
                
                <template v-if="column.key === 'action'">
                  <a-button
                    type="link"
                    size="small"
                    @click="viewBatchResult(record)"
                    :disabled="record.status !== 'success'"
                  >
                    查看详情
                  </a-button>
                </template>
              </template>
            </a-table>
          </a-card>
        </a-col>
      </a-row>
    </div>

    <!-- 批改详情弹窗 -->
    <a-modal
      v-model:visible="detailModalVisible"
      title="批改详情"
      width="700px"
      :footer="null"
    >
      <div v-if="selectedResult" class="detail-content">
        <div class="detail-header">
          <div class="detail-title">{{ selectedResult.studentName }}</div>
          <div class="detail-score">得分：{{ selectedResult.score }} / {{ gradingForm.maxScore || 100 }}</div>
        </div>
        
        <a-divider />
        
        <div class="detail-feedback">
          <h3>批改反馈</h3>
          <div v-html="formatFeedback(selectedResult.feedback)"></div>
        </div>
        
        <a-divider />
        
        <div class="detail-analysis">
          <h3>详细分析</h3>
          <div v-html="formatAnalysis(selectedResult.analysis)"></div>
        </div>
        
        <a-divider />
        
        <div class="detail-footer">
          <a-button type="primary" @click="detailModalVisible = false">关闭</a-button>
        </div>
      </div>
    </a-modal>
  </div>
</template>

<script setup lang="ts">
import { ref, reactive, computed, onMounted } from 'vue'
import { message, Modal } from 'ant-design-vue'
import { UploadOutlined } from '@ant-design/icons-vue'
import axios from 'axios'
import dayjs from 'dayjs'

// 状态变量
const assignments = ref<any[]>([])
const submissions = ref<any[]>([])
const assignmentsLoading = ref(false)
const submissionsLoading = ref(false)
const grading = ref(false)
const batchGrading = ref(false)
const fileList = ref<any[]>([])
const detailModalVisible = ref(false)
const selectedResult = ref<any>(null)

// 批改表单
const gradingForm = reactive({
  assignmentId: null as number | null,
  submissionId: null as number | null,
  referenceAnswer: '',
  gradingCriteria: 'normal',
  submittedFile: null as File | null,
  maxScore: 100,
  mode: 'text' // 添加mode属性，默认为text模式
})

// 批改结果
const gradingResult = reactive({
  success: false,
  score: 0,
  feedback: '',
  analysis: '',
  error: ''
})

// 批量批改结果
const batchResults = ref<any[]>([])

// 批量批改表格列
const batchColumns = [
  {
    title: '学生姓名',
    dataIndex: 'studentName',
    key: 'studentName'
  },
  {
    title: '提交时间',
    dataIndex: 'submitTime',
    key: 'submitTime',
    render: (text: string) => formatSubmitTime(text)
  },
  {
    title: '状态',
    dataIndex: 'status',
    key: 'status'
  },
  {
    title: '得分',
    dataIndex: 'score',
    key: 'score'
  },
  {
    title: '操作',
    key: 'action'
  }
]

// 计算是否可以进行批改
const canGrade = computed(() => {
  if (gradingForm.mode === 'file') {
    return gradingForm.assignmentId && 
           gradingForm.submissionId && 
           gradingForm.referenceAnswer && 
           (gradingForm.submittedFile || fileList.value.length > 0)
  } else {
    return gradingForm.assignmentId && 
           gradingForm.submissionId && 
           gradingForm.referenceAnswer
  }
})

// 格式化提交时间
const formatSubmitTime = (time: string) => {
  if (!time) return '-'
  return dayjs(time).format('YYYY-MM-DD HH:mm')
}

// 获取分数颜色
const getScoreColor = (score: number, maxScore: number = 100) => {
  const percentage = (score / maxScore) * 100
  if (percentage >= 90) return 'green'
  if (percentage >= 60) return 'blue'
  return 'red'
}

// 获取分数等级
const getScoreLevel = (score: number, maxScore: number = 100) => {
  const percentage = (score / maxScore) * 100
  if (percentage >= 90) return '优秀'
  if (percentage >= 80) return '良好'
  if (percentage >= 70) return '中等'
  if (percentage >= 60) return '及格'
  return '不及格'
}

// 格式化反馈内容（将换行符转换为HTML换行）
const formatFeedback = (feedback: string) => {
  if (!feedback) return ''
  return feedback.replace(/\n/g, '<br />')
}

// 格式化分析内容（将换行符转换为HTML换行）
const formatAnalysis = (analysis: string) => {
  if (!analysis) return ''
  return analysis.replace(/\n/g, '<br />')
}

// 上传前检查文件
const beforeUpload = (file: File) => {
  gradingForm.submittedFile = file
  return false // 阻止自动上传
}

// 加载作业列表
const loadAssignments = async () => {
  assignmentsLoading.value = true
  try {
    // 获取token
    const token = localStorage.getItem('token') || localStorage.getItem('user-token')
    
    const response = await axios.get('/api/teacher/assignments', {
      params: {
        status: 1 // 只获取已发布的作业
      },
      headers: {
        'Authorization': token ? `Bearer ${token}` : ''
      }
    })
    
    if (response.data && response.data.code === 200) {
      const data = response.data.data
      assignments.value = data.records || []
    } else {
      message.error('获取作业列表失败')
    }
  } catch (error) {
    console.error('加载作业列表失败:', error)
    message.error('获取作业列表失败，请检查网络连接')
  } finally {
    assignmentsLoading.value = false
  }
}

// 加载提交记录
const loadSubmissions = async (assignmentId: number) => {
  submissionsLoading.value = true
  try {
    // 获取token
    const token = localStorage.getItem('token') || localStorage.getItem('user-token')
    
    const response = await axios.get(`/api/teacher/assignments/${assignmentId}/submissions`, {
      headers: {
        'Authorization': token ? `Bearer ${token}` : ''
      }
    })
    
    if (response.data && response.data.code === 200) {
      submissions.value = response.data.data || []
    } else {
      message.error('获取提交记录失败')
      submissions.value = []
    }
  } catch (error) {
    console.error('加载提交记录失败:', error)
    message.error('获取提交记录失败，请检查网络连接')
    submissions.value = []
  } finally {
    submissionsLoading.value = false
  }
}

// 处理作业变更
const handleAssignmentChange = (assignmentId: number) => {
  gradingForm.submissionId = null
  submissions.value = []
  
  if (assignmentId) {
    // 获取作业详情，包括参考答案
    getAssignmentDetail(assignmentId)
    // 加载提交记录
    loadSubmissions(assignmentId)
  }
}

// 获取作业详情
const getAssignmentDetail = async (assignmentId: number) => {
  try {
    // 获取token
    const token = localStorage.getItem('token') || localStorage.getItem('user-token')
    
    const response = await axios.get(`/api/teacher/assignments/${assignmentId}`, {
      headers: {
        'Authorization': token ? `Bearer ${token}` : ''
      }
    })
    
    if (response.data && response.data.code === 200) {
      const assignment = response.data.data
      gradingForm.referenceAnswer = assignment.referenceAnswer || ''
      gradingForm.maxScore = assignment.totalScore || 100
    }
  } catch (error) {
    console.error('获取作业详情失败:', error)
  }
}

// 处理提交记录变更
const handleSubmissionChange = (submissionId: number) => {
  if (submissionId) {
    const submission = submissions.value.find(item => item.id === submissionId)
    if (submission) {
      // 如果有提交内容，可以自动填充
      if (submission.content) {
        // 这里可以根据需要处理提交内容
      }
    }
  }
}

// 执行智能批改
const handleGrade = async () => {
  // 表单验证
  if (!gradingForm.assignmentId) {
    message.error('请选择要批改的作业')
    return
  }
  if (!gradingForm.submissionId) {
    message.error('请选择要批改的提交记录')
    return
  }
  if (!gradingForm.referenceAnswer) {
    message.error('请输入参考答案')
    return
  }
  
  grading.value = true
  gradingResult.success = false
  gradingResult.error = ''
  
  try {
    // 获取token
    const token = localStorage.getItem('token') || localStorage.getItem('user-token')
    
    // 创建表单数据
    const formData = new FormData()
    formData.append('submissionId', String(gradingForm.submissionId))
    formData.append('referenceAnswer', gradingForm.referenceAnswer)
    
    // 添加文件（如果有）
    if (gradingForm.submittedFile) {
      formData.append('submittedFile', gradingForm.submittedFile)
    } else if (fileList.value.length > 0 && fileList.value[0].originFileObj) {
      formData.append('submittedFile', fileList.value[0].originFileObj)
    }
    
    // 调用智能批改API
    const response = await axios.post(
      `/api/teacher/assignments/submissions/${gradingForm.submissionId}/ai-grade`,
      formData,
      {
        headers: {
          'Authorization': token ? `Bearer ${token}` : '',
          'Content-Type': 'multipart/form-data'
        }
      }
    )
    
    if (response.data && response.data.code === 200) {
      const result = response.data.data
      
      // 更新批改结果
      gradingResult.success = result.success
      gradingResult.score = result.score || 0
      gradingResult.feedback = result.feedback || '未提供反馈'
      gradingResult.analysis = result.analysis || '未提供分析'
      
      if (result.success) {
        message.success('智能批改完成')
      } else {
        gradingResult.error = result.error || '批改失败，但未提供具体错误信息'
        message.error('智能批改失败: ' + gradingResult.error)
      }
    } else {
      gradingResult.error = response.data?.message || '批改请求失败'
      message.error('智能批改失败: ' + gradingResult.error)
    }
  } catch (error: any) {
    console.error('智能批改失败:', error)
    gradingResult.error = error.message || '未知错误'
    message.error('智能批改失败: ' + gradingResult.error)
  } finally {
    grading.value = false
  }
}

// 批量批改
const handleBatchGrade = async () => {
  if (!gradingForm.assignmentId) {
    message.error('请选择要批改的作业')
    return
  }
  if (!gradingForm.referenceAnswer) {
    message.error('请输入参考答案')
    return
  }
  
  Modal.confirm({
    title: '批量批改确认',
    content: '确定要对该作业的所有提交记录进行批量批改吗？这可能需要一些时间。',
    onOk: async () => {
      batchGrading.value = true
      try {
        // 获取token
        const token = localStorage.getItem('token') || localStorage.getItem('user-token')
        
        const response = await axios.post(
          `/api/teacher/assignments/${gradingForm.assignmentId}/ai-grade-batch`,
          {
            referenceAnswer: gradingForm.referenceAnswer
          },
          {
            headers: {
              'Authorization': token ? `Bearer ${token}` : ''
            }
          }
        )
        
        if (response.data && response.data.code === 200) {
          const result = response.data.data
          
          // 更新批量批改结果
          batchResults.value = result.results || []
          
          message.success(`批量批改完成，成功: ${result.success || 0}, 失败: ${result.failed || 0}`)
        } else {
          message.error(response.data?.message || '批量批改请求失败')
        }
      } catch (error: any) {
        console.error('批量批改失败:', error)
        message.error('批量批改失败: ' + (error.message || '未知错误'))
      } finally {
        batchGrading.value = false
      }
    }
  })
}

// 保存批改结果
const saveGradingResult = async () => {
  if (!gradingForm.submissionId || !gradingResult.success) {
    message.error('没有可保存的批改结果')
    return
  }
  
  try {
    // 获取token
    const token = localStorage.getItem('token') || localStorage.getItem('user-token')
    
    const response = await axios.put(
      `/api/teacher/submissions/${gradingForm.submissionId}/grade`,
      {
        score: gradingResult.score,
        feedback: gradingResult.feedback
      },
      {
        headers: {
          'Authorization': token ? `Bearer ${token}` : ''
        }
      }
    )
    
    if (response.data && response.data.code === 200) {
      message.success('批改结果保存成功')
      
      // 更新提交记录状态
      const submissionIndex = submissions.value.findIndex(item => item.id === gradingForm.submissionId)
      if (submissionIndex !== -1) {
        submissions.value[submissionIndex].status = 2 // 已批改
        submissions.value[submissionIndex].score = gradingResult.score
      }
    } else {
      message.error(response.data?.message || '保存批改结果失败')
    }
  } catch (error) {
    console.error('保存批改结果失败:', error)
    message.error('保存批改结果失败，请检查网络连接')
  }
}

// 查看批量批改详情
const viewBatchResult = (record: any) => {
  selectedResult.value = record
  detailModalVisible.value = true
}

// 初始化
onMounted(() => {
  loadAssignments()
})
</script>

<style scoped>
.ai-grading-page {
  padding: 24px;
  background-color: #f0f2f5;
  min-height: 100vh;
}

.page-header {
  margin-bottom: 24px;
}

.page-header h1 {
  margin-bottom: 8px;
  font-size: 24px;
  font-weight: 600;
}

.description {
  color: rgba(0, 0, 0, 0.45);
}

.content-container {
  max-width: 1200px;
  margin: 0 auto;
}

.grading-form-card,
.grading-result-card,
.batch-result-card {
  border-radius: 8px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.09);
  margin-bottom: 24px;
}

.grading-loading {
  display: flex;
  justify-content: center;
  align-items: center;
  padding: 40px 0;
}

.loading-content {
  margin-top: 24px;
  text-align: center;
}

.grading-empty {
  padding: 40px 0;
  text-align: center;
}

.grading-success {
  padding: 16px;
}

.result-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
}

.score-section {
  display: flex;
  align-items: baseline;
}

.score {
  font-size: 48px;
  font-weight: 600;
  color: #1890ff;
}

.total-score {
  font-size: 20px;
  color: rgba(0, 0, 0, 0.45);
  margin-left: 8px;
}

.feedback-section,
.analysis-section {
  margin-top: 16px;
}

.feedback-content,
.analysis-content {
  background-color: #f9f9f9;
  border-radius: 4px;
  padding: 16px;
  margin-top: 8px;
}

.grading-error {
  padding: 24px 0;
}

.detail-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.detail-title {
  font-size: 18px;
  font-weight: 600;
}

.detail-score {
  font-size: 16px;
  font-weight: 500;
  color: #1890ff;
}

.detail-feedback,
.detail-analysis {
  margin-top: 16px;
}

.detail-footer {
  margin-top: 24px;
  text-align: right;
}
</style> 