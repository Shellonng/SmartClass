<template>
  <div class="assignment-detail">
    <div class="page-header">
      <div class="back-link">
        <router-link to="/teacher/assignments">
          <LeftOutlined /> 返回作业列表
        </router-link>
      </div>
      <h2>{{ assignment?.title || '作业详情' }}</h2>
    </div>

    <a-spin :spinning="loading">
      <div v-if="assignment" class="assignment-content">
        <div class="assignment-info">
          <div class="info-row">
            <div class="info-item">
              <div class="info-label">所属课程</div>
              <div class="info-value">{{ assignment.courseName || '未指定课程' }}</div>
            </div>
            <div class="info-item">
              <div class="info-label">发布状态</div>
              <div class="info-value">
                <a-tag :color="assignment.status === 0 ? 'orange' : 'green'">
                  {{ assignment.status === 0 ? '未发布' : '已发布' }}
                </a-tag>
              </div>
            </div>
          </div>
          <div class="info-row">
            <div class="info-item">
              <div class="info-label">开始时间</div>
              <div class="info-value">{{ formatDate(assignment.startTime) }}</div>
            </div>
            <div class="info-item">
              <div class="info-label">结束时间</div>
              <div class="info-value">{{ formatDate(assignment.endTime) }}</div>
            </div>
            <div class="info-item">
              <div class="info-label">作业时长</div>
              <div class="info-value">{{ assignment.duration }} 分钟</div>
            </div>
          </div>
          <div class="info-row">
            <div class="info-item full-width">
              <div class="info-label">作业说明</div>
              <div class="info-value description">{{ assignment.description || '无' }}</div>
            </div>
          </div>
        </div>

        <a-tabs v-model:activeKey="activeTabKey" class="detail-tabs">
          <a-tab-pane key="questions" tab="题目列表">
            <div class="questions-section">
              <div v-if="!assignment.questions || assignment.questions.length === 0" class="no-questions">
                <a-empty description="暂无题目" />
              </div>
              <div v-else class="questions-list">
                <div v-for="(question, index) in assignment.questions" :key="question.id" class="question-card">
                  <div class="question-header">
                    <div class="question-title">
                      <span class="question-number">{{ index + 1 }}.</span>
                      <span class="question-type-tag">{{ question.questionTypeDesc }}</span>
                      <span>{{ question.title }}</span>
                    </div>
                    <div class="question-score">{{ question.score }}分</div>
                  </div>
                  
                  <!-- 选择题选项 -->
                  <div v-if="['single', 'multiple', 'true_false'].includes(question.questionType)" class="question-options">
                    <div v-for="option in question.options" :key="option.id" class="option-item">
                      <span class="option-label">{{ option.optionLabel }}.</span>
                      <span class="option-text">{{ option.optionText }}</span>
                    </div>
                  </div>
                  
                  <!-- 题目答案和解析 -->
                  <div class="question-answer">
                    <div class="answer-item">
                      <span class="answer-label">正确答案:</span>
                      <span class="answer-content">{{ question.correctAnswer }}</span>
                    </div>
                    <div v-if="question.explanation" class="answer-item">
                      <span class="answer-label">解析:</span>
                      <span class="answer-content">{{ question.explanation }}</span>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </a-tab-pane>
          
          <!-- 提交记录选项卡 -->
          <a-tab-pane key="submissions" tab="提交记录">
            <div class="submissions-section">
              <div class="filter-bar">
                <div class="filter-left">
                  <a-select 
                    v-model:value="submissionFilter" 
                    style="width: 150px" 
                    placeholder="提交状态"
                    @change="handleFilterChange"
                  >
                    <a-select-option value="">全部状态</a-select-option>
                    <a-select-option value="0">未提交</a-select-option>
                    <a-select-option value="1">已提交未批改</a-select-option>
                    <a-select-option value="2">已批改</a-select-option>
                  </a-select>
                </div>
                <div class="filter-right">
                  <a-button @click="refreshSubmissions">
                    <ReloadOutlined />
                    刷新数据
                  </a-button>
                </div>
              </div>
              
              <a-table
                :dataSource="submissions"
                :columns="submissionColumns"
                :pagination="pagination"
                :loading="submissionsLoading"
                rowKey="id"
                @change="handleTableChange"
              >
                <!-- 学生姓名 -->
                <template #bodyCell="{ column, record }">
                  <template v-if="column.dataIndex === 'studentName'">
                    <span>{{ record.studentName }}</span>
                  </template>
                  
                  <!-- 得分 -->
                  <template v-else-if="column.dataIndex === 'score'">
                    <span v-if="record.status === 2">{{ record.score || 0 }}/{{ assignment.totalScore || 100 }}</span>
                    <span v-else>-</span>
                  </template>
                  
                  <!-- 提交状态 -->
                  <template v-else-if="column.dataIndex === 'status'">
                    <a-tag :color="getStatusColor(record.status)">
                      {{ getStatusText(record.status) }}
                    </a-tag>
                  </template>
                  
                  <!-- 提交时间 -->
                  <template v-else-if="column.dataIndex === 'submitTime'">
                    <span v-if="record.submitTime">{{ formatDate(record.submitTime) }}</span>
                    <span v-else>-</span>
                  </template>
                  
                  <!-- 操作 -->
                  <template v-else-if="column.dataIndex === 'action'">
                    <div class="action-buttons">
                      <a-tooltip v-if="record.status === 1" title="批改">
                        <a-button type="link" size="small" @click="handleGradeSubmission(record)">
                          <CheckOutlined />
                        </a-button>
                      </a-tooltip>
                      <a-tooltip v-if="record.status === 2" title="查看">
                        <a-button type="link" size="small" @click="handleViewSubmission(record)">
                          <EyeOutlined />
                        </a-button>
                      </a-tooltip>
                      <a-tooltip title="删除">
                        <a-popconfirm
                          title="确定要删除这条提交记录吗？"
                          @confirm="handleDeleteSubmission(record.id)"
                          ok-text="确定"
                          cancel-text="取消"
                        >
                          <a-button type="link" danger size="small">
                            <DeleteOutlined />
                          </a-button>
                        </a-popconfirm>
                      </a-tooltip>
                    </div>
                  </template>
                </template>
              </a-table>
              
              <!-- 没有提交记录时显示的空状态 -->
              <div v-if="submissions.length === 0 && !submissionsLoading" class="empty-state">
                <a-empty description="暂无提交记录" />
                <p class="empty-tip">可能是因为该课程还没有学生选择，或者学生尚未提交作业。</p>
              </div>
            </div>
          </a-tab-pane>
        </a-tabs>

        <div class="action-buttons">
          <a-button type="primary" @click="editAssignment">编辑作业</a-button>
          <a-button v-if="assignment.status === 0" type="primary" @click="publishAssignment" style="margin-left: 10px">
            发布作业
          </a-button>
          <a-button v-else type="default" @click="unpublishAssignment" style="margin-left: 10px">
            取消发布
          </a-button>
          <a-popconfirm
            title="确定要删除这个作业吗？"
            description="删除后将无法恢复，包括作业题目关联数据也会被删除。"
            @confirm="handleDeleteAssignment"
            ok-text="确定"
            cancel-text="取消"
          >
            <a-button danger style="margin-left: 10px">删除作业</a-button>
          </a-popconfirm>
        </div>
      </div>
    </a-spin>
    
    <!-- 批改作业弹窗 -->
    <a-modal
      v-model:open="gradeModalVisible"
      title="批改作业"
      width="700px"
      :maskClosable="false"
      @ok="submitGrade"
      :okButtonProps="{ loading: submitting }"
      :okText="submitting ? '提交中...' : '提交'"
    >
      <div v-if="currentSubmission" class="grade-form">
        <div class="student-info">
          <div class="info-item">
            <span class="info-label">学生姓名：</span>
            <span class="info-value">{{ currentSubmission.studentName }}</span>
          </div>
          <div class="info-item">
            <span class="info-label">提交时间：</span>
            <span class="info-value">{{ formatDate(currentSubmission.submitTime) }}</span>
          </div>
        </div>
        
        <a-form layout="vertical">
          <a-form-item label="分数">
            <a-input-number 
              v-model:value="gradeForm.score" 
              :min="0" 
              :max="assignment?.totalScore || 100" 
              style="width: 200px"
            />
            <span class="max-score">满分：{{ assignment?.totalScore || 100 }}分</span>
          </a-form-item>
          
          <a-form-item label="评语">
            <a-textarea 
              v-model:value="gradeForm.feedback" 
              :rows="4" 
              placeholder="输入对学生作业的评价和建议..."
            />
          </a-form-item>
        </a-form>
      </div>
    </a-modal>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted, watch } from 'vue'
import { useRouter, useRoute } from 'vue-router'
import { message } from 'ant-design-vue'
import dayjs from 'dayjs'
import { LeftOutlined, DeleteOutlined, ReloadOutlined, CheckOutlined, EyeOutlined } from '@ant-design/icons-vue'
import assignmentApi from '@/api/assignment'

const props = defineProps({
  id: {
    type: Number,
    required: true
  }
})

const router = useRouter()
const route = useRoute() // 获取当前路由对象
const loading = ref(false)
const assignment = ref<any>(null)
const activeTabKey = ref('questions') // 默认显示题目列表

// 提交记录相关
const submissions = ref<any[]>([])
const submissionsLoading = ref(false)
const submissionFilter = ref('')
const pagination = ref({
  current: 1,
  pageSize: 10,
  total: 0,
  showSizeChanger: true,
  showTotal: (total: number) => `共 ${total} 条记录`
})

// 批改弹窗相关
const gradeModalVisible = ref(false)
const currentSubmission = ref<any>(null)
const submitting = ref(false)
const gradeForm = ref({
  score: 0,
  feedback: ''
})

// 表格列定义
const submissionColumns = [
  {
    title: '学生姓名',
    dataIndex: 'studentName',
    key: 'studentName',
    width: '15%'
  },
  {
    title: '得分',
    dataIndex: 'score',
    key: 'score',
    width: '15%'
  },
  {
    title: '状态',
    dataIndex: 'status',
    key: 'status',
    width: '15%'
  },
  {
    title: '提交时间',
    dataIndex: 'submitTime',
    key: 'submitTime',
    width: '20%'
  },
  {
    title: '操作',
    dataIndex: 'action',
    key: 'action',
    width: '20%'
  }
]

// 获取作业详情
const fetchAssignmentDetail = async () => {
  loading.value = true
  try {
    const response = await assignmentApi.getAssignmentDetail(props.id)
    if (response.code === 200) {
      assignment.value = response.data
      
      // 如果没有课程名称但有课程ID，则手动查询课程名称
      if ((!assignment.value.courseName || assignment.value.courseName === '') && assignment.value.courseId) {
        try {
          const courseResponse = await fetch(`/api/teacher/courses/${assignment.value.courseId}`)
          const courseData = await courseResponse.json()
          if (courseData.code === 200 && courseData.data) {
            assignment.value.courseName = courseData.data.name
            console.log('手动查询到课程名称:', assignment.value.courseName)
          }
        } catch (courseError) {
          console.error('查询课程名称失败:', courseError)
        }
      }
      
      // 获取作业详情后，加载提交记录
      fetchSubmissions()
    } else {
      message.error(response.message || '获取作业详情失败')
    }
  } catch (error) {
    console.error('获取作业详情出错:', error)
    message.error('获取作业详情失败，请检查网络连接')
  } finally {
    loading.value = false
  }
}

// 获取提交记录列表
const fetchSubmissions = async () => {
  submissionsLoading.value = true
  try {
    const params = {
      current: pagination.value.current,
      pageSize: pagination.value.pageSize,
      status: submissionFilter.value || undefined
    }
    
    const response = await assignmentApi.getAssignmentSubmissions(props.id, params)
    
    if (response.code === 200) {
      submissions.value = response.data.records || []
      pagination.value.total = response.data.total || 0
      console.log('获取到的提交记录:', submissions.value.length, '条记录')
    } else {
      message.error(response.message || '获取提交记录失败')
      submissions.value = []
      pagination.value.total = 0
    }
  } catch (error) {
    console.error('获取提交记录出错:', error)
    message.error('获取提交记录失败，请检查网络连接')
    submissions.value = []
    pagination.value.total = 0
  } finally {
    submissionsLoading.value = false
  }
}

// 刷新提交记录
const refreshSubmissions = () => {
  fetchSubmissions()
}

// 处理表格变更（分页、排序等）
const handleTableChange = (pag: any) => {
  pagination.value.current = pag.current
  pagination.value.pageSize = pag.pageSize
  fetchSubmissions()
}

// 处理筛选条件变更
const handleFilterChange = () => {
  pagination.value.current = 1
  fetchSubmissions()
}

// 获取状态文本
const getStatusText = (status: number) => {
  // 如果status是字符串，转换为数字
  const statusNum = typeof status === 'string' ? parseInt(status) : status
  
  const statusMap: Record<number, string> = {
    0: '未提交',
    1: '未批改',
    2: '已批改'
  }
  
  // 如果记录中已经有statusText字段，优先使用
  return statusMap[statusNum] || '未知状态'
}

// 获取状态颜色
const getStatusColor = (status: number) => {
  const colorMap: Record<number, string> = {
    0: 'default',
    1: 'orange',
    2: 'green'
  }
  return colorMap[status] || 'default'
}

// 处理批改作业
const handleGradeSubmission = (record: any) => {
  currentSubmission.value = record
  gradeForm.value.score = 0
  gradeForm.value.feedback = ''
  gradeModalVisible.value = true
}

// 查看已批改作业
const handleViewSubmission = (record: any) => {
  // 可以实现查看已批改作业的功能
  router.push(`/teacher/submissions/${record.id}`)
}

// 提交批改
const submitGrade = async () => {
  if (!currentSubmission.value) return
  
  submitting.value = true
  try {
    const data = {
      score: gradeForm.value.score,
      feedback: gradeForm.value.feedback
    }
    
    const response = await assignmentApi.gradeSubmission(currentSubmission.value.id, data)
    
    if (response.code === 200) {
      message.success('批改成功')
      gradeModalVisible.value = false
      fetchSubmissions() // 刷新提交列表
    } else {
      message.error(response.message || '批改失败')
    }
  } catch (error) {
    console.error('批改作业出错:', error)
    message.error('批改失败，请检查网络连接')
  } finally {
    submitting.value = false
  }
}

// 删除提交记录
const handleDeleteSubmission = async (id: number) => {
  try {
    const response = await assignmentApi.deleteSubmission(id)
    
    if (response.code === 200) {
      message.success('删除提交记录成功')
      fetchSubmissions() // 刷新提交列表
    } else {
      message.error(response.message || '删除提交记录失败')
    }
  } catch (error) {
    console.error('删除提交记录出错:', error)
    message.error('删除提交记录失败，请检查网络连接')
  }
}

// 格式化日期
const formatDate = (dateStr: string) => {
  if (!dateStr) return '未设置'
  return dayjs(dateStr).format('YYYY-MM-DD HH:mm')
}

// 编辑作业
const editAssignment = () => {
  router.push(`/teacher/assignments/${props.id}/edit`)
}

// 发布作业
const publishAssignment = async () => {
  try {
    const response = await assignmentApi.publishAssignment(props.id)
    if (response.code === 200) {
      message.success('作业发布成功')
      fetchAssignmentDetail()
    } else {
      message.error(response.message || '作业发布失败')
    }
  } catch (error) {
    console.error('发布作业出错:', error)
    message.error('作业发布失败，请检查网络连接')
  }
}

// 取消发布作业
const unpublishAssignment = async () => {
  try {
    const response = await assignmentApi.unpublishAssignment(props.id)
    if (response.code === 200) {
      message.success('取消发布成功')
      fetchAssignmentDetail()
    } else {
      message.error(response.message || '取消发布失败')
    }
  } catch (error) {
    console.error('取消发布作业出错:', error)
    message.error('取消发布失败，请检查网络连接')
  }
}

// 删除作业
const handleDeleteAssignment = async () => {
  try {
    const response = await assignmentApi.deleteAssignment(props.id)
    if (response.code === 200) {
      message.success('作业删除成功')
      router.push('/teacher/assignments')
    } else {
      message.error(response.message || '作业删除失败')
    }
  } catch (error) {
    console.error('删除作业出错:', error)
    message.error('作业删除失败，请检查网络连接')
  }
}

onMounted(() => {
  // 如果路由有tab参数，设置激活的标签页
  if (route.query.tab && typeof route.query.tab === 'string') {
    activeTabKey.value = route.query.tab
  }
  
  fetchAssignmentDetail()
})

// 监听route.query变化，更新activeTabKey
watch(() => route.query.tab, (newTab) => {
  if (newTab && typeof newTab === 'string') {
    activeTabKey.value = newTab
  }
})
</script>

<style scoped>
.assignment-detail {
  padding: 24px;
}

.page-header {
  margin-bottom: 24px;
}

.back-link {
  margin-bottom: 16px;
}

.back-link a {
  display: inline-flex;
  align-items: center;
  color: #1890ff;
}

.page-header h2 {
  margin-bottom: 0;
  font-size: 20px;
  font-weight: 600;
}

.assignment-content {
  background: #fff;
  padding: 24px;
  border-radius: 4px;
}

.assignment-info {
  margin-bottom: 24px;
}

.info-row {
  display: flex;
  flex-wrap: wrap;
  margin-bottom: 16px;
}

.info-item {
  flex: 1;
  min-width: 180px;
  margin-bottom: 8px;
}

.info-item.full-width {
  flex: 100%;
}

.info-label {
  font-weight: 600;
  margin-bottom: 4px;
}

.info-value {
  color: #666;
}

.info-value.description {
  white-space: pre-line;
  background: #f5f5f5;
  padding: 12px;
  border-radius: 4px;
}

.detail-tabs {
  margin-bottom: 24px;
}

.questions-section,
.submissions-section {
  margin-bottom: 24px;
}

.no-questions {
  padding: 40px 0;
  text-align: center;
}

.question-card {
  margin-bottom: 20px;
  padding: 16px;
  border: 1px solid #e8e8e8;
  border-radius: 4px;
  background: #f9f9f9;
}

.question-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 12px;
}

.question-title {
  font-weight: 600;
}

.question-number {
  margin-right: 8px;
}

.question-type-tag {
  background: #e6f7ff;
  color: #1890ff;
  padding: 2px 8px;
  border-radius: 4px;
  font-size: 12px;
  margin-right: 8px;
}

.question-score {
  color: #ff4d4f;
  font-weight: 600;
}

.question-options {
  margin-bottom: 16px;
}

.option-item {
  margin-bottom: 8px;
  padding-left: 24px;
}

.option-label {
  margin-right: 8px;
  font-weight: 600;
}

.question-answer {
  background: #f0f0f0;
  padding: 12px;
  border-radius: 4px;
}

.answer-item {
  margin-bottom: 8px;
}

.answer-item:last-child {
  margin-bottom: 0;
}

.answer-label {
  font-weight: 600;
  margin-right: 8px;
  color: #1890ff;
}

/* 操作按钮样式 */
.action-buttons {
  margin-top: 24px;
  padding-top: 16px;
  border-top: 1px solid #f0f0f0;
}

/* 表格中的操作按钮样式 */
.submissions-section .action-buttons {
  display: flex;
  justify-content: center;
  align-items: center;
  gap: 8px;
  margin-top: 0;
  padding-top: 0;
  border-top: none;
}

.submissions-section .action-buttons .ant-btn-link {
  padding: 4px 8px;
  height: 32px;
  display: flex;
  align-items: center;
  justify-content: center;
}

/* 提交记录相关样式 */
.filter-bar {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 16px;
}

.empty-state {
  padding: 40px 0;
  text-align: center;
}

.empty-tip {
  color: #999;
  margin-top: 16px;
}

/* 批改弹窗样式 */
.grade-form {
  padding: 16px 0;
}

.student-info {
  margin-bottom: 24px;
  padding: 12px;
  background: #f5f5f5;
  border-radius: 4px;
  display: flex;
}

.student-info .info-item {
  margin-right: 24px;
}

.max-score {
  margin-left: 12px;
  color: #999;
}
</style> 