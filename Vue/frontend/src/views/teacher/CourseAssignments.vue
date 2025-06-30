<template>
  <div class="course-assignments">
    <div class="page-header">
      <h2>作业管理</h2>
      <a-button type="primary" @click="showAddAssignmentModal">
        <PlusOutlined />
        添加作业
      </a-button>
    </div>

    <div class="filter-section">
      <div class="filter-row">
        <div class="filter-left">
          <div class="filter-item">
            <span class="filter-label">状态：</span>
            <a-select 
              v-model:value="filters.status" 
              style="width: 120px" 
              placeholder="全部状态"
              allowClear
              @change="handleFilterChange"
            >
              <a-select-option value="not_started">未开始</a-select-option>
              <a-select-option value="in_progress">进行中</a-select-option>
              <a-select-option value="ended">已结束</a-select-option>
            </a-select>
          </div>
          <div class="filter-item">
            <span class="filter-label">类型：</span>
            <a-select 
              v-model:value="filters.type" 
              style="width: 120px" 
              placeholder="全部类型"
              allowClear
              @change="handleFilterChange"
            >
              <a-select-option value="QUIZ">测验</a-select-option>
              <a-select-option value="ESSAY">论文</a-select-option>
              <a-select-option value="PROJECT">项目</a-select-option>
              <a-select-option value="OTHER">其他</a-select-option>
            </a-select>
          </div>
        </div>
        <div class="filter-right">
          <div class="filter-item search-box">
            <a-input-search
              v-model:value="filters.keyword"
              placeholder="搜索作业名称"
              style="width: 250px"
              @search="handleSearch"
              enter-button
            />
          </div>
          <a-button @click="resetFilters" size="middle">
            重置筛选
          </a-button>
        </div>
      </div>
    </div>

    <div class="assignment-content">
      <a-spin :spinning="loading">
        <a-empty v-if="assignments.length === 0" description="暂无作业" />
        
        <a-table
          v-else
          :dataSource="assignments"
          :columns="columns"
          :pagination="pagination"
          :rowKey="(record) => record.id"
          @change="handleTableChange"
        >
          <!-- 作业名称 -->
          <template #bodyCell="{ column, record }">
            <template v-if="column.dataIndex === 'title'">
              <div class="assignment-title">
                <span>{{ record.title }}</span>
                <a-tag v-if="record.type" class="assignment-type-tag">{{ getTypeText(record.type) }}</a-tag>
              </div>
            </template>

            <!-- 作业状态 -->
            <template v-else-if="column.dataIndex === 'status'">
              <a-tag :color="getStatusColor(record.status)">{{ getStatusText(record.status) }}</a-tag>
            </template>

            <!-- 作业时间 -->
            <template v-else-if="column.dataIndex === 'assignmentTime'">
              <div>
                <div>开始：{{ formatDate(record.startTime) }}</div>
                <div>截止：{{ formatDate(record.endTime) }}</div>
              </div>
            </template>

            <!-- 提交情况 -->
            <template v-else-if="column.dataIndex === 'submissionRate'">
              <a-progress 
                :percent="record.submissionRate * 100" 
                :format="(percent: number) => `${Math.round(percent)}%`"
                :status="getSubmissionStatus(record.submissionRate)"
              />
              <div class="submission-count">{{ record.submittedCount }}/{{ record.totalCount }}</div>
            </template>

            <!-- 操作 -->
            <template v-else-if="column.dataIndex === 'action'">
              <div class="assignment-actions">
                <a-tooltip title="查看">
                  <a-button type="link" @click="viewAssignment(record)">
                    <EyeOutlined />
                  </a-button>
                </a-tooltip>
                <a-tooltip title="批改">
                  <a-button type="link" @click="reviewAssignment(record)">
                    <CheckOutlined />
                  </a-button>
                </a-tooltip>
                <a-tooltip title="编辑">
                  <a-button type="link" @click="editAssignment(record)">
                    <EditOutlined />
                  </a-button>
                </a-tooltip>
                <a-tooltip title="删除">
                  <a-popconfirm
                    title="确定要删除这个作业吗？"
                    @confirm="handleDeleteAssignment(record.id)"
                    ok-text="确定"
                    cancel-text="取消"
                  >
                    <a-button type="link" danger>
                      <DeleteOutlined />
                    </a-button>
                  </a-popconfirm>
                </a-tooltip>
              </div>
            </template>
          </template>
        </a-table>
      </a-spin>
    </div>

    <!-- 添加/编辑作业弹窗 -->
    <a-modal
      v-model:open="assignmentModalVisible"
      :title="isEditing ? '编辑作业' : '添加作业'"
      :maskClosable="false"
      @ok="handleSaveAssignment"
      :okButtonProps="{ loading: saving }"
      :okText="saving ? '保存中...' : '保存'"
      width="700px"
    >
      <a-form :model="assignmentForm" layout="vertical">
        <a-form-item label="作业名称" required>
          <a-input v-model:value="assignmentForm.title" placeholder="请输入作业名称" />
        </a-form-item>
        <a-form-item label="作业时间" required>
          <a-range-picker 
            v-model:value="assignmentTimeRange" 
            :show-time="{ format: 'HH:mm' }" 
            format="YYYY-MM-DD HH:mm"
            @change="handleTimeRangeChange"
          />
        </a-form-item>
        <a-form-item label="总分值" required>
          <a-input-number v-model:value="assignmentForm.totalScore" :min="1" :max="100" />
        </a-form-item>
        <a-form-item label="作业类型" required>
          <a-select v-model:value="assignmentForm.type" placeholder="请选择作业类型">
            <a-select-option value="QUIZ">测验</a-select-option>
            <a-select-option value="ESSAY">论文</a-select-option>
            <a-select-option value="PROJECT">项目</a-select-option>
            <a-select-option value="OTHER">其他</a-select-option>
          </a-select>
        </a-form-item>
        <a-form-item label="作业说明">
          <a-textarea v-model:value="assignmentForm.description" placeholder="请输入作业说明" :rows="4" />
        </a-form-item>
      </a-form>
    </a-modal>

    <!-- 查看作业详情弹窗 -->
    <a-modal
      v-model:open="viewModalVisible"
      title="作业详情"
      :footer="null"
      width="700px"
    >
      <div v-if="currentAssignment" class="assignment-detail">
        <div class="assignment-detail-header">
          <a-tag :color="getStatusColor(currentAssignment.status)" class="status-tag">
            {{ getStatusText(currentAssignment.status) }}
          </a-tag>
          <a-tag class="type-tag">{{ getTypeText(currentAssignment.type) }}</a-tag>
        </div>
        
        <div class="assignment-detail-item">
          <div class="assignment-detail-label">作业名称：</div>
          <div class="assignment-detail-value">{{ currentAssignment.title }}</div>
        </div>
        
        <div class="assignment-detail-item">
          <div class="assignment-detail-label">作业时间：</div>
          <div class="assignment-detail-value">
            <div>开始时间：{{ formatDate(currentAssignment.startTime) }}</div>
            <div>截止时间：{{ formatDate(currentAssignment.endTime) }}</div>
          </div>
        </div>
        
        <div class="assignment-detail-item">
          <div class="assignment-detail-label">总分值：</div>
          <div class="assignment-detail-value">{{ currentAssignment.totalScore }} 分</div>
        </div>
        
        <div class="assignment-detail-item" v-if="currentAssignment.description">
          <div class="assignment-detail-label">作业说明：</div>
          <div class="assignment-detail-value">{{ currentAssignment.description }}</div>
        </div>
        
        <div class="assignment-detail-item">
          <div class="assignment-detail-label">提交状态：</div>
          <div class="assignment-detail-value">
            <a-progress 
              :percent="currentAssignment.submissionRate * 100" 
              :format="(percent: number) => `${Math.round(percent)}% (${currentAssignment.submittedCount}/${currentAssignment.totalCount})`" 
              :status="getSubmissionStatus(currentAssignment.submissionRate)"
            />
          </div>
        </div>
        
        <div class="assignment-detail-item">
          <div class="assignment-detail-label">创建时间：</div>
          <div class="assignment-detail-value">{{ formatDate(currentAssignment.createTime) }}</div>
        </div>
        
        <div class="assignment-detail-actions">
          <a-button type="primary" @click="reviewAssignment(currentAssignment)">批改作业</a-button>
          <a-button type="default" @click="editAssignment(currentAssignment)">编辑作业</a-button>
          <a-button @click="viewModalVisible = false">关闭</a-button>
        </div>
      </div>
    </a-modal>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted, watch, defineProps } from 'vue'
import { message } from 'ant-design-vue'
import {
  PlusOutlined,
  EyeOutlined,
  EditOutlined,
  DeleteOutlined,
  CheckOutlined
} from '@ant-design/icons-vue'
import { formatDate } from '@/utils/date'
import axios from 'axios'
import type { Dayjs } from 'dayjs'

const props = defineProps({
  courseId: {
    type: Number,
    required: true
  }
})

// 作业状态
const assignmentStatus = {
  NOT_STARTED: 'not_started',
  IN_PROGRESS: 'in_progress',
  ENDED: 'ended'
}

// 作业类型
const assignmentType = {
  QUIZ: 'QUIZ',
  ESSAY: 'ESSAY',
  PROJECT: 'PROJECT',
  OTHER: 'OTHER'
}

// 状态定义
const assignments = ref<any[]>([])
const loading = ref(false)
const pagination = ref({
  current: 1,
  pageSize: 10,
  total: 0,
  showSizeChanger: true,
  showTotal: (total: number) => `共 ${total} 条`
})

// 筛选条件
const filters = ref({
  status: undefined as string | undefined,
  type: undefined as string | undefined,
  keyword: ''
})

// 表格列定义
const columns = [
  {
    title: '作业名称',
    dataIndex: 'title',
    key: 'title',
    ellipsis: true,
    width: '25%'
  },
  {
    title: '状态',
    dataIndex: 'status',
    key: 'status',
    width: '10%'
  },
  {
    title: '作业时间',
    dataIndex: 'assignmentTime',
    key: 'assignmentTime',
    width: '20%'
  },
  {
    title: '提交情况',
    dataIndex: 'submissionRate',
    key: 'submissionRate',
    width: '20%'
  },
  {
    title: '操作',
    dataIndex: 'action',
    key: 'action',
    width: '15%'
  }
]

// 添加/编辑作业相关状态
const assignmentModalVisible = ref(false)
const isEditing = ref(false)
const saving = ref(false)
const assignmentTimeRange = ref<[Dayjs, Dayjs] | null>(null)
const assignmentForm = ref({
  id: undefined as number | undefined,
  title: '',
  courseId: props.courseId,
  startTime: '',
  endTime: '',
  totalScore: 100,
  type: assignmentType.QUIZ,
  description: '',
  status: assignmentStatus.NOT_STARTED
})

// 查看作业相关状态
const viewModalVisible = ref(false)
const currentAssignment = ref<any | null>(null)

// 监听课程ID变化
watch(() => props.courseId, (newId) => {
  if (newId) {
    fetchAssignments()
  }
})

// 生命周期钩子
onMounted(() => {
  fetchAssignments()
})

// 获取作业列表
const fetchAssignments = async () => {
  loading.value = true
  try {
    // 模拟API请求
    setTimeout(() => {
      // 这里是模拟数据，实际项目中应该从API获取
      assignments.value = [
        {
          id: 1,
          title: '第一次作业',
          courseId: props.courseId,
          startTime: '2025-07-01 09:00:00',
          endTime: '2025-07-07 23:59:59',
          totalScore: 100,
          type: assignmentType.QUIZ,
          description: '完成课本第一章习题',
          status: assignmentStatus.IN_PROGRESS,
          createTime: '2025-06-20 14:30:00',
          submissionRate: 0.75,
          submittedCount: 30,
          totalCount: 40
        },
        {
          id: 2,
          title: '第二次作业',
          courseId: props.courseId,
          startTime: '2025-07-08 09:00:00',
          endTime: '2025-07-14 23:59:59',
          totalScore: 100,
          type: assignmentType.ESSAY,
          description: '网络协议分析报告',
          status: assignmentStatus.NOT_STARTED,
          createTime: '2025-06-22 10:15:00',
          submissionRate: 0,
          submittedCount: 0,
          totalCount: 40
        }
      ]
      pagination.value.total = assignments.value.length
      loading.value = false
    }, 500)
    
    // 实际项目中的API调用示例
    // const res = await axios.get(`/api/teacher/courses/${props.courseId}/assignments`, { params: filters.value })
    // assignments.value = res.data.data.records
    // pagination.value.total = res.data.data.total
  } catch (error) {
    console.error('获取作业列表失败:', error)
    message.error('获取作业列表失败，请稍后再试')
  } finally {
    loading.value = false
  }
}

// 筛选变化处理
const handleFilterChange = () => {
  pagination.value.current = 1
  fetchAssignments()
}

// 搜索处理
const handleSearch = () => {
  pagination.value.current = 1
  fetchAssignments()
}

// 重置筛选条件
const resetFilters = () => {
  filters.value = {
    status: undefined,
    type: undefined,
    keyword: ''
  }
  pagination.value.current = 1
  fetchAssignments()
}

// 表格变化事件
const handleTableChange = (pag: any) => {
  pagination.value.current = pag.current
  pagination.value.pageSize = pag.pageSize
  fetchAssignments()
}

// 作业时间范围变化
const handleTimeRangeChange = (dates: [Dayjs, Dayjs] | null) => {
  if (dates) {
    assignmentForm.value.startTime = dates[0].format('YYYY-MM-DD HH:mm:ss')
    assignmentForm.value.endTime = dates[1].format('YYYY-MM-DD HH:mm:ss')
  } else {
    assignmentForm.value.startTime = ''
    assignmentForm.value.endTime = ''
  }
}

// 显示添加作业弹窗
const showAddAssignmentModal = () => {
  isEditing.value = false
  assignmentForm.value = {
    id: undefined,
    title: '',
    courseId: props.courseId,
    startTime: '',
    endTime: '',
    totalScore: 100,
    type: assignmentType.QUIZ,
    description: '',
    status: assignmentStatus.NOT_STARTED
  }
  assignmentTimeRange.value = null
  assignmentModalVisible.value = true
}

// 查看作业
const viewAssignment = (assignment: any) => {
  currentAssignment.value = assignment
  viewModalVisible.value = true
}

// 编辑作业
const editAssignment = (assignment: any) => {
  isEditing.value = true
  assignmentForm.value = { ...assignment }
  // 这里应该从API获取完整的作业详情
  assignmentTimeRange.value = null // 应该根据startTime和endTime设置
  assignmentModalVisible.value = true
  
  // 如果是从详情弹窗点击的编辑按钮，关闭详情弹窗
  if (viewModalVisible.value) {
    viewModalVisible.value = false
  }
}

// 批改作业
const reviewAssignment = (assignment: any) => {
  message.info(`暂未实现批改功能：${assignment.title}`)
  // 这里应该跳转到批改页面，或者打开批改弹窗
}

// 删除作业
const handleDeleteAssignment = async (id: number) => {
  try {
    // 模拟API请求
    setTimeout(() => {
      message.success('作业删除成功')
      fetchAssignments()
    }, 500)
    
    // 实际项目中的API调用示例
    // await axios.delete(`/api/teacher/courses/${props.courseId}/assignments/${id}`)
    // message.success('作业删除成功')
    // fetchAssignments()
  } catch (error) {
    console.error('作业删除失败:', error)
    message.error('作业删除失败')
  }
}

// 保存作业
const handleSaveAssignment = async () => {
  // 表单验证
  if (!assignmentForm.value.title) {
    message.error('请输入作业名称')
    return
  }
  if (!assignmentForm.value.startTime || !assignmentForm.value.endTime) {
    message.error('请选择作业时间')
    return
  }
  if (!assignmentForm.value.type) {
    message.error('请选择作业类型')
    return
  }

  saving.value = true
  try {
    // 模拟API请求
    setTimeout(() => {
      if (isEditing.value) {
        message.success('作业更新成功')
      } else {
        message.success('作业添加成功')
      }
      assignmentModalVisible.value = false
      fetchAssignments()
      saving.value = false
    }, 1000)
    
    // 实际项目中的API调用示例
    // if (isEditing.value) {
    //   await axios.put(`/api/teacher/courses/${props.courseId}/assignments/${assignmentForm.value.id}`, assignmentForm.value)
    //   message.success('作业更新成功')
    // } else {
    //   await axios.post(`/api/teacher/courses/${props.courseId}/assignments`, assignmentForm.value)
    //   message.success('作业添加成功')
    // }
    // assignmentModalVisible.value = false
    // fetchAssignments()
  } catch (error) {
    console.error('保存作业失败:', error)
    message.error('保存作业失败，请稍后再试')
  } finally {
    saving.value = false
  }
}

// 获取提交状态
const getSubmissionStatus = (rate: number): string => {
  if (rate === 0) return 'exception';
  if (rate < 0.5) return 'active';
  if (rate < 1) return 'normal';
  return 'success';
}

// 获取状态显示文本
const getStatusText = (status: string): string => {
  const statusMap: Record<string, string> = {
    [assignmentStatus.NOT_STARTED]: '未开始',
    [assignmentStatus.IN_PROGRESS]: '进行中',
    [assignmentStatus.ENDED]: '已结束'
  }
  return statusMap[status] || '未知状态'
}

// 获取状态标签颜色
const getStatusColor = (status: string): string => {
  const colorMap: Record<string, string> = {
    [assignmentStatus.NOT_STARTED]: 'blue',
    [assignmentStatus.IN_PROGRESS]: 'green',
    [assignmentStatus.ENDED]: 'gray'
  }
  return colorMap[status] || 'default'
}

// 获取作业类型文本
const getTypeText = (type: string): string => {
  const typeMap: Record<string, string> = {
    [assignmentType.QUIZ]: '测验',
    [assignmentType.ESSAY]: '论文',
    [assignmentType.PROJECT]: '项目',
    [assignmentType.OTHER]: '其他'
  }
  return typeMap[type] || '未知类型'
}
</script>

<style scoped>
.course-assignments {
  padding: 24px;
  background-color: #fff;
}

.page-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 24px;
}

.page-header h2 {
  margin: 0;
  font-size: 20px;
  font-weight: 600;
}

.filter-section {
  background-color: #f5f7fa;
  padding: 16px;
  border-radius: 8px;
  margin-bottom: 24px;
}

.filter-row {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  justify-content: space-between;
}

.filter-left {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
}

.filter-right {
  display: flex;
  align-items: center;
  gap: 16px;
}

.filter-item {
  display: flex;
  align-items: center;
  margin-right: 24px;
  margin-bottom: 8px;
}

.filter-label {
  margin-right: 8px;
  white-space: nowrap;
}

.search-box {
  flex-grow: 1;
}

.assignment-content {
  background-color: #fff;
}

.assignment-title {
  display: flex;
  align-items: center;
  gap: 8px;
}

.assignment-type-tag {
  font-size: 12px;
  line-height: 1;
  padding: 2px 6px;
  border-radius: 4px;
  background-color: #f5f5f5;
  color: #666;
}

.assignment-actions {
  display: flex;
  gap: 8px;
}

.submission-count {
  font-size: 12px;
  color: #999;
  margin-top: 4px;
  text-align: center;
}

.assignment-detail {
  padding: 16px;
}

.assignment-detail-header {
  display: flex;
  align-items: center;
  gap: 16px;
  margin-bottom: 20px;
  padding-bottom: 12px;
  border-bottom: 1px solid #f0f0f0;
}

.status-tag, .type-tag {
  font-size: 14px;
  padding: 2px 12px;
}

.assignment-detail-item {
  margin-bottom: 16px;
}

.assignment-detail-label {
  font-weight: 600;
  margin-bottom: 8px;
}

.assignment-detail-value {
  white-space: pre-line;
}

.assignment-detail-actions {
  display: flex;
  justify-content: flex-end;
  gap: 12px;
  margin-top: 24px;
  padding-top: 16px;
  border-top: 1px solid #f0f0f0;
}
</style> 