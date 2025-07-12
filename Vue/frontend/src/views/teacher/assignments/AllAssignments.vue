<template>
  <div class="all-assignments">
    <div class="assignments-wrapper">
      <a-table
        :dataSource="assignments"
        :columns="columns"
        :pagination="pagination"
        :loading="loading"
        @change="handleTableChange"
        rowKey="id"
      >
        <!-- 作业名称 -->
        <template #bodyCell="{ column, record }">
          <template v-if="column.dataIndex === 'title'">
            <div class="assignment-title">
              <span>{{ record.title }}</span>
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
              <div>结束：{{ formatDate(record.endTime) }}</div>
            </div>
          </template>

          <!-- 提交情况 -->
          <template v-else-if="column.dataIndex === 'submissionRate'">
            <a-progress
              :percent="record.submissionRate || 0"
              size="small"
              :status="getSubmissionStatus(record.submissionRate)"
            />
            <div class="submission-text">
              {{ record.submissionRate || 0 }}% 已提交
            </div>
          </template>

          <!-- 操作 -->
          <template v-else-if="column.dataIndex === 'action'">
            <div class="action-buttons">
              <a-button type="primary" size="small" @click="viewAssignment(record)">
                <EyeOutlined />
                查看
              </a-button>
              <a-button type="default" size="small" @click="editAssignment(record)">
                <EditOutlined />
                编辑
              </a-button>
              <a-popconfirm
                title="确定要删除此作业吗？"
                ok-text="确定"
                cancel-text="取消"
                @confirm="deleteAssignment(record.id)"
              >
                <a-button type="danger" size="small">
                  <DeleteOutlined />
                  删除
                </a-button>
              </a-popconfirm>
            </div>
          </template>
        </template>
      </a-table>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, reactive, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { message } from 'ant-design-vue'
import { EyeOutlined, EditOutlined, DeleteOutlined } from '@ant-design/icons-vue'
import dayjs from 'dayjs'
import axios from 'axios'

// 路由实例
const router = useRouter()

// 状态
const loading = ref(false)
const assignments = ref([])
const pagination = reactive({
  current: 1,
  pageSize: 10,
  total: 0,
  showSizeChanger: true,
  showQuickJumper: true,
  showTotal: (total: number) => `共 ${total} 条记录`
})

// 表格列定义
const columns = [
  {
    title: '作业名称',
    dataIndex: 'title',
    key: 'title',
    width: '20%'
  },
  {
    title: '所属课程',
    dataIndex: 'courseName',
    key: 'courseName',
    width: '15%'
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
    width: '15%'
  },
  {
    title: '操作',
    dataIndex: 'action',
    key: 'action',
    width: '20%'
  }
]

// 生命周期钩子
onMounted(() => {
  fetchAssignments()
})

// 获取作业列表
const fetchAssignments = async () => {
  loading.value = true
  try {
    const response = await axios.get('/api/teacher/assignments', {
      params: {
        pageNum: pagination.current,
        pageSize: pagination.pageSize
      }
    })
    
    if (response.data && response.data.code === 200) {
      assignments.value = response.data.data.list || []
      pagination.total = response.data.data.total || 0
    } else {
      message.error(response.data?.message || '获取作业列表失败')
    }
  } catch (error) {
    console.error('获取作业列表失败:', error)
    message.error('获取作业列表失败，请重试')
  } finally {
    loading.value = false
  }
}

// 表格变化事件
const handleTableChange = (pag: any) => {
  pagination.current = pag.current
  pagination.pageSize = pag.pageSize
  fetchAssignments()
}

// 格式化日期
const formatDate = (dateString: string) => {
  if (!dateString) return '未设置'
  return dayjs(dateString).format('YYYY-MM-DD HH:mm')
}

// 获取状态颜色
const getStatusColor = (status: string | number) => {
  if (typeof status === 'number') {
    // 数字状态码
    switch (status) {
      case 0: return 'default' // 未发布
      case 1: return 'green'   // 已发布
      default: return 'default'
    }
  } else {
    // 字符串状态
    switch (status) {
      case 'not_started': return 'orange'
      case 'in_progress': return 'blue'
      case 'ended': return 'green'
      default: return 'default'
    }
  }
}

// 获取状态文本
const getStatusText = (status: string | number) => {
  if (typeof status === 'number') {
    // 数字状态码
    switch (status) {
      case 0: return '未发布'
      case 1: return '已发布'
      default: return '未知状态'
    }
  } else {
    // 字符串状态
    switch (status) {
      case 'not_started': return '未开始'
      case 'in_progress': return '进行中'
      case 'ended': return '已结束'
      default: return '未知状态'
    }
  }
}

// 获取提交状态
const getSubmissionStatus = (rate: number) => {
  if (rate >= 80) return 'success'
  if (rate >= 40) return 'active'
  return 'exception'
}

// 查看作业
const viewAssignment = (assignment: any) => {
  router.push(`/teacher/assignments/${assignment.id}`)
}

// 编辑作业
const editAssignment = (assignment: any) => {
  router.push(`/teacher/assignments/${assignment.id}/edit`)
}

// 删除作业
const deleteAssignment = async (id: number) => {
  try {
    const response = await axios.delete(`/api/teacher/assignments/${id}`)
    if (response.data && response.data.code === 200) {
      message.success('删除成功')
      fetchAssignments()
    } else {
      message.error(response.data?.message || '删除失败')
    }
  } catch (error) {
    console.error('删除作业失败:', error)
    message.error('删除作业失败，请重试')
  }
}
</script>

<style scoped>
.all-assignments {
  padding: 20px 0;
}

.assignments-wrapper {
  background-color: #fff;
  padding: 24px;
  border-radius: 4px;
  box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);
}

.assignment-title {
  font-weight: 500;
}

.action-buttons {
  display: flex;
  gap: 8px;
}

.submission-text {
  font-size: 12px;
  color: rgba(0, 0, 0, 0.65);
  margin-top: 4px;
}
</style> 