<template>
  <div class="assignments">
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
          :rowKey="(record: any) => record.id"
          @change="handleTableChange"
        >
          <!-- 作业名称 -->
          <template #bodyCell="{ column, record }">
            <template v-if="column.dataIndex === 'title'">
              <div class="assignment-title">
                <span>{{ record.title }}</span>
              </div>
            </template>

            <!-- 作业状态 -->
            <template v-else-if="column.dataIndex === 'assignmentState'">
              <a-tag :color="getStatusColor(record.assignmentState)">{{ getStatusText(record.assignmentState) }}</a-tag>
            </template>

            <!-- 作业时间 -->
            <template v-else-if="column.dataIndex === 'assignmentTime'">
              <div>
                <div>开始：{{ formatDate(record.startTime) }}</div>
                <div>结束：{{ formatDate(record.endTime) }}</div>
              </div>
            </template>

            <!-- 发布状态 -->
            <template v-else-if="column.dataIndex === 'publishStatus'">
              <a-tag :color="record.status === 0 ? 'orange' : 'green'">
                {{ record.status === 0 ? '未发布' : '已发布' }}
              </a-tag>
            </template>
            
            <!-- 操作 -->
            <template v-else-if="column.dataIndex === 'action'">
              <div class="assignment-actions">
                <a-tooltip title="查看">
                  <a-button type="link" @click="viewAssignmentDetail(record)">
                    <EyeOutlined />
                  </a-button>
                </a-tooltip>
                <a-tooltip title="编辑">
                  <a-button type="link" @click="editAssignment(record)">
                    <EditOutlined />
                  </a-button>
                </a-tooltip>
                <a-tooltip title="发布" v-if="record.status === 0">
                  <a-button type="link" @click="publishAssignment(record)" style="color: #52c41a">
                    <CheckOutlined />
                  </a-button>
                </a-tooltip>
                <a-tooltip title="取消发布" v-if="record.status === 1">
                  <a-button type="link" @click="unpublishAssignment(record)" style="color: #faad14">
                    <CloseCircleOutlined />
                  </a-button>
                </a-tooltip>
                <a-tooltip title="删除">
                  <a-popconfirm
                    title="确定要删除这个作业吗？"
                    description="删除后将无法恢复，包括作业题目关联数据也会被删除。"
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
  </div>
</template>

<script setup lang="ts">
import { ref, reactive, onMounted, computed, defineComponent } from 'vue'
import { useRouter } from 'vue-router'
import { message, notification } from 'ant-design-vue'
import dayjs from 'dayjs'
import assignment from '@/api/assignment'

// 定义组件名称，便于调试
defineComponent({
  name: 'TeacherAssignments'
})

import {
  PlusOutlined,
  EditOutlined,
  DeleteOutlined,
  EyeOutlined,
  CheckOutlined,
  CloseCircleOutlined
} from '@ant-design/icons-vue'

const router = useRouter()

// 表格列定义
const columns = [
  {
    title: '作业名称',
    dataIndex: 'title',
    key: 'title',
    width: '25%'
  },
  {
    title: '状态',
    dataIndex: 'assignmentState',
    key: 'assignmentState',
    width: '15%'
  },
  {
    title: '作业时间',
    dataIndex: 'assignmentTime',
    key: 'assignmentTime',
    width: '20%'
  },
  {
    title: '发布状态',
    dataIndex: 'publishStatus',
    key: 'publishStatus',
    width: '15%'
  },
  {
    title: '操作',
    dataIndex: 'action',
    key: 'action',
    width: '25%'
  }
]

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

// 筛选条件
const filters = reactive({
  status: undefined,
  keyword: ''
})

// 加载作业列表
const loadAssignments = async () => {
  loading.value = true
  try {
    const params = {
      current: pagination.current,
      pageSize: pagination.pageSize,
      keyword: filters.keyword,
      // 根据筛选条件的状态值转换为对应的数字状态码
      status: filters.status ? undefined : undefined
    }
    
    const response = await assignment.getAssignmentList(params)
    
    if (response.code === 200) {
      const { records, total, current, size } = response.data
      
      // 处理每个作业记录，添加状态字段
      assignments.value = records.map((item: any) => {
        const now = dayjs()
        const startTime = dayjs(item.startTime)
        const endTime = dayjs(item.endTime)
        
        let assignmentState = 'not_started'
        if (now.isAfter(endTime)) {
          assignmentState = 'ended'
        } else if (now.isAfter(startTime)) {
          assignmentState = 'in_progress'
        }
        
        return {
          ...item,
          assignmentState
        }
      })
      
      pagination.total = total
      pagination.current = current
      pagination.pageSize = size
    } else {
      message.error(response.message || '获取作业列表失败')
    }
  } catch (error) {
    console.error('获取作业列表出错:', error)
    message.error('获取作业列表失败，请检查网络连接')
  } finally {
    loading.value = false
  }
}

// 格式化日期
const formatDate = (dateStr: string) => {
  if (!dateStr) return ''
  return dayjs(dateStr).format('YYYY-MM-DD HH:mm')
}

// 获取状态文本
const getStatusText = (status: string) => {
  const statusMap: Record<string, string> = {
    'not_started': '未开始',
    'in_progress': '进行中',
    'ended': '已结束'
  }
  return statusMap[status] || '未知状态'
}

// 获取状态颜色
const getStatusColor = (status: string) => {
  const colorMap: Record<string, string> = {
    'not_started': 'blue',
    'in_progress': 'green',
    'ended': 'gray'
  }
  return colorMap[status] || 'default'
}

// 处理表格变更（分页、排序等）
const handleTableChange = (pag: any) => {
  pagination.current = pag.current
  pagination.pageSize = pag.pageSize
  loadAssignments()
}

// 处理筛选条件变更
const handleFilterChange = () => {
  pagination.current = 1
  loadAssignments()
}

// 处理搜索
const handleSearch = () => {
  pagination.current = 1
  loadAssignments()
}

// 重置筛选条件
const resetFilters = () => {
  filters.status = undefined
  filters.keyword = ''
  pagination.current = 1
  loadAssignments()
}

// 发布作业
const publishAssignment = async (record: any) => {
  try {
    const response = await assignment.publishAssignment(record.id)
    if (response.code === 200) {
      message.success('作业发布成功')
      loadAssignments()
    } else {
      message.error(response.message || '作业发布失败')
    }
  } catch (error) {
    console.error('发布作业出错:', error)
    message.error('作业发布失败，请检查网络连接')
  }
}

// 取消发布作业
const unpublishAssignment = async (record: any) => {
  try {
    const response = await assignment.unpublishAssignment(record.id)
    if (response.code === 200) {
      message.success('取消发布成功')
      loadAssignments()
    } else {
      message.error(response.message || '取消发布失败')
    }
  } catch (error) {
    console.error('取消发布作业出错:', error)
    message.error('取消发布失败，请检查网络连接')
  }
}

// 删除作业
const handleDeleteAssignment = async (id: number) => {
  try {
    const response = await assignment.deleteAssignment(id)
    if (response.code === 200) {
      message.success('作业删除成功')
      loadAssignments()
    } else {
      message.error(response.message || '作业删除失败')
    }
  } catch (error) {
    console.error('删除作业出错:', error)
    message.error('作业删除失败，请检查网络连接')
  }
}

// 查看作业详情
const viewAssignmentDetail = (record: any) => {
  router.push(`/teacher/assignments/${record.id}`)
}

// 编辑作业
const editAssignment = (record: any) => {
  router.push(`/teacher/assignments/${record.id}/edit`)
}

// 添加作业
const showAddAssignmentModal = () => {
  router.push('/teacher/assignments/create')
}

// 初始化
onMounted(() => {
  loadAssignments()
})
</script>

<style scoped>
.assignments {
  padding: 24px;
}

.page-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 20px;
}

.page-header h2 {
  margin-bottom: 0;
  font-size: 20px;
  font-weight: 600;
}

.filter-section {
  background: #f5f5f5;
  padding: 16px;
  border-radius: 4px;
  margin-bottom: 20px;
}

.filter-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.filter-left {
  display: flex;
  gap: 16px;
}

.filter-right {
  display: flex;
  gap: 16px;
}

.filter-item {
  display: flex;
  align-items: center;
}

.filter-label {
  margin-right: 8px;
  white-space: nowrap;
}

.search-box {
  margin-right: 8px;
}

.assignment-content {
  background: #fff;
  padding: 20px;
  border-radius: 4px;
}

.assignment-title {
  font-weight: 500;
}

.assignment-actions {
  display: flex;
  justify-content: space-around;
}
</style> 