<template>
  <div class="exam-management">
    <div class="page-header">
      <h2>考试管理</h2>
      <a-button type="primary" @click="showAddExamModal">
        <PlusOutlined />
        添加考试
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
            <span class="filter-label">课程：</span>
            <a-select 
              v-model:value="filters.courseId" 
              style="width: 180px" 
              placeholder="全部课程"
              allowClear
              @change="handleFilterChange"
            >
              <a-select-option v-for="course in courses" :key="course.id" :value="course.id">
                {{ course.name }}
              </a-select-option>
            </a-select>
          </div>
        </div>
        <div class="filter-right">
          <div class="filter-item search-box">
            <a-input-search
              v-model:value="filters.keyword"
              placeholder="搜索考试名称"
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

    <div class="exam-content">
      <a-spin :spinning="loading">
        <a-empty v-if="exams.length === 0" description="暂无考试" />
        
        <a-table
          v-else
          :dataSource="exams"
          :columns="columns"
          :pagination="pagination"
          :rowKey="(record: any) => record.id"
          @change="handleTableChange"
        >
          <!-- 所属课程 -->
          <template #bodyCell="{ column, record }">
            <template v-if="column.dataIndex === 'courseName'">
              {{ record.courseName || (record.courseId ? `课程ID: ${record.courseId}` : '未关联课程') }}
            </template>

            <!-- 考试状态 -->
            <template v-else-if="column.dataIndex === 'status'">
              <a-tag :color="getStatusColor(record.status)">{{ getStatusText(record.status) }}</a-tag>
            </template>

            <!-- 考试时间 -->
            <template v-else-if="column.dataIndex === 'examTime'">
              <div>
                <div>开始：{{ formatDate(record.startTime) }}</div>
                <div>结束：{{ formatDate(record.endTime) }}</div>
              </div>
            </template>
            
            <!-- 考试时长 -->
            <template v-else-if="column.dataIndex === 'duration'">
              {{ record.duration }} 分钟
            </template>

            <!-- 操作 -->
            <template v-else-if="column.dataIndex === 'action'">
              <div class="exam-actions">
                <a-tooltip title="查看">
                  <a-button type="link" @click="viewExam(record)">
                    <EyeOutlined />
                  </a-button>
                </a-tooltip>
                <a-tooltip title="编辑">
                  <a-button type="link" @click="editExam(record)">
                    <EditOutlined />
                  </a-button>
                </a-tooltip>
                <a-tooltip title="删除">
                  <a-popconfirm
                    title="确定要删除这个考试吗？"
                    @confirm="handleDeleteExam(record.id)"
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

    <!-- 添加/编辑考试弹窗 -->
    <a-modal
      v-model:visible="examModalVisible"
      :title="isEditing ? '编辑考试' : '添加考试'"
      :confirmLoading="saving"
      @ok="handleSaveExam"
      width="700px"
    >
      <a-form :model="examForm" layout="vertical">
        <a-form-item label="考试名称" required>
          <a-input v-model:value="examForm.title" placeholder="请输入考试名称" />
        </a-form-item>
        <a-form-item label="所属课程" required>
          <a-select 
            v-model:value="examForm.courseId" 
            placeholder="请选择课程"
            style="width: 100%"
          >
            <a-select-option v-for="course in courses" :key="course.id" :value="course.id">
              {{ course.name }}
            </a-select-option>
          </a-select>
        </a-form-item>
        <a-form-item label="考试时间" required>
          <a-range-picker 
            v-model:value="examTimeRange" 
            :show-time="{ format: 'HH:mm' }" 
            format="YYYY-MM-DD HH:mm"
            @change="handleTimeRangeChange"
            style="width: 100%"
          />
        </a-form-item>
        <a-form-item label="考试说明">
          <a-textarea v-model:value="examForm.description" placeholder="请输入考试说明" :rows="4" />
        </a-form-item>
      </a-form>
    </a-modal>

    <!-- 查看考试详情弹窗 -->
    <a-modal
      v-model:visible="viewModalVisible"
      title="考试详情"
      :footer="null"
      width="700px"
    >
      <div v-if="currentExam" class="exam-detail">
        <div class="exam-detail-header">
          <a-tag :color="getStatusColor(currentExam.status)" class="status-tag">
            {{ getStatusText(currentExam.status) }}
          </a-tag>
        </div>
        
        <div class="exam-detail-item">
          <div class="exam-detail-label">考试名称：</div>
          <div class="exam-detail-value">{{ currentExam.title }}</div>
        </div>
        
        <div class="exam-detail-item">
          <div class="exam-detail-label">所属课程：</div>
          <div class="exam-detail-value">{{ currentExam.courseName }}</div>
        </div>
        
        <div class="exam-detail-item">
          <div class="exam-detail-label">考试时间：</div>
          <div class="exam-detail-value">
            <div>开始时间：{{ formatDate(currentExam.startTime) }}</div>
            <div>结束时间：{{ formatDate(currentExam.endTime) }}</div>
          </div>
        </div>
        
        <div class="exam-detail-item">
          <div class="exam-detail-label">考试时长：</div>
          <div class="exam-detail-value">{{ currentExam.duration }} 分钟</div>
        </div>
        
        <div class="exam-detail-item">
          <div class="exam-detail-label">总分值：</div>
          <div class="exam-detail-value">{{ currentExam.totalScore }} 分</div>
        </div>
        
        <div class="exam-detail-item" v-if="currentExam.description">
          <div class="exam-detail-label">考试说明：</div>
          <div class="exam-detail-value">{{ currentExam.description }}</div>
        </div>
        
        <div class="exam-detail-item">
          <div class="exam-detail-label">创建时间：</div>
          <div class="exam-detail-value">{{ formatDate(String(currentExam.createTime)) }}</div>
        </div>
      </div>
    </a-modal>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { message } from 'ant-design-vue'
import {
  PlusOutlined,
  EyeOutlined,
  EditOutlined,
  DeleteOutlined
} from '@ant-design/icons-vue'
import { formatDate } from '@/utils/date'
import axios from 'axios'
import type { Dayjs } from 'dayjs'
import request from '@/utils/request'
import type { ApiResponse } from '@/utils/request'
import { getTeacherCourses, type Course } from '@/api/teacher'

// 定义考试接口
interface Exam {
  id: number;
  title: string;
  courseId?: number;
  courseName?: string;
  startTime: string | Date | any[];
  endTime: string | Date | any[];
  duration?: number;
  totalScore?: number;
  description?: string;
  status: string;
  createTime?: string | Date | any[];
  updateTime?: string | Date | any[];
}

// 考试状态
const examStatus = {
  NOT_STARTED: 'not_started',
  IN_PROGRESS: 'in_progress',
  ENDED: 'ended'
}

// 状态定义
const exams = ref<Exam[]>([])
const courses = ref<{id: number, name: string}[]>([])
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
  courseId: undefined as number | undefined,
  keyword: ''
})

// 表格列定义
const columns = [
  {
    title: '考试名称',
    dataIndex: 'title',
    key: 'title',
    ellipsis: true,
    width: '25%'
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
    title: '考试时间',
    dataIndex: 'examTime',
    key: 'examTime',
    width: '20%'
  },
  {
    title: '考试时长',
    dataIndex: 'duration',
    key: 'duration',
    width: '10%'
  },
  {
    title: '操作',
    dataIndex: 'action',
    key: 'action',
    width: '15%'
  }
]

// 添加/编辑考试相关状态
const examModalVisible = ref(false)
const isEditing = ref(false)
const saving = ref(false)
const examTimeRange = ref<[Dayjs, Dayjs] | null>(null)
const examForm = ref({
  id: undefined as number | undefined,
  title: '',
  courseId: undefined as number | undefined,
  startTime: '',
  endTime: '',
  duration: 60,
  totalScore: 100,
  description: '',
  status: examStatus.NOT_STARTED
})

// 查看考试相关状态
const viewModalVisible = ref(false)
const currentExam = ref<any | null>(null)

// 生命周期钩子
onMounted(() => {
  // 先获取课程列表，再获取考试列表
  fetchCourses().then(() => {
  fetchExams()
  })
})

// 更新考试的课程名称
const updateExamCourseNames = () => {
  exams.value.forEach(exam => {
    if (exam.courseId && !exam.courseName) {
      const course = courses.value.find(c => c.id === exam.courseId);
      exam.courseName = course ? course.name : `课程ID: ${exam.courseId}`;
    }
  });
}

// 获取考试列表
const fetchExams = async () => {
  loading.value = true;
  try {
    const params = {
      courseId: filters.value.courseId,
      status: filters.value.status,
      keyword: filters.value.keyword,
      current: pagination.value.current,
      pageSize: pagination.value.pageSize
    };
    
    console.log('发送请求参数:', params);

    const response = await axios.get('/api/teacher/exams', { params });
    console.log('API响应:', response.data);
    
    if (response.data && response.data.code === 200) {
      const { records, total } = response.data.data;
      console.log('原始考试数据:', records);
      
      // 处理考试数据
      exams.value = records.map((exam: any) => {
        // 计算考试状态
        const now = new Date().getTime();
        const startTime = Array.isArray(exam.startTime) 
          ? new Date(exam.startTime[0], exam.startTime[1]-1, exam.startTime[2], exam.startTime[3], exam.startTime[4], exam.startTime[5]).getTime()
          : new Date(exam.startTime).getTime();
        const endTime = Array.isArray(exam.endTime)
          ? new Date(exam.endTime[0], exam.endTime[1]-1, exam.endTime[2], exam.endTime[3], exam.endTime[4], exam.endTime[5]).getTime()
          : new Date(exam.endTime).getTime();
        
        console.log(`考试ID: ${exam.id}, 标题: ${exam.title}, 开始时间: ${startTime}, 结束时间: ${endTime}, 当前时间: ${now}`);
        
        let status;
        if (now < startTime) {
          status = examStatus.NOT_STARTED;
          console.log(`考试 ${exam.id} 状态: 未开始`);
        } else if (now >= startTime && now <= endTime) {
          status = examStatus.IN_PROGRESS;
          console.log(`考试 ${exam.id} 状态: 进行中`);
        } else {
          status = examStatus.ENDED;
          console.log(`考试 ${exam.id} 状态: 已结束`);
        }
        
        // 计算考试时长（分钟）
        const durationMinutes = Math.round((endTime - startTime) / (1000 * 60));
        
        return {
          ...exam,
          status,
          duration: exam.duration || durationMinutes,
          examTime: {
            startTime: exam.startTime,
            endTime: exam.endTime
          }
        };
      });
      
      pagination.value.total = total;
      console.log('处理后的考试数据:', exams.value);
      
      // 更新考试的课程名称
      updateExamCourseNames();
      
      return Promise.resolve();
    } else {
      message.error(response.data?.message || '获取考试列表失败');
      return Promise.reject(response.data?.message || '获取考试列表失败');
    }
  } catch (error) {
    console.error('获取考试列表失败:', error);
    message.error('获取考试列表失败，请稍后再试');
    return Promise.reject(error);
  } finally {
    loading.value = false;
  }
}

// 获取课程列表
const fetchCourses = async () => {
  try {
    const response = await getTeacherCourses();
    
    if (response && response.data) {
      // 处理API返回的数据
      let coursesData = [];
      
      // 根据返回数据结构处理
      if (Array.isArray(response.data)) {
        coursesData = response.data;
      } else if (response.data.records || response.data.content || response.data.list) {
        coursesData = response.data.records || response.data.content || response.data.list;
      } else if (response.data.code === 200 && response.data.data) {
        if (Array.isArray(response.data.data)) {
          coursesData = response.data.data;
        } else if (response.data.data.records || response.data.data.content || response.data.data.list) {
          coursesData = response.data.data.records || response.data.data.content || response.data.data.list;
        }
      }
      
      // 格式化课程数据
      courses.value = coursesData.map((course: Course) => ({
        id: course.id,
        name: course.title || course.courseName || '未命名课程'
      }));
      
      console.log('获取到课程列表:', courses.value);
      return Promise.resolve();
    } else {
      message.error('获取课程列表失败');
      return Promise.reject('获取课程列表失败');
    }
  } catch (error) {
    console.error('获取课程列表失败:', error);
    message.error('获取课程列表失败');
    return Promise.reject(error);
  }
}

// 筛选变化处理
const handleFilterChange = () => {
  console.log('筛选条件变化:', filters.value);
  pagination.value.current = 1
  fetchExams().then(() => {
    updateExamCourseNames()
  })
}

// 搜索处理
const handleSearch = () => {
  pagination.value.current = 1
  fetchExams().then(() => {
    updateExamCourseNames()
  })
}

// 重置筛选条件
const resetFilters = () => {
  filters.value = {
    status: undefined,
    courseId: undefined,
    keyword: ''
  }
  pagination.value.current = 1
  fetchExams().then(() => {
    updateExamCourseNames()
  })
}

// 表格变化事件
const handleTableChange = (pag: any) => {
  pagination.value.current = pag.current
  pagination.value.pageSize = pag.pageSize
  fetchExams()
}

// 考试时间范围变化
const handleTimeRangeChange = (dates: [Dayjs, Dayjs] | null) => {
  if (dates) {
    examForm.value.startTime = dates[0].format('YYYY-MM-DD HH:mm:ss')
    examForm.value.endTime = dates[1].format('YYYY-MM-DD HH:mm:ss')
    
    // 自动计算考试时长（分钟）
    const startTime = dates[0]
    const endTime = dates[1]
    const durationMinutes = endTime.diff(startTime, 'minute')
    examForm.value.duration = durationMinutes
  } else {
    examForm.value.startTime = ''
    examForm.value.endTime = ''
    examForm.value.duration = 60 // 默认60分钟
  }
}

// 显示添加考试弹窗
const showAddExamModal = () => {
  isEditing.value = false
  examForm.value = {
    id: undefined,
    title: '',
    courseId: undefined,
    startTime: '',
    endTime: '',
    duration: 60,
    totalScore: 100,
    description: '',
    status: examStatus.NOT_STARTED
  }
  examTimeRange.value = null
  examModalVisible.value = true
}

// 查看考试
const viewExam = (exam: any) => {
  currentExam.value = exam
  viewModalVisible.value = true
}

// 编辑考试
const editExam = (exam: any) => {
  isEditing.value = true
  examForm.value = { ...exam }
  // 这里应该从API获取完整的考试详情
  examTimeRange.value = null // 应该根据startTime和endTime设置
  examModalVisible.value = true
}

// 删除考试
const handleDeleteExam = async (id: number) => {
  try {
    // 模拟API请求
    setTimeout(() => {
      message.success('考试删除成功')
      fetchExams()
    }, 500)
    
    // 实际项目中的API调用示例
    // await axios.delete(`/api/teacher/exams/${id}`)
    // message.success('考试删除成功')
    // fetchExams()
  } catch (error) {
    console.error('考试删除失败:', error)
    message.error('考试删除失败')
  }
}

// 保存考试
const handleSaveExam = async () => {
  // 表单验证
  if (!examForm.value.title) {
    message.error('请输入考试名称')
    return
  }
  if (!examForm.value.courseId) {
    message.error('请选择所属课程')
    return
  }
  if (!examForm.value.startTime || !examForm.value.endTime) {
    message.error('请选择考试时间')
    return
  }

  saving.value = true
  try {
    // 模拟API请求
    setTimeout(() => {
      if (isEditing.value) {
        message.success('考试更新成功')
      } else {
        message.success('考试添加成功')
      }
      examModalVisible.value = false
      fetchExams()
      saving.value = false
    }, 1000)
    
    // 实际项目中的API调用示例
    // if (isEditing.value) {
    //   await axios.put('/api/teacher/exams', examForm.value)
    //   message.success('考试更新成功')
    // } else {
    //   await axios.post('/api/teacher/exams', examForm.value)
    //   message.success('考试添加成功')
    // }
    // examModalVisible.value = false
    // fetchExams()
  } catch (error) {
    console.error('保存考试失败:', error)
    message.error('保存考试失败，请稍后再试')
  } finally {
    saving.value = false
  }
}

// 获取状态显示文本
const getStatusText = (status: string): string => {
  const statusMap: Record<string, string> = {
    [examStatus.NOT_STARTED]: '未开始',
    [examStatus.IN_PROGRESS]: '进行中',
    [examStatus.ENDED]: '已结束'
  }
  return statusMap[status] || '未知状态'
}

// 获取状态标签颜色
const getStatusColor = (status: string): string => {
  const colorMap: Record<string, string> = {
    [examStatus.NOT_STARTED]: 'blue',
    [examStatus.IN_PROGRESS]: 'green',
    [examStatus.ENDED]: 'gray'
  }
  return colorMap[status] || 'default'
}
</script>

<style scoped>
.exam-management {
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

.exam-content {
  background-color: #fff;
}

.exam-actions {
  display: flex;
  gap: 8px;
}

.exam-detail {
  padding: 16px;
}

.exam-detail-header {
  display: flex;
  align-items: center;
  gap: 16px;
  margin-bottom: 20px;
  padding-bottom: 12px;
  border-bottom: 1px solid #f0f0f0;
}

.status-tag {
  font-size: 14px;
  padding: 2px 12px;
}

.exam-detail-item {
  margin-bottom: 16px;
}

.exam-detail-label {
  font-weight: 600;
  margin-bottom: 8px;
}

.exam-detail-value {
  white-space: pre-line;
}
</style> 