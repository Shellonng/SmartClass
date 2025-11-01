<template>
  <div class="teacher-courses">
    <!-- 页面头部 -->
    <div class="page-header">
      <div class="header-content">
        <h1 class="page-title">
          <BookOutlined />
          <span v-if="!isStudent">课程管理</span>
          <span v-else>我的课程</span>
        </h1>
        <p class="page-description" v-if="!isStudent">管理您的教学课程，掌控教学进度</p>
        <p class="page-description" v-else>探索知识海洋，开启学习之旅</p>
      </div>
      <div class="header-actions">
      <a-button type="primary" @click="showCreateModal = true" v-if="!isStudent">
        <PlusOutlined />
        新建课程
      </a-button>
        <a-button @click="exportCourses">
          <DownloadOutlined />
          导出课程
        </a-button>
      </div>
    </div>

    <!-- 课程概览 -->
    <div class="courses-overview">
      <div class="overview-card total">
        <div class="card-icon">
          <BookOutlined />
        </div>
        <div class="card-content">
          <div class="card-title">总课程数</div>
          <div class="card-value">{{ totalCourses }}</div>
          <div class="card-subtitle">本学期 {{ activeCourses }} 门</div>
        </div>
      </div>

      <div class="overview-card students">
        <div class="card-icon">
          <TeamOutlined />
        </div>
        <div class="card-content">
          <div class="card-title">学生总数</div>
          <div class="card-value">{{ totalStudents }}</div>
          <div class="card-subtitle">平均 {{ averageStudentsPerCourse }} 人/课程</div>
        </div>
      </div>

      <div class="overview-card progress">
        <div class="card-icon">
          <ClockCircleOutlined />
        </div>
        <div class="card-content">
          <div class="card-title">平均进度</div>
          <div class="card-value">{{ averageProgress }}%</div>
          <div class="card-subtitle">{{ completedLessons }}/{{ totalLessons }} 课时</div>
        </div>
      </div>

      <div class="overview-card performance">
        <div class="card-icon">
          <TrophyOutlined />
        </div>
        <div class="card-content">
          <div class="card-title">平均成绩</div>
          <div class="card-value">{{ averageScore }}</div>
          <div class="card-subtitle">较上期 {{ scoreChange > 0 ? '+' : '' }}{{ scoreChange.toFixed(1) }}</div>
        </div>
      </div>
    </div>
    
    <!-- 课程列表 -->
    <div class="courses-content">
      <!-- 筛选和搜索 -->
      <div class="filter-section">
        <div class="filter-controls">
          <a-select
            v-model:value="semesterFilter"
            placeholder="选择学期"
            style="width: 150px"
            @change="handleFilter"
          >
            <a-select-option value="">全部学期</a-select-option>
            <a-select-option v-for="semester in semesters" :key="semester" :value="semester">
              {{ semester }}
            </a-select-option>
          </a-select>

          <a-select
            v-model:value="statusFilter"
            placeholder="课程状态"
            style="width: 120px"
            @change="handleFilter"
          >
            <a-select-option value="">全部状态</a-select-option>
            <a-select-option value="PUBLISHED">已发布</a-select-option>
            <a-select-option value="DRAFT">草稿</a-select-option>
            <a-select-option value="ARCHIVED">已归档</a-select-option>
          </a-select>

          <a-select
            v-model:value="typeFilter"
            placeholder="课程类型"
            style="width: 120px"
            @change="handleFilter"
          >
            <a-select-option value="">全部类型</a-select-option>
            <a-select-option value="REQUIRED">必修课</a-select-option>
            <a-select-option value="ELECTIVE">选修课</a-select-option>
            <a-select-option value="PUBLIC">公共课</a-select-option>
          </a-select>

          <a-input-search
            v-model:value="searchKeyword"
            placeholder="搜索课程名称或代码..."
            style="width: 250px"
            @search="handleSearch"
          />
        </div>

        <div class="view-controls">
          <a-radio-group v-model:value="viewMode" @change="handleViewChange">
            <a-radio-button value="table">
              <TableOutlined />
              表格视图
            </a-radio-button>
            <a-radio-button value="card">
              <AppstoreOutlined />
              卡片视图
            </a-radio-button>
          </a-radio-group>
        </div>
      </div>
      
      <!-- 视图容器 - 固定宽度 -->
      <div class="fixed-width-container">
        <!-- 表格视图 -->
        <div v-if="viewMode === 'table'" class="view-container table-view">
          <a-table
            :columns="columns"
            :data-source="courseList"
            :loading="loading"
            :pagination="pagination"
            :scroll="{ x: 'max-content' }"
            row-key="id"
            @change="handleTableChange"
          >
            <template #bodyCell="{ column, record }">
              <template v-if="column.key === 'courseName'">
                <div class="course-info">
                  <div class="course-name">{{ record.title || record.courseName }}</div>
                  <div class="course-meta">{{ record.courseType || record.category }} · {{ record.credit }}学分</div>
                </div>
              </template>

              <template v-else-if="column.key === 'semester'">
                {{ record.term || record.semester || '-' }}
              </template>

              <template v-else-if="column.key === 'status'">
                <a-tag :class="getStatusClass(record.status)">
                  {{ getStatusText(record.status) }}
                </a-tag>
              </template>

              <template v-else-if="column.key === 'students'">
                <div class="students-cell">
                  <div class="students-count">{{ record.studentCount || 0 }}</div>
                  <div class="students-text">名学生</div>
                </div>
              </template>

              <template v-else-if="column.key === 'progress'">
                <div class="progress-cell">
                  <a-progress 
                    :percent="calculateProgress(record.startTime, record.endTime)" 
                    size="small"
                    :stroke-color="getProgressColor(calculateProgress(record.startTime, record.endTime))"
                  />
                  <div class="progress-text">{{ calculateProgress(record.startTime, record.endTime) }}%</div>
                </div>
              </template>
              
              <template v-else-if="column.key === 'performance'">
                <div class="performance-cell">
                  <div class="average-score">{{ record.averageScore || '-' }}</div>
                </div>
              </template>
              
              <template v-else-if="column.key === 'startTime'">
                {{ formatDate(record.startTime) }}
              </template>

              <template v-else-if="column.key === 'action'">
                <a-button-group size="small">
                  <a-button @click="viewCourse(record)">
                    <EyeOutlined />
                    查看
                  </a-button>
                  <a-button @click="editCourse(record)">
                    <EditOutlined />
                    编辑
                  </a-button>
                  <a-button danger @click="confirmDelete(record)">
                    <DeleteOutlined />
                    删除
                  </a-button>
                </a-button-group>
              </template>
            </template>
          </a-table>
        </div>

        <!-- 卡片视图 -->
        <div v-else class="view-container card-view">
          <div class="course-cards">
            <div
              v-for="course in courseList"
              :key="course.id"
              class="course-card"
            >
              <div class="card-header">
                <div 
                  class="course-cover" 
                  :style="getCourseBackground(course)"
                ></div>
                <div class="course-status">
                  <a-tag :class="getStatusClass(course.status)">
                    {{ getStatusText(course.status) }}
                  </a-tag>
                </div>
                <div class="card-menu">
                  <a-dropdown :trigger="['hover']">
                    <EllipsisOutlined class="ellipsis-icon" />
                    <template #overlay>
                      <a-menu>
                        <a-menu-item key="edit" @click="editCourse(course)">
                          <EditOutlined />
                          编辑
                        </a-menu-item>
                        <a-menu-item key="delete" @click="confirmDelete(course)">
                          <DeleteOutlined />
                          <span class="danger">删除</span>
                        </a-menu-item>
                      </a-menu>
                    </template>
                  </a-dropdown>
                </div>
              </div>
              
              <div class="course-content">
                <h3 class="course-title">{{ course.title || course.courseName }}</h3>
                <p class="course-description">{{ course.description || '暂无描述' }}</p>
                
                <div class="course-stats">
                  <div class="stat-item">
                    <span class="stat-label">学分：</span>
                    <span class="stat-value">{{ course.credit || '-' }}</span>
                  </div>
                  <div class="stat-item">
                    <span class="stat-label">学生数：</span>
                    <span class="stat-value">{{ course.studentCount || 0 }}</span>
                  </div>
                  <div class="stat-item">
                    <span class="stat-label">课程进度：</span>
                    <a-progress 
                      :percent="calculateProgress(course.startTime, course.endTime)" 
                      size="small"
                      :stroke-color="getProgressColor(calculateProgress(course.startTime, course.endTime))"
                    />
                  </div>
                  <div class="stat-item">
                    <span class="stat-label">平均成绩：</span>
                    <span class="stat-value">{{ course.averageScore || '-' }}</span>
                  </div>
                </div>
              </div>
              
              <div class="card-footer">
                <div class="course-type">
                  <a-tag class="type-tag">
                    {{ course.courseType === 'REQUIRED' || course.courseType === '必修课' ? '必修' : 
                       course.courseType === 'ELECTIVE' || course.courseType === '选修课' ? '选修' : '必修' }}
                  </a-tag>
                </div>
                <div class="card-actions">
                  <a-button type="link" class="detail-button" @click="viewCourse(course)">
                    <EyeOutlined />
                    查看详情
                  </a-button>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- 创建课程弹窗 -->
    <a-modal
      v-model:open="showCreateModal"
      :title="isEditing ? '编辑课程' : '新建课程'"
      width="600px"
      @ok="handleCreateOrUpdateCourse"
      @cancel="resetCreateForm"
    >
      <a-form
        ref="createFormRef"
        :model="createForm"
        :rules="createRules"
        layout="vertical"
      >
        <a-form-item label="课程名称" name="courseName">
          <a-input 
            v-model:value="createForm.courseName" 
            placeholder="请输入课程名称" 
            @input="(value: string) => { createForm.title = value }"
          />
        </a-form-item>
        
        <a-form-item label="课程描述" name="description">
          <a-textarea 
            v-model:value="createForm.description" 
            placeholder="请输入课程描述"
            :rows="3"
          />
        </a-form-item>
        
        <a-form-item label="课程封面" name="coverImage">
          <div class="course-cover-upload">
            <a-upload
              v-model:file-list="coverFileList"
              list-type="picture-card"
              :show-upload-list="true"
              :before-upload="beforeCoverUpload"
              :customRequest="handleCoverUpload"
              :maxCount="1"
            >
              <div v-if="!createForm.coverImage">
                <upload-outlined />
                <div style="margin-top: 8px">上传封面</div>
              </div>
            </a-upload>
            <div class="cover-preview" v-if="createForm.coverImage">
              <img :src="createForm.coverImage" alt="课程封面预览" />
            </div>
          </div>
          <div class="upload-hint">建议上传16:9比例的图片，大小不超过2MB</div>
        </a-form-item>
        
        <a-row :gutter="16">
          <a-col :span="12">
            <a-form-item label="学分" name="credit">
              <a-input-number 
                v-model:value="createForm.credit" 
                :min="1" 
                :max="10" 
                style="width: 100%"
                placeholder="学分"
              />
            </a-form-item>
          </a-col>
          <a-col :span="12">
            <a-form-item label="课程类型" name="category">
              <a-select v-model:value="createForm.category" placeholder="选择课程类型">
                <a-select-option value="REQUIRED">必修课</a-select-option>
                <a-select-option value="ELECTIVE">选修课</a-select-option>
                <a-select-option value="PUBLIC">公共课</a-select-option>
              </a-select>
            </a-form-item>
          </a-col>
        </a-row>
        
        <a-row :gutter="16">
          <a-col :span="12">
            <a-form-item label="开始时间" name="startTime">
              <a-date-picker 
                v-model:value="createForm.startTime" 
                style="width: 100%"
                placeholder="选择开始时间"
              />
            </a-form-item>
          </a-col>
          <a-col :span="12">
            <a-form-item label="结束时间" name="endTime">
              <a-date-picker 
                v-model:value="createForm.endTime" 
                style="width: 100%"
                placeholder="选择结束时间"
              />
            </a-form-item>
          </a-col>
        </a-row>

        <a-row :gutter="16">
          <a-col :span="12">
            <a-form-item label="学年" name="year">
              <a-select
                v-model:value="selectedYear"
                placeholder="选择年份"
                style="width: 100%"
                @change="updateSemester"
              >
                <a-select-option v-for="year in yearOptions" :key="year" :value="year">
                  {{ year }}
                </a-select-option>
              </a-select>
            </a-form-item>
          </a-col>
          <a-col :span="12">
            <a-form-item label="季度" name="term">
              <a-select
                v-model:value="selectedTerm"
                placeholder="选择季度"
                style="width: 100%"
                @change="updateSemester"
              >
                <a-select-option v-for="term in termOptions" :key="term.value" :value="term.value">
                  {{ term.label }}
                </a-select-option>
              </a-select>
            </a-form-item>
          </a-col>
        </a-row>
      </a-form>
    </a-modal>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { message, Modal } from 'ant-design-vue'
import { useAuthStore } from '@/stores/auth'
import {
  BookOutlined,
  PlusOutlined,
  DownloadOutlined,
  TeamOutlined,
  ClockCircleOutlined,
  TrophyOutlined,
  TableOutlined,
  AppstoreOutlined,
  EyeOutlined,
  EditOutlined,
  MoreOutlined,
  ArrowUpOutlined,
  ArrowDownOutlined,
  MinusOutlined,
  DeleteOutlined,
  FileTextOutlined,
  UploadOutlined,
  EllipsisOutlined
} from '@ant-design/icons-vue'
import { getCourses, createCourse, deleteCourse, updateCourse, getStudentEnrolledCourses, type Course, type CourseCreateRequest } from '@/api/course'
import dayjs, { Dayjs } from 'dayjs'
import axios from 'axios'

const router = useRouter()
const authStore = useAuthStore()

// 检查用户是否是学生
const isStudent = computed(() => authStore.user?.role?.toUpperCase() === 'STUDENT')

// 响应式数据
const loading = ref(false)
const viewMode = ref('card')
const searchKeyword = ref('')
const semesterFilter = ref('')
const statusFilter = ref('')
const typeFilter = ref('')

// 课程数据
const courseList = ref<Course[]>([])
const pagination = ref({
  current: 1,
  pageSize: 10,
  total: 0,
  showSizeChanger: true,
  showQuickJumper: true,
  showTotal: (total: number) => `共 ${total} 条记录`
})

// 弹窗状态
const showCreateModal = ref(false)
const isEditing = ref(false)
const currentEditingCourseId = ref<number | undefined>(undefined)

// 封面上传相关
const coverFileList = ref<any[]>([])
const uploadLoading = ref(false)

// 创建表单
interface CreateForm {
  title: string;
  courseName: string;
  description: string;
  coverImage: string;
  credit: number | string;
  category: string;
  courseType: string;
  startTime: string | Dayjs;
  endTime: string | Dayjs;
  term: string;
  semester: string;
}

// 表单数据
const createForm = ref<CreateForm>({
  title: '',
  courseName: '',
  description: '',
  coverImage: '',
  credit: 3,
  category: 'REQUIRED',
  courseType: '必修课',
  startTime: '',
  endTime: '',
  term: '',
  semester: ''
})

// 表单验证规则
const createRules = {
  courseName: [
    { required: true, message: '请输入课程名称', trigger: 'blur' }
  ],
  category: [
    { required: true, message: '请选择课程类型', trigger: 'change' }
  ],
  semester: [
    { required: true, message: '请选择学期', trigger: 'change' }
  ]
}

// 学期相关数据
const yearOptions = ref([
  '2023-2024', '2024-2025', '2025-2026', '2026-2027', '2027-2028', '2028-2029'
])
const termOptions = ref([
  { label: '秋季', value: '1' },
  { label: '春季', value: '2' },
  { label: '夏季', value: '3' }
])
const selectedYear = ref('2024-2025') // 设置默认值
const selectedTerm = ref('1') // 设置默认值

// 监听学期选择变化
const updateSemester = () => {
  if (selectedYear.value && selectedTerm.value) {
    createForm.value.semester = `${selectedYear.value}-${selectedTerm.value}`
    console.log('学期已更新:', createForm.value.semester)
  } else {
    createForm.value.semester = ''
  }
}

const semesters = ref(['2024-2025-1', '2024-2025-2', '2023-2024-1', '2023-2024-2'])

// 表格列定义
const columns = [
  {
    title: '课程名称',
    dataIndex: 'title',
    key: 'courseName',
    width: 180,
    ellipsis: true
  },
  {
    title: '学分',
    dataIndex: 'credit',
    key: 'credit',
    width: 60,
    align: 'center'
  },
  {
    title: '学期',
    dataIndex: 'term',
    key: 'semester',
    width: 100
  },
  {
    title: '状态',
    dataIndex: 'status',
    key: 'status',
    width: 80
  },
  {
    title: '学生数',
    dataIndex: 'studentCount',
    key: 'students',
    width: 80,
    align: 'center'
  },
  {
    title: '课程进度',
    key: 'progress',
    width: 180
  },
  {
    title: '平均成绩',
    dataIndex: 'averageScore',
    key: 'performance',
    width: 80,
    align: 'center'
  },
  {
    title: '开课时间',
    dataIndex: 'startTime',
    key: 'startTime',
    width: 100
  },
  {
    title: '操作',
    key: 'action',
    width: 180,
    fixed: 'right',
    className: 'action-column'
  }
]

// 计算属性
const totalCourses = computed(() => courseList.value.length)
const activeCourses = computed(() => courseList.value.filter(c => c.status === 'PUBLISHED').length)
const totalStudents = computed(() => courseList.value.reduce((sum, c) => sum + (c.studentCount || 0), 0))
const averageStudentsPerCourse = computed(() => {
  return activeCourses.value > 0 ? Math.round(totalStudents.value / activeCourses.value) : 0
})
const averageProgress = computed(() => {
  // 计算所有课程进度的平均值
  if (courseList.value.length === 0) return 0
  
  const totalProgress = courseList.value.reduce((sum, course) => {
    return sum + calculateProgress(course.startTime, course.endTime)
  }, 0)
  
  return Math.round(totalProgress / courseList.value.length)
})
const completedLessons = computed(() => {
  // 这里可以根据实际需求计算完成的课时
  return courseList.value.reduce((sum, c) => sum + (c.chapterCount || 0), 0)
})
const totalLessons = computed(() => {
  // 这里可以根据实际需求计算总课时
  return completedLessons.value + 20 // 临时计算
})
const averageScore = computed(() => {
  const validScores = courseList.value.filter(c => c.averageScore !== undefined && c.averageScore !== null)
  if (validScores.length === 0) return '0.0'
  const total = validScores.reduce((sum, c) => sum + (c.averageScore || 0), 0)
  return (total / validScores.length).toFixed(1)
})
const scoreChange = computed(() => {
  // 这里可以根据实际需求计算成绩变化
  return 2.3 // 临时值
})

// 方法
const loadCourses = async () => {
  try {
    loading.value = true
    const params = {
      page: pagination.value.current,
      size: pagination.value.pageSize,
      keyword: searchKeyword.value || undefined,
      status: statusFilter.value || undefined,
      term: semesterFilter.value || undefined
    }
    
    let response;
    if (isStudent.value) {
      // 学生查看自己的课程
      response = await getStudentEnrolledCourses(params);
      console.log('获取学生课程响应:', response);
    } else {
      // 教师查看自己的课程
      response = await getCourses(params);
    }
    
    if (response.data && response.data.code === 200) {
      const result = response.data.data
      courseList.value = result.records || result.list || []
      pagination.value.total = result.total || 0
      
      console.log('获取到的课程列表:', courseList.value)
    } else {
      message.error(response.data?.message || '获取课程列表失败')
      courseList.value = []
    }
  } catch (error) {
    console.error('获取课程列表失败:', error)
    message.error('获取课程列表失败，请检查网络连接')
    courseList.value = []
  } finally {
    loading.value = false
  }
}

const handleCreateOrUpdateCourse = async () => {
  try {
    loading.value = true;
    
    // 表单验证
    if (!createForm.value.courseName || createForm.value.courseName.trim() === '') {
      message.error('请输入课程名称');
      loading.value = false;
      return;
    }
    
    // 确保学分是数字类型
    let creditValue = createForm.value.credit;
    if (typeof creditValue === 'string') {
      creditValue = parseFloat(creditValue);
    }
    
    // 格式化开始和结束时间
    const startTime = createForm.value.startTime 
      ? (typeof createForm.value.startTime === 'string' 
        ? createForm.value.startTime 
        : dayjs(createForm.value.startTime).format('YYYY-MM-DD HH:mm:ss')) 
      : undefined;
    
    const endTime = createForm.value.endTime 
      ? (typeof createForm.value.endTime === 'string' 
        ? createForm.value.endTime 
        : dayjs(createForm.value.endTime).format('YYYY-MM-DD HH:mm:ss')) 
      : undefined;
    
    // 如果没有选择开始时间，使用当前时间
    const formattedStartTime = startTime || dayjs().format('YYYY-MM-DD HH:mm:ss');
    // 如果没有选择结束时间，使用开始时间后3个月
    const formattedEndTime = endTime || dayjs(formattedStartTime).add(3, 'month').format('YYYY-MM-DD HH:mm:ss');
    
    // 确保学期格式正确
    let term = createForm.value.semester || '2024-2025-1';
    if (selectedYear.value && selectedTerm.value) {
      term = `${selectedYear.value}-${selectedTerm.value}`;
    }
    
    const formData = {
      title: createForm.value.courseName, // 使用courseName作为title
      courseName: createForm.value.courseName,
      description: createForm.value.description || '',
      coverImage: createForm.value.coverImage || '',
      credit: creditValue,
      category: createForm.value.category || 'REQUIRED',
      courseType: createForm.value.courseType || getCourseTypeFromCategory(createForm.value.category),
      startTime: formattedStartTime,
      endTime: formattedEndTime,
      term: term,
      semester: term,
      status: isEditing.value ? undefined : '未开始' // 如果是编辑，保留原状态
    };
    
    let response;
    
    if (isEditing.value && currentEditingCourseId.value) {
      // 更新课程
      console.log('更新课程数据:', JSON.stringify(formData));
      response = await updateCourseAPI(currentEditingCourseId.value, formData);
    } else {
      // 创建新课程
      console.log('提交的课程数据:', JSON.stringify(formData));
      response = await createCourse(formData);
    }
    
    if (response.data.code === 200) {
      message.success(isEditing.value ? '课程更新成功' : '课程创建成功');
      showCreateModal.value = false;
      resetCreateForm();
      loadCourses();
    } else {
      message.error(response.data.message || (isEditing.value ? '课程更新失败' : '课程创建失败'));
    }
  } catch (error: any) {
    console.error(isEditing.value ? '更新课程失败:' : '创建课程失败:', error);
    
    // 处理不同类型的错误
    if (error.response) {
      // 服务器返回了错误响应
      if (error.response.status === 401) {
        message.error('请先登录后再操作课程');
        setTimeout(() => {
          router.push('/login');
        }, 1500);
      } else if (error.response.data && error.response.data.message) {
        message.error(error.response.data.message);
      } else {
        message.error(`操作失败 (${error.response.status})`);
      }
    } else if (error.request) {
      // 请求已经发出，但没有收到响应
      message.error('服务器无响应，请检查网络连接');
    } else {
      // 请求设置时发生错误
      message.error('请求错误: ' + error.message);
    }
  } finally {
    loading.value = false;
  }
}

// 根据category获取courseType
const getCourseTypeFromCategory = (category?: string) => {
  if (!category) return '必修课'
  
  const categoryMap: Record<string, string> = {
    'REQUIRED': '必修课',
    'ELECTIVE': '选修课',
    'PUBLIC': '公共课'
  }
  
  return categoryMap[category] || '必修课'
}

// 重置创建表单
const resetCreateForm = () => {
  createForm.value = {
    title: '',
    courseName: '',
    description: '',
    coverImage: '',
    credit: 3,
    category: 'REQUIRED',
    courseType: '必修课',
    startTime: '',
    endTime: '',
    term: '',
    semester: ''
  }
  selectedYear.value = '2024-2025'
  selectedTerm.value = '1'
  coverFileList.value = []
}

const confirmDelete = (course: Course) => {
  if (!course || !course.id) {
    message.error('无效的课程数据，无法删除');
    console.error('无效的课程数据:', course);
    return;
  }

  // 将课程ID转换为字符串，避免JavaScript的数值精度问题
  const courseIdStr = String(course.id);
  console.log(`准备删除课程，ID: ${course.id}，转换为字符串: ${courseIdStr}，名称: ${course.title || course.courseName}`);
  
  Modal.confirm({
    title: '确认删除',
    content: `确定要删除课程"${course.title || course.courseName}"吗？此操作不可恢复。`,
    okText: '确认',
    cancelText: '取消',
    onOk: () => {
      console.log(`用户确认删除课程，ID: ${courseIdStr}`);
      handleDeleteCourse(courseIdStr);
    }
  });
}

const handleDeleteCourse = async (courseId: string | number) => {
  try {
    loading.value = true
    
    // 将课程ID转换为字符串，避免JavaScript的数值精度问题
    const courseIdStr = String(courseId);
    console.log(`开始删除课程，ID: ${courseId}，转换为字符串: ${courseIdStr}`);
    
    // 确保courseId不为空
    if (!courseIdStr) {
      message.error('无效的课程ID');
      console.error('无效的课程ID:', courseId);
      loading.value = false;
      return;
    }
    
    const response = await deleteCourse(courseIdStr);
    
    if (response.data && response.data.code === 200) {
      message.success('课程删除成功');
      console.log('课程删除成功，重新加载课程列表');
      await loadCourses(); // 重新加载课程列表
    } else {
      message.error(response.data?.message || '课程删除失败');
      console.error('删除课程失败，服务器返回:', response.data);
    }
  } catch (error: any) {
    console.error('删除课程失败:', error);
    
    // 处理不同类型的错误
    if (error.response) {
      // 服务器返回了错误响应
      if (error.response.status === 401) {
        message.error('请先登录后再操作');
        setTimeout(() => {
          router.push('/login');
        }, 1500);
      } else if (error.response.data && error.response.data.message) {
        message.error(error.response.data.message);
      } else {
        message.error(`删除课程失败 (${error.response.status})`);
      }
    } else if (error.request) {
      // 请求已经发出，但没有收到响应
      message.error('服务器无响应，请检查网络连接');
    } else {
      // 请求设置时发生错误
      message.error('删除课程请求错误: ' + error.message);
    }
  } finally {
    loading.value = false;
  }
}

const viewCourse = (course: Course) => {
  router.push(`/teacher/courses/${course.id}`);
}

const editCourse = (course: Course) => {
  // 复制课程数据到编辑表单
  createForm.value = {
    title: course.title || course.courseName || '',
    courseName: course.title || course.courseName || '',
    description: course.description || '',
    coverImage: course.coverImage || '',
    credit: course.credit || 3,
    category: course.category || course.courseType || 'REQUIRED',
    courseType: course.courseType || '必修课',
    startTime: course.startTime ? dayjs(course.startTime) : '',
    endTime: course.endTime ? dayjs(course.endTime) : '',
    term: course.term || course.semester || '',
    semester: course.semester || course.term || ''
  };
  
  // 解析学期信息
  if (course.term || course.semester) {
    const termStr = course.term || course.semester || '';
    const parts = termStr.split('-');
    if (parts.length >= 3) {
      selectedYear.value = `${parts[0]}-${parts[1]}`;
      selectedTerm.value = parts[2];
    }
  }
  
  // 设置编辑状态和当前编辑的课程ID
  isEditing.value = true;
  currentEditingCourseId.value = course.id;
  
  // 显示创建/编辑弹窗
  showCreateModal.value = true;
}

const handleFilter = () => {
  pagination.value.current = 1
  loadCourses()
}

const handleSearch = () => {
  pagination.value.current = 1
  loadCourses()
}

const handleViewChange = () => {
  // 视图模式切换
}

const handleTableChange = (pag: any) => {
  pagination.value.current = pag.current
  pagination.value.pageSize = pag.pageSize
  loadCourses()
}

const exportCourses = () => {
  message.info('导出功能开发中...')
}

// 工具方法
const getStatusColor = (status: string) => {
  const colors: Record<string, string> = {
    'PUBLISHED': 'green',
    'DRAFT': 'orange',
    'ARCHIVED': 'red',
    '进行中': 'green',
    '未开始': 'orange',
    '已结束': 'red'
  }
  return colors[status] || 'default'
}

const getStatusText = (status: string) => {
  const texts: Record<string, string> = {
    'PUBLISHED': '已发布',
    'DRAFT': '草稿',
    'ARCHIVED': '已归档',
    '进行中': '进行中',
    '未开始': '未开始',
    '已结束': '已结束'
  }
  return texts[status] || status
}

const formatDate = (dateString: string) => {
  if (!dateString) return '-'
  return dayjs(dateString).format('YYYY-MM-DD')
}

// 获取进度条颜色
const getProgressColor = (progress: number) => {
  if (progress < 30) return '#ff4d4f' // 红色
  if (progress < 60) return '#faad14' // 黄色
  if (progress < 90) return '#1890ff' // 蓝色
  return '#52c41a' // 绿色
}

// 计算课程进度
const calculateProgress = (startTime: string | undefined, endTime: string | undefined): number => {
  if (!startTime || !endTime) {
    return 0;
  }
  
  const start = dayjs(startTime);
  const end = dayjs(endTime);
  const now = dayjs();
  
  // 如果当前时间在开始时间之前，进度为0
  if (now.isBefore(start)) {
    return 0;
  }
  
  // 如果当前时间在结束时间之后，进度为100
  if (now.isAfter(end)) {
    return 100;
  }
  
  // 计算总时长（以毫秒为单位）
  const totalDuration = end.diff(start);
  if (totalDuration <= 0) {
    return 0; // 避免除以零
  }
  
  // 计算已经过去的时长
  const elapsedDuration = now.diff(start);
  
  // 计算百分比
  const progress = Math.round((elapsedDuration / totalDuration) * 100);
  
  // 确保进度在0-100之间
  return Math.min(100, Math.max(0, progress));
}

// 处理封面上传前的校验
const beforeCoverUpload = (file: File) => {
  // 检查文件类型
  const isImage = file.type === 'image/jpeg' || file.type === 'image/png' || file.type === 'image/gif'
  if (!isImage) {
    message.error('只能上传JPG/PNG/GIF格式的图片!')
    return false
  }
  
  // 检查文件大小
  const isLt2M = file.size / 1024 / 1024 < 2
  if (!isLt2M) {
    message.error('图片大小不能超过2MB!')
    return false
  }
  
  return true
}

// 自定义上传方法
const handleCoverUpload = (options: any) => {
  const { file, onSuccess, onError } = options
  uploadLoading.value = true
  
  console.log('准备上传课程封面图片', file.name, file.size, file.type);
  
  // 创建FormData对象
  const formData = new FormData()
  formData.append('file', file)
  
  // 获取token
  const token = localStorage.getItem('token') || localStorage.getItem('user-token') || '';
  
  // 调用新的上传API - 使用photo目录
  fetch('http://localhost:8080/api/common/files/upload/course-photo', {
    method: 'POST',
    body: formData,
    credentials: 'include', // 确保包含凭证
    headers: {
      'Authorization': token ? `Bearer ${token}` : ''
    }
  })
    .then(response => {
      if (!response.ok) {
        console.error('上传响应状态异常:', response.status, response.statusText);
        return response.text().then(text => {
          try {
            return JSON.parse(text);
          } catch (e) {
            throw new Error(`上传失败: ${response.status} ${response.statusText} - ${text}`);
          }
        });
      }
      return response.json();
    })
    .then(result => {
      console.log('上传响应数据:', result);
      if (result.code === 200) {
        createForm.value.coverImage = result.data.url
        message.success('封面上传成功')
        onSuccess(result)
      } else {
        message.error(result.message || '封面上传失败')
        onError(new Error(result.message || '封面上传失败'))
      }
    })
    .catch(error => {
      console.error('上传异常:', error);
      message.error('封面上传失败: ' + error.message)
      onError(error)
    })
    .finally(() => {
      uploadLoading.value = false
    })
}

// 获取课程背景
const getCourseBackground = (course: Course) => {
  // 如果有封面图，则使用封面图
  if (course.coverImage) {
    // 直接使用图片URL，新格式的URL已经是/api/photo/...的形式，无需转换
    let imageUrl = course.coverImage;
    console.log('课程封面图片URL:', imageUrl);
    
    return {
      backgroundImage: `url(${imageUrl})`,
      backgroundSize: 'cover',
      backgroundPosition: 'center'
    };
  }
  
  // 否则根据课程状态生成渐变背景
  let gradient: string;
  
  switch (course.status) {
    case '已结束':
      gradient = 'linear-gradient(135deg, #f783ac 0%, #e64980 100%)';
      break;
    case '进行中':
      gradient = 'linear-gradient(135deg, #9775fa 0%, #7048e8 100%)';
      break;
    case '未开始':
      // 根据ID选择不同的蓝色或橙红色渐变
      if (course.id && course.id % 2 === 0) {
        gradient = 'linear-gradient(135deg, #74c0fc 0%, #4dabf7 100%)';
      } else {
        gradient = 'linear-gradient(135deg, #ffa8a8 0%, #ff6b6b 100%)';
      }
      break;
    default:
      gradient = 'linear-gradient(135deg, #74c0fc 0%, #4dabf7 100%)';
  }
  
  return {
    background: gradient
  };
}

// 简单的哈希函数，用于生成一个数字
const hashCode = (str: string): number => {
  let hash = 0;
  for (let i = 0; i < str.length; i++) {
    hash = ((hash << 5) - hash) + str.charCodeAt(i);
    hash |= 0; // 转换为32位整数
  }
  return hash;
}

// 获取课程类型颜色
const getCourseTypeColor = (courseType: string | undefined): string => {
  if (!courseType) return 'blue';
  
  const typeMap: Record<string, string> = {
    '必修课': 'blue',
    'REQUIRED': 'blue',
    '选修课': 'green',
    'ELECTIVE': 'green',
    '公共课': 'purple',
    'PUBLIC': 'purple'
  };
  
  return typeMap[courseType] || 'blue';
}

// 生命周期
onMounted(() => {
  loadCourses()
})

// 修改updateCourse API调用，使用正确的API路径
const updateCourseAPI = (courseId: number, data: any) => {
  // 获取token和用户ID
  const token = localStorage.getItem('user-token') || localStorage.getItem('token');
  const userInfo = localStorage.getItem('user-info');
  let userId = '';
  
  if (userInfo) {
    try {
      const userObj = JSON.parse(userInfo);
      userId = userObj.id || '';
    } catch (e) {
      console.error('解析用户信息失败:', e);
    }
  }
  
  // 使用简化的token格式
  const authToken = userId ? `Bearer token-${userId}` : (token ? `Bearer ${token}` : '');
  
  return axios.put(`/api/teacher/courses/${courseId}`, data, {
    headers: {
      'Authorization': authToken
    }
  });
}

// 获取状态标签的样式类
const getStatusClass = (status: string): string => {
  switch (status) {
    case 'PUBLISHED':
    case '已发布':
    case '进行中':
      return 'status-in-progress';
    case 'DRAFT':
    case '草稿':
    case '未开始':
      return 'status-not-started';
    case 'ARCHIVED':
    case '已归档':
    case '已结束':
      return 'status-completed';
    default:
      return 'status-not-started';
  }
}
</script>

<style scoped>
.teacher-courses {
  padding: 24px;
  background-color: #f5f7fa;
  min-height: calc(100vh - 64px);
}

/* 表格中的状态标签样式 */
:deep(.ant-table .ant-tag.status-completed) {
  background-color: #ff6b6b;
  color: white;
  border: none;
  font-size: 12px;
  padding: 0 8px;
  height: 24px;
  line-height: 22px;
  border-radius: 4px;
}

:deep(.ant-table .ant-tag.status-in-progress) {
  background-color: #67c23a;
  color: white;
  border: none;
  font-size: 12px;
  padding: 0 8px;
  height: 24px;
  line-height: 22px;
  border-radius: 4px;
}

:deep(.ant-table .ant-tag.status-not-started) {
  background-color: #ff9800;
  color: white;
  border: none;
  font-size: 12px;
  padding: 0 8px;
  height: 24px;
  line-height: 22px;
  border-radius: 4px;
}

/* 固定宽度容器 - 确保两种视图使用相同的宽度 */
.fixed-width-container {
  width: 100%;
  min-width: 1200px; /* 设置一个固定的最小宽度 */
  padding: 0 24px;
  box-sizing: border-box;
}

/* 表格视图和卡片视图共享样式 */
.view-container {
  width: 100%;
  min-width: 1200px; /* 与容器相同的最小宽度 */
  box-sizing: border-box;
  padding-bottom: 24px;
}

/* 卡片视图 */
.card-view {
  width: 100%;
}

/* 卡片网格 */
.course-cards {
  display: grid;
  grid-template-columns: repeat(4, 1fr); /* 固定4列 */
  gap: 16px;
  width: 100%;
}

/* 卡片样式 */
.course-card {
  background: white;
  border-radius: 8px;
  box-shadow: 0 2px 6px rgba(0, 0, 0, 0.08);
  overflow: hidden;
  transition: transform 0.2s, box-shadow 0.2s;
  display: flex;
  flex-direction: column;
  height: 100%;
  position: relative;
}

.course-card:hover {
  transform: translateY(-4px);
  box-shadow: 0 8px 24px rgba(0, 0, 0, 0.15);
}

.card-header {
  position: relative;
  height: 140px;
  overflow: hidden;
}

.course-cover {
  width: 100%;
  height: 100%;
  background-size: cover;
  background-position: center;
  transition: transform 0.3s;
}

.course-status {
  position: absolute;
  top: 10px;
  left: 10px;
  z-index: 2;
}

.course-status .ant-tag {
  font-size: 12px;
  padding: 0 8px;
  height: 24px;
  line-height: 22px;
  border-radius: 4px;
  border: none;
}

.course-status .ant-tag.status-completed {
  background-color: #ff6b6b;
  color: white;
}

.course-status .ant-tag.status-in-progress {
  background-color: #67c23a;
  color: white;
}

.course-status .ant-tag.status-not-started {
  background-color: #ff9800;
  color: white;
}

.card-menu {
  position: absolute;
  top: 10px;
  right: 10px;
  z-index: 2;
}

.ellipsis-icon {
  font-size: 20px;
  color: white;
  background-color: rgba(0, 0, 0, 0.3);
  border-radius: 50%;
  width: 32px;
  height: 32px;
  cursor: pointer;
  transition: background-color 0.2s;
  display: flex;
  align-items: center;
  justify-content: center;
}

.ellipsis-icon:hover {
  background-color: rgba(0, 0, 0, 0.5);
}

.course-content {
  padding: 16px;
  display: flex;
  flex-direction: column;
  flex-grow: 1;
}

.course-title {
  font-size: 18px;
  font-weight: 600;
  color: #303133;
  margin-bottom: 8px;
  overflow: hidden;
  text-overflow: ellipsis;
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
}

.course-description {
  font-size: 14px;
  color: #606266;
  margin-bottom: 16px;
  overflow: hidden;
  text-overflow: ellipsis;
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
}

.course-stats {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.stat-item {
  display: flex;
  justify-content: space-between;
  align-items: center;
  font-size: 14px;
}

.stat-label {
  color: #606266;
}

.stat-value {
  color: #303133;
  font-weight: 500;
}

/* 进度条样式 */
.ant-progress {
  margin-right: 0;
}

.ant-progress-bg {
  height: 6px !important;
  border-radius: 3px !important;
}

.course-meta {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-top: 12px;
  padding-top: 8px;
  border-top: 1px solid #f3f4f6;
  font-size: 12px;
  color: #6b7280;
}

.create-time {
  font-size: 12px;
  color: #9ca3af;
}

.card-footer {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 8px 12px;
  margin-top: auto;
}

.course-type {
  display: flex;
  align-items: center;
}

.type-tag {
  border: 1px solid #f56c6c;
  background-color: transparent;
  color: #f56c6c;
  border-radius: 4px;
  font-size: 12px;
  padding: 0 8px;
  height: 24px;
  line-height: 22px;
}

.card-actions {
  display: flex;
  align-items: center;
}

.detail-button {
  color: #606266;
  font-size: 14px;
  padding: 0;
  display: flex;
  align-items: center;
  gap: 4px;
}

.detail-button:hover {
  color: #409eff;
}

/* 概览卡片 */
.courses-overview {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 20px;
  margin-bottom: 24px;
  width: 100%;
  min-width: 1200px; /* 与其他容器相同的最小宽度 */
  box-sizing: border-box;
}

/* 内容区域 */
.courses-content {
  background: white;
  border-radius: 12px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
  overflow: hidden;
  width: 100%;
  min-width: 1200px; /* 与其他容器相同的最小宽度 */
  box-sizing: border-box;
}

/* 页面头部 */
.page-header {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  margin-bottom: 24px;
  background: white;
  padding: 24px;
  border-radius: 12px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
  width: 100%;
  min-width: 1200px; /* 与其他容器相同的最小宽度 */
  box-sizing: border-box;
}

.header-content {
  flex: 1;
}

.page-title {
  font-size: 28px;
  font-weight: 600;
  color: #1f2937;
  margin: 0 0 8px 0;
  display: flex;
  align-items: center;
  gap: 12px;
}

.page-description {
  color: #6b7280;
  font-size: 16px;
  margin: 0;
}

.header-actions {
  display: flex;
  gap: 12px;
}

/* 概览卡片 */
.overview-card {
  background: white;
  padding: 24px;
  border-radius: 12px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
  display: flex;
  align-items: center;
  gap: 16px;
  transition: transform 0.2s, box-shadow 0.2s;
}

.overview-card:hover {
  transform: translateY(-2px);
  box-shadow: 0 4px 16px rgba(0, 0, 0, 0.15);
}

.card-icon {
  width: 48px;
  height: 48px;
  border-radius: 12px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 20px;
  color: white;
}

.total .card-icon { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }
.students .card-icon { background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); }
.progress .card-icon { background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); }
.performance .card-icon { background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%); }

.card-content {
  flex: 1;
}

.card-title {
  font-size: 14px;
  color: #6b7280;
  margin-bottom: 4px;
}

.card-value {
  font-size: 24px;
  font-weight: 600;
  color: #1f2937;
  margin-bottom: 4px;
}

.card-subtitle {
  font-size: 12px;
  color: #9ca3af;
}

/* 筛选区域 */
.filter-section {
  padding: 20px 24px;
  border-bottom: 1px solid #f3f4f6;
  display: flex;
  justify-content: space-between;
  align-items: center;
  flex-wrap: wrap;
  gap: 16px;
  width: 100%;
  box-sizing: border-box;
}

.filter-controls {
  display: flex;
  gap: 12px;
  flex-wrap: wrap;
}

.view-controls {
  display: flex;
  gap: 8px;
}

/* 表格相关样式 */
.course-info {
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.course-name {
  font-weight: 600;
  color: #1f2937;
}

.course-meta {
  font-size: 12px;
  color: #6b7280;
}

.progress-cell {
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.progress-text {
  font-size: 12px;
  color: #6b7280;
}

.students-cell {
  text-align: center;
}

.students-count {
  font-weight: 600;
  font-size: 16px;
  color: #1f2937;
}

.students-text {
  font-size: 12px;
  color: #6b7280;
}

.performance-cell {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 4px;
}

.average-score {
  font-weight: 600;
  font-size: 16px;
  color: #1f2937;
}

.score-trend {
  font-size: 12px;
  display: flex;
  align-items: center;
  gap: 2px;
}

.score-trend.positive { color: #10b981; }
.score-trend.negative { color: #ef4444; }

/* 危险操作样式 */
:deep(.danger) {
  color: #ef4444 !important;
}

:deep(.danger:hover) {
  background-color: #fef2f2 !important;
}
</style> 
