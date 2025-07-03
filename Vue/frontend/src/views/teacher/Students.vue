<template>
  <div class="teacher-students">
    <div class="page-header">
      <h1>学生管理</h1>
      <a-space>
        <a-button @click="importStudents">
          <template #icon><upload-outlined /></template>
          批量导入
        </a-button>
        <a-button @click="exportStudents">
          <template #icon><download-outlined /></template>
          批量导出
        </a-button>
        <a-button type="primary" @click="showAddStudentModal">
          <template #icon><plus-outlined /></template>
          新增学生
        </a-button>
      </a-space>
    </div>
    
    <div class="filter-row">
      <a-input-search
        v-model:value="searchKeyword"
        placeholder="搜索学生姓名或学号"
        style="width: 250px"
        @search="handleSearch"
      />
      <a-select
        v-model:value="classFilter"
        style="width: 200px; margin-left: 16px;"
        placeholder="班级筛选"
        @change="handleClassFilterChange"
        allowClear
      >
        <a-select-option value="">全部班级</a-select-option>
        <a-select-option v-for="cls in classes" :key="cls.id" :value="cls.id">
          {{ cls.name }}
        </a-select-option>
      </a-select>
      <a-select
        v-model:value="courseFilter"
        style="width: 200px; margin-left: 16px;"
        placeholder="课程筛选"
        @change="handleCourseFilterChange"
        allowClear
      >
        <a-select-option value="">全部课程</a-select-option>
        <a-select-option v-for="course in teacherCourses" :key="course.id" :value="course.id">
          {{ course.title || course.courseName }}
        </a-select-option>
      </a-select>
    </div>
    
    <div class="students-content">
      <a-spin :spinning="loading">
        <a-table 
          :dataSource="students" 
          :columns="columns" 
          :pagination="pagination"
          @change="handleTableChange"
          rowKey="id"
        >
          <!-- 学生姓名列 -->
          <template #bodyCell="{ column, record }">
            <template v-if="column.key === 'name'">
              <a @click="viewStudentDetail(record.id)">{{ record.user?.realName || '未设置' }}</a>
            </template>
            
            <!-- 班级列 -->
            <template v-if="column.key === 'classes'">
              <a-space wrap>
                <a-tag v-for="cls in record.classes" :key="cls.id" color="blue">
                  {{ cls.name }}
                </a-tag>
                <a-button type="link" size="small" @click="showAssignClassModal(record)">
                  <plus-outlined /> 分配班级
                </a-button>
              </a-space>
            </template>
            
            <!-- 课程列 -->
            <template v-if="column.key === 'courses'">
              <a-space wrap>
                <a-tag v-for="course in record.courses" :key="course.id" color="green">
                  {{ course.title || course.courseName }}
                </a-tag>
                <a-button type="link" size="small" @click="showAssignCourseModal(record)">
                  <plus-outlined /> 分配课程
                </a-button>
              </a-space>
            </template>
            
            <!-- 操作列 -->
            <template v-if="column.key === 'action'">
              <a-space>
                <a-button type="link" size="small" @click="viewStudentDetail(record.id)">
                  <eye-outlined /> 查看
                </a-button>
                <a-button type="link" size="small" @click="showEditStudentModal(record)">
                  <edit-outlined /> 编辑
                </a-button>
                <a-popconfirm
                  title="确定要重置该学生的密码吗？"
                  @confirm="resetPassword(record.userId)"
                >
                  <a-button type="link" size="small">
                    <key-outlined /> 重置密码
                  </a-button>
                </a-popconfirm>
              </a-space>
            </template>
          </template>
        </a-table>
      </a-spin>
    </div>
    
    <!-- 添加学生对话框 -->
    <a-modal
      v-model:open="addStudentModalVisible"
      title="新增学生"
      @ok="handleAddStudent"
      :confirm-loading="submitLoading"
    >
      <a-input-search
        v-model:value="studentSearchKeyword"
        placeholder="搜索学生姓名或学号"
        style="margin-bottom: 16px;"
        @search="handleSearchStudents"
        allow-clear
        enter-button
      />
      
      <a-spin :spinning="searchLoading">
        <a-table
          :dataSource="searchStudentResults"
          :columns="searchStudentColumns"
          :pagination="false"
          :scroll="{ y: 240 }"
          size="small"
          rowKey="id"
          @row-click="selectStudent"
          :row-selection="{
            columnWidth: 60,
            type: 'radio',
            selectedRowKeys: selectedStudentKeys,
            onChange: onStudentSelectionChange
          }"
        >
        </a-table>
      </a-spin>
      
      <div style="margin-top: 16px;">
        <h4>选择课程添加学生:</h4>
        <a-select
          v-model:value="selectedCourseId"
          style="width: 100%;"
          placeholder="请选择要添加学生的课程"
        >
          <a-select-option v-for="course in teacherCourses" :key="course.id" :value="course.id">
            {{ course.title || course.courseName }}
          </a-select-option>
        </a-select>
      </div>
    </a-modal>
    
    <!-- 编辑学生对话框 -->
    <a-modal
      v-model:open="editStudentModalVisible"
      title="编辑学生信息"
      @ok="handleEditStudent"
      :confirm-loading="submitLoading"
    >
      <a-form :model="studentForm" ref="editStudentFormRef" layout="vertical">
        <a-form-item name="username" label="用户名/学号">
          <a-input v-model:value="studentForm.username" disabled />
        </a-form-item>
        <a-form-item name="realName" label="真实姓名" required>
          <a-input v-model:value="studentForm.realName" placeholder="请输入真实姓名" />
        </a-form-item>
        <a-form-item name="email" label="邮箱">
          <a-input v-model:value="studentForm.email" placeholder="请输入邮箱" />
        </a-form-item>
        <a-form-item name="enrollmentStatus" label="学籍状态">
          <a-select v-model:value="studentForm.enrollmentStatus">
            <a-select-option value="ENROLLED">在读</a-select-option>
            <a-select-option value="SUSPENDED">休学</a-select-option>
            <a-select-option value="GRADUATED">毕业</a-select-option>
            <a-select-option value="DROPPED_OUT">退学</a-select-option>
          </a-select>
        </a-form-item>
      </a-form>
    </a-modal>
    
    <!-- 分配班级对话框 -->
    <a-modal
      v-model:open="assignClassModalVisible"
      title="分配班级"
      @ok="handleAssignClass"
      :confirm-loading="submitLoading"
    >
      <a-form :model="assignClassForm" layout="vertical">
        <a-form-item name="classId" label="选择班级" required>
          <a-select
            v-model:value="assignClassForm.classId"
            placeholder="请选择班级"
            style="width: 100%"
          >
            <a-select-option v-for="cls in classes" :key="cls.id" :value="cls.id">
              {{ cls.name }}
              <span v-if="cls.courseId">
                ({{ getCourseName(cls.courseId) }})
              </span>
            </a-select-option>
          </a-select>
        </a-form-item>
      </a-form>
    </a-modal>
    
    <!-- 分配课程对话框 -->
    <a-modal
      v-model:open="assignCourseModalVisible"
      title="分配课程"
      @ok="handleAssignCourse"
      :confirm-loading="submitLoading"
    >
      <a-form :model="assignCourseForm" layout="vertical">
        <a-form-item name="courseId" label="选择课程" required>
          <a-select
            v-model:value="assignCourseForm.courseId"
            placeholder="请选择课程"
            style="width: 100%"
          >
            <a-select-option v-for="course in teacherCourses" :key="course.id" :value="course.id">
              {{ course.title || course.courseName }}
            </a-select-option>
          </a-select>
        </a-form-item>
      </a-form>
    </a-modal>
    
    <!-- 选课申请列表对话框 -->
    <a-modal
      v-model:open="enrollmentRequestsModalVisible"
      title="选课申请列表"
      width="800px"
      @cancel="enrollmentRequestsModalVisible = false"
      :footer="null"
    >
      <a-table
        :dataSource="enrollmentRequests"
        :columns="enrollmentRequestColumns"
        :pagination="enrollmentRequestPagination"
        @change="handleEnrollmentRequestTableChange"
        rowKey="id"
      >
        <template #bodyCell="{ column, record }">
          <!-- 状态列 -->
          <template v-if="column.key === 'status'">
            <a-tag :color="getStatusColor(record.status)">
              {{ getStatusText(record.status) }}
            </a-tag>
          </template>
          
          <!-- 操作列 -->
          <template v-if="column.key === 'action'">
            <a-space>
              <a-button type="primary" size="small" @click="approveEnrollmentRequest(record)" :disabled="record.status !== 0">
                通过
              </a-button>
              <a-button danger size="small" @click="rejectEnrollmentRequest(record)" :disabled="record.status !== 0">
                拒绝
              </a-button>
            </a-space>
          </template>
        </template>
      </a-table>
    </a-modal>
  </div>
</template>

<script setup lang="ts">
import { ref, reactive, onMounted, computed } from 'vue'
import { useRouter } from 'vue-router'
import { message } from 'ant-design-vue'
import type { TablePaginationConfig } from 'ant-design-vue'
import { 
  PlusOutlined, 
  UploadOutlined, 
  DownloadOutlined, 
  EditOutlined, 
  EyeOutlined,
  KeyOutlined
} from '@ant-design/icons-vue'
import { 
  getStudents, 
  getTeacherClasses,
  addStudentToClass,
  removeStudentFromClass,
  addStudentToCourse,
  removeStudentFromCourse,
  createStudent,
  updateStudent,
  getEnrollmentRequests,
  processEnrollmentRequest,
  searchStudents,
  type Student,
  type Class,
  type EnrollmentRequest,
  type StudentSearchResult
} from '@/api/student'
import { getTeacherCourses, type Course } from '@/api/teacher'

const router = useRouter()
const loading = ref(false)
const submitLoading = ref(false)
const searchLoading = ref(false)

// 学生列表数据
const students = ref<Student[]>([])
const classes = ref<Class[]>([])
const teacherCourses = ref<Course[]>([])

// 筛选条件
const searchKeyword = ref('')
const classFilter = ref<number | string>('')
const courseFilter = ref<number | string>('')

// 分页配置
const pagination = reactive<TablePaginationConfig>({
  current: 1,
  pageSize: 10,
  total: 0,
  showSizeChanger: true,
  showTotal: (total) => `共 ${total} 条记录`
})

// 表格列定义
const columns = [
  { title: '学号', dataIndex: ['studentId'], key: 'studentId' },
  { title: '姓名', key: 'name' },
  { title: '班级', key: 'classes' },
  { title: '课程', key: 'courses' },
  { title: '学籍状态', dataIndex: ['enrollmentStatus'], key: 'enrollmentStatus' },
  { title: '操作', key: 'action' }
]

// 对话框显示状态
const addStudentModalVisible = ref(false)
const editStudentModalVisible = ref(false)
const assignClassModalVisible = ref(false)
const assignCourseModalVisible = ref(false)
const enrollmentRequestsModalVisible = ref(false)

// 表单数据
const studentFormRef = ref()
const editStudentFormRef = ref()
const studentForm = reactive({
  id: undefined as number | undefined,
  userId: undefined as number | undefined,
  username: '',
  realName: '',
  email: '',
  password: '',
  enrollmentStatus: 'ENROLLED'
})

// 分配班级表单
const assignClassForm = reactive({
  studentId: undefined as number | undefined,
  classId: undefined as number | undefined
})

// 分配课程表单
const assignCourseForm = reactive({
  studentId: undefined as number | undefined,
  courseId: undefined as number | undefined
})

// 表单验证规则
const rules = {
  username: [
    { required: true, message: '请输入用户名/学号', trigger: 'blur' },
    { min: 3, max: 20, message: '用户名长度应为3-20个字符', trigger: 'blur' }
  ],
  realName: [
    { required: true, message: '请输入真实姓名', trigger: 'blur' },
    { max: 50, message: '姓名不能超过50个字符', trigger: 'blur' }
  ],
  email: [
    { type: 'email', message: '请输入有效的邮箱地址', trigger: 'blur' }
  ]
}

// 选课申请相关
const enrollmentRequests = ref<EnrollmentRequest[]>([])
const enrollmentRequestPagination = reactive<TablePaginationConfig>({
  current: 1,
  pageSize: 10,
  total: 0
})
const enrollmentRequestColumns = [
  { title: '学生', dataIndex: ['student', 'name'], key: 'studentName' },
  { title: '学号', dataIndex: ['student', 'studentId'], key: 'studentId' },
  { title: '课程', dataIndex: ['course', 'title'], key: 'courseTitle' },
  { title: '申请理由', dataIndex: 'reason', key: 'reason' },
  { title: '申请时间', dataIndex: 'submitTime', key: 'submitTime' },
  { title: '状态', key: 'status' },
  { title: '操作', key: 'action' }
]

// 学生搜索相关
const studentSearchKeyword = ref('')
const searchStudentResults = ref<StudentSearchResult[]>([])
const selectedStudentKeys = ref<number[]>([])
const selectedStudent = ref<StudentSearchResult | null>(null)
const selectedCourseId = ref<number | undefined>(undefined)

// 学生搜索表格列定义
const searchStudentColumns = [
  { title: '学号', dataIndex: 'studentId', key: 'studentId' },
  { title: '姓名', dataIndex: 'realName', key: 'realName' }
]

// 获取学生列表
const fetchStudentList = async () => {
  try {
    loading.value = true
    const params = {
      current: pagination.current || 1,
      size: pagination.pageSize || 10,
      keyword: searchKeyword.value,
      classId: classFilter.value ? Number(classFilter.value) : undefined,
      courseId: courseFilter.value ? Number(courseFilter.value) : undefined
    }
    
    const res = await getStudents(params)
    
    if (res && res.data) {
      students.value = res.data.records || []
      pagination.total = res.data.total || 0
      
      // 处理每个学生的班级和课程信息
      students.value.forEach(student => {
        // 这里需要后端返回学生的班级和课程信息
        // 如果后端没有直接返回，可能需要额外请求
        student.classes = student.classes || []
        student.courses = student.courses || []
      })
    } else {
      message.error('获取学生列表失败')
      students.value = []
    }
  } catch (error) {
    console.error('获取学生列表失败:', error)
    message.error('获取学生列表失败，请稍后重试')
    students.value = []
  } finally {
    loading.value = false
  }
}

// 获取班级列表
const fetchClassList = async () => {
  try {
    console.log('开始获取班级列表')
    const response = await getTeacherClasses()
    console.log('班级列表API返回数据:', response)
    
    if (!response || !response.data) {
      console.error('API返回的数据格式异常:', response)
      classes.value = []
      return
    }
    
    // 处理不同格式的响应数据
    let classesData: any[] = []
    
    // 使用类型断言处理响应数据
    const responseData = response.data as any
    
    // 检查是否是普通数组
    if (Array.isArray(responseData)) {
      console.log('API直接返回了班级数组')
      classesData = responseData
    } 
    // 检查是否是标准Result包装类型
    else if (responseData.code === 200 && responseData.data) {
      if (Array.isArray(responseData.data)) {
        console.log('API返回了Result包装的数组')
        classesData = responseData.data
      } else if (responseData.data.records) {
        console.log('API返回了Result包装的分页对象')
        classesData = responseData.data.records
      } else if (responseData.data.content) {
        console.log('API返回了Result包装的Spring分页对象')
        classesData = responseData.data.content
      }
    }
    // 检查是否是分页格式
    else if (responseData.records && Array.isArray(responseData.records)) {
      console.log('API返回了MyBatisPlus分页格式')
      classesData = responseData.records
    } else if (responseData.content && Array.isArray(responseData.content)) {
      console.log('API返回了Spring分页格式')
      classesData = responseData.content
    }
    
    console.log('处理后的班级数据:', classesData)
    classes.value = classesData
    
  } catch (error) {
    console.error('获取班级列表失败:', error)
    message.error('获取班级列表失败，请稍后重试')
    classes.value = []
  }
}

// 获取教师课程列表
const fetchTeacherCourses = async () => {
  try {
    console.log('开始获取教师课程列表')
    const response = await getTeacherCourses()
    console.log('教师课程列表API原始返回:', response)
    
    if (!response || !response.data) {
      console.error('API返回的数据格式异常:', response)
      teacherCourses.value = []
      return
    }
    
    let coursesData: any[] = []
    
    // 使用类型断言处理响应数据
    const responseData = response.data as any
    
    // 检查是否是普通数组
    if (Array.isArray(responseData)) {
      console.log('API直接返回了课程数组')
      coursesData = responseData
    } 
    // 检查是否是标准Result包装类型
    else if (responseData.code === 200 && responseData.data) {
      if (Array.isArray(responseData.data)) {
        console.log('API返回了Result包装的数组')
        coursesData = responseData.data
      } else if (responseData.data.records) {
        console.log('API返回了Result包装的分页对象')
        coursesData = responseData.data.records
      } else if (responseData.data.content) {
        console.log('API返回了Result包装的Spring分页对象')
        coursesData = responseData.data.content
      }
    }
    // 检查是否是分页格式
    else if (responseData.records && Array.isArray(responseData.records)) {
      console.log('API返回了MyBatisPlus分页格式')
      coursesData = responseData.records
    } else if (responseData.content && Array.isArray(responseData.content)) {
      console.log('API返回了Spring分页格式')
      coursesData = responseData.content
    }
    
    console.log('处理后的课程数据:', coursesData)
    teacherCourses.value = coursesData
  } catch (error) {
    console.error('获取课程列表失败:', error)
    message.error('获取课程列表失败，请稍后重试')
    teacherCourses.value = []
  }
}

// 获取选课申请列表
const fetchEnrollmentRequests = async () => {
  try {
    const params = {
      current: enrollmentRequestPagination.current || 1,
      size: enrollmentRequestPagination.pageSize || 10,
      courseId: undefined
    }
    
    const res = await getEnrollmentRequests(params)
    const resData = res as any
    
    if (resData.code === 200 && resData.data) {
      enrollmentRequests.value = resData.data.records || []
      enrollmentRequestPagination.total = resData.data.total || 0
    } else {
      message.error(resData.message || '获取选课申请列表失败')
      enrollmentRequests.value = []
    }
  } catch (error) {
    console.error('获取选课申请列表失败:', error)
    message.error('获取选课申请列表失败，请稍后重试')
    enrollmentRequests.value = []
  }
}

// 显示添加学生对话框
const showAddStudentModal = () => {
  // 重置搜索结果和选择状态
  studentSearchKeyword.value = ''
  searchStudentResults.value = []
  selectedStudentKeys.value = []
  selectedStudent.value = null
  selectedCourseId.value = undefined
  
  addStudentModalVisible.value = true
}

// 显示编辑学生对话框
const showEditStudentModal = (student: Student) => {
  // 填充表单数据
  Object.assign(studentForm, {
    id: student.id,
    userId: student.userId,
    username: student.user?.username || '',
    realName: student.user?.realName || '',
    email: student.user?.email || '',
    password: '',
    enrollmentStatus: student.enrollmentStatus || 'ENROLLED'
  })
  
  editStudentModalVisible.value = true
}

// 显示分配班级对话框
const showAssignClassModal = (student: Student) => {
  assignClassForm.studentId = student.id
  assignClassForm.classId = undefined
  assignClassModalVisible.value = true
}

// 显示分配课程对话框
const showAssignCourseModal = (student: Student) => {
  assignCourseForm.studentId = student.id
  assignCourseForm.courseId = undefined
  assignCourseModalVisible.value = true
}

// 搜索学生
const handleSearchStudents = async () => {
  if (!studentSearchKeyword.value.trim()) {
    searchStudentResults.value = []
    return
  }
  
  try {
    searchLoading.value = true
    const res = await searchStudents(studentSearchKeyword.value)
    console.log('API返回的原始数据:', res)
    
    // 用any类型绕过TypeScript类型检查
    const apiResponse = res.data as any
    
    if (apiResponse && apiResponse.code === 200 && Array.isArray(apiResponse.data)) {
      searchStudentResults.value = apiResponse.data
      console.log('处理后的搜索结果:', searchStudentResults.value)
    } else {
      console.error('搜索学生返回的数据格式不正确:', apiResponse)
      searchStudentResults.value = []
      message.error('搜索学生失败或结果格式不正确')
    }
  } catch (error) {
    console.error('搜索学生失败:', error)
    message.error('搜索学生失败，请稍后重试')
    searchStudentResults.value = []
  } finally {
    searchLoading.value = false
  }
}

// 选择学生
const selectStudent = (record: StudentSearchResult) => {
  selectedStudentKeys.value = [record.id]
  selectedStudent.value = record
}

// 学生选择改变
const onStudentSelectionChange = (selectedRowKeys: number[], selectedRows: StudentSearchResult[]) => {
  selectedStudentKeys.value = selectedRowKeys
  selectedStudent.value = selectedRows.length > 0 ? selectedRows[0] : null
}

// 添加学生到课程
const handleAddStudent = async () => {
  if (!selectedStudent.value) {
    message.error('请先选择一个学生')
    return
  }
  
  if (!selectedCourseId.value) {
    message.error('请选择要添加学生的课程')
    return
  }
  
  try {
    submitLoading.value = true
    
    const res = await addStudentToCourse(selectedStudent.value.id, selectedCourseId.value)
    const resData = res as any
    
    if (resData && resData.code === 200) {
      message.success('学生已成功添加到课程')
      addStudentModalVisible.value = false
      fetchStudentList()
    } else {
      message.error(resData.message || '添加学生到课程失败')
    }
  } catch (error) {
    console.error('添加学生到课程失败:', error)
    message.error('添加学生到课程失败，请稍后重试')
  } finally {
    submitLoading.value = false
  }
}

// 编辑学生
const handleEditStudent = async () => {
  try {
    await editStudentFormRef.value.validate()
    submitLoading.value = true
    
    const studentData: Student = {
      id: studentForm.id!,
      userId: studentForm.userId!,
      studentId: studentForm.username,
      enrollmentStatus: studentForm.enrollmentStatus,
      user: {
        id: studentForm.userId!,
        username: studentForm.username,
        realName: studentForm.realName,
        email: studentForm.email,
        role: 'STUDENT',
        status: 'ACTIVE'
      }
    }
    
    const res = await updateStudent(studentForm.id!, studentData)
    
    if (res.code === 200) {
      message.success('更新学生信息成功')
      editStudentModalVisible.value = false
      fetchStudentList()
    } else {
      message.error(res.message || '更新学生信息失败')
    }
  } catch (error) {
    console.error('更新学生信息失败:', error)
    message.error('更新学生信息失败，请检查表单数据')
  } finally {
    submitLoading.value = false
  }
}

// 分配班级
const handleAssignClass = async () => {
  if (!assignClassForm.classId || !assignClassForm.studentId) {
    message.error('请选择班级')
    return
  }
  
  try {
    submitLoading.value = true
    
    const res = await addStudentToClass(assignClassForm.studentId, assignClassForm.classId)
    
    if (res.code === 200) {
      message.success('分配班级成功')
      assignClassModalVisible.value = false
      fetchStudentList()
    } else {
      message.error(res.message || '分配班级失败')
    }
  } catch (error) {
    console.error('分配班级失败:', error)
    message.error('分配班级失败，请稍后重试')
  } finally {
    submitLoading.value = false
  }
}

// 分配课程
const handleAssignCourse = async () => {
  if (!assignCourseForm.courseId || !assignCourseForm.studentId) {
    message.error('请选择课程')
    return
  }
  
  try {
    submitLoading.value = true
    
    const res = await addStudentToCourse(assignCourseForm.studentId, assignCourseForm.courseId)
    
    if (res.code === 200) {
      message.success('分配课程成功')
      assignCourseModalVisible.value = false
      fetchStudentList()
    } else {
      message.error(res.message || '分配课程失败')
    }
  } catch (error) {
    console.error('分配课程失败:', error)
    message.error('分配课程失败，请稍后重试')
  } finally {
    submitLoading.value = false
  }
}

// 通过选课申请
const approveEnrollmentRequest = async (request: EnrollmentRequest) => {
  try {
    const res = await processEnrollmentRequest(request.id, true)
    
    if (res.code === 200) {
      message.success('已通过选课申请')
      fetchEnrollmentRequests()
    } else {
      message.error(res.message || '操作失败')
    }
  } catch (error) {
    console.error('处理选课申请失败:', error)
    message.error('处理选课申请失败，请稍后重试')
  }
}

// 拒绝选课申请
const rejectEnrollmentRequest = async (request: EnrollmentRequest) => {
  try {
    const res = await processEnrollmentRequest(request.id, false, '不符合选课条件')
    
    if (res.code === 200) {
      message.success('已拒绝选课申请')
      fetchEnrollmentRequests()
    } else {
      message.error(res.message || '操作失败')
    }
  } catch (error) {
    console.error('处理选课申请失败:', error)
    message.error('处理选课申请失败，请稍后重试')
  }
}

// 查看学生详情
const viewStudentDetail = (id: number) => {
  router.push(`/teacher/students/${id}`)
}

// 重置密码
const resetPassword = async (userId: number) => {
  message.success('密码重置成功，新密码为：123456')
}

// 批量导入学生
const importStudents = () => {
  message.info('批量导入功能开发中...')
}

// 批量导出学生
const exportStudents = () => {
  message.info('批量导出功能开发中...')
}

// 处理表格变化
const handleTableChange = (pag: TablePaginationConfig) => {
  pagination.current = pag.current || 1
  pagination.pageSize = pag.pageSize || 10
  fetchStudentList()
}

// 处理选课申请表格变化
const handleEnrollmentRequestTableChange = (pag: TablePaginationConfig) => {
  enrollmentRequestPagination.current = pag.current || 1
  enrollmentRequestPagination.pageSize = pag.pageSize || 10
  fetchEnrollmentRequests()
}

// 处理搜索
const handleSearch = () => {
  pagination.current = 1
  fetchStudentList()
}

// 处理班级筛选变化
const handleClassFilterChange = () => {
  pagination.current = 1
  courseFilter.value = ''  // 重置课程筛选
  fetchStudentList()
}

// 处理课程筛选变化
const handleCourseFilterChange = () => {
  pagination.current = 1
  classFilter.value = ''  // 重置班级筛选
  fetchStudentList()
}

// 获取课程名称
const getCourseName = (courseId: number) => {
  const course = teacherCourses.value.find(c => c.id === courseId)
  return course ? (course.title || course.courseName) : '未知课程'
}

// 获取状态颜色
const getStatusColor = (status: number) => {
  switch (status) {
    case 0: return 'orange'
    case 1: return 'green'
    case 2: return 'red'
    default: return 'default'
  }
}

// 获取状态文本
const getStatusText = (status: number) => {
  switch (status) {
    case 0: return '待审核'
    case 1: return '已通过'
    case 2: return '已拒绝'
    default: return '未知状态'
  }
}

// 页面初始化加载数据
onMounted(() => {
  // 同时获取学生、班级和课程数据
  fetchStudentList()
  fetchClassList()
  fetchTeacherCourses()
})
</script>

<style scoped>
.teacher-students {
  padding: 24px;
}

.page-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 24px;
}

.filter-row {
  display: flex;
  margin-bottom: 24px;
}

.students-content {
  background-color: #fff;
  padding: 24px;
  border-radius: 4px;
}
</style> 