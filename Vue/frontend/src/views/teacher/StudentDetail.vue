<template>
  <div class="student-detail">
    <div class="page-header">
      <div class="header-left">
        <a-button @click="goBack">
          <template #icon><left-outlined /></template>
          返回
        </a-button>
    <h1>学生详情</h1>
      </div>
      <div class="header-right">
        <a-button type="primary" @click="showEditModal">
          <template #icon><edit-outlined /></template>
          编辑信息
        </a-button>
      </div>
    </div>
    
    <a-spin :spinning="loading">
      <div class="detail-content" v-if="student">
        <a-row :gutter="24">
          <a-col :span="8">
            <a-card title="基本信息" :bordered="false">
              <a-descriptions :column="1">
                <a-descriptions-item label="姓名">{{ student.user?.realName || '未设置' }}</a-descriptions-item>
                <a-descriptions-item label="学号">{{ student.studentId }}</a-descriptions-item>
                <a-descriptions-item label="邮箱">{{ student.user?.email || '未设置' }}</a-descriptions-item>
                <a-descriptions-item label="学籍状态">
                  <a-tag :color="getStatusColor(student.enrollmentStatus)">
                    {{ getStatusText(student.enrollmentStatus) }}
                  </a-tag>
                </a-descriptions-item>
                <a-descriptions-item label="GPA">{{ student.gpa || '暂无' }}</a-descriptions-item>
                <a-descriptions-item label="GPA等级">{{ student.gpaLevel || '暂无' }}</a-descriptions-item>
              </a-descriptions>
            </a-card>
          </a-col>
          
          <a-col :span="16">
            <a-card title="班级信息" :bordered="false" style="margin-bottom: 24px;">
              <div class="card-action">
                <a-button type="primary" size="small" @click="showAssignClassModal">
                  <template #icon><plus-outlined /></template>
                  分配班级
                </a-button>
              </div>
              
              <a-spin :spinning="classesLoading">
                <a-empty v-if="studentClasses.length === 0" description="暂无班级信息" />
                <a-table
                  v-else
                  :dataSource="studentClasses"
                  :columns="classColumns"
                  :pagination="false"
                  rowKey="id"
                  size="small"
                >
                  <template #bodyCell="{ column, record }">
                    <template v-if="column.key === 'course'">
                      <span v-if="record.course">{{ record.course.title || record.course.courseName }}</span>
                      <span v-else>未绑定课程</span>
                    </template>
                    <template v-if="column.key === 'action'">
                      <a-popconfirm
                        title="确定要将该学生从班级中移除吗？"
                        @confirm="removeFromClass(record.id)"
                      >
                        <a-button type="link" danger size="small">移除</a-button>
                      </a-popconfirm>
                    </template>
                  </template>
                </a-table>
              </a-spin>
            </a-card>
            
            <a-card title="课程信息" :bordered="false">
              <div class="card-action">
                <a-button type="primary" size="small" @click="showAssignCourseModal">
                  <template #icon><plus-outlined /></template>
                  分配课程
                </a-button>
              </div>
              
              <a-spin :spinning="coursesLoading">
                <a-empty v-if="studentCourses.length === 0" description="暂无课程信息" />
                <a-table
                  v-else
                  :dataSource="studentCourses"
                  :columns="courseColumns"
                  :pagination="false"
                  rowKey="id"
                  size="small"
                >
                  <template #bodyCell="{ column, record }">
                    <template v-if="column.key === 'action'">
                      <a-popconfirm
                        title="确定要将该学生从课程中移除吗？"
                        @confirm="removeFromCourse(record.id)"
                      >
                        <a-button type="link" danger size="small">移除</a-button>
                      </a-popconfirm>
                    </template>
                  </template>
                </a-table>
              </a-spin>
            </a-card>
          </a-col>
        </a-row>
      </div>
    </a-spin>
    
    <!-- 编辑学生对话框 -->
    <a-modal
      v-model:open="editModalVisible"
      title="编辑学生信息"
      @ok="handleEditStudent"
      :confirm-loading="submitLoading"
    >
      <a-form :model="studentForm" ref="studentFormRef" layout="vertical">
        <a-form-item name="studentId" label="学号">
          <a-input v-model:value="studentForm.studentId" disabled />
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
      @cancel="() => assignClassModalVisible = false"
      :confirm-loading="submitLoading"
    >
      <div v-if="classesLoading" style="text-align: center; padding: 20px;">
        <a-spin />
        <div style="margin-top: 10px;">加载班级数据中...</div>
      </div>
      <a-form v-else :model="assignClassForm" layout="vertical">
        <div v-if="classes.length === 0" style="text-align: center; color: #ff4d4f; margin-bottom: 16px;">
          <a-alert
            message="暂无可分配的班级"
            description="系统中还没有创建任何班级，请先创建班级后再尝试分配。"
            type="warning"
            show-icon
          />
        </div>
        <a-form-item v-else name="classId" label="选择班级" required>
          <a-select
            v-model:value="assignClassForm.classId"
            placeholder="请选择班级"
            style="width: 100%"
            :options="classes.map(cls => ({ 
              value: cls.id, 
              label: cls.name + (cls.courseId ? ` (${getCourseName(cls.courseId)})` : '')
            }))"
          />
          <div style="margin-top: 8px; color: #1890ff; cursor: pointer;" @click="refreshClasses">
            <a-button type="link" size="small" style="padding: 0;">
              <reload-outlined /> 刷新班级列表
            </a-button>
          </div>
        </a-form-item>
      </a-form>
    </a-modal>
    
    <!-- 分配课程对话框 -->
    <a-modal
      v-model:open="assignCourseModalVisible"
      title="分配课程"
      @ok="handleAssignCourse"
      @cancel="() => assignCourseModalVisible = false"
      :confirm-loading="submitLoading"
    >
      <div v-if="coursesLoading" style="text-align: center; padding: 20px;">
        <a-spin />
        <div style="margin-top: 10px;">加载课程数据中...</div>
      </div>
      <a-form v-else :model="assignCourseForm" layout="vertical">
        <div v-if="teacherCourses.length === 0" style="text-align: center; color: #ff4d4f; margin-bottom: 16px;">
          <a-alert
            message="暂无可分配的课程"
            description="系统中还没有创建任何课程，请先创建课程后再尝试分配。"
            type="warning"
            show-icon
          />
        </div>
        <a-form-item v-else name="courseId" label="选择课程" required>
          <a-select
            v-model:value="assignCourseForm.courseId"
            placeholder="请选择课程"
            style="width: 100%"
            :options="teacherCourses.map(course => ({ 
              value: course.id, 
              label: course.title || course.courseName || `课程${course.id}`
            }))"
          />
          <div style="margin-top: 8px; color: #1890ff; cursor: pointer;" @click="refreshCourses">
            <a-button type="link" size="small" style="padding: 0;">
              <reload-outlined /> 刷新课程列表
            </a-button>
          </div>
        </a-form-item>
      </a-form>
    </a-modal>
  </div>
</template>

<script setup lang="ts">
import { ref, reactive, onMounted } from 'vue'
import { useRouter, useRoute } from 'vue-router'
import { message } from 'ant-design-vue'
import { 
  LeftOutlined,
  EditOutlined,
  PlusOutlined,
  ReloadOutlined
} from '@ant-design/icons-vue'
import { 
  getStudentDetail,
  getTeacherClasses,
  addStudentToClass,
  removeStudentFromClass,
  addStudentToCourse,
  removeStudentFromCourse,
  updateStudent,
  type Student,
  type Class
} from '@/api/student'
import { getTeacherCourses, type Course } from '@/api/teacher'
import axios from 'axios'

const router = useRouter()
const route = useRoute()
const loading = ref(false)
const submitLoading = ref(false)
const classesLoading = ref(false)
const coursesLoading = ref(true)

// 学生ID
const studentId = ref<number>(Number(route.params.id))

// 学生数据
const student = ref<Student>()
const studentClasses = ref<Class[]>([])
const studentCourses = ref<Course[]>([])
const classes = ref<Class[]>([])
const teacherCourses = ref<Course[]>([])

// 对话框显示状态
const editModalVisible = ref(false)
const assignClassModalVisible = ref(false)
const assignCourseModalVisible = ref(false)

// 表单数据
const studentFormRef = ref()
const studentForm = reactive({
  id: undefined as number | undefined,
  userId: undefined as number | undefined,
  studentId: '',
  realName: '',
  email: '',
  enrollmentStatus: 'ENROLLED'
})

// 分配班级表单
const assignClassForm = reactive({
  classId: undefined as number | undefined
})

// 分配课程表单
const assignCourseForm = reactive({
  courseId: undefined as number | undefined
})

// 班级表格列定义
const classColumns = [
  { title: '班级名称', dataIndex: 'name', key: 'name' },
  { title: '课程', key: 'course' },
  { title: '加入时间', dataIndex: 'joinTime', key: 'joinTime' },
  { title: '操作', key: 'action', width: 100 }
]

// 课程表格列定义
const courseColumns = [
  { title: '课程名称', dataIndex: 'title', key: 'title' },
  { title: '课程类型', dataIndex: 'courseType', key: 'courseType' },
  { title: '学期', dataIndex: 'term', key: 'term' },
  { title: '选课时间', dataIndex: 'enrollTime', key: 'enrollTime' },
  { title: '操作', key: 'action', width: 100 }
]

// 获取学生详情
const fetchStudentDetail = async () => {
  try {
    loading.value = true
    console.log('开始获取学生详情, ID:', studentId.value)
    const response = await getStudentDetail(studentId.value)
    console.log('学生详情API返回:', response)
    
    // 处理不同的响应格式
    let studentData = null
    
    if (response && response.data) {
      if (response.data.code === 200 && response.data.data) {
        studentData = response.data.data
      } else if (typeof response.data === 'object' && !response.data.code) {
        studentData = response.data
      } else {
        console.error('无法识别的学生数据格式:', response.data)
        message.error('获取学生信息失败: 数据格式错误')
        return
      }
      
      console.log('处理后的学生数据:', studentData)
      student.value = studentData
      
      // 获取学生的班级信息 - 通过class_student表
      fetchStudentClasses()
      
      // 获取学生的课程信息 - 通过course_student表
      fetchStudentCourses()
    } else {
      message.error('获取学生详情失败: 无返回数据')
    }
  } catch (error: any) {
    console.error('获取学生详情失败:', error)
    message.error(`获取学生详情失败: ${error.response?.data?.message || error.message || '请稍后重试'}`)
  } finally {
    loading.value = false
  }
}

// 获取班级列表
const fetchClassList = async () => {
  console.log('开始获取班级列表')
  classesLoading.value = true
  
  try {
    const response = await getTeacherClasses()
    console.log('班级列表API返回:', response)
    
    if (!response || !response.data) {
      console.error('API返回的数据格式异常:', response)
      classes.value = []
      return
    }
    
    let classesData = null
    
    // 检查是否是普通数组
    if (Array.isArray(response.data)) {
      console.log('API直接返回了班级数组')
      classesData = response.data
    } 
    // 检查是否是标准分页格式
    else if (response.data.content && Array.isArray(response.data.content)) {
      console.log('API返回了标准Spring分页格式')
      classesData = response.data.content
    } 
    // 检查是否是MyBatisPlus分页格式
    else if (response.data.records && Array.isArray(response.data.records)) {
      console.log('API返回了MyBatisPlus分页格式')
      classesData = response.data.records
    }
    // 检查是否是自定义列表格式
    else if (response.data.list && Array.isArray(response.data.list)) {
      console.log('API返回了自定义列表格式')
      classesData = response.data.list
    }
    // 检查是否是包含code和data的标准格式
    else if (response.data.code === 200 && response.data.data) {
      if (Array.isArray(response.data.data)) {
        console.log('API返回了标准成功数据格式，内容是数组')
        classesData = response.data.data
      } else if (response.data.data.content && Array.isArray(response.data.data.content)) {
        console.log('API返回了标准成功数据格式，内容是Spring分页')
        classesData = response.data.data.content
      } else if (response.data.data.records && Array.isArray(response.data.data.records)) {
        console.log('API返回了标准成功数据格式，内容是MyBatisPlus分页')
        classesData = response.data.data.records
      } else if (response.data.data.list && Array.isArray(response.data.data.list)) {
        console.log('API返回了标准成功数据格式，内容是自定义列表')
        classesData = response.data.data.list
      } else {
        console.log('API返回了标准成功数据格式，内容是单个对象，将其转为数组')
        classesData = [response.data.data]
      }
    }
    // 尝试读取其他可能的结构
    else {
      console.warn('无法识别的班级数据格式，尝试提取可能的数据')
      if (typeof response.data === 'object') {
        if (response.data.data && Array.isArray(response.data.data)) {
          console.log('从data字段找到数组数据')
          classesData = response.data.data
        } else {
          console.log('将整个响应对象视为单个班级')
          classesData = [response.data]
        }
      }
    }
    
    // 检查和处理班级数据
    if (classesData && Array.isArray(classesData)) {
      // 过滤并转换班级数据
      const validClasses = classesData
        .filter(cls => cls && cls.id)
        .map(cls => ({
          id: cls.id,
          name: cls.name || `班级${cls.id}`,
          courseId: cls.courseId || null,
          ...cls
        }));
      
      console.log(`找到 ${validClasses.length} 个有效班级`, validClasses)
      classes.value = validClasses
    } else {
      console.warn('提取的班级数据无效或为空')
      classes.value = []
    }
  } catch (error) {
    console.error('获取班级列表失败:', error)
    message.error('获取班级列表失败，请稍后重试')
    classes.value = []
  } finally {
    classesLoading.value = false
  }
}

// 获取教师课程列表
const fetchTeacherCourses = async () => {
  console.log('开始获取教师课程列表')
  coursesLoading.value = true
  
  try {
    const response = await getTeacherCourses()
    console.log('教师课程列表API原始返回:', response)
    
    if (!response || !response.data) {
      console.error('API返回的数据格式异常:', response)
      teacherCourses.value = []
      return
    }
    
    let coursesData = null
    
    // 检查是否是普通数组
    if (Array.isArray(response.data)) {
      console.log('API直接返回了课程数组')
      coursesData = response.data
    } 
    // 检查是否是标准分页格式
    else if (response.data.content && Array.isArray(response.data.content)) {
      console.log('API返回了标准Spring分页格式')
      coursesData = response.data.content
    } 
    // 检查是否是MyBatisPlus分页格式
    else if (response.data.records && Array.isArray(response.data.records)) {
      console.log('API返回了MyBatisPlus分页格式')
      coursesData = response.data.records
    }
    // 检查是否是自定义列表格式
    else if (response.data.list && Array.isArray(response.data.list)) {
      console.log('API返回了自定义列表格式')
      coursesData = response.data.list
    }
    // 检查是否是包含code和data的标准格式
    else if (response.data.code === 200 && response.data.data) {
      if (Array.isArray(response.data.data)) {
        console.log('API返回了标准成功数据格式，内容是数组')
        coursesData = response.data.data
      } else if (response.data.data.content && Array.isArray(response.data.data.content)) {
        console.log('API返回了标准成功数据格式，内容是Spring分页')
        coursesData = response.data.data.content
      } else if (response.data.data.records && Array.isArray(response.data.data.records)) {
        console.log('API返回了标准成功数据格式，内容是MyBatisPlus分页')
        coursesData = response.data.data.records
      } else if (response.data.data.list && Array.isArray(response.data.data.list)) {
        console.log('API返回了标准成功数据格式，内容是自定义列表')
        coursesData = response.data.data.list
      } else {
        console.log('API返回了标准成功数据格式，内容是单个对象，将其转为数组')
        coursesData = [response.data.data]
      }
    }
    // 尝试读取其他可能的结构
    else {
      console.warn('无法识别的课程数据格式，尝试提取可能的数据')
      if (typeof response.data === 'object') {
        if (response.data.data && Array.isArray(response.data.data)) {
          console.log('从data字段找到数组数据')
          coursesData = response.data.data
        } else {
          console.log('将整个响应对象视为单个课程')
          coursesData = [response.data]
        }
      }
    }
    
    // 检查和处理课程数据
    if (coursesData && Array.isArray(coursesData)) {
      // 过滤并转换课程数据
      const validCourses = coursesData
        .filter(course => course && course.id)
        .map(course => ({
          id: course.id,
          title: course.title || course.courseName || `课程${course.id}`,
          courseName: course.courseName || course.title,
          courseType: course.courseType || '未知类型',
          term: course.term || course.semester || '未知学期',
          ...course
        }));
      
      console.log(`找到 ${validCourses.length} 个有效课程`, validCourses)
      teacherCourses.value = validCourses
    } else {
      console.warn('提取的课程数据无效或为空')
      teacherCourses.value = []
    }
  } catch (error) {
    console.error('获取课程列表失败:', error)
    message.error('获取课程列表失败，请稍后重试')
    teacherCourses.value = []
  } finally {
    coursesLoading.value = false
  }
}

// 获取学生所属班级列表
const fetchStudentClasses = async () => {
  if (!studentId.value) return
  
  console.log('开始获取学生班级信息, 学生ID:', studentId.value)
  classesLoading.value = true
  
  try {
    // 获取token和用户ID
    const token = localStorage.getItem('user-token') || localStorage.getItem('token')
    const userInfo = localStorage.getItem('user-info')
    let userId = ''
    
    if (userInfo) {
      try {
        const userObj = JSON.parse(userInfo)
        userId = userObj.id || ''
      } catch (e) {
        console.error('解析用户信息失败:', e)
      }
    }
    
    // 使用token构建授权头
    const authToken = userId ? `Bearer token-${userId}` : (token ? `Bearer ${token}` : '')
    
    const response = await axios.get(`/api/teacher/students/${studentId.value}/classes`, {
      headers: {
        'Authorization': authToken
      }
    })
    
    console.log('学生班级API返回:', response)
    
    // 处理不同的响应格式
    if (response && response.data) {
      let classesData = null
      
      if (response.data.code === 200 && response.data.data) {
        classesData = response.data.data
      } else if (Array.isArray(response.data)) {
        classesData = response.data
      } else if (typeof response.data === 'object' && !response.data.code) {
        classesData = response.data
      } else if (response.data.data && Array.isArray(response.data.data)) {
        classesData = response.data.data
      } else {
        console.error('无法识别的班级数据格式:', response.data)
        studentClasses.value = []
        return
      }
      
      console.log('处理后的学生班级数据:', classesData)
      studentClasses.value = classesData || []
    } else {
      console.error('获取学生班级信息失败:', response)
      studentClasses.value = []
    }
  } catch (error: any) {
    console.error('获取学生班级信息失败:', error)
    message.error(`获取班级信息失败: ${error.response?.data?.message || error.message || '请稍后重试'}`)
    studentClasses.value = []
  } finally {
    classesLoading.value = false
  }
}

// 获取学生课程列表
const fetchStudentCourses = async () => {
  if (!studentId.value) return
  
  console.log('开始获取学生课程信息, 学生ID:', studentId.value)
  coursesLoading.value = true
  
  try {
    // 获取token和用户ID
    const token = localStorage.getItem('user-token') || localStorage.getItem('token')
    const userInfo = localStorage.getItem('user-info')
    let userId = ''
    
    if (userInfo) {
      try {
        const userObj = JSON.parse(userInfo)
        userId = userObj.id || ''
      } catch (e) {
        console.error('解析用户信息失败:', e)
      }
    }
    
    // 使用token构建授权头
    const authToken = userId ? `Bearer token-${userId}` : (token ? `Bearer ${token}` : '')
    
    const response = await axios.get(`/api/teacher/students/${studentId.value}/courses`, {
      headers: {
        'Authorization': authToken
      }
    })
    
    console.log('学生课程API返回:', response)
    
    // 处理不同的响应格式
    if (response && response.data) {
      let coursesData = null
      
      if (response.data.code === 200 && response.data.data) {
        coursesData = response.data.data
      } else if (Array.isArray(response.data)) {
        coursesData = response.data
      } else if (typeof response.data === 'object' && !response.data.code) {
        coursesData = response.data
      } else if (response.data.data && Array.isArray(response.data.data)) {
        coursesData = response.data.data
      } else {
        console.error('无法识别的课程数据格式:', response.data)
        studentCourses.value = []
        return
      }
      
      console.log('处理后的学生课程数据:', coursesData)
      studentCourses.value = coursesData || []
    } else {
      console.error('获取学生课程信息失败:', response)
      studentCourses.value = []
    }
  } catch (error: any) {
    console.error('获取学生课程信息失败:', error)
    message.error(`获取课程信息失败: ${error.response?.data?.message || error.message || '请稍后重试'}`)
    studentCourses.value = []
  } finally {
    coursesLoading.value = false
  }
}

// 显示编辑对话框
const showEditModal = () => {
  if (!student.value) return
  
  // 填充表单数据
  Object.assign(studentForm, {
    id: student.value.id,
    userId: student.value.userId,
    studentId: student.value.studentId,
    realName: student.value.user?.realName || '',
    email: student.value.user?.email || '',
    enrollmentStatus: student.value.enrollmentStatus || 'ENROLLED'
  })
  
  editModalVisible.value = true
}

// 显示分配班级对话框
const showAssignClassModal = async () => {
  console.log('开始显示分配班级对话框')
  classesLoading.value = true
  assignClassModalVisible.value = true
  
  try {
    // 确保已加载班级列表
    await fetchClassList()
    
    console.log('加载的班级数据:', classes.value)
    
    if (!classes.value || classes.value.length === 0) {
      message.warning('暂无可分配的班级，请先创建班级')
    }
  } catch (error) {
    console.error('加载班级列表失败:', error)
    message.error('加载班级列表失败，请稍后重试')
  } finally {
    classesLoading.value = false
  }
}

// 显示分配课程对话框
const showAssignCourseModal = async () => {
  console.log('开始显示分配课程对话框')
  coursesLoading.value = true
  assignCourseModalVisible.value = true
  
  try {
    // 确保已加载课程列表
    await fetchTeacherCourses()
    
    console.log('加载的课程数据:', teacherCourses.value)
    
    if (!teacherCourses.value || teacherCourses.value.length === 0) {
      message.warning('暂无可分配的课程，请先创建课程')
    }
  } catch (error) {
    console.error('加载课程列表失败:', error)
    message.error('加载课程列表失败，请稍后重试')
  } finally {
    coursesLoading.value = false
  }
}

// 编辑学生信息
const handleEditStudent = async () => {
  if (!student.value) return
  
  try {
    submitLoading.value = true
    
    const studentData: Student = {
      id: student.value.id,
      userId: student.value.userId,
      studentId: studentForm.studentId,
      enrollmentStatus: studentForm.enrollmentStatus,
      user: {
        id: student.value.userId,
        username: student.value.user?.username || '',
        realName: studentForm.realName,
        email: studentForm.email,
        role: 'STUDENT',
        status: 'ACTIVE'
      }
    }
    
    console.log('提交学生更新数据:', studentData)
    
    const response = await updateStudent(student.value.id, studentData)
    console.log('更新学生API返回:', response)
    
    let success = false
    let errorMsg = ''
    
    if (response && response.data) {
      if (response.data.code === 200) {
        success = true
      } else if (response.data.success === true) {
        success = true
      } else {
        errorMsg = response.data.message || '更新学生信息失败'
      }
    }
    
    if (success) {
      message.success('更新学生信息成功')
      editModalVisible.value = false
      fetchStudentDetail() // 重新加载学生详情
    } else {
      message.error(errorMsg || '更新学生信息失败')
    }
  } catch (error: any) {
    console.error('更新学生信息失败:', error)
    message.error(`更新学生信息失败: ${error.response?.data?.message || error.message || '请检查表单数据'}`)
  } finally {
    submitLoading.value = false
  }
}

// 分配班级
const handleAssignClass = async () => {
  if (!assignClassForm.classId || !studentId.value) {
    message.error('请选择班级')
    return
  }
  
  try {
    submitLoading.value = true
    
    const response = await addStudentToClass(studentId.value, assignClassForm.classId)
    console.log('分配班级API返回:', response)
    
    let success = false
    let errorMsg = ''
    
    if (response && response.data) {
      if (response.data.code === 200) {
        success = true
      } else if (response.data.success === true) {
        success = true
      } else {
        errorMsg = response.data.message || '分配班级失败'
      }
    }
    
    if (success) {
      message.success('分配班级成功')
      assignClassModalVisible.value = false
      fetchStudentClasses() // 只重新加载班级数据
    } else {
      message.error(errorMsg || '分配班级失败')
    }
  } catch (error: any) {
    console.error('分配班级失败:', error)
    message.error(`分配班级失败: ${error.response?.data?.message || error.message || '请稍后重试'}`)
  } finally {
    submitLoading.value = false
  }
}

// 分配课程
const handleAssignCourse = async () => {
  if (!assignCourseForm.courseId || !studentId.value) {
    message.error('请选择课程')
    return
  }
  
  try {
    submitLoading.value = true
    
    const response = await addStudentToCourse(studentId.value, assignCourseForm.courseId)
    console.log('分配课程API返回:', response)
    
    let success = false
    let errorMsg = ''
    
    if (response && response.data) {
      if (response.data.code === 200) {
        success = true
      } else if (response.data.success === true) {
        success = true
      } else {
        errorMsg = response.data.message || '分配课程失败'
      }
    }
    
    if (success) {
      message.success('分配课程成功')
      assignCourseModalVisible.value = false
      fetchStudentCourses() // 只重新加载课程数据
    } else {
      message.error(errorMsg || '分配课程失败')
    }
  } catch (error: any) {
    console.error('分配课程失败:', error)
    message.error(`分配课程失败: ${error.response?.data?.message || error.message || '请稍后重试'}`)
  } finally {
    submitLoading.value = false
  }
}

// 从班级中移除学生
const removeFromClass = async (classId: number) => {
  if (!studentId.value) return
  
  try {
    const response = await removeStudentFromClass(studentId.value, classId)
    console.log('从班级移除学生API返回:', response)
    
    let success = false
    let errorMsg = ''
    
    if (response && response.data) {
      if (response.data.code === 200) {
        success = true
      } else if (response.data.success === true) {
        success = true
      } else {
        errorMsg = response.data.message || '操作失败'
      }
    }
    
    if (success) {
      message.success('从班级移除学生成功')
      fetchStudentClasses() // 只重新加载班级数据
    } else {
      message.error(errorMsg || '操作失败')
    }
  } catch (error: any) {
    console.error('从班级移除学生失败:', error)
    message.error(`从班级移除学生失败: ${error.response?.data?.message || error.message || '请稍后重试'}`)
  }
}

// 从课程中移除学生
const removeFromCourse = async (courseId: number) => {
  if (!studentId.value) return
  
  try {
    const response = await removeStudentFromCourse(studentId.value, courseId)
    console.log('从课程移除学生API返回:', response)
    
    let success = false
    let errorMsg = ''
    
    if (response && response.data) {
      if (response.data.code === 200) {
        success = true
      } else if (response.data.success === true) {
        success = true
      } else {
        errorMsg = response.data.message || '操作失败'
      }
    }
    
    if (success) {
      message.success('从课程移除学生成功')
      fetchStudentCourses() // 只重新加载课程数据
    } else {
      message.error(errorMsg || '操作失败')
    }
  } catch (error: any) {
    console.error('从课程移除学生失败:', error)
    message.error(`从课程移除学生失败: ${error.response?.data?.message || error.message || '请稍后重试'}`)
  }
}

// 返回上一页
const goBack = () => {
  router.back()
}

// 获取课程名称
const getCourseName = (courseId: number | undefined): string => {
  if (!courseId) return ''
  
  // 在教师课程列表中查找
  const course = teacherCourses.value.find(c => c.id === courseId)
  if (course) {
    return course.title || course.courseName || `课程${course.id}`
  }
  
  // 在学生课程列表中查找
  const studentCourse = studentCourses.value.find(c => c.id === courseId)
  if (studentCourse) {
    return studentCourse.title || studentCourse.courseName || `课程${studentCourse.id}`
  }
  
  return `课程${courseId}`
}

// 获取状态颜色
const getStatusColor = (status?: string) => {
  switch (status) {
    case 'ENROLLED': return 'green'
    case 'SUSPENDED': return 'orange'
    case 'GRADUATED': return 'blue'
    case 'DROPPED_OUT': return 'red'
    default: return 'default'
  }
}

// 获取状态文本
const getStatusText = (status?: string) => {
  switch (status) {
    case 'ENROLLED': return '在读'
    case 'SUSPENDED': return '休学'
    case 'GRADUATED': return '毕业'
    case 'DROPPED_OUT': return '退学'
    default: return '未知状态'
  }
}

// 刷新课程列表
const refreshCourses = async () => {
  console.log('开始刷新课程列表')
  
  try {
    coursesLoading.value = true
    await fetchTeacherCourses()
    console.log('课程列表刷新完成')
  } catch (error: any) {
    console.error('刷新课程列表失败:', error)
    message.error('刷新课程列表失败，请稍后重试')
  } finally {
    coursesLoading.value = false
  }
}

// 刷新班级列表
const refreshClasses = async () => {
  console.log('开始刷新班级列表')
  
  try {
    classesLoading.value = true
    await fetchClassList()
    console.log('班级列表刷新完成')
  } catch (error: any) {
    console.error('刷新班级列表失败:', error)
    message.error('刷新班级列表失败，请稍后重试')
  } finally {
    classesLoading.value = false
  }
}

// 组件挂载时获取数据
onMounted(async () => {
  console.log('组件挂载，开始加载数据')
  loading.value = true
  
  try {
    // 并行加载数据以提高性能
    await fetchStudentDetail()
    
    // 并行加载班级和课程数据
    await Promise.all([
      fetchClassList(),
      fetchTeacherCourses()
    ])
    
    console.log('所有数据加载完成')
  } catch (error) {
    console.error('数据加载过程中发生错误:', error)
    message.error('加载数据失败，请刷新页面重试')
  } finally {
    loading.value = false
  }
})
</script>

<style scoped>
.student-detail {
  padding: 24px;
}

.page-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 24px;
}

.header-left {
  display: flex;
  align-items: center;
}

.header-left h1 {
  margin: 0 0 0 16px;
}

.detail-content {
  margin-top: 24px;
}

.card-action {
  display: flex;
  justify-content: flex-end;
  margin-bottom: 16px;
}
</style> 