<template>
  <div class="teacher-classes">
    <div class="page-header">
      <h1>班级管理</h1>
      <div>
        <a-button style="margin-right: 8px;" @click="refreshData">
          <template #icon><ReloadOutlined /></template>
          刷新数据
        </a-button>
      <a-button type="primary" @click="showCreateModal">
          <template #icon><PlusOutlined /></template>
        创建班级
      </a-button>
      </div>
    </div>
    
    <div class="classes-content">
      <a-spin :spinning="loading">
        <!-- 班级筛选 -->
        <div class="filter-row">
          <a-input-search
            v-model:value="searchKeyword"
            placeholder="搜索班级名称"
            style="width: 250px"
            @search="handleSearch"
          />
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

        <!-- 班级列表 -->
        <a-table
          :columns="columns"
          :dataSource="classes"
          :pagination="pagination"
          :loading="loading"
          rowKey="id"
          @change="handleTableChange"
        ></a-table>
      </a-spin>
    </div>

    <!-- 创建班级对话框 -->
    <a-modal
      v-model:open="createModalVisible"
      title="创建班级"
      @ok="handleCreateClass"
      :confirm-loading="submitLoading"
    >
      <a-form :model="form" :rules="rules" ref="formRef">
        <a-form-item name="name" label="班级名称" required>
          <a-input v-model:value="form.name" placeholder="请输入班级名称，例如：2025春A班" />
        </a-form-item>
        <a-form-item name="description" label="班级说明">
          <a-textarea v-model:value="form.description" placeholder="请输入班级说明" :rows="4" />
        </a-form-item>
        <a-form-item name="courseId" label="绑定课程">
          <a-select
            v-model:value="form.courseId"
            style="width: 100%"
            placeholder="选择关联课程（可选）"
            allowClear
            :loading="coursesLoading"
            :options-height="500"
            show-search
            :filter-option="(input: string, option: any) => 
              (option?.label?.toString() || '').toLowerCase().includes(input.toLowerCase())"
          >
            <a-empty v-if="teacherCourses.length === 0" description="暂无课程" />
            <template v-else>
              <a-select-option v-for="course in teacherCourses" :key="course.id" :value="course.id" :label="course.title || course.courseName">
                <div class="course-option">
                  <div class="course-title">{{ course.title || course.courseName }}</div>
                  <div class="course-info">
                    <span>类别: {{ course.courseType || '未设置' }}</span>
                    <span style="margin-left: 10px">状态: {{ course.status || '未设置' }}</span>
                  </div>
                </div>
              </a-select-option>
            </template>
          </a-select>
        </a-form-item>
        <a-form-item name="isDefault" v-if="form.courseId">
          <a-checkbox v-model:checked="form.isDefault">
            设为该课程的默认班级
          </a-checkbox>
        </a-form-item>
      </a-form>
    </a-modal>

    <!-- 编辑班级对话框 -->
    <a-modal
      v-model:open="editModalVisible"
      title="编辑班级"
      @ok="handleUpdateClass"
      :confirm-loading="submitLoading"
    >
      <a-form :model="form" :rules="rules" ref="editFormRef">
        <a-form-item name="name" label="班级名称" required>
          <a-input v-model:value="form.name" placeholder="请输入班级名称，例如：2025春A班" />
        </a-form-item>
        <a-form-item name="description" label="班级说明">
          <a-textarea v-model:value="form.description" placeholder="请输入班级说明" :rows="4" />
        </a-form-item>
        <a-form-item name="courseId" label="绑定课程">
          <a-select
            v-model:value="form.courseId"
            style="width: 100%"
            placeholder="选择关联课程（可选）"
            allowClear
            :loading="coursesLoading"
            :options-height="500"
            show-search
            :filter-option="(input: string, option: any) => 
              (option?.label?.toString() || '').toLowerCase().includes(input.toLowerCase())"
          >
            <a-empty v-if="teacherCourses.length === 0" description="暂无课程" />
            <template v-else>
              <a-select-option v-for="course in teacherCourses" :key="course.id" :value="course.id" :label="course.title || course.courseName">
                <div class="course-option">
                  <div class="course-title">{{ course.title || course.courseName }}</div>
                  <div class="course-info">
                    <span>类别: {{ course.courseType || '未设置' }}</span>
                    <span style="margin-left: 10px">状态: {{ course.status || '未设置' }}</span>
                  </div>
                </div>
              </a-select-option>
            </template>
          </a-select>
        </a-form-item>
        <a-form-item name="isDefault" v-if="form.courseId">
          <a-checkbox v-model:checked="form.isDefault">
            设为该课程的默认班级
          </a-checkbox>
        </a-form-item>
      </a-form>
    </a-modal>

    <!-- 绑定课程对话框 -->
    <a-modal
      v-model:open="bindCourseModalVisible"
      title="绑定课程"
      @ok="handleBindCourse"
      :confirm-loading="submitLoading"
    >
      <a-form :model="bindCourseForm" ref="bindCourseFormRef">
        <a-form-item name="courseId" label="选择课程" required>
          <a-select
            v-model:value="bindCourseForm.courseId"
            placeholder="请选择要绑定的课程"
            @change="handleBindCourseChange"
            style="width: 100%"
            :loading="coursesLoading"
            :options-height="500"
            show-search
            :filter-option="(input: string, option: any) => 
              (option?.label?.toString() || '').toLowerCase().includes(input.toLowerCase())"
          >
            <a-empty v-if="teacherCourses.length === 0" description="暂无课程" />
            <template v-else>
              <a-select-option v-for="course in teacherCourses" :key="course.id" :value="course.id" :label="course.title || course.courseName">
                <div class="course-option">
                  <div class="course-title">{{ course.title || course.courseName }}</div>
                  <div class="course-info">
                    <span>类别: {{ course.courseType || '未设置' }}</span>
                    <span style="margin-left: 10px">状态: {{ course.status || '未设置' }}</span>
                  </div>
                </div>
              </a-select-option>
            </template>
          </a-select>
        </a-form-item>
        <a-form-item name="isDefault">
          <a-checkbox v-model:checked="bindCourseForm.isDefault">
            设为该课程的默认班级
          </a-checkbox>
        </a-form-item>
      </a-form>
    </a-modal>
  </div>
</template>

<script setup lang="ts">
import { ref, reactive, onMounted, computed, h } from 'vue'
import { useRouter } from 'vue-router'
import { message } from 'ant-design-vue'
import type { TablePaginationConfig } from 'ant-design-vue'
import { 
  PlusOutlined, 
  ReloadOutlined, 
  EditOutlined, 
  DeleteOutlined, 
  LinkOutlined,
  EyeOutlined,
  CheckOutlined
} from '@ant-design/icons-vue'
import { 
  getClasses, 
  createClass, 
  updateClass, 
  deleteClass, 
  getTeacherCourses,
  type Class,
  type Course
} from '@/api/teacher'

const router = useRouter()
const loading = ref(false)
const submitLoading = ref(false)
const createModalVisible = ref(false)
const editModalVisible = ref(false)
const bindCourseModalVisible = ref(false)
const classes = ref<Class[]>([])
const teacherCourses = ref<Course[]>([])
const currentClassId = ref<number | null>(null)
const searchKeyword = ref('')
const courseFilter = ref<number | string>('')
const coursesLoading = ref(false)

// 表单相关
const formRef = ref()
const editFormRef = ref()
const bindCourseFormRef = ref()
const form = reactive({
  name: '',
  description: '',
  courseId: undefined as number | undefined,
  isDefault: false
})

const bindCourseForm = reactive({
  courseId: undefined as number | undefined,
  isDefault: false
})

// 表单验证规则
const rules = {
  name: [
    { required: true, message: '请输入班级名称', trigger: 'blur' },
    { max: 100, message: '班级名称不能超过100个字符', trigger: 'blur' }
  ]
}

// 班级列表表格
const columns = [
  {
    title: '班级名称',
    dataIndex: 'name',
    key: 'name'
  },
  {
    title: '课程绑定',
    key: 'courseBinding',
    customRender: ({ record }: { record: Class }) => {
      if (record.courseId) {
        return [
          h('a-tag', { color: 'green' }, '已绑定'),
          record.isDefault ? h('a-tag', { color: 'blue' }, '默认班级') : null
        ]
      } else {
        return h('a-tag', { color: 'orange' }, '未绑定课程')
      }
    }
  },
  {
    title: '学生数量',
    dataIndex: 'studentCount',
    key: 'studentCount'
  },
  {
    title: '创建时间',
    dataIndex: 'createTime',
    key: 'createTime'
  },
  {
    title: '操作',
    key: 'operation',
    width: '300px', /* 增加宽度以适应更大的图标和间距 */
    align: 'center',
    customRender: ({ record }: { record: Class }) => {
      return h('div', { class: 'action-buttons' }, [
        h('a', {
          class: 'action-icon',
          style: { 
            color: '#52c41a',
            fontSize: '16px',
            margin: '0 20px'
          },
          onClick: (e) => {
            e.preventDefault();
            viewClass(record.id);
          },
          title: '查看'
        }, [h(EyeOutlined)]),
        
        h('a', {
          class: 'action-icon',
          style: { 
            color: '#1890ff',
            fontSize: '16px',
            margin: '0 20px'
          },
          onClick: (e) => {
            e.preventDefault();
            showEditModal(record);
          },
          title: '编辑'
        }, [h(EditOutlined)]),
        
        h('a', {
          class: 'action-icon',
          style: { 
            color: '#ff4d4f',
            fontSize: '16px',
            margin: '0 20px'
          },
          onClick: (e) => {
            e.preventDefault();
            deleteClassItem(record.id);
          },
          title: '删除'
        }, [h(DeleteOutlined)])
      ])
    }
  }
]

// 分页配置
const pagination = reactive<TablePaginationConfig>({
  current: 1,
  pageSize: 10,
  total: 0,
  showSizeChanger: true,
  showTotal: (total) => `共 ${total} 条记录`
})

// 获取班级列表数据
const fetchClassList = async () => {
  try {
    loading.value = true
    const res = await getClasses({
      page: pagination.current,
      size: pagination.pageSize,
      keyword: searchKeyword.value,
      courseId: courseFilter.value === '' ? undefined : (typeof courseFilter.value === 'string' ? parseInt(courseFilter.value) : courseFilter.value)
    })
    
    console.log('班级列表原始响应:', res)
    
    // 处理不同格式的响应
    let classData = null
    let totalItems = 0
    
    // 检查数据是否在result.data中
    if (res.data && (res.data.records || res.data.content || res.data.list)) {
      classData = res.data.records || res.data.content || res.data.list
      totalItems = res.data.total || res.data.totalElements || 0
    }
    // 检查数据是否直接在外层
    else if (res.records || res.content || res.list) {
      classData = res.records || res.content || res.list
      totalItems = res.total || res.totalElements || 0
    }
    // 检查是否为Result包装类型
    else if (res.code === 200 && res.data) {
      if (Array.isArray(res.data)) {
        classData = res.data
        totalItems = res.data.length
      } else if (res.data.records || res.data.content || res.data.list) {
        classData = res.data.records || res.data.content || res.data.list
        totalItems = res.data.total || res.data.totalElements || 0
      }
    }
    
    if (classData) {
      classes.value = classData
      pagination.total = totalItems
    } else {
      console.error('无法识别的班级数据格式:', res)
      classes.value = []
      pagination.total = 0
    }
    
    console.log('处理后的班级数据:', classes.value)
  } catch (error) {
    console.error('获取班级列表失败:', error)
    message.error('获取班级列表失败，请稍后重试')
    classes.value = []
  } finally {
    loading.value = false
  }
}

// 获取教师课程列表
const fetchTeacherCourses = async () => {
  try {
    coursesLoading.value = true
    console.log('开始获取教师课程列表')
    
    const response = await getTeacherCourses()
    console.log('教师课程列表原始响应:', response)
    
    if (response && response.data) {
      // 处理不同的响应结构
      const responseData = response.data
      console.log('响应数据类型:', typeof responseData, '是否为数组:', Array.isArray(responseData))
      console.log('响应数据:', responseData)
      
      if (Array.isArray(responseData)) {
        // 直接是数组
        console.log('数据是数组，长度:', responseData.length)
        teacherCourses.value = responseData
      } else if (responseData.records || responseData.content || responseData.list) {
        // 分页响应
        console.log('数据是分页对象')
        teacherCourses.value = responseData.records || responseData.content || responseData.list || []
      } else if (responseData.code === 200 && responseData.data) {
        // Result包装的数据
        console.log('数据是Result包装的数据:', responseData.data)
        if (Array.isArray(responseData.data)) {
          teacherCourses.value = responseData.data
        } else if (responseData.data.records || responseData.data.content || responseData.data.list) {
          teacherCourses.value = responseData.data.records || responseData.data.content || responseData.data.list || []
        } else {
          teacherCourses.value = []
        }
      } else {
        // 其他情况
        console.warn('未能识别的课程数据结构:', responseData)
        teacherCourses.value = []
      }
    } else {
      console.warn('未获取到课程数据')
      teacherCourses.value = []
    }
    
    console.log('处理后的教师课程列表:', teacherCourses.value)
  } catch (error: any) {
    console.error('获取教师课程列表失败:', error)
    message.error('获取课程列表失败: ' + (error.response?.data?.message || error.message || '未知错误'))
    teacherCourses.value = []
  } finally {
    coursesLoading.value = false
  }
}

// 显示创建班级对话框
const showCreateModal = () => {
  // 重置表单
  Object.assign(form, {
    name: '',
    description: '',
    courseId: undefined,
    isDefault: false
  })
  createModalVisible.value = true
}

// 显示编辑班级对话框
const showEditModal = (record: Class) => {
  currentClassId.value = record.id
  Object.assign(form, {
    name: record.name,
    description: record.description || '',
    courseId: record.courseId,
    isDefault: record.isDefault || false
  })
  editModalVisible.value = true
}

// 显示绑定课程对话框
const showBindCourseModal = (record: Class) => {
  currentClassId.value = record.id
  Object.assign(bindCourseForm, {
    courseId: undefined,
    isDefault: false
  })
  bindCourseModalVisible.value = true
}

// 处理创建班级
const handleCreateClass = async () => {
  try {
    await formRef.value.validate()
    submitLoading.value = true
    
    // 准备提交的数据
    const formData = {
      name: form.name,
      description: form.description,
      courseId: form.courseId || null, // 确保courseId为null而不是undefined
      isDefault: form.courseId ? form.isDefault : false // 如果没有选择课程，则isDefault为false
    }
    
    // 调用实际API创建班级
    await createClass(formData)
    message.success('创建班级成功')
    createModalVisible.value = false
    fetchClassList()
  } catch (error: any) {
    console.error('创建班级失败:', error)
    message.error('创建班级失败: ' + (error.response?.data?.message || error.message || '未知错误'))
  } finally {
    submitLoading.value = false
  }
}

// 处理更新班级
const handleUpdateClass = async () => {
  if (!currentClassId.value) return
  
  try {
    await editFormRef.value.validate()
    submitLoading.value = true
    
    // 调用实际API更新班级
    await updateClass(currentClassId.value, form)
    message.success('更新班级成功')
    editModalVisible.value = false
    fetchClassList()
  } catch (error) {
    console.error('更新班级失败:', error)
    message.error('更新班级失败')
  } finally {
    submitLoading.value = false
  }
}

// 处理绑定课程
const handleBindCourse = async () => {
  if (!currentClassId.value || !bindCourseForm.courseId) {
    message.error('请选择要绑定的课程')
    return
  }
  
  try {
    submitLoading.value = true
    
    // 调用实际API绑定课程
    await updateClass(currentClassId.value, {
      courseId: bindCourseForm.courseId,
      isDefault: bindCourseForm.isDefault
    })
    
    message.success('课程绑定成功')
    bindCourseModalVisible.value = false
    fetchClassList()
  } catch (error) {
    console.error('绑定课程失败:', error)
    message.error('绑定课程失败')
  } finally {
    submitLoading.value = false
  }
}

// 删除班级
const deleteClassItem = async (id: number) => {
  try {
    loading.value = true
    
    // 调用实际API删除班级
    await deleteClass(id)
    message.success('删除班级成功')
    fetchClassList()
  } catch (error) {
    console.error('删除班级失败:', error)
    message.error('删除班级失败')
  } finally {
    loading.value = false
  }
}

// 查看班级详情
const viewClass = (id: number) => {
  router.push(`/teacher/classes/${id}`)
}

// 处理表格变化
const handleTableChange = (pag: TablePaginationConfig) => {
  pagination.current = pag.current || 1
  pagination.pageSize = pag.pageSize || 10
  fetchClassList()
}

// 处理搜索
const handleSearch = () => {
  pagination.current = 1
  fetchClassList()
}

// 处理课程筛选变化
const handleCourseFilterChange = () => {
  pagination.current = 1
  fetchClassList()
}

// 处理课程选择变化
const handleCourseChange = (value: number | undefined) => {
  form.courseId = value
  if (!value) {
    form.isDefault = false
  }
}

// 处理绑定课程变化
const handleBindCourseChange = (value: number | undefined) => {
  bindCourseForm.courseId = value
}

// 根据课程ID获取课程名称
const getCourseNameById = (courseId: number) => {
  const course = teacherCourses.value.find(c => c.id === courseId)
  return course ? (course.title || course.courseName) : '未知课程'
}

// 刷新数据
const refreshData = () => {
  fetchClassList()
}

// 组件挂载时获取数据
onMounted(() => {
  console.log('Classes组件已挂载')
  // 获取班级列表
  fetchClassList()
  // 获取教师课程列表
  fetchTeacherCourses()

  // Debug计时器，每3秒打印当前数据状态
  const timer = setInterval(() => {
    console.log('---当前数据状态---')
    console.log('班级列表:', classes.value)
    console.log('教师课程列表:', teacherCourses.value)
    console.log('加载状态:', loading.value, coursesLoading.value)
    console.log('----------------')
  }, 3000)

  // 30秒后清除计时器
  setTimeout(() => clearInterval(timer), 30000)
})
</script>

<style scoped>
.teacher-classes {
  padding: 20px;
}

.page-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 20px;
}

.filter-row {
  display: flex;
  margin-bottom: 20px;
}

.course-option {
  padding: 4px 0;
}

.course-title {
  font-weight: 500;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.course-info {
  font-size: 12px;
  color: #666;
  margin-top: 4px;
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}

.action-buttons {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 100px; /* 控制图标之间的间距，可以根据需要调整 */
}

.action-icon {
  font-size: 28px; /* 控制图标大小，增加到28px使其更明显 */
  cursor: pointer;
  transition: all 0.2s;
  display: flex;
  align-items: center;
  justify-content: center;
  width: 40px; /* 确保图标有足够的点击区域 */
  height: 40px; /* 确保图标有足够的点击区域 */
}

.action-icon:hover {
  opacity: 0.8;
  transform: scale(1.1); /* 鼠标悬停时稍微放大图标 */
}
</style> 