<template>
  <div class="class-members">
    <a-spin :spinning="loading">
      <a-row>
        <a-col :span="24">
          <a-card class="members-card">
            <template #title>
              <div class="members-title">
                <span>同学列表</span>
                <a-tag color="blue">{{ students.length }}人</a-tag>
              </div>
            </template>
            <template #extra>
              <a-button type="primary" @click="goToClassInfo">
                <template #icon><InfoCircleOutlined /></template>
                班级信息
              </a-button>
            </template>
            
            <a-input-search
              v-model:value="searchText"
              placeholder="搜索同学"
              style="margin-bottom: 16px; width: 300px;"
              @search="onSearch"
              allowClear
            />
            
            <a-table
              :dataSource="filteredStudents"
              :columns="columns"
              :pagination="{ pageSize: 10, showSizeChanger: true, pageSizeOptions: ['10', '20', '50', '100'] }"
              :rowKey="(record: ClassStudent) => record.id"
              :loading="loading"
            >
              <template #bodyCell="{ column, record }: { column: any, record: ClassStudent }">
                <template v-if="column.dataIndex === 'avatar'">
                  <a-avatar :src="record.user?.avatar" :size="40">
                    {{ record.user?.realName?.charAt(0) || record.user?.username?.charAt(0) || '?' }}
                  </a-avatar>
                </template>
                
                <template v-else-if="column.dataIndex === 'name'">
                  <div class="student-name">
                    <span class="real-name">{{ record.user?.realName || '未设置姓名' }}</span>
                    <span class="username">({{ record.user?.username }})</span>
                  </div>
                </template>
                
                <template v-else-if="column.dataIndex === 'studentId'">
                  {{ getStudentId(record) }}
                </template>
                
                <template v-else-if="column.dataIndex === 'enrollmentStatus'">
                  <a-tag :color="getStatusColor(record.enrollmentStatus)">
                    {{ getStatusText(record.enrollmentStatus) }}
                  </a-tag>
                </template>
                
                <template v-else-if="column.dataIndex === 'gpa'">
                  <div class="gpa-info">
                    <span>{{ record.gpa || '-' }}</span>
                    <a-tag v-if="record.gpaLevel" :color="getGpaLevelColor(record.gpaLevel)">
                      {{ record.gpaLevel }}
                    </a-tag>
                  </div>
                </template>
                

              </template>
            </a-table>
          </a-card>
        </a-col>
      </a-row>
      
      <a-result
        v-if="!loading && error"
        status="warning"
        :title="error"
        sub-title="您可能尚未加入任何班级"
      >
        <template #extra>
          <a-button type="primary" @click="reload">
            重试
          </a-button>
        </template>
      </a-result>
    </a-spin>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { InfoCircleOutlined } from '@ant-design/icons-vue'
import { getClassMembers } from '@/api/student'
import type { ClassStudent } from '@/api/student'
import { message } from 'ant-design-vue'

const router = useRouter()
const loading = ref(true)
const students = ref<ClassStudent[]>([])
const classId = ref<number | null>(null)
const error = ref<string | null>(null)
const searchText = ref('')

// 表格列定义
const columns = [
  {
    title: '头像',
    dataIndex: 'avatar',
    width: 80,
  },
  {
    title: '姓名',
    dataIndex: 'name',
    sorter: (a: ClassStudent, b: ClassStudent) => {
      const aName = a.user?.realName || a.user?.username || ''
      const bName = b.user?.realName || b.user?.username || ''
      return aName.localeCompare(bName)
    },
  },
  {
    title: '学号',
    dataIndex: 'studentId',
    sorter: (a: ClassStudent, b: ClassStudent) => {
      const aId = getStudentId(a)
      const bId = getStudentId(b)
      return aId.localeCompare(bId)
    },
  },
  {
    title: '学籍状态',
    dataIndex: 'enrollmentStatus',
    filters: [
      { text: '在读', value: 'ENROLLED' },
      { text: '休学', value: 'SUSPENDED' },
      { text: '毕业', value: 'GRADUATED' },
      { text: '退学', value: 'DROPPED_OUT' },
    ],
    onFilter: (value: string, record: ClassStudent) => record.enrollmentStatus === value,
  },
  {
    title: 'GPA',
    dataIndex: 'gpa',
    sorter: (a: ClassStudent, b: ClassStudent) => {
      const aGpa = a.gpa || 0
      const bGpa = b.gpa || 0
      return aGpa - bGpa
    },
  },

]

// 获取学生ID
const getStudentId = (student: ClassStudent): string => {
  return student.id.toString()
}

// 过滤后的学生列表
const filteredStudents = computed(() => {
  if (!searchText.value) {
    return students.value
  }
  
  const search = searchText.value.toLowerCase()
  return students.value.filter(student => {
    const realName = student.user?.realName?.toLowerCase() || ''
    const username = student.user?.username?.toLowerCase() || ''
    const studentId = getStudentId(student).toLowerCase()
    
    return realName.includes(search) || 
           username.includes(search) || 
           studentId.includes(search)
  })
})

// 获取班级同学列表
const fetchClassMembers = async () => {
  loading.value = true
  error.value = null
  
  try {
    const data = await getClassMembers()
    students.value = data.students
    classId.value = data.classId
  } catch (err: any) {
    error.value = err.message || '获取班级同学列表失败'
    message.error(error.value)
  } finally {
    loading.value = false
  }
}

// 获取学籍状态颜色
const getStatusColor = (status?: string) => {
  switch (status) {
    case 'ENROLLED': return 'green'
    case 'SUSPENDED': return 'orange'
    case 'GRADUATED': return 'blue'
    case 'DROPPED_OUT': return 'red'
    default: return 'default'
  }
}

// 获取学籍状态文本
const getStatusText = (status?: string) => {
  switch (status) {
    case 'ENROLLED': return '在读'
    case 'SUSPENDED': return '休学'
    case 'GRADUATED': return '毕业'
    case 'DROPPED_OUT': return '退学'
    default: return status || '未知'
  }
}

// 获取GPA等级颜色
const getGpaLevelColor = (level?: string) => {
  switch (level) {
    case 'A': return 'green'
    case 'B': return 'cyan'
    case 'C': return 'blue'
    case 'D': return 'orange'
    case 'F': return 'red'
    default: return 'default'
  }
}

// 搜索处理
const onSearch = (value: string) => {
  searchText.value = value
}



// 跳转到班级信息页面
const goToClassInfo = () => {
  router.push('/student/classes/info')
}

// 重新加载
const reload = () => {
  fetchClassMembers()
}

onMounted(() => {
  fetchClassMembers()
})
</script>

<style scoped>
.class-members {
  padding: 20px;
}

.members-card {
  margin-bottom: 20px;
}

.members-title {
  display: flex;
  align-items: center;
  gap: 10px;
}

.student-name {
  display: flex;
  flex-direction: column;
}

.real-name {
  font-weight: bold;
}

.username {
  color: #666;
  font-size: 12px;
}

.gpa-info {
  display: flex;
  align-items: center;
  gap: 8px;
}
</style> 