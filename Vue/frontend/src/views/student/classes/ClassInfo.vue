<template>
  <div class="class-info">
    <a-spin :spinning="loading">
      <a-row v-if="classInfo">
        <a-col :span="24">
          <a-card class="class-card">
            <template #title>
              <div class="class-title">
                <span class="class-name">{{ classInfo.name }}</span>
                <a-tag v-if="classInfo.isDefault" color="blue">默认班级</a-tag>
              </div>
            </template>
            <template #extra>
              <a-button type="primary" @click="goToMembers">
                <template #icon><TeamOutlined /></template>
                查看同学
              </a-button>
            </template>
            
            <a-descriptions bordered :column="{ xxl: 4, xl: 3, lg: 3, md: 3, sm: 2, xs: 1 }">
              <a-descriptions-item label="班级名称">
                {{ classInfo.name }}
              </a-descriptions-item>
              <a-descriptions-item label="学生人数">
                {{ classInfo.studentCount || 0 }}人
              </a-descriptions-item>
              <a-descriptions-item label="创建时间">
                {{ formatDate(classInfo.createTime) }}
              </a-descriptions-item>
              <a-descriptions-item label="班级描述" :span="3">
                {{ classInfo.description || '暂无描述' }}
              </a-descriptions-item>
            </a-descriptions>

            <a-divider orientation="left">关联课程</a-divider>
            
            <div v-if="classInfo.course" class="course-info">
              <a-card hoverable class="course-card">
                <a-row>
                  <a-col :xs="24" :sm="8" :md="6" :lg="4">
                    <div class="course-image">
                      <img 
                        :src="classInfo.course.coverImage || 'https://via.placeholder.com/200x150?text=课程封面'" 
                        :alt="classInfo.course.title"
                      />
                    </div>
                  </a-col>
                  <a-col :xs="24" :sm="16" :md="18" :lg="20">
                    <div class="course-details">
                      <h3 class="course-title">{{ classInfo.course.title }}</h3>
                      <p class="course-description">{{ classInfo.course.description || '暂无课程描述' }}</p>
                      <a-button type="primary" @click="goToCourse(classInfo.course.id)">
                        查看课程
                      </a-button>
                    </div>
                  </a-col>
                </a-row>
              </a-card>
            </div>
            <a-empty v-else description="暂无关联课程" />
          </a-card>
        </a-col>
      </a-row>
      
      <a-result
        v-else-if="!loading && error"
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
import { ref, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { TeamOutlined } from '@ant-design/icons-vue'
import { getClassInfo } from '@/api/student'
import type { ClassInfo } from '@/api/student'
import { message } from 'ant-design-vue'

const router = useRouter()
const loading = ref(true)
const classInfo = ref<ClassInfo | null>(null)
const error = ref<string | null>(null)

// 获取班级信息
const fetchClassInfo = async () => {
  loading.value = true
  error.value = null
  
  try {
    const data = await getClassInfo()
    classInfo.value = data
  } catch (err: any) {
    error.value = err.message || '获取班级信息失败'
    message.error(error.value)
  } finally {
    loading.value = false
  }
}

// 格式化日期
const formatDate = (dateStr?: string) => {
  if (!dateStr) return '未知'
  
  try {
    const date = new Date(dateStr)
    return date.toLocaleString('zh-CN', {
      year: 'numeric',
      month: '2-digit',
      day: '2-digit',
      hour: '2-digit',
      minute: '2-digit'
    })
  } catch (e) {
    return dateStr
  }
}

// 跳转到课程页面
const goToCourse = (courseId: number) => {
  router.push(`/student/courses/${courseId}`)
}

// 跳转到班级成员页面
const goToMembers = () => {
  router.push('/student/classes/members')
}

// 重新加载
const reload = () => {
  fetchClassInfo()
}

onMounted(() => {
  fetchClassInfo()
})
</script>

<style scoped>
.class-info {
  padding: 20px;
}

.class-card {
  margin-bottom: 20px;
}

.class-title {
  display: flex;
  align-items: center;
}

.class-name {
  font-size: 18px;
  font-weight: bold;
  margin-right: 10px;
}

.course-info {
  margin-top: 20px;
}

.course-card {
  background-color: #fafafa;
}

.course-image {
  padding: 10px;
  display: flex;
  justify-content: center;
  align-items: center;
}

.course-image img {
  max-width: 100%;
  max-height: 150px;
  object-fit: cover;
  border-radius: 4px;
}

.course-details {
  padding: 10px 20px;
}

.course-title {
  font-size: 16px;
  font-weight: bold;
  margin-bottom: 10px;
}

.course-description {
  color: #666;
  margin-bottom: 15px;
  display: -webkit-box;
  -webkit-line-clamp: 3;
  -webkit-box-orient: vertical;
  overflow: hidden;
  text-overflow: ellipsis;
}
</style> 