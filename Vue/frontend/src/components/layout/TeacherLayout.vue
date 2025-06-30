<template>
  <a-layout class="teacher-layout">
    <!-- 侧边栏 -->
    <a-layout-sider
      v-model:collapsed="collapsed"
      :trigger="null"
      collapsible
      class="sidebar"
      :width="200"
      :collapsed-width="80"
    >
      <div class="logo">
        <span v-if="!collapsed" class="logo-text">SmartClass</span>
        <div class="logo-spacer"></div>
        <a-button 
          v-if="!collapsed"
          type="text"
          @click="collapsed = !collapsed"
          class="sidebar-collapse-btn"
        >
          <MenuFoldOutlined />
        </a-button>
        <img src="/logo-mini.svg" alt="Logo" v-else />
      </div>
      
      <!-- 侧边栏折叠按钮，仅在收起状态显示 -->
      <a-menu
        v-if="collapsed"
        theme="dark"
        mode="inline"
        class="sidebar-menu"
      >
        <a-menu-item key="collapse" @click="collapsed = !collapsed">
          <template #icon>
            <MenuUnfoldOutlined />
          </template>
        </a-menu-item>
      </a-menu>
      
      <!-- 课程详情页侧边栏 -->
      <template v-if="isCourseDetailPage">
        <div class="course-sidebar-header">
          <BookOutlined />
          <span v-if="!collapsed" class="course-title">{{ currentCourse?.title || currentCourse?.courseName || '课程详情' }}</span>
        </div>
        <div v-if="!collapsed" class="course-sidebar-info">
          <div class="course-semester">{{ currentCourse?.semester || currentCourse?.term || '未设置学期' }}</div>
          <span>已完成进度: {{ calculateProgress(currentCourse?.startTime, currentCourse?.endTime) }}%</span>
        </div>
        <a-menu
          v-model:selectedKeys="courseSelectedKeys"
          mode="inline"
          theme="dark"
          class="sidebar-menu"
          @click="handleCourseMenuClick"
        >
          <a-sub-menu key="tasks">
            <template #icon>
              <FileTextOutlined />
            </template>
            <template #title>任务</template>
            <a-menu-item key="exams">考试</a-menu-item>
            <a-menu-item key="assignments">作业</a-menu-item>
          </a-sub-menu>
          <a-menu-item key="chapters">
            <template #icon>
              <OrderedListOutlined />
            </template>
            <span>章节</span>
          </a-menu-item>
          <a-menu-item key="discussions">
            <template #icon>
              <CommentOutlined />
            </template>
            <span>讨论</span>
          </a-menu-item>
          <a-menu-item key="resources">
            <template #icon>
              <FolderOutlined />
            </template>
            <span>资料</span>
          </a-menu-item>
          <a-menu-item key="wrongbook">
            <template #icon>
              <EditOutlined />
            </template>
            <span>错题集</span>
          </a-menu-item>
          <a-menu-item key="records">
            <template #icon>
              <HistoryOutlined />
            </template>
            <span>学习记录</span>
          </a-menu-item>
          <a-menu-item key="knowledge-map">
            <template #icon>
              <NodeIndexOutlined />
            </template>
            <span>知识图谱</span>
          </a-menu-item>
        </a-menu>
      </template>
      
      <!-- 普通侧边栏 -->
      <a-menu
        v-else
        v-model:selectedKeys="selectedKeys"
        v-model:openKeys="openKeys"
        mode="inline"
        theme="dark"
        class="sidebar-menu"
        @click="handleMenuClick"
      >
        <a-menu-item key="dashboard">
          <template #icon>
            <DashboardOutlined />
          </template>
          <span>工作台</span>
        </a-menu-item>
        
        <a-sub-menu key="classes">
          <template #icon>
            <TeamOutlined />
          </template>
          <template #title>班级管理</template>
          <a-menu-item key="classes-list">班级列表</a-menu-item>
          <a-menu-item key="classes-create">创建班级</a-menu-item>
        </a-sub-menu>
        
        <a-sub-menu key="courses">
          <template #icon>
            <BookOutlined />
          </template>
          <template #title>课程管理</template>
          <a-menu-item key="courses-list">课程列表</a-menu-item>
          <a-menu-item key="courses-create">创建课程</a-menu-item>
          <a-menu-item key="courses-chapters">章节管理</a-menu-item>
        </a-sub-menu>
        
        <a-sub-menu key="assignments">
          <template #icon>
            <FileTextOutlined />
          </template>
          <template #title>作业管理</template>
          <a-menu-item key="assignments-list">作业列表</a-menu-item>
          <a-menu-item key="assignments-create">布置作业</a-menu-item>
          <a-menu-item key="assignments-review">批改作业</a-menu-item>
        </a-sub-menu>
        
        <a-sub-menu key="students">
          <template #icon>
            <UserOutlined />
          </template>
          <template #title>学生管理</template>
          <a-menu-item key="students-list">学生列表</a-menu-item>
          <a-menu-item key="students-grades">成绩管理</a-menu-item>
        </a-sub-menu>
        
        <a-sub-menu key="resources">
          <template #icon>
            <FolderOutlined />
          </template>
          <template #title>教学资源</template>
          <a-menu-item key="resources-list">资源库</a-menu-item>
          <a-menu-item key="resources-upload">上传资源</a-menu-item>
        </a-sub-menu>
        
        <a-menu-item key="analytics">
          <template #icon>
            <BarChartOutlined />
          </template>
          <span>数据分析</span>
        </a-menu-item>
        
        <a-menu-item key="ai-assistant">
          <template #icon>
            <RobotOutlined />
          </template>
          <span>AI助手</span>
        </a-menu-item>
      </a-menu>
    </a-layout-sider>
    
    <!-- 主内容区 -->
    <a-layout class="main-layout">
      <!-- 顶部导航 -->
      <a-layout-header class="header">
        <div class="header-left">
          <div class="page-title">
            <a-breadcrumb class="breadcrumb">
              <a-breadcrumb-item v-for="item in breadcrumbItems" :key="item.path">
                <router-link v-if="item.path" :to="item.path">{{ item.title }}</router-link>
                <span v-else>{{ item.title }}</span>
              </a-breadcrumb-item>
            </a-breadcrumb>
          </div>
        </div>
        
        <div class="header-right">
          <!-- 通知 -->
          <a-badge :count="notificationCount" class="notification-badge">
            <a-button type="text" shape="circle" @click="showNotifications">
              <BellOutlined />
            </a-button>
          </a-badge>
          
          <!-- 用户菜单 -->
          <a-dropdown placement="bottomRight">
            <a-button type="text" class="user-info">
              <a-avatar :src="userInfo?.avatar" :size="32">
                {{ userInfo?.realName?.charAt(0) }}
              </a-avatar>
              <span class="username">{{ userInfo?.realName }}</span>
              <DownOutlined />
            </a-button>
            <template #overlay>
              <a-menu @click="handleUserMenuClick">
                <a-menu-item key="profile">
                  <UserOutlined />
                  个人资料
                </a-menu-item>
                <a-menu-item key="settings">
                  <SettingOutlined />
                  系统设置
                </a-menu-item>
                <a-menu-divider />
                <a-menu-item key="logout">
                  <LogoutOutlined />
                  退出登录
                </a-menu-item>
              </a-menu>
            </template>
          </a-dropdown>
        </div>
      </a-layout-header>
      
      <!-- 内容区域 -->
      <a-layout-content class="content">
        <div class="content-wrapper">
          <router-view />
        </div>
      </a-layout-content>
      
      <!-- AI助手悬浮按钮 -->
      <div class="ai-assistant-float" v-if="showAIAssistant">
        <a-button
          type="primary"
          shape="circle"
          size="large"
          @click="toggleAIAssistant"
          class="ai-button"
        >
          <RobotOutlined />
        </a-button>
      </div>
    </a-layout>
    
    <!-- 通知抽屉 -->
    <a-drawer
      v-model:open="notificationDrawerVisible"
      title="通知消息"
      placement="right"
      :width="400"
    >
      <a-list
        :data-source="notifications"
        :loading="notificationsLoading"
      >
        <template #renderItem="{ item }">
          <a-list-item>
            <a-list-item-meta
              :title="item.title"
              :description="item.description"
            >
              <template #avatar>
                <a-avatar :style="{ backgroundColor: item.color }">
                  <component :is="item.icon" />
                </a-avatar>
              </template>
            </a-list-item-meta>
            <template #actions>
              <span class="notification-time">{{ item.time }}</span>
            </template>
          </a-list-item>
        </template>
      </a-list>
    </a-drawer>
  </a-layout>
</template>

<script setup lang="ts">
import { ref, computed, watch, onMounted } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { useAuthStore } from '@/stores/auth'
import axios from 'axios'
import dayjs from 'dayjs'
import {
  DashboardOutlined,
  TeamOutlined,
  BookOutlined,
  FileTextOutlined,
  UserOutlined,
  FolderOutlined,
  BarChartOutlined,
  RobotOutlined,
  MenuUnfoldOutlined,
  MenuFoldOutlined,
  BellOutlined,
  DownOutlined,
  SettingOutlined,
  LogoutOutlined,
  OrderedListOutlined,
  CommentOutlined,
  EditOutlined,
  HistoryOutlined,
  NodeIndexOutlined
} from '@ant-design/icons-vue'
import { message } from 'ant-design-vue'

const route = useRoute()
const router = useRouter()
const authStore = useAuthStore()

// 侧边栏状态
const collapsed = ref(false)
const selectedKeys = ref<string[]>(['dashboard'])
const openKeys = ref<string[]>([])
const courseSelectedKeys = ref<string[]>(['chapters'])

// 用户信息
const userInfo = computed(() => authStore.user)

// 判断是否是课程详情页
const isCourseDetailPage = computed(() => {
  return /^\/teacher\/courses\/\d+$/.test(route.path);
})

// 当前课程信息
const currentCourse = ref<any>(null)

// 计算课程进度
const calculateProgress = (startTime: string | undefined, endTime: string | undefined): number => {
  if (!startTime || !endTime) return 0

  try {
    const start = new Date(startTime).getTime()
    const end = new Date(endTime).getTime()
    const now = Date.now()

    if (now <= start) return 0
    if (now >= end) return 100

    const total = end - start
    const current = now - start
    return Math.round((current / total) * 100)
  } catch (e) {
    console.error('计算进度错误:', e)
    return 0
  }
}

// 获取课程信息
const fetchCourseInfo = async (courseId: number) => {
  try {
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
    
    const response = await axios.get(`/api/teacher/courses/${courseId}`, {
      headers: {
        'Authorization': authToken
      }
    });
    
    if (response.data && response.data.code === 200) {
      currentCourse.value = response.data.data;
      console.log('获取到的课程信息:', currentCourse.value);
    } else {
      message.error(response.data?.message || '获取课程信息失败');
    }
  } catch (error) {
    console.error('获取课程信息失败:', error);
    message.error('获取课程信息失败，请检查网络连接');
  }
}

// 监听路由变化，获取课程信息和设置正确的菜单选中状态
watch(() => route.params.id, (newId) => {
  if (isCourseDetailPage.value && newId) {
    fetchCourseInfo(Number(newId));
    
    // 根据URL中的view参数设置正确的菜单选中状态
    const viewParam = route.query.view as string;
    if (viewParam) {
      courseSelectedKeys.value = [viewParam];
    } else {
      courseSelectedKeys.value = ['chapters'];
    }
  }
}, { immediate: true });

// 监听路由query参数变化，更新菜单选中状态
watch(() => route.query.view, (newView) => {
  if (isCourseDetailPage.value && newView) {
    courseSelectedKeys.value = [newView as string];
  }
}, { immediate: true });

// 面包屑导航
const breadcrumbItems = computed(() => {
  const items = [{ title: '首页', path: '/teacher' }]
  
  if (route.meta?.breadcrumb) {
    items.push(...route.meta.breadcrumb as any[])
  }
  
  return items
})

// 通知相关
const notificationCount = ref(5)
const notificationDrawerVisible = ref(false)
const notificationsLoading = ref(false)
const notifications = ref([
  {
    id: 1,
    title: '新的作业提交',
    description: '张三提交了数学作业',
    time: '2分钟前',
    icon: 'FileTextOutlined',
    color: '#1890ff'
  },
  {
    id: 2,
    title: '班级申请',
    description: '李四申请加入高一(1)班',
    time: '10分钟前',
    icon: 'TeamOutlined',
    color: '#52c41a'
  }
])

// AI助手
const showAIAssistant = ref(true)

// 监听路由变化
watch(
  () => route.fullPath,
  () => {
    updateSelectedKeys()
    
    // 处理课程详情页的视图切换
    if (isCourseDetailPage.value) {
      const view = route.query.view as string
      if (view) {
        courseSelectedKeys.value = [view]
      } else {
        courseSelectedKeys.value = ['chapters']
      }
    }
  },
  { immediate: true }
)

// 更新选中的菜单项
function updateSelectedKeys() {
  const pathSegments = route.path.split('/').filter(Boolean)
  if (pathSegments.length >= 2) {
    const key = pathSegments.slice(1).join('-')
    selectedKeys.value = [key || 'dashboard']
    
    // 自动展开父菜单
    if (key.includes('-')) {
      const parentKey = key.split('-')[0]
      if (!openKeys.value.includes(parentKey)) {
        openKeys.value.push(parentKey)
      }
    }
  }
}

// 菜单点击处理
function handleMenuClick({ key }: { key: string }) {
  const routeMap: Record<string, string> = {
    'dashboard': '/teacher',
    'classes-list': '/teacher/classes',
    'classes-create': '/teacher/classes/create',
    'courses-list': '/teacher/courses',
    'courses-create': '/teacher/courses/create',
    'courses-chapters': '/teacher/courses/chapters',
    'assignments-list': '/teacher/assignments',
    'assignments-create': '/teacher/assignments/create',
    'assignments-review': '/teacher/assignments/review',
    'students-list': '/teacher/students',
    'students-grades': '/teacher/students/grades',
    'resources-list': '/teacher/resources',
    'resources-upload': '/teacher/resources/upload',
    'analytics': '/teacher/analytics',
    'ai-assistant': '/teacher/ai-assistant'
  }
  
  const targetRoute = routeMap[key]
  if (targetRoute && targetRoute !== route.path) {
    router.push(targetRoute)
  }
}

// 用户菜单点击处理
function handleUserMenuClick({ key }: { key: string }) {
  switch (key) {
    case 'profile':
      router.push('/teacher/profile')
      break
    case 'settings':
      router.push('/teacher/settings')
      break
    case 'logout':
      handleLogout()
      break
  }
}

// 退出登录
const handleLogout = async () => {
  try {
    await authStore.logout()
    message.success('退出登录成功')
    router.push('/login')
  } catch (error) {
    console.error('退出登录失败:', error)
    message.error('退出登录失败')
  }
}

// 显示通知
function showNotifications() {
  notificationDrawerVisible.value = true
}

// 切换AI助手
function toggleAIAssistant() {
  // TODO: 实现AI助手功能
  message.info('AI助手功能开发中...')
}

// 课程详情页菜单点击处理
function handleCourseMenuClick({ key }: { key: string }) {
  const courseId = route.params.id
  if (!courseId) return
  
  // 更新路由查询参数，保持在同一页面但切换视图
  router.push({
    path: `/teacher/courses/${courseId}`,
    query: { view: key }
  })
}

onMounted(() => {
  // 初始化时设置正确的菜单选中状态
  updateSelectedKeys()
})
</script>

<style scoped>
.teacher-layout {
  min-height: 100vh;
}

.sidebar {
  position: fixed;
  left: 0;
  top: 0;
  bottom: 0;
  z-index: 100;
  box-shadow: 2px 0 8px rgba(0, 0, 0, 0.15);
  width: 200px !important;
}

.logo {
  height: 64px;
  display: flex;
  align-items: center;
  padding-left: 16px;
  padding-right: 16px;
  margin-bottom: 16px;
}

.logo img {
  height: 32px;
}

.logo-text {
  font-size: 24px;
  font-weight: 600;
  color: white;
  letter-spacing: 1px;
}

.logo-spacer {
  flex: 1;
}

.sidebar-collapse-btn {
  color: rgba(255, 255, 255, 0.65);
  font-size: 16px;
  padding: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  background: transparent;
  border: none;
}

.sidebar-collapse-btn:hover {
  color: #fff;
}

.sidebar-menu {
  border-right: none;
}

.main-layout {
  margin-left: 200px;
  transition: margin-left 0.2s;
  min-width: 800px;
  max-width: 1800px;
  width: calc(100vw - 200px);
}

.teacher-layout :deep(.ant-layout-sider-collapsed) + .main-layout {
  margin-left: 80px;
  width: calc(100vw - 80px);
}

.header {
  background: #fff;
  padding: 0 24px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);
  position: sticky;
  top: 0;
  z-index: 99;
}

.header-left {
  display: flex;
  align-items: center;
  gap: 16px;
}

.header-trigger {
  font-size: 18px;
  cursor: pointer;
  transition: color 0.3s;
  display: flex;
  align-items: center;
  justify-content: center;
  color: #333;
}

.header-trigger:hover {
  color: #1890ff;
}

.page-title {
  margin-left: 16px;
  display: flex;
  align-items: center;
}

.breadcrumb {
  font-size: 14px;
  color: #333;
}

.header-right {
  display: flex;
  align-items: center;
  gap: 16px;
}

.notification-badge {
  cursor: pointer;
}

.user-info {
  display: flex;
  align-items: center;
  gap: 8px;
  height: 40px;
  padding: 0 12px;
  border-radius: 8px;
  transition: background-color 0.3s;
}

.user-info:hover {
  background-color: #f5f5f5;
}

.username {
  font-weight: 500;
  margin-left: 4px;
}

.content {
  margin: 24px;
  padding: 0;
  min-height: calc(100vh - 112px);
  max-width: 1600px;
  margin: 24px auto;
}

.content-wrapper {
  background: #fff;
  border-radius: 12px;
  min-height: 100%;
  overflow: hidden;
  padding: 24px;
}

.ai-assistant-float {
  position: fixed;
  right: 24px;
  bottom: 24px;
  z-index: 1000;
}

.ai-button {
  width: 56px;
  height: 56px;
  box-shadow: 0 4px 12px rgba(24, 144, 255, 0.3);
  font-size: 24px;
}

.notification-time {
  color: #999;
  font-size: 12px;
}

/* 响应式设计 */
@media (min-width: 1400px) {
  .main-layout {
    max-width: none;
  }
  
  .content {
    max-width: 1800px;
  }
}

@media (max-width: 1200px) {
  .main-layout {
    min-width: 800px;
  }
  
  .content {
    max-width: 100%;
    margin: 16px;
  }
}

.course-sidebar-header {
  display: flex;
  align-items: center;
  padding: 16px;
  color: white;
}

.course-title {
  font-size: 18px;
  font-weight: 500;
  margin-left: 8px;
  color: white;
}

.course-sidebar-info {
  padding: 0 16px 16px;
  color: rgba(255, 255, 255, 0.85);
}

.course-semester {
  margin-bottom: 12px;
  font-size: 14px;
}

.course-progress {
  margin-bottom: 16px;
}

.course-progress :deep(.ant-progress-bg) {
  background-color: #1890ff;
}

.course-progress span {
  display: block;
  margin-top: 8px;
  font-size: 13px;
  color: rgba(255, 255, 255, 0.65);
}
</style>