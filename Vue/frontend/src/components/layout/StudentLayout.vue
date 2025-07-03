<template>
  <a-layout class="student-layout">
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
      </div>
      
      <!-- 主菜单 -->
      <a-menu
        v-model:selectedKeys="selectedKeys"
        v-model:openKeys="openKeys"
        mode="inline"
        theme="dark"
        class="sidebar-menu"
        @click="handleMenuClick"
      >
        <!-- 折叠按钮，仅在收起状态显示 -->
        <a-menu-item v-if="collapsed" key="collapse" @click="collapsed = !collapsed" class="collapse-menu-item">
          <template #icon>
            <MenuUnfoldOutlined />
          </template>
        </a-menu-item>
        <a-menu-item key="dashboard">
          <template #icon>
            <HomeOutlined />
          </template>
          <span>首页</span>
        </a-menu-item>
        
        <a-menu-item key="courses">
          <template #icon>
            <ReadOutlined />
          </template>
          <span>我的课程</span>
        </a-menu-item>
        
        <a-menu-item key="assignments">
          <template #icon>
            <FileTextOutlined />
          </template>
          <span>任务中心</span>
        </a-menu-item>
        
        <a-menu-item key="grades">
          <template #icon>
            <TrophyOutlined />
          </template>
          <span>成绩查询</span>
        </a-menu-item>
        
        <a-sub-menu key="classes">
          <template #icon>
            <TeamOutlined />
          </template>
          <template #title>班级</template>
          <a-menu-item key="classes-info">班级信息</a-menu-item>
          <a-menu-item key="classes-members">同学列表</a-menu-item>
        </a-sub-menu>
        
        <a-sub-menu key="resources">
          <template #icon>
            <BookOutlined />
          </template>
          <template #title>学习资源</template>
          <a-menu-item key="resources-library">资源库</a-menu-item>
        </a-sub-menu>
        
        <a-menu-item key="schedule">
          <template #icon>
            <CalendarOutlined />
          </template>
          <span>学习计划</span>
        </a-menu-item>
        
        <a-menu-item key="ai-tutor">
          <template #icon>
            <RobotOutlined />
          </template>
          <span>AI学习助手</span>
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
                  学习设置
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
      
      <!-- AI学习助手悬浮按钮 -->
      <div class="ai-tutor-float" v-if="showAITutor">
        <a-button
          type="primary"
          shape="circle"
          size="large"
          @click="toggleAITutor"
          class="ai-button"
        >
          <RobotOutlined />
        </a-button>
        <div class="ai-tooltip" v-if="showAITooltip">
          有学习问题？点击我获得帮助！
        </div>
      </div>
    </a-layout>
    
    <!-- 通知抽屉 -->
    <a-drawer
      v-model:open="notificationDrawerVisible"
      title="消息通知"
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
    
    <!-- 学习提醒弹窗 -->
    <a-modal
      v-model:open="studyReminderVisible"
      title="学习提醒"
      :footer="null"
      width="400"
    >
      <div class="study-reminder">
        <div class="reminder-icon">
          <ClockCircleOutlined style="font-size: 48px; color: #1890ff;" />
        </div>
        <div class="reminder-content">
          <h3>该休息一下了！</h3>
          <p>您已经连续学习了 {{ studyDuration }} 分钟，建议休息 10-15 分钟后继续学习。</p>
        </div>
        <div class="reminder-actions">
          <a-button @click="studyReminderVisible = false">继续学习</a-button>
          <a-button type="primary" @click="takeBreak">休息一下</a-button>
        </div>
      </div>
    </a-modal>
  </a-layout>
</template>

<script setup lang="ts">
import { ref, computed, watch, onMounted, onUnmounted } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { useAuthStore } from '@/stores/auth'
import {
  DashboardOutlined,
  FileTextOutlined,
  TrophyOutlined,
  TeamOutlined,
  BookOutlined,
  CalendarOutlined,
  RobotOutlined,
  MenuUnfoldOutlined,
  MenuFoldOutlined,
  BellOutlined,
  DownOutlined,
  UserOutlined,
  SettingOutlined,
  LogoutOutlined,
  ClockCircleOutlined,
  HomeOutlined,
  ReadOutlined
} from '@ant-design/icons-vue'
import { message } from 'ant-design-vue'
import { emitter, APP_EVENTS } from '@/App.vue'

const route = useRoute()
const router = useRouter()
const authStore = useAuthStore()

// 侧边栏状态
const collapsed = ref(false)
const selectedKeys = ref<string[]>(['dashboard'])
const openKeys = ref<string[]>([])

// 用户信息
const userInfo = computed(() => authStore.user)

// 学习进度
const todayProgress = ref(65)

// 面包屑导航
const breadcrumbItems = computed(() => {
  // 始终显示首页和全部课程两个面包屑项
  let items = [
    { title: '首页', path: '/student' },
    { title: '全部课程', path: '/courses' }
  ];
  
  // 添加额外的面包屑项（如果有）
  if (route.meta?.breadcrumb) {
    items.push(...route.meta.breadcrumb as any[])
  }
  
  return items
})

// 通知相关
const notificationCount = ref(3)
const notificationDrawerVisible = ref(false)
const notificationsLoading = ref(false)
const notifications = ref([
  {
    id: 1,
    title: '新作业通知',
    description: '数学老师布置了新的作业,',
    time: '5分钟?,',
    icon: 'FileTextOutlined',
    color: '#1890ff'
  },
  {
    id: 2,
    title: '成绩发布',
    description: '英语测试成绩已发布,',
    time: '1小时?,',
    icon: 'TrophyOutlined',
    color: '#52c41a'
  },
  {
    id: 3,
    title: '学习提醒',
    description: '今日学习计划还有2项未完成',
    time: '2小时�?,',
    icon: 'CalendarOutlined',
    color: '#faad14'
  }
])

// AI学习助手
const showAITutor = ref(true)
const showAITooltip = ref(false)

// 学习提醒
const studyReminderVisible = ref(false)
const studyDuration = ref(45)
let studyTimer: number | null = null

// 监听路由变化更新选中菜单
watch(
  () => route.path,
  (newPath) => {
    updateSelectedKeys(newPath)
  },
  { immediate: true }
)

// 更新选中的菜单项
function updateSelectedKeys(path: string) {
  const pathSegments = path.split('/').filter(Boolean)
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
    'dashboard': '/student',
    'courses': '/student/courses',
    'assignments': '/student/assignments',
    'grades': '/student/grades',
    'classes-info': '/student/classes/info',
    'classes-members': '/student/classes/members',
    'resources-library': '/student/resources',
    'schedule': '/student/schedule',
    'ai-tutor': '/student/ai-tutor'
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
      router.push('/student/profile')
      break
    case 'settings':
      router.push('/student/settings')
      break
    case 'logout':
      handleLogout()
      break
  }
}

// 退出登录
function handleLogout() {
  authStore.logout()
  message.success('已退出登录')
  router.push('/login')
}

// 显示通知
function showNotifications() {
  notificationDrawerVisible.value = true
}

// 切换AI学习助手
function toggleAITutor() {
  // TODO: 实现AI学习助手功能
  message.info('AI学习助手功能开发中...')
}

// 休息提醒
function takeBreak() {
  studyReminderVisible.value = false
  message.success('记得适当休息，保护视力哦！')
  // 可以添加休息计时
}

// 开始学习计时
function startStudyTimer() {
  studyTimer = setInterval(() => {
    studyDuration.value += 1
    
    // 45分钟提醒一次
    if (studyDuration.value % 45 === 0) {
      studyReminderVisible.value = true
    }
  }, 60000) // 每分钟更新一次
}

// 显示AI助手提示
function showAITooltipTemporarily() {
  showAITooltip.value = true
  setTimeout(() => {
    showAITooltip.value = false
  }, 3000)
}

onMounted(() => {
  // 初始化时设置正确的菜单选中状态
  updateSelectedKeys(route.path)
  
  // 开始学习计时
  startStudyTimer()
  
  // 5秒后显示AI助手提示
  setTimeout(() => {
    showAITooltipTemporarily()
  }, 5000)

  // 监听侧边栏收起事件
  emitter.on(APP_EVENTS.COLLAPSE_SIDEBAR, () => {
    collapsed.value = true
  })
})

onUnmounted(() => {
  if (studyTimer) {
    clearInterval(studyTimer)
  }
  
  // 移除事件监听
})
</script>

<style scoped>
.student-layout {
  min-height: 100vh;
  background: #f5f5f5;
}

.sidebar {
  position: fixed;
  left: 0;
  top: 0;
  bottom: 0;
  z-index: 100;
  overflow: auto;
  height: 100vh;
  box-shadow: 2px 0 8px rgba(0, 0, 0, 0.15);
}

.logo {
  display: flex;
  align-items: center;
  height: 64px;
  padding: 0 16px;
  background: #001529;
  position: relative;
}

.student-layout :deep(.ant-layout-sider-collapsed) .logo {
  padding: 0;
  justify-content: center;
}

.logo img {
  height: 32px;
}

.logo-text {
  font-size: 18px;
  font-weight: 500;
  margin-left: 8px;
  color: white;
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

/* 确保折叠状态下的图标正确显示 */
.student-layout :deep(.ant-layout-sider-collapsed) .sidebar-menu .ant-menu-item {
  padding: 0 calc(50% - 16px) !important;
  display: flex;
  justify-content: center;
}

.student-layout :deep(.ant-layout-sider-collapsed) .sidebar-menu .ant-menu-submenu-title {
  padding: 0 calc(50% - 16px) !important;
  display: flex;
  justify-content: center;
}

.student-layout :deep(.ant-layout-sider-collapsed) .ant-menu-item .anticon,
.student-layout :deep(.ant-layout-sider-collapsed) .ant-menu-submenu-title .anticon {
  margin-right: 0;
  font-size: 16px;
}

.main-layout {
  margin-left: 200px;
  transition: margin-left 0.2s;
  min-width: 800px;
  max-width: 1800px;
  width: calc(100vw - 200px);
}

.student-layout :deep(.ant-layout-sider-collapsed) + .main-layout {
  margin-left: 80px;
  width: calc(100vw - 80px);
}

/* 折叠状态下隐藏文本 */
.student-layout :deep(.ant-layout-sider-collapsed) .ant-menu-item span:not(.anticon),
.student-layout :deep(.ant-layout-sider-collapsed) .ant-menu-submenu-title span:not(.anticon) {
  display: none;
}

/* 折叠状态下的展开按钮样式 */
.collapse-menu-item {
  margin-top: 8px !important;
  margin-bottom: 8px !important;
}

.student-layout :deep(.ant-layout-sider-collapsed) .collapse-menu-item .anticon {
  font-size: 18px;
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
  margin: 0;
  padding: 0;
  min-height: calc(100vh - 64px);
  width: 100%;
}

.content-wrapper {
  background: #fff;
  min-height: 100%;
  overflow: hidden;
  padding: 24px;
  border-radius: 0;
}

.ai-tutor-float {
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

.ai-tooltip {
  position: absolute;
  right: 70px;
  bottom: 10px;
  background: #1890ff;
  color: white;
  padding: 8px 12px;
  border-radius: 8px;
  font-size: 12px;
  white-space: nowrap;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.15);
  animation: fadeInOut 3s ease-in-out;
}

.ai-tooltip::after {
  content: '';
  position: absolute;
  right: -6px;
  top: 50%;
  transform: translateY(-50%);
  border: 6px solid transparent;
  border-left-color: #1890ff;
}

@keyframes fadeInOut {
  0%, 100% {
    opacity: 0;
    transform: translateX(10px);
  }
  10%, 90% {
    opacity: 1;
    transform: translateX(0);
  }
}

.notification-time {
  color: #999;
  font-size: 12px;
}

.study-reminder {
  text-align: center;
  padding: 20px;
}

.reminder-icon {
  margin-bottom: 16px;
}

.reminder-content h3 {
  margin-bottom: 8px;
  color: #1890ff;
}

.reminder-content p {
  color: #666;
  margin-bottom: 20px;
}

.reminder-actions {
  display: flex;
  gap: 12px;
  justify-content: center;
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
</style>
