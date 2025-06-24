<template>
  <a-layout class="teacher-layout">
    <!-- 侧边栏 -->
    <a-layout-sider
      v-model:collapsed="collapsed"
      :trigger="null"
      collapsible
      class="sidebar"
      :width="240"
      :collapsed-width="80"
    >
      <div class="logo">
        <img src="/logo.svg" alt="Logo" v-if="!collapsed" />
        <img src="/logo-mini.svg" alt="Logo" v-else />
      </div>
      
      <a-menu
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
          <a-button
            type="text"
            @click="collapsed = !collapsed"
            class="trigger"
          >
            <MenuUnfoldOutlined v-if="collapsed" />
            <MenuFoldOutlined v-else />
          </a-button>
          
          <a-breadcrumb class="breadcrumb">
            <a-breadcrumb-item v-for="item in breadcrumbItems" :key="item.path">
              <router-link v-if="item.path" :to="item.path">{{ item.title }}</router-link>
              <span v-else>{{ item.title }}</span>
            </a-breadcrumb-item>
          </a-breadcrumb>
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
import { useAuthStore, type User } from '@/stores/auth'
import {
  DashboardOutlined,
  TeamOutlined,
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
  LogoutOutlined
} from '@ant-design/icons-vue'
import { message } from 'ant-design-vue'

const route = useRoute()
const router = useRouter()
const authStore = useAuthStore()

// 侧边栏状态
const collapsed = ref(false)
const selectedKeys = ref<string[]>(['dashboard'])
const openKeys = ref<string[]>([])

// 用户信息
const userInfo = computed(() => authStore.user as User | null)

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
    'dashboard': '/teacher',
    'classes-list': '/teacher/classes',
    'classes-create': '/teacher/classes/create',
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
function handleLogout() {
  authStore.logout()
  message.success('已退出登录')
  router.push('/login')
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

onMounted(() => {
  // 初始化时设置正确的菜单选中状态
  updateSelectedKeys(route.path)
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
}

.logo {
  height: 64px;
  display: flex;
  align-items: center;
  justify-content: center;
  background: rgba(255, 255, 255, 0.1);
  margin: 16px;
  border-radius: 8px;
}

.logo img {
  height: 32px;
}

.sidebar-menu {
  border-right: none;
}

.main-layout {
  margin-left: 240px;
  transition: margin-left 0.2s;
  min-width: 1000px;
  max-width: 1800px;
  width: calc(100vw - 240px);
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

.trigger {
  font-size: 18px;
  line-height: 64px;
  cursor: pointer;
  transition: color 0.3s;
}

.trigger:hover {
  color: #1890ff;
}

.breadcrumb {
  margin-left: 16px;
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


</style>