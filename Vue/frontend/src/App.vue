<script lang="ts">
// 创建事件总线
import mitt from 'mitt'
export const emitter = mitt()

// 导出供其他组件使用
export const APP_EVENTS = {
  COLLAPSE_SIDEBAR: 'collapse-sidebar'
}

// 扩展Window接口，添加difyChatbotConfig属性
declare global {
  interface Window {
    difyChatbotConfig?: {
      token: string;
      baseUrl: string;
      systemVariables?: Record<string, any>;
      userVariables?: Record<string, any>;
    }
  }
}
</script>

<script setup lang="ts">
import { RouterView } from 'vue-router'
import { onMounted, onUnmounted } from 'vue'
import { useAuthStore } from './stores/auth'

const authStore = useAuthStore()

onMounted(() => {
  // 初始化认证状态
  authStore.init()
  
  // 加载Dify聊天机器人
  loadDifyChatbot()
})

onUnmounted(() => {
  // 清理聊天机器人
  cleanupDifyChatbot()
})

// 加载聊天机器人
function loadDifyChatbot() {
  // 设置聊天机器人配置
  window.difyChatbotConfig = {
    token: 'SKiyotVrMpqPW2Sp',
    baseUrl: 'http://219.216.65.108',
    systemVariables: {
      user_id: authStore.user?.id?.toString() || '',
      user_role: authStore.user?.role || 'user'
    },
    userVariables: {
      avatar_url: authStore.user?.avatar || '',
      name: authStore.user?.realName || authStore.user?.username || '用户'
    }
  }
  
  // 添加样式
  const style = document.createElement('style')
  style.textContent = `
    #dify-chatbot-bubble-button {
      background-color: #1C64F2 !important;
      z-index: 10000 !important;
    }
    #dify-chatbot-bubble-window {
      width: 24rem !important;
      height: 40rem !important;
      z-index: 10000 !important;
    }
  `
  document.head.appendChild(style)
  
  // 加载脚本
  const script = document.createElement('script')
  script.src = 'http://219.216.65.108/embed.min.js'
  script.id = 'SKiyotVrMpqPW2Sp'
  script.defer = true
  document.body.appendChild(script)
}

// 清理聊天机器人
function cleanupDifyChatbot() {
  // 移除脚本
  const script = document.getElementById('SKiyotVrMpqPW2Sp')
  if (script) script.remove()
  
  // 移除配置
  if (window.difyChatbotConfig) {
    delete window.difyChatbotConfig
  }
}
</script>

<template>
  <div id="app">
    <RouterView />
  </div>
</template>

<style>
* {
  margin: 0;
  padding: 0;
  box-sizing: border-box;
}

html, body {
  width: 100%;
  height: 100%;
  margin: 0;
  padding: 0;
  overflow-x: hidden;
}

body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'PingFang SC', 'Hiragino Sans GB',
    'Microsoft YaHei', 'Helvetica Neue', Helvetica, Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  background-color: #f5f5f5;
}

#app {
  width: 100%;
  height: 100vh;
  margin: 0;
  padding: 0;
}

/* 全局滚动条样式 */
::-webkit-scrollbar {
  width: 6px;
  height: 6px;
}

::-webkit-scrollbar-track {
  background: #f1f1f1;
  border-radius: 3px;
}

::-webkit-scrollbar-thumb {
  background: #c1c1c1;
  border-radius: 3px;
}

::-webkit-scrollbar-thumb:hover {
  background: #a8a8a8;
}

/* Ant Design 组件样式覆盖 */
.ant-btn {
  border-radius: 8px;
  font-weight: 500;
}

.ant-input,
.ant-input-password {
  border-radius: 8px;
}

.ant-select .ant-select-selector {
  border-radius: 8px;
}

.ant-card {
  border-radius: 12px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);
}

.ant-modal .ant-modal-content {
  border-radius: 12px;
}

.ant-message {
  top: 80px;
}

/* Dify聊天机器人全局样式 */
#dify-chatbot-bubble-button {
  z-index: 10000 !important;
  position: fixed !important;
  bottom: 20px !important;
  right: 20px !important;
  visibility: visible !important;
  opacity: 1 !important;
  width: 56px !important;
  height: 56px !important;
  display: flex !important;
  align-items: center !important;
  justify-content: center !important;
  cursor: pointer !important;
}

#dify-chatbot-bubble-window {
  z-index: 10000 !important;
  position: fixed !important;
  bottom: 90px !important;
  right: 20px !important;
  visibility: visible !important;
  opacity: 1 !important;
  width: 400px !important;
  height: 600px !important;
  box-shadow: 0 5px 40px rgba(0, 0, 0, 0.16) !important;
  border-radius: 8px !important;
}

/* 聊天机器人挂载点 */
.chatbot-mount-point {
  position: fixed;
  z-index: 9999;
  bottom: 0;
  right: 0;
  width: 1px;
  height: 1px;
}
</style>
