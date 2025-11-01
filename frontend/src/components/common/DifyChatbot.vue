<template>
  <div id="dify-chatbot-container">
    <!-- Dify聊天机器人将自动注入到这里 -->
  </div>
</template>

<script setup lang="ts">
import { onMounted, onUnmounted } from 'vue'

interface DifyChatbotConfig {
  token: string
  baseUrl: string
  systemVariables?: {
    user_id?: string
    conversation_id?: string
  }
  userVariables?: {
    avatar_url?: string
    name?: string
  }
}

const props = defineProps<{
  userId?: string
  userName?: string
  avatarUrl?: string
  conversationId?: string
}>()

let chatbotScript: HTMLScriptElement | null = null

onMounted(() => {
  initializeDifyChatbot()
})

onUnmounted(() => {
  // 清理聊天机器人资源
  if (chatbotScript) {
    document.body.removeChild(chatbotScript)
  }
  
  // 清理全局配置
  if (window.difyChatbotConfig) {
    delete window.difyChatbotConfig
  }
})

function initializeDifyChatbot() {
  try {
    // 配置Dify聊天机器人
    const config: DifyChatbotConfig = {
      token: 'SKiyotVrMpqPW2Sp',
      baseUrl: 'http://219.216.65.108',
      systemVariables: {
        ...(props.userId && { user_id: props.userId }),
        ...(props.conversationId && { conversation_id: props.conversationId })
      },
      userVariables: {
        ...(props.avatarUrl && { avatar_url: props.avatarUrl }),
        ...(props.userName && { name: props.userName })
      }
    }

    // 设置全局配置
    window.difyChatbotConfig = config

    // 加载聊天机器人脚本
    loadChatbotScript()
    
    // 添加自定义样式
    addCustomStyles()
    
  } catch (error) {
    console.error('初始化Dify聊天机器人失败:', error)
  }
}

function loadChatbotScript() {
  // 检查是否已经加载
  if (document.getElementById('SKiyotVrMpqPW2Sp')) {
    return
  }

  chatbotScript = document.createElement('script')
  chatbotScript.src = 'http://219.216.65.108/embed.min.js'
  chatbotScript.id = 'SKiyotVrMpqPW2Sp'
  chatbotScript.defer = true
  
  chatbotScript.onload = () => {
    console.log('Dify聊天机器人加载成功')
  }
  
  chatbotScript.onerror = () => {
    console.error('Dify聊天机器人加载失败')
  }
  
  document.body.appendChild(chatbotScript)
}

function addCustomStyles() {
  // 检查是否已经添加样式
  if (document.getElementById('dify-chatbot-styles')) {
    return
  }

  const style = document.createElement('style')
  style.id = 'dify-chatbot-styles'
  style.innerHTML = `
    #dify-chatbot-bubble-button {
      background-color: #1C64F2 !important;
      box-shadow: 0 4px 12px rgba(28, 100, 242, 0.3) !important;
      transition: all 0.3s ease !important;
    }
    
    #dify-chatbot-bubble-button:hover {
      background-color: #1E56D6 !important;
      transform: scale(1.05) !important;
    }
    
    #dify-chatbot-bubble-window {
      width: 24rem !important;
      height: 40rem !important;
      border-radius: 12px !important;
      box-shadow: 0 10px 40px rgba(0, 0, 0, 0.15) !important;
      border: 1px solid #e5e7eb !important;
    }
    
    @media (max-width: 768px) {
      #dify-chatbot-bubble-window {
        width: 90vw !important;
        height: 80vh !important;
        max-width: 350px !important;
      }
    }
  `
  
  document.head.appendChild(style)
}

// 声明全局类型
declare global {
  interface Window {
    difyChatbotConfig?: DifyChatbotConfig
  }
}
</script>

<style scoped>
#dify-chatbot-container {
  position: relative;
  z-index: 1000;
}
</style> 