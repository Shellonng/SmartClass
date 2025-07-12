<template>
  <div class="dify-chatbot-container">
    <!-- 聊天机器人将在这里自动加载 -->
    <div v-if="debugMode" class="debug-info">
      <h4>调试信息：</h4>
      <p>脚本加载状态: {{ scriptLoaded ? '成功' : '未加载' }}</p>
      <p>聊天窗口状态: {{ chatWindowFound ? '已创建' : '未创建' }}</p>
      <p>聊天按钮状态: {{ chatButtonFound ? '已创建' : '未创建' }}</p>
      <p>错误信息: {{ errorMsg }}</p>
      <button @click="checkElements">检查元素</button>
      <button @click="reloadChatbot">重新加载</button>
      <button @click="toggleDebugMode">{{ debugMode ? '关闭' : '开启' }}调试</button>
    </div>
  </div>
</template>

<script setup lang="ts">
import { onMounted, onUnmounted, ref } from 'vue'
import type { DifyChatbotConfig } from '@/types/dify'

// 调试相关状态
const debugMode = ref(true) // 设为true开启调试模式
const scriptLoaded = ref(false)
const chatWindowFound = ref(false)
const chatButtonFound = ref(false)
const errorMsg = ref('')

// 定义组件 props
interface Props {
  token?: string
  baseUrl?: string
  systemVariables?: Record<string, any>
  userVariables?: Record<string, any>
}

const props = withDefaults(defineProps<Props>(), {
  token: 'SKiyotVrMpqPW2Sp',
  baseUrl: 'http://219.216.65.108',
  systemVariables: () => ({}),
  userVariables: () => ({})
})

// 动态加载 Dify 聊天机器人
const loadDifyChatbot = () => {
  try {
    console.log('开始加载Dify聊天机器人...')
    console.log('配置信息:', {
      token: props.token,
      baseUrl: props.baseUrl,
      systemVariables: props.systemVariables,
      userVariables: props.userVariables
    })
    
  // 设置配置
  const config: DifyChatbotConfig = {
    token: props.token,
    baseUrl: props.baseUrl,
    systemVariables: props.systemVariables,
    userVariables: props.userVariables
  }
  
  ;(window as any).difyChatbotConfig = config
    console.log('已设置全局配置:', (window as any).difyChatbotConfig)

  // 添加样式
  const style = document.createElement('style')
  style.textContent = `
    #dify-chatbot-bubble-button {
      background-color: #1C64F2 !important;
        z-index: 10000 !important;
        position: fixed !important;
        bottom: 20px !important;
        right: 20px !important;
        display: flex !important;
    }
    #dify-chatbot-bubble-window {
      width: 24rem !important;
      height: 40rem !important;
        z-index: 10000 !important;
        position: fixed !important;
        bottom: 100px !important;
        right: 20px !important;
        display: block !important;
        visibility: visible !important;
        opacity: 1 !important;
    }
  `
  document.head.appendChild(style)
    console.log('已添加自定义样式')

  // 检查是否已经加载过脚本
  const existingScript = document.getElementById(props.token)
  if (existingScript) {
      console.log('移除已存在的脚本')
    existingScript.remove()
  }

    // 移除可能存在的旧元素
    const oldButton = document.getElementById('dify-chatbot-bubble-button')
    const oldWindow = document.getElementById('dify-chatbot-bubble-window')
    if (oldButton) oldButton.remove()
    if (oldWindow) oldWindow.remove()

  // 加载聊天机器人脚本
  const script = document.createElement('script')
  script.src = `${props.baseUrl}/embed.min.js`
  script.id = props.token
  script.defer = true
  script.onload = () => {
      console.log('Dify 聊天机器人脚本加载成功')
      scriptLoaded.value = true
      
      // 脚本加载成功后延迟检查元素
      setTimeout(() => {
        checkElements()
        
        // 尝试手动打开聊天窗口
        const chatButton = document.getElementById('dify-chatbot-bubble-button')
        if (chatButton && chatButton instanceof HTMLElement) {
          console.log('尝试点击聊天按钮...')
          chatButton.click()
        }
      }, 2000)
  }
    script.onerror = (e) => {
      console.error('Dify 聊天机器人脚本加载失败', e)
      errorMsg.value = '脚本加载失败: ' + e
    }
    
    // 将脚本添加到body而不是head
    document.body.appendChild(script)
    console.log('已添加脚本到body')
  } catch (e) {
    console.error('加载聊天机器人时出错:', e)
    errorMsg.value = '加载错误: ' + e
  }
}

// 检查聊天窗口元素
const checkElements = () => {
  console.log('检查聊天元素...')
  const chatButton = document.getElementById('dify-chatbot-bubble-button')
  const chatWindow = document.getElementById('dify-chatbot-bubble-window')
  
  chatButtonFound.value = !!chatButton
  chatWindowFound.value = !!chatWindow
  
  console.log('聊天按钮:', chatButton)
  console.log('聊天窗口:', chatWindow)
  
  if (chatButton) {
    console.log('聊天按钮样式:', getComputedStyle(chatButton))
  }
  
  if (chatWindow) {
    console.log('聊天窗口样式:', getComputedStyle(chatWindow))
  }
  
  // 检查是否正确加载了Dify脚本
  if (!chatButton && !chatWindow && scriptLoaded.value) {
    console.warn('脚本已加载但聊天元素未创建，可能是Dify服务连接问题')
    errorMsg.value = 'Dify服务连接问题，聊天元素未创建'
  }
}

// 重新加载聊天机器人
const reloadChatbot = () => {
  cleanupDifyChatbot()
  setTimeout(() => {
    loadDifyChatbot()
  }, 500)
}

// 清理聊天机器人
const cleanupDifyChatbot = () => {
  console.log('清理聊天机器人资源...')
  try {
  // 移除脚本
  const script = document.getElementById(props.token)
  if (script) {
    script.remove()
      console.log('已移除脚本')
  }

  // 移除聊天机器人相关DOM元素
  const chatbotButton = document.getElementById('dify-chatbot-bubble-button')
  if (chatbotButton) {
    chatbotButton.remove()
      console.log('已移除聊天按钮')
  }

  const chatbotWindow = document.getElementById('dify-chatbot-bubble-window')
  if (chatbotWindow) {
    chatbotWindow.remove()
      console.log('已移除聊天窗口')
  }

  // 清理配置
  if ((window as any).difyChatbotConfig) {
    delete (window as any).difyChatbotConfig
      console.log('已清理配置')
    }
    
    // 重置状态
    scriptLoaded.value = false
    chatButtonFound.value = false
    chatWindowFound.value = false
    errorMsg.value = ''
  } catch (e) {
    console.error('清理资源时出错:', e)
    errorMsg.value = '清理错误: ' + e
  }
}

// 切换调试模式
const toggleDebugMode = () => {
  debugMode.value = !debugMode.value
}

onMounted(() => {
  console.log('DifyChatbot组件已挂载')
  loadDifyChatbot()
})

onUnmounted(() => {
  console.log('DifyChatbot组件将卸载')
  cleanupDifyChatbot()
})
</script>

<style scoped>
.dify-chatbot-container {
  position: relative;
  z-index: 1000;
}

.debug-info {
  position: fixed;
  top: 100px;
  left: 20px;
  background-color: rgba(255, 255, 255, 0.9);
  padding: 10px;
  border: 1px solid #ccc;
  border-radius: 5px;
  z-index: 9999;
  max-width: 300px;
  font-size: 12px;
}

.debug-info h4 {
  margin: 0 0 10px;
  color: #333;
}

.debug-info p {
  margin: 5px 0;
}

.debug-info button {
  margin: 5px;
  padding: 4px 8px;
  background: #1C64F2;
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
}
</style> 