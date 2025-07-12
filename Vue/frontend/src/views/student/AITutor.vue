<template>
  <div class="ai-tutor-page">
    <div class="page-header">
    <h1>AI学习助手</h1>
    </div>

    <div class="tutor-content">
      <a-row :gutter="[24, 24]">
        <!-- 左侧功能介绍 -->
        <a-col :xs="24" :md="10">
          <a-card title="学习助手功能" class="features-card">
            <div class="feature-list">
              <div class="feature-item">
                <a-icon><BulbOutlined /></a-icon>
                <div class="feature-info">
                  <h3>学习指导</h3>
                  <p>针对不同学科提供个性化学习建议和解题技巧</p>
                </div>
              </div>
              <div class="feature-item">
                <a-icon><QuestionCircleOutlined /></a-icon>
                <div class="feature-info">
                  <h3>疑难解答</h3>
                  <p>解答学习过程中遇到的问题和困惑</p>
                </div>
              </div>
              <div class="feature-item">
                <a-icon><BookOutlined /></a-icon>
                <div class="feature-info">
                  <h3>知识讲解</h3>
                  <p>详细解释课程中的重点难点内容</p>
                </div>
              </div>
              <div class="feature-item">
                <a-icon><ScheduleOutlined /></a-icon>
                <div class="feature-info">
                  <h3>学习规划</h3>
                  <p>帮助制定合理的学习计划和复习策略</p>
                </div>
              </div>
            </div>
          </a-card>

          <a-card title="使用指南" class="guide-card" style="margin-top: 24px">
            <div class="guide-steps">
              <div class="step">
                <div class="step-number">1</div>
                <div class="step-content">
                  <h4>选择学科</h4>
                  <p>先告诉AI你需要哪个学科的帮助</p>
                </div>
              </div>
              <div class="step">
                <div class="step-number">2</div>
                <div class="step-content">
                  <h4>描述问题</h4>
                  <p>清晰描述你的学习问题或需求</p>
                </div>
              </div>
              <div class="step">
                <div class="step-number">3</div>
                <div class="step-content">
                  <h4>获取帮助</h4>
                  <p>AI助手将提供个性化的学习指导</p>
                </div>
              </div>
            </div>
          </a-card>
        </a-col>

        <!-- 右侧聊天区域 -->
        <a-col :xs="24" :md="14">
          <a-card title="与AI助手对话" class="chat-card">
            <div class="chatbot-container">
              <!-- 聊天机器人将由全局脚本加载 -->
              <a-button 
                type="primary" 
                size="large" 
                block
                @click="openChatbot"
              >
                <RobotOutlined />
                打开AI助手对话
              </a-button>
              <p class="chat-tip">
                点击上方按钮或右下角悬浮图标打开对话窗口
              </p>
            </div>
          </a-card>
        </a-col>
      </a-row>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { 
  BulbOutlined, 
  QuestionCircleOutlined, 
  BookOutlined, 
  ScheduleOutlined,
  RobotOutlined
} from '@ant-design/icons-vue'
import { useAuthStore } from '@/stores/auth'
import { message } from 'ant-design-vue'

const authStore = useAuthStore()

// 打开聊天机器人
function openChatbot() {
  const chatButton = document.getElementById('dify-chatbot-bubble-button')
  if (chatButton && chatButton instanceof HTMLElement) {
    chatButton.click()
  } else {
    message.info('正在初始化AI助手，请稍后再试...')
  }
}

// 更新聊天机器人配置
onMounted(() => {
  // 更新聊天机器人上下文
  if (window.difyChatbotConfig) {
    window.difyChatbotConfig.systemVariables = {
      ...window.difyChatbotConfig.systemVariables,
      context: 'ai_tutor',
      user_role: 'student'
    }
    
    if (authStore.user) {
      window.difyChatbotConfig.systemVariables!.user_id = authStore.user.id?.toString() || ''
      window.difyChatbotConfig.userVariables = {
        name: authStore.user.realName || authStore.user.username || '',
        avatar_url: authStore.user.avatar || ''
      }
    }
  }
  
  // 自动打开聊天窗口
  setTimeout(openChatbot, 1000)
})
</script> 

<style scoped>
.ai-tutor-page {
  padding: 24px;
  background-color: #f5f7fa;
  min-height: calc(100vh - 64px);
}

.page-header {
  margin-bottom: 24px;
}

.page-header h1 {
  font-size: 24px;
  font-weight: 500;
  margin: 0;
}

.feature-list {
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.feature-item {
  display: flex;
  align-items: flex-start;
  gap: 12px;
}

.feature-item .ant-icon {
  font-size: 20px;
  color: #1890ff;
  padding-top: 2px;
}

.feature-info h3 {
  margin: 0 0 4px;
  font-size: 16px;
  font-weight: 500;
}

.feature-info p {
  margin: 0;
  color: #666;
}

.guide-steps {
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.step {
  display: flex;
  gap: 12px;
}

.step-number {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 28px;
  height: 28px;
  background-color: #1890ff;
  color: white;
  border-radius: 50%;
  font-weight: bold;
}

.step-content h4 {
  margin: 0 0 4px;
  font-size: 16px;
  font-weight: 500;
}

.step-content p {
  margin: 0;
  color: #666;
}

.chatbot-container {
  height: 600px;
  position: relative;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
}

.chat-tip {
  margin-top: 16px;
  color: #999;
  text-align: center;
}

@media (max-width: 768px) {
  .chatbot-container {
    height: 450px;
  }
}
</style> 