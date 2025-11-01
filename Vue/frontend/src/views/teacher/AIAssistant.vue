<template>
  <div class="ai-assistant-page">
    <div class="page-header">
      <h2>AI智能助手</h2>
      <div class="header-actions">
        <a-button type="primary" @click="toggleChatbot">
          <RobotOutlined />
          {{ showChatbot ? '隐藏' : '显示' }}助手
        </a-button>
      </div>
    </div>

    <div class="content">
      <div class="assistant-info">
        <a-card title="智能助手功能">
          <div class="feature-list">
            <div class="feature-item">
              <h4>📝 作业批改</h4>
              <p>提供智能批改建议，自动评分和详细反馈</p>
            </div>
            <div class="feature-item">
              <h4>📊 考试管理</h4>
              <p>协助创建试卷、管理考试和分析考试结果</p>
            </div>
            <div class="feature-item">
              <h4>💬 智能问答</h4>
              <p>回答教学相关问题，提供教学建议</p>
            </div>
            <div class="feature-item">
              <h4>📈 数据分析</h4>
              <p>分析学生成绩和学习情况，生成报告</p>
            </div>
          </div>
        </a-card>
      </div>

      <div class="usage-guide">
        <a-card title="使用指南">
          <div class="guide-steps">
            <div class="step">
              <div class="step-number">1</div>
              <div class="step-content">
                <h4>启动助手</h4>
                <p>点击右下角的聊天按钮启动智能助手</p>
              </div>
            </div>
            <div class="step">
              <div class="step-number">2</div>
              <div class="step-content">
                <h4>提出问题</h4>
                <p>在聊天窗口中输入您的问题或需求</p>
              </div>
            </div>
            <div class="step">
              <div class="step-number">3</div>
              <div class="step-content">
                <h4>获取帮助</h4>
                <p>AI助手会提供专业的回答和建议</p>
              </div>
            </div>
          </div>
        </a-card>
      </div>
    </div>

    <!-- 智能体聊天机器人 -->
    <DifyChatbot 
      v-if="showChatbot"
      :token="chatbotConfig.token"
      :baseUrl="chatbotConfig.baseUrl"
      :systemVariables="chatbotConfig.systemVariables"
      :userVariables="chatbotConfig.userVariables"
    />
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue'
import { RobotOutlined } from '@ant-design/icons-vue'
import DifyChatbot from '@/components/DifyChatbot.vue'

// 智能体聊天机器人配置
const showChatbot = ref(true)
const chatbotConfig = ref({
  token: 'SKiyotVrMpqPW2Sp',
  baseUrl: 'http://219.216.65.108',
  systemVariables: {
    context: 'ai_assistant',
    user_role: 'teacher'
  },
  userVariables: {
    page: 'ai_assistant'
  }
})

const toggleChatbot = () => {
  showChatbot.value = !showChatbot.value
}
</script>

<style scoped>
.ai-assistant-page {
  padding: 24px;
  background-color: #f5f7fa;
  min-height: 100vh;
}

.page-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 24px;
}

.page-header h2 {
  margin: 0;
  font-size: 24px;
  font-weight: 600;
  color: #1f2937;
}

.header-actions {
  display: flex;
  gap: 12px;
}

.content {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 24px;
  margin-bottom: 24px;
}

.assistant-info .feature-list {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 16px;
}

.feature-item {
  padding: 16px;
  background-color: #f8fafc;
  border-radius: 8px;
  border-left: 4px solid #1C64F2;
}

.feature-item h4 {
  margin: 0 0 8px 0;
  font-size: 16px;
  font-weight: 600;
  color: #1f2937;
}

.feature-item p {
  margin: 0;
  color: #6b7280;
  font-size: 14px;
}

.usage-guide .guide-steps {
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.step {
  display: flex;
  align-items: flex-start;
  gap: 16px;
}

.step-number {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 32px;
  height: 32px;
  background-color: #1C64F2;
  color: white;
  border-radius: 50%;
  font-weight: 600;
  font-size: 14px;
  flex-shrink: 0;
}

.step-content h4 {
  margin: 0 0 8px 0;
  font-size: 16px;
  font-weight: 600;
  color: #1f2937;
}

.step-content p {
  margin: 0;
  color: #6b7280;
  font-size: 14px;
}

@media (max-width: 768px) {
  .content {
    grid-template-columns: 1fr;
  }
  
  .assistant-info .feature-list {
    grid-template-columns: 1fr;
  }
}
</style> 