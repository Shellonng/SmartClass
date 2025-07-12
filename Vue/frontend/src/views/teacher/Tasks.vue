<template>
  <div class="teacher-tasks">
    <div class="page-header">
      <h1>任务管理</h1>
      <div class="header-actions">
        <a-button type="primary" @click="publishTask">
          <PlusOutlined />
          发布任务
        </a-button>
        <a-button type="default" @click="toggleChatbot" :icon="showChatbot ? 'EyeInvisibleOutlined' : 'RobotOutlined'">
          {{ showChatbot ? '隐藏' : '显示' }}智能助手
        </a-button>
      </div>
    </div>
    
    <div class="tasks-content">
      <a-table :dataSource="tasks" :columns="columns" />
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
import { PlusOutlined, EyeInvisibleOutlined, RobotOutlined } from '@ant-design/icons-vue'
import DifyChatbot from '@/components/DifyChatbot.vue'

const tasks = ref([
  { id: 1, title: '第一章作业', course: '高等数学', deadline: '2024-02-01' },
  { id: 2, title: '期中测验', course: '程序设计', deadline: '2024-02-15' }
])

const columns = [
  { title: '任务标题', dataIndex: 'title', key: 'title' },
  { title: '课程', dataIndex: 'course', key: 'course' },
  { title: '截止时间', dataIndex: 'deadline', key: 'deadline' }
]

// 智能体聊天机器人配置
const showChatbot = ref(true)
const chatbotConfig = ref({
  token: 'SKiyotVrMpqPW2Sp',
  baseUrl: 'http://219.216.65.108',
  systemVariables: {
    // 可以根据需要添加系统变量
    context: 'assignment_management',
    user_role: 'teacher'
  },
  userVariables: {
    // 可以根据需要添加用户变量
    page: 'tasks'
  }
})

const publishTask = () => {
  console.log('发布任务')
}

const toggleChatbot = () => {
  showChatbot.value = !showChatbot.value
}
</script>

<style scoped>
.teacher-tasks {
  padding: 24px;
}

.page-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 24px;
}

.header-actions {
  display: flex;
  gap: 12px;
}
</style> 