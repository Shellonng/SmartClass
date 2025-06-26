<template>
  <div class="teacher-dashboard">
    <a-spin :spinning="loading" tip="加载中...">
      <!-- 顶部欢迎区域 -->
      <div class="dashboard-header">
        <div class="header-content">
          <div class="welcome-section">
            <div class="avatar-section">
              <a-avatar :size="64" :src="authStore.user?.avatar" class="teacher-avatar">
                {{ (authStore.user?.realName || authStore.user?.username || 'T').charAt(0) }}
              </a-avatar>
              <div class="welcome-info">
                <h1 class="welcome-title">
                  {{ greeting }}，{{ authStore.user?.realName || authStore.user?.username || '老师' }}
                </h1>
                <p class="welcome-subtitle">
                  {{ formatDate(currentDate) }} · 今天也要充满活力地教学哦！
                </p>
                <div class="teacher-badges">
                  <a-tag color="blue">高级讲师</a-tag>
                  <a-tag color="green">优秀教师</a-tag>
                  <a-tag color="orange">AI教学先锋</a-tag>
                </div>
              </div>
            </div>
          </div>
          
          <div class="header-actions">
            <a-space size="large">
              <a-button 
                type="primary" 
                size="large" 
                @click="showCreateCourseModal = true"
                class="action-btn"
              >
                <template #icon><PlusOutlined /></template>
                创建课程
              </a-button>
              <a-button 
                size="large" 
                @click="showCreateTaskModal = true"
                class="action-btn"
              >
                <template #icon><FileTextOutlined /></template>
                布置作业
              </a-button>
              <a-button 
                size="large" 
                @click="showCreateClassModal = true"
                class="action-btn"
              >
                <template #icon><TeamOutlined /></template>
                创建班级
              </a-button>
            </a-space>
          </div>
        </div>
      </div>

      <!-- 核心统计卡片 -->
      <div class="stats-overview">
        <div class="stats-grid">
          <div class="stat-card courses">
            <div class="stat-icon">
              <BookOutlined />
            </div>
            <div class="stat-content">
              <div class="stat-number">{{ stats.courseCount }}</div>
              <div class="stat-label">我的课程</div>
              <div class="stat-trend">
                <ArrowUpOutlined class="trend-up" />
                <span>较上月 +{{ stats.courseGrowth }}%</span>
              </div>
            </div>
          </div>
          
          <div class="stat-card students">
            <div class="stat-icon">
              <UserOutlined />
            </div>
            <div class="stat-content">
              <div class="stat-number">{{ stats.studentCount }}</div>
              <div class="stat-label">教授学生</div>
              <div class="stat-trend">
                <ArrowUpOutlined class="trend-up" />
                <span>较上月 +{{ stats.studentGrowth }}%</span>
              </div>
            </div>
          </div>
          
          <div class="stat-card assignments">
            <div class="stat-icon">
              <FileTextOutlined />
            </div>
            <div class="stat-content">
              <div class="stat-number">{{ stats.assignmentCount }}</div>
              <div class="stat-label">布置作业</div>
              <div class="stat-trend">
                <ArrowUpOutlined class="trend-up" />
                <span>本周 +{{ stats.weeklyAssignments }}</span>
              </div>
            </div>
          </div>
          
          <div class="stat-card pending">
            <div class="stat-icon">
              <ClockCircleOutlined />
            </div>
            <div class="stat-content">
              <div class="stat-number">{{ stats.pendingCount }}</div>
              <div class="stat-label">待批改</div>
              <div class="stat-trend urgent">
                <ExclamationCircleOutlined class="trend-urgent" />
                <span>需要关注</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      <!-- 主要内容区域 -->
      <div class="dashboard-content">
        <a-row :gutter="24">
          <!-- 左侧主要内容 -->
          <a-col :span="16">
            <!-- 我的课程 -->
            <div class="content-section">
              <div class="section-header">
                <h3 class="section-title">
                  <BookOutlined />
                  我的课程
                </h3>
                <div class="section-actions">
                  <a-select v-model:value="courseFilter" style="width: 120px" size="small">
                    <a-select-option value="all">全部课程</a-select-option>
                    <a-select-option value="active">进行中</a-select-option>
                    <a-select-option value="completed">已结束</a-select-option>
                  </a-select>
                  <a-button type="link">
                    查看全部 <ArrowRightOutlined />
                  </a-button>
                </div>
              </div>
              
              <div class="course-grid">
                <div 
                  v-for="course in mockCourses" 
                  :key="course.id"
                  class="course-card"
                >
                  <div class="course-header">
                    <div class="course-cover" :style="{ background: getCourseGradient(course.subject) }">
                      <div class="course-category">{{ course.subject }}</div>
                      <div class="course-level">{{ course.difficulty }}</div>
                    </div>
                    <div class="course-actions">
                      <a-dropdown>
                        <a-button type="text" size="small">
                          <MoreOutlined />
                        </a-button>
                        <template #overlay>
                          <a-menu>
                            <a-menu-item key="edit">编辑课程</a-menu-item>
                            <a-menu-item key="students">学生管理</a-menu-item>
                            <a-menu-item key="analytics">数据分析</a-menu-item>
                            <a-menu-item key="settings">课程设置</a-menu-item>
                          </a-menu>
                        </template>
                      </a-dropdown>
                    </div>
                  </div>
                  
                  <div class="course-content">
                    <h4 class="course-title">{{ course.name }}</h4>
                    <p class="course-description">{{ course.description }}</p>
                    
                    <div class="course-meta">
                      <div class="meta-item">
                        <UserOutlined />
                        <span>{{ course.studentCount }}名学生</span>
                      </div>
                      <div class="meta-item">
                        <PlayCircleOutlined />
                        <span>{{ course.chapterCount }}个章节</span>
                      </div>
                      <div class="meta-item">
                        <StarOutlined />
                        <span>{{ course.rating }}分</span>
                      </div>
                    </div>
                    
                    <div class="course-progress">
                      <div class="progress-header">
                        <span>教学进度</span>
                        <span>{{ course.progress }}%</span>
                      </div>
                      <a-progress 
                        :percent="course.progress" 
                        size="small" 
                        :show-info="false"
                        :stroke-color="getCourseColor(course.subject)"
                      />
                    </div>
                    
                    <div class="course-stats">
                      <div class="stat-item">
                        <span class="stat-value">{{ course.weeklyActive }}</span>
                        <span class="stat-label">周活跃</span>
                      </div>
                      <div class="stat-item">
                        <span class="stat-value">{{ course.completionRate }}%</span>
                        <span class="stat-label">完成率</span>
                      </div>
                      <div class="stat-item">
                        <span class="stat-value">{{ course.avgScore }}</span>
                        <span class="stat-label">平均分</span>
                      </div>
                    </div>
                  </div>
                  
                  <div class="course-footer">
                    <a-space>
                      <a-button type="primary" size="small">
                        进入课程
                      </a-button>
                      <a-button size="small">
                        查看数据
                      </a-button>
                    </a-space>
                  </div>
                </div>
                
                <!-- 创建课程卡片 -->
                <div class="course-card create-card" @click="showCreateCourseModal = true">
                  <div class="create-content">
                    <PlusOutlined class="create-icon" />
                    <h4>创建新课程</h4>
                    <p>开始构建你的知识体系</p>
                  </div>
                </div>
              </div>
            </div>

            <!-- 近期作业 -->
            <div class="content-section">
              <div class="section-header">
                <h3 class="section-title">
                  <FileTextOutlined />
                  近期作业
                </h3>
                <div class="section-actions">
                  <a-select v-model:value="assignmentFilter" style="width: 120px" size="small">
                    <a-select-option value="all">全部作业</a-select-option>
                    <a-select-option value="pending">待批改</a-select-option>
                    <a-select-option value="graded">已批改</a-select-option>
                  </a-select>
                  <a-button type="link">
                    查看全部 <ArrowRightOutlined />
                  </a-button>
                </div>
              </div>
              
              <div class="assignment-list">
                <div 
                  v-for="assignment in mockAssignments" 
                  :key="assignment.id"
                  class="assignment-card"
                >
                  <div class="assignment-header">
                    <div class="assignment-priority">
                      <div class="priority-indicator" :class="assignment.priority"></div>
                      <span class="assignment-course">{{ assignment.courseName }}</span>
                    </div>
                    <div class="assignment-deadline" :class="{ urgent: isUrgent(assignment.dueDate) }">
                      <ClockCircleOutlined />
                      {{ formatDeadline(assignment.dueDate) }}
                    </div>
                  </div>
                  
                  <h4 class="assignment-title">{{ assignment.title }}</h4>
                  <p class="assignment-description">{{ assignment.description }}</p>
                  
                  <div class="assignment-progress">
                    <div class="progress-stats">
                      <div class="stat">
                        <span class="number">{{ assignment.submittedCount }}</span>
                        <span class="label">已提交</span>
                      </div>
                      <div class="stat">
                        <span class="number">{{ assignment.totalStudents }}</span>
                        <span class="label">总人数</span>
                      </div>
                      <div class="stat">
                        <span class="number">{{ assignment.gradedCount }}</span>
                        <span class="label">已批改</span>
                      </div>
                    </div>
                    <div class="progress-bar">
                      <div class="progress-info">
                        <span>提交进度</span>
                        <span>{{ Math.round((assignment.submittedCount / assignment.totalStudents) * 100) }}%</span>
                      </div>
                      <a-progress 
                        :percent="Math.round((assignment.submittedCount / assignment.totalStudents) * 100)" 
                        size="small" 
                        :show-info="false"
                      />
                    </div>
                  </div>
                  
                  <div class="assignment-actions">
                    <a-space>
                      <a-button size="small" type="primary">
                        查看提交
                      </a-button>
                      <a-button size="small" v-if="assignment.submittedCount > 0">
                        开始批改
                      </a-button>
                      <a-button size="small" type="text">
                        统计分析
                      </a-button>
                    </a-space>
                  </div>
                </div>
                
                <div v-if="mockAssignments.length === 0" class="empty-state">
                  <FileTextOutlined />
                  <p>暂无作业</p>
                  <a-button type="primary" @click="showCreateTaskModal = true">
                    布置第一个作业
                  </a-button>
                </div>
              </div>
            </div>
          </a-col>

          <!-- 右侧辅助内容 -->
          <a-col :span="8">
            <!-- 教学日历 -->
            <div class="content-section">
              <div class="section-header">
                <h3 class="section-title">
                  <CalendarOutlined />
                  教学日历
                </h3>
              </div>
              
              <div class="calendar-widget">
                <a-calendar 
                  v-model:value="selectedDate"
                  :fullscreen="false"
                />
                
                <div class="calendar-events">
                  <h4>今日安排</h4>
                  <div class="event-list">
                    <div 
                      v-for="event in todayEvents" 
                      :key="event.id"
                      class="event-item"
                    >
                      <div class="event-time">{{ event.time }}</div>
                      <div class="event-content">
                        <div class="event-title">{{ event.title }}</div>
                        <div class="event-desc">{{ event.description }}</div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>

            <!-- AI教学助手 -->
            <div class="content-section ai-section">
              <div class="section-header">
                <h3 class="section-title">
                  <RobotOutlined />
                  AI教学助手
                </h3>
                <a-badge :count="3" size="small">
                  <BellOutlined />
                </a-badge>
              </div>
              
              <div class="ai-recommendations">
                <div class="ai-card featured">
                  <div class="ai-header">
                    <BulbOutlined class="ai-icon" />
                    <span class="ai-title">智能推荐</span>
                  </div>
                  <div class="ai-content">
                    <p>基于学生学习数据，为您推荐3个需要重点关注的知识点</p>
                    <a-button type="primary" size="small" @click="handleAIRecommendation">
                      查看详情
                    </a-button>
                  </div>
                </div>
                
                <div class="ai-features-grid">
                  <div class="ai-feature" @click="handleAIFeature('analysis')">
                    <BarChartOutlined />
                    <span>学情分析</span>
                  </div>
                  <div class="ai-feature" @click="handleAIFeature('grading')">
                    <EditOutlined />
                    <span>智能批改</span>
                  </div>
                  <div class="ai-feature" @click="handleAIFeature('content')">
                    <FileAddOutlined />
                    <span>内容生成</span>
                  </div>
                  <div class="ai-feature" @click="handleAIFeature('question')">
                    <QuestionCircleOutlined />
                    <span>题目推荐</span>
                  </div>
                </div>
              </div>
            </div>

            <!-- 数据洞察 -->
            <div class="content-section">
              <div class="section-header">
                <h3 class="section-title">
                  <BarChartOutlined />
                  数据洞察
                </h3>
              </div>
              
              <div class="insights-widget">
                <div class="insight-item">
                  <div class="insight-header">
                    <span class="insight-title">学生活跃度</span>
                    <span class="insight-trend up">↗ 12%</span>
                  </div>
                  <div class="insight-chart">
                    <div class="mini-chart">
                      <div class="chart-bar" style="height: 60%"></div>
                      <div class="chart-bar" style="height: 80%"></div>
                      <div class="chart-bar" style="height: 70%"></div>
                      <div class="chart-bar" style="height: 90%"></div>
                      <div class="chart-bar" style="height: 100%"></div>
                      <div class="chart-bar" style="height: 85%"></div>
                      <div class="chart-bar" style="height: 95%"></div>
                    </div>
                  </div>
                </div>
                
                <div class="insight-item">
                  <div class="insight-header">
                    <span class="insight-title">作业完成率</span>
                    <span class="insight-trend down">↘ 5%</span>
                  </div>
                  <div class="insight-value">
                    <span class="value">87.5%</span>
                    <span class="unit">平均完成率</span>
                  </div>
                </div>
                
                <div class="insight-item">
                  <div class="insight-header">
                    <span class="insight-title">课程评分</span>
                    <span class="insight-trend up">↗ 0.2</span>
                  </div>
                  <div class="insight-rating">
                    <a-rate :value="4.8" disabled allow-half />
                    <span class="rating-value">4.8</span>
                  </div>
                </div>
              </div>
            </div>

            <!-- 快速操作 -->
            <div class="content-section">
              <div class="section-header">
                <h3 class="section-title">
                  <ThunderboltOutlined />
                  快速操作
                </h3>
              </div>
              
              <div class="quick-actions-grid">
                <div class="quick-action" @click="showCreateTaskModal = true">
                  <FileTextOutlined />
                  <span>布置作业</span>
                </div>
                <div class="quick-action">
                  <EditOutlined />
                  <span>批改作业</span>
                </div>
                <div class="quick-action">
                  <UserOutlined />
                  <span>学生管理</span>
                </div>
                <div class="quick-action">
                  <BarChartOutlined />
                  <span>数据分析</span>
                </div>
                <div class="quick-action">
                  <FolderOutlined />
                  <span>资源库</span>
                </div>
                <div class="quick-action">
                  <SettingOutlined />
                  <span>设置</span>
                </div>
              </div>
            </div>
          </a-col>
        </a-row>
      </div>

      <!-- 创建课程弹窗 -->
      <a-modal
        v-model:open="showCreateCourseModal"
        title="创建新课程"
        width="600px"
        @ok="handleCreateCourse"
      >
        <a-form :model="courseForm" layout="vertical">
          <a-row :gutter="16">
            <a-col :span="12">
              <a-form-item label="课程名称" required>
                <a-input v-model:value="courseForm.name" placeholder="请输入课程名称" />
              </a-form-item>
            </a-col>
            <a-col :span="12">
              <a-form-item label="课程代码" required>
                <a-input v-model:value="courseForm.code" placeholder="如：CS101" />
              </a-form-item>
            </a-col>
          </a-row>
          
          <a-row :gutter="16">
            <a-col :span="12">
              <a-form-item label="学科类别">
                <a-select v-model:value="courseForm.subject" placeholder="请选择学科">
                  <a-select-option value="计算机科学">计算机科学</a-select-option>
                  <a-select-option value="数学">数学</a-select-option>
                  <a-select-option value="物理">物理</a-select-option>
                  <a-select-option value="化学">化学</a-select-option>
                  <a-select-option value="生物">生物</a-select-option>
                  <a-select-option value="英语">英语</a-select-option>
                  <a-select-option value="历史">历史</a-select-option>
                  <a-select-option value="地理">地理</a-select-option>
                </a-select>
              </a-form-item>
            </a-col>
            <a-col :span="12">
              <a-form-item label="难度等级">
                <a-select v-model:value="courseForm.difficulty" placeholder="请选择难度">
                  <a-select-option value="初级">初级</a-select-option>
                  <a-select-option value="中级">中级</a-select-option>
                  <a-select-option value="高级">高级</a-select-option>
                </a-select>
              </a-form-item>
            </a-col>
          </a-row>
          
          <a-form-item label="课程描述">
            <a-textarea 
              v-model:value="courseForm.description" 
              placeholder="请输入课程描述，包括课程目标、内容概述等"
              :rows="4"
            />
          </a-form-item>
        </a-form>
      </a-modal>

      <!-- 创建班级弹窗 -->
      <a-modal
        v-model:open="showCreateClassModal"
        title="创建班级"
        @ok="handleCreateClass"
      >
        <a-form :model="classForm" layout="vertical">
          <a-form-item label="班级名称" required>
            <a-input v-model:value="classForm.name" placeholder="请输入班级名称" />
          </a-form-item>
          <a-form-item label="班级代码" required>
            <a-input v-model:value="classForm.code" placeholder="请输入班级代码" />
          </a-form-item>
          <a-form-item label="专业">
            <a-input v-model:value="classForm.major" placeholder="请输入专业" />
          </a-form-item>
          <a-form-item label="年级">
            <a-select v-model:value="classForm.grade" placeholder="请选择年级">
              <a-select-option value="2024">2024级</a-select-option>
              <a-select-option value="2023">2023级</a-select-option>
              <a-select-option value="2022">2022级</a-select-option>
              <a-select-option value="2021">2021级</a-select-option>
            </a-select>
          </a-form-item>
          <a-form-item label="班级描述">
            <a-textarea v-model:value="classForm.description" placeholder="请输入班级描述" />
          </a-form-item>
        </a-form>
      </a-modal>
      
      <!-- 布置作业弹窗 -->
      <a-modal
        v-model:open="showCreateTaskModal"
        title="布置作业"
        width="700px"
        @ok="handleCreateTask"
      >
        <a-form :model="taskForm" layout="vertical">
          <a-row :gutter="16">
            <a-col :span="16">
              <a-form-item label="作业标题" required>
                <a-input v-model:value="taskForm.title" placeholder="请输入作业标题" />
              </a-form-item>
            </a-col>
            <a-col :span="8">
              <a-form-item label="作业类型" required>
                <a-select v-model:value="taskForm.type" placeholder="选择类型">
                  <a-select-option value="HOMEWORK">课后作业</a-select-option>
                  <a-select-option value="QUIZ">随堂测验</a-select-option>
                  <a-select-option value="PROJECT">项目作业</a-select-option>
                  <a-select-option value="EXAM">考试</a-select-option>
                </a-select>
              </a-form-item>
            </a-col>
          </a-row>
          
          <a-row :gutter="16">
            <a-col :span="12">
              <a-form-item label="选择课程" required>
                <a-select v-model:value="taskForm.courseId" placeholder="请选择课程">
                  <a-select-option 
                    v-for="course in mockCourses" 
                    :key="course.id"
                    :value="course.id"
                  >
                    {{ course.name }}
                  </a-select-option>
                </a-select>
              </a-form-item>
            </a-col>
            <a-col :span="12">
              <a-form-item label="总分">
                <a-input-number 
                  v-model:value="taskForm.totalScore" 
                  placeholder="总分"
                  style="width: 100%"
                />
              </a-form-item>
            </a-col>
          </a-row>
          
          <a-row :gutter="16">
            <a-col :span="12">
              <a-form-item label="开始时间">
                <a-date-picker 
                  v-model:value="taskForm.startTime" 
                  show-time 
                  placeholder="请选择开始时间"
                  style="width: 100%"
                />
              </a-form-item>
            </a-col>
            <a-col :span="12">
              <a-form-item label="截止时间" required>
                <a-date-picker 
                  v-model:value="taskForm.endTime" 
                  show-time 
                  placeholder="请选择截止时间"
                  style="width: 100%"
                />
              </a-form-item>
            </a-col>
          </a-row>
          
          <a-form-item label="作业要求">
            <a-textarea 
              v-model:value="taskForm.requirements" 
              placeholder="请详细描述作业要求、评分标准等"
              :rows="4"
            />
          </a-form-item>
          
          <a-form-item label="提交方式">
            <a-radio-group v-model:value="taskForm.submitType">
              <a-radio value="TEXT">文本提交</a-radio>
              <a-radio value="FILE">文件上传</a-radio>
              <a-radio value="BOTH">文本+文件</a-radio>
            </a-radio-group>
          </a-form-item>
        </a-form>
      </a-modal>
    </a-spin>
  </div>
</template>

<script setup lang="ts">
import { ref, reactive, onMounted, computed } from 'vue'
import { message } from 'ant-design-vue'
import dayjs, { Dayjs } from 'dayjs'
import {
  PlusOutlined,
  BookOutlined,
  UserOutlined,
  FileTextOutlined,
  ClockCircleOutlined,
  ArrowRightOutlined,
  ArrowUpOutlined,
  ExclamationCircleOutlined,
  TeamOutlined,
  MoreOutlined,
  PlayCircleOutlined,
  StarOutlined,
  CalendarOutlined,
  RobotOutlined,
  BellOutlined,
  BulbOutlined,
  BarChartOutlined,
  EditOutlined,
  FileAddOutlined,
  QuestionCircleOutlined,
  ThunderboltOutlined,
  FolderOutlined,
  SettingOutlined
} from '@ant-design/icons-vue'
import { useAuthStore } from '@/stores/auth'

const authStore = useAuthStore()

// 响应式数据
const loading = ref(false)
const courseFilter = ref('all')
const assignmentFilter = ref('all')
const selectedDate = ref<Dayjs>(dayjs())

// 当前日期和问候语
const currentDate = computed(() => new Date())
const greeting = computed(() => {
  const hour = new Date().getHours()
  if (hour < 12) return '早上好'
  if (hour < 18) return '下午好'
  return '晚上好'
})

// 统计数据
const stats = reactive({
  courseCount: 6,
  studentCount: 148,
  assignmentCount: 12,
  pendingCount: 8,
  courseGrowth: 15,
  studentGrowth: 12,
  weeklyAssignments: 3
})

// 模拟课程数据
const mockCourses = ref([
  {
    id: 1,
    name: '高等数学A',
    description: '本课程是理工科学生的必修课程，主要讲授微积分、线性代数等内容',
    subject: '数学',
    difficulty: '中级',
    studentCount: 45,
    chapterCount: 12,
    rating: '4.8',
    progress: 75,
    weeklyActive: 38,
    completionRate: 85,
    avgScore: '82.5'
  },
  {
    id: 2,
    name: '数据结构与算法',
    description: '深入学习各种数据结构和算法设计思想，培养编程思维',
    subject: '计算机科学',
    difficulty: '高级',
    studentCount: 32,
    chapterCount: 15,
    rating: '4.9',
    progress: 60,
    weeklyActive: 28,
    completionRate: 78,
    avgScore: '85.2'
  },
  {
    id: 3,
    name: '大学英语',
    description: '提升英语听说读写能力，培养国际化视野',
    subject: '英语',
    difficulty: '中级',
    studentCount: 52,
    chapterCount: 10,
    rating: '4.6',
    progress: 90,
    weeklyActive: 45,
    completionRate: 92,
    avgScore: '78.9'
  }
])

// 模拟作业数据
const mockAssignments = ref([
  {
    id: 1,
    title: '微积分练习题集',
    description: '完成第三章导数相关练习题，包括基础题和提高题',
    courseName: '高等数学A',
    dueDate: '2024-12-30',
    priority: 'high',
    submittedCount: 38,
    totalStudents: 45,
    gradedCount: 25
  },
  {
    id: 2,
    title: '二叉树遍历算法实现',
    description: '用Java或Python实现二叉树的前序、中序、后序遍历',
    courseName: '数据结构与算法',
    dueDate: '2025-01-05',
    priority: 'medium',
    submittedCount: 28,
    totalStudents: 32,
    gradedCount: 15
  },
  {
    id: 3,
    title: '英语口语展示',
    description: '准备5分钟的英语口语展示，主题自选',
    courseName: '大学英语',
    dueDate: '2025-01-08',
    priority: 'low',
    submittedCount: 48,
    totalStudents: 52,
    gradedCount: 40
  }
])

// 今日事件
const todayEvents = ref([
  {
    id: 1,
    time: '09:00',
    title: '高等数学课程',
    description: '第三章 导数与微分'
  },
  {
    id: 2,
    time: '14:30',
    title: '作业批改',
    description: '数据结构作业批改'
  },
  {
    id: 3,
    time: '16:00',
    title: '学生答疑',
    description: '在线答疑时间'
  }
])

// 弹窗状态
const showCreateCourseModal = ref(false)
const showCreateClassModal = ref(false)
const showCreateTaskModal = ref(false)

// 表单数据
const courseForm = reactive({
  name: '',
  code: '',
  subject: '',
  difficulty: '中级',
  description: ''
})

const classForm = reactive({
  name: '',
  code: '',
  major: '',
  grade: '',
  description: ''
})

const taskForm = reactive({
  title: '',
  type: 'HOMEWORK',
  courseId: null,
  totalScore: 100,
  startTime: null,
  endTime: null,
  requirements: '',
  submitType: 'BOTH'
})

// 工具函数
const formatDate = (date: Date) => {
  return date.toLocaleDateString('zh-CN', {
    year: 'numeric',
    month: 'long',
    day: 'numeric',
    weekday: 'long'
  })
}

const formatDeadline = (dateStr: string) => {
  const date = dayjs(dateStr)
  const now = dayjs()
  const diffDays = date.diff(now, 'day')
  
  if (diffDays < 0) return '已过期'
  if (diffDays === 0) return '今天到期'
  if (diffDays === 1) return '明天到期'
  if (diffDays <= 7) return `${diffDays}天后到期`
  return date.format('MM月DD日')
}

const isUrgent = (dateStr: string) => {
  const date = dayjs(dateStr)
  const now = dayjs()
  return date.diff(now, 'day') <= 2
}

const getCourseGradient = (subject: string) => {
  const gradients: Record<string, string> = {
    '计算机科学': 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
    '数学': 'linear-gradient(135deg, #f093fb 0%, #f5576c 100%)',
    '物理': 'linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)',
    '化学': 'linear-gradient(135deg, #43e97b 0%, #38f9d7 100%)',
    '英语': 'linear-gradient(135deg, #fa709a 0%, #fee140 100%)',
    '历史': 'linear-gradient(135deg, #a8edea 0%, #fed6e3 100%)',
    '地理': 'linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%)',
    '生物': 'linear-gradient(135deg, #a8e6cf 0%, #dcedc1 100%)'
  }
  return gradients[subject] || 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)'
}

const getCourseColor = (subject: string) => {
  const colors: Record<string, string> = {
    '计算机科学': '#667eea',
    '数学': '#f5576c',
    '物理': '#4facfe',
    '化学': '#43e97b',
    '英语': '#fa709a',
    '历史': '#a8edea',
    '地理': '#ffecd2',
    '生物': '#a8e6cf'
  }
  return colors[subject] || '#667eea'
}

// 事件处理
const handleCreateCourse = async () => {
  message.success('课程创建成功')
  showCreateCourseModal.value = false
  Object.assign(courseForm, {
    name: '',
    code: '',
    subject: '',
    difficulty: '中级',
    description: ''
  })
}

const handleCreateClass = async () => {
  message.success('班级创建成功')
  showCreateClassModal.value = false
  Object.assign(classForm, {
    name: '',
    code: '',
    major: '',
    grade: '',
    description: ''
  })
}

const handleCreateTask = async () => {
  message.success('作业布置成功')
  showCreateTaskModal.value = false
  Object.assign(taskForm, {
    title: '',
    type: 'HOMEWORK',
    courseId: null,
    totalScore: 100,
    startTime: null,
    endTime: null,
    requirements: '',
    submitType: 'BOTH'
  })
}

const handleAIRecommendation = () => {
  message.info('AI推荐功能开发中...')
}

const handleAIFeature = (type: string) => {
  message.info(`AI${type}功能开发中...`)
}

// 初始化
onMounted(() => {
  // 模拟数据加载
  loading.value = true
  setTimeout(() => {
    loading.value = false
  }, 1000)
})
</script>

<style scoped>
/* 全局样式 */
.teacher-dashboard {
  min-height: 100vh;
  background: #f5f7fa;
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
}

/* 顶部欢迎区域样式 */
.dashboard-header {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  padding: 40px 0;
  margin-bottom: 32px;
  position: relative;
  overflow: hidden;
}

.dashboard-header::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 20"><defs><radialGradient id="a" cx="50%" cy="0%" r="100%"><stop offset="0%" style="stop-color:rgb(255,255,255);stop-opacity:0.1" /><stop offset="100%" style="stop-color:rgb(255,255,255);stop-opacity:0" /></radialGradient></defs><rect width="100" height="20" fill="url(%23a)" /></svg>') repeat-x;
  opacity: 0.3;
}

.header-content {
  max-width: 1200px;
  margin: 0 auto;
  padding: 0 24px;
  display: flex;
  justify-content: space-between;
  align-items: center;
  position: relative;
  z-index: 1;
}

.avatar-section {
  display: flex;
  align-items: center;
  gap: 20px;
}

.teacher-avatar {
  background: rgba(255, 255, 255, 0.2);
  color: white;
  border: 3px solid rgba(255, 255, 255, 0.3);
  font-size: 24px;
  font-weight: 600;
}

.welcome-info {
  flex: 1;
}

.welcome-title {
  font-size: 2rem;
  font-weight: 700;
  margin: 0 0 8px 0;
  letter-spacing: -0.02em;
}

.welcome-subtitle {
  font-size: 1rem;
  opacity: 0.9;
  margin: 0 0 16px 0;
  font-weight: 400;
}

.teacher-badges {
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
}

.header-actions {
  display: flex;
  align-items: center;
}

.action-btn {
  background: rgba(255, 255, 255, 0.15);
  border: 1px solid rgba(255, 255, 255, 0.3);
  color: white;
  backdrop-filter: blur(10px);
  border-radius: 8px;
  font-weight: 500;
  transition: all 0.3s ease;
}

.action-btn:hover {
  background: rgba(255, 255, 255, 0.25);
  border-color: rgba(255, 255, 255, 0.5);
  color: white;
  transform: translateY(-2px);
}

/* 核心统计卡片样式 */
.stats-overview {
  background: white;
  padding: 32px;
  border-radius: 20px;
  box-shadow: 0 8px 32px rgba(0, 0, 0, 0.08);
  border: 1px solid rgba(0, 0, 0, 0.04);
  margin-bottom: 32px;
  position: relative;
  overflow: hidden;
}

.stats-overview::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  height: 4px;
  background: linear-gradient(90deg, #667eea, #764ba2, #f093fb, #f5576c);
}

.stats-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
  gap: 32px;
}

.stat-card {
  background: linear-gradient(135deg, #f8fafc 0%, #ffffff 100%);
  padding: 24px;
  border-radius: 16px;
  border: 1px solid #e2e8f0;
  transition: all 0.3s ease;
  position: relative;
  overflow: hidden;
}

.stat-card::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  height: 3px;
}

.stat-card.courses::before {
  background: linear-gradient(90deg, #667eea, #764ba2);
}

.stat-card.students::before {
  background: linear-gradient(90deg, #4facfe, #00f2fe);
}

.stat-card.assignments::before {
  background: linear-gradient(90deg, #43e97b, #38f9d7);
}

.stat-card.pending::before {
  background: linear-gradient(90deg, #fa709a, #fee140);
}

.stat-card:hover {
  transform: translateY(-4px);
  box-shadow: 0 12px 40px rgba(0, 0, 0, 0.12);
}

.stat-icon {
  width: 56px;
  height: 56px;
  border-radius: 14px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 24px;
  color: white;
  margin-bottom: 16px;
}

.stat-card.courses .stat-icon {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
}

.stat-card.students .stat-icon {
  background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
}

.stat-card.assignments .stat-icon {
  background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
}

.stat-card.pending .stat-icon {
  background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
}

.stat-content {
  flex: 1;
}

.stat-number {
  font-size: 2.5rem;
  font-weight: 700;
  color: #1a202c;
  line-height: 1;
  margin-bottom: 8px;
}

.stat-label {
  font-size: 0.875rem;
  color: #64748b;
  font-weight: 500;
  margin-bottom: 12px;
}

.stat-trend {
  display: flex;
  align-items: center;
  gap: 4px;
  font-size: 0.75rem;
  font-weight: 500;
}

.trend-up {
  color: #10b981;
}

.trend-down {
  color: #ef4444;
}

.trend-urgent {
  color: #f59e0b;
  font-weight: 600;
}

/* 主要内容区域样式 */
.dashboard-content {
  max-width: 1200px;
  margin: 0 auto;
  padding: 0 24px;
}

/* 内容区块通用样式 */
.content-section {
  background: white;
  border-radius: 20px;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
  border: 1px solid rgba(0, 0, 0, 0.04);
  margin-bottom: 32px;
  overflow: hidden;
  transition: all 0.3s ease;
}

.content-section:hover {
  box-shadow: 0 8px 30px rgba(0, 0, 0, 0.12);
}

.section-header {
  padding: 24px 32px 0 32px;
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 24px;
}

.section-title {
  font-size: 1.25rem;
  font-weight: 600;
  color: #1a202c;
  display: flex;
  align-items: center;
  gap: 8px;
  margin: 0;
}

.section-title .anticon {
  color: #667eea;
}

.section-actions {
  display: flex;
  align-items: center;
  gap: 12px;
}

/* 课程网格样式 */
.course-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
  gap: 24px;
  padding: 0 32px 32px 32px;
}

.course-card {
  background: white;
  border-radius: 16px;
  border: 1px solid #e2e8f0;
  overflow: hidden;
  transition: all 0.3s ease;
  cursor: pointer;
  position: relative;
}

.course-card:hover {
  transform: translateY(-6px);
  box-shadow: 0 12px 40px rgba(0, 0, 0, 0.15);
  border-color: #667eea;
}

.course-header {
  position: relative;
  height: 120px;
  overflow: hidden;
}

.course-cover {
  width: 100%;
  height: 100%;
  display: flex;
  flex-direction: column;
  justify-content: space-between;
  padding: 16px;
  color: white;
  position: relative;
}

.course-cover::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: rgba(0, 0, 0, 0.1);
}

.course-category {
  font-size: 0.75rem;
  font-weight: 500;
  background: rgba(255, 255, 255, 0.2);
  padding: 4px 8px;
  border-radius: 6px;
  align-self: flex-start;
  backdrop-filter: blur(10px);
  z-index: 1;
  position: relative;
}

.course-level {
  font-size: 0.75rem;
  font-weight: 500;
  background: rgba(255, 255, 255, 0.2);
  padding: 4px 8px;
  border-radius: 6px;
  align-self: flex-end;
  backdrop-filter: blur(10px);
  z-index: 1;
  position: relative;
}

.course-actions {
  position: absolute;
  top: 16px;
  right: 16px;
  z-index: 2;
}

.course-content {
  padding: 20px;
}

.course-title {
  font-size: 1.125rem;
  font-weight: 600;
  color: #1a202c;
  margin: 0 0 8px 0;
  line-height: 1.4;
}

.course-description {
  font-size: 0.875rem;
  color: #64748b;
  margin: 0 0 16px 0;
  line-height: 1.5;
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
  overflow: hidden;
}

.course-meta {
  display: flex;
  gap: 16px;
  margin-bottom: 16px;
  flex-wrap: wrap;
}

.meta-item {
  display: flex;
  align-items: center;
  gap: 4px;
  font-size: 0.75rem;
  color: #64748b;
}

.meta-item .anticon {
  color: #94a3b8;
}

.course-progress {
  margin-bottom: 16px;
}

.progress-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 8px;
  font-size: 0.75rem;
  color: #64748b;
  font-weight: 500;
}

.course-stats {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 16px;
  margin-bottom: 20px;
  padding: 16px;
  background: #f8fafc;
  border-radius: 12px;
}

.course-stats .stat-item {
  text-align: center;
}

.stat-value {
  display: block;
  font-size: 1.125rem;
  font-weight: 600;
  color: #1a202c;
  line-height: 1;
}

.stat-label {
  display: block;
  font-size: 0.75rem;
  color: #64748b;
}

.course-footer {
  padding: 0 20px 20px 20px;
}

.create-card {
  background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
  border: 2px dashed #cbd5e0;
  display: flex;
  align-items: center;
  justify-content: center;
  min-height: 300px;
  transition: all 0.3s ease;
}

.create-card:hover {
  background: linear-gradient(135deg, #e2e8f0 0%, #cbd5e0 100%);
  border-color: #667eea;
  transform: translateY(-4px);
}

.create-content {
  text-align: center;
  color: #64748b;
}

.create-icon {
  font-size: 3rem;
  margin-bottom: 16px;
  color: #94a3b8;
}

.create-content h4 {
  font-size: 1.125rem;
  font-weight: 600;
  margin: 0 0 8px 0;
  color: #475569;
}

.create-content p {
  font-size: 0.875rem;
  margin: 0;
  color: #64748b;
}

/* 作业列表样式 */
.assignment-list {
  padding: 0 32px 32px 32px;
}

.assignment-card {
  background: #f8fafc;
  border-radius: 16px;
  padding: 20px;
  margin-bottom: 16px;
  border: 1px solid #e2e8f0;
  transition: all 0.3s ease;
  cursor: pointer;
}

.assignment-card:hover {
  transform: translateX(4px);
  box-shadow: 0 8px 25px rgba(102, 126, 234, 0.15);
  border-color: #667eea;
  background: white;
}

.assignment-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 12px;
}

.assignment-priority {
  display: flex;
  align-items: center;
  gap: 8px;
}

.priority-indicator {
  width: 8px;
  height: 8px;
  border-radius: 50%;
}

.priority-indicator.high {
  background: #ef4444;
}

.priority-indicator.medium {
  background: #f59e0b;
}

.priority-indicator.low {
  background: #10b981;
}

.assignment-course {
  font-size: 0.75rem;
  color: #64748b;
  font-weight: 500;
}

.assignment-deadline {
  display: flex;
  align-items: center;
  gap: 4px;
  font-size: 0.75rem;
  color: #64748b;
  font-weight: 500;
}

.assignment-deadline.urgent {
  color: #ef4444;
  font-weight: 600;
}

.assignment-title {
  font-size: 1rem;
  font-weight: 600;
  color: #1a202c;
  margin: 0 0 8px 0;
  line-height: 1.4;
}

.assignment-description {
  font-size: 0.875rem;
  color: #64748b;
  margin: 0 0 16px 0;
  line-height: 1.5;
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
  overflow: hidden;
}

.assignment-progress {
  margin-bottom: 16px;
}

.progress-stats {
  display: flex;
  gap: 24px;
  margin-bottom: 12px;
}

.progress-stats .stat {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 4px;
}

.progress-stats .number {
  font-size: 1.25rem;
  font-weight: 600;
  color: #1a202c;
  line-height: 1;
}

.progress-stats .label {
  font-size: 0.75rem;
  color: #64748b;
}

.progress-bar {
  margin-bottom: 8px;
}

.progress-info {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 8px;
  font-size: 0.75rem;
  color: #64748b;
  font-weight: 500;
}

.assignment-actions {
  display: flex;
  justify-content: flex-end;
  gap: 8px;
}

/* AI教学助手样式 */
.ai-section {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
}

.ai-section .section-header {
  color: white;
}

.ai-section .section-title {
  color: white;
}

.ai-section .section-title .anticon {
  color: rgba(255, 255, 255, 0.9);
}

.ai-recommendations {
  padding: 0 32px 32px 32px;
}

.ai-card {
  background: rgba(255, 255, 255, 0.1);
  border-radius: 16px;
  padding: 20px;
  margin-bottom: 16px;
  backdrop-filter: blur(10px);
  border: 1px solid rgba(255, 255, 255, 0.2);
  transition: all 0.3s ease;
  cursor: pointer;
}

.ai-card:hover {
  background: rgba(255, 255, 255, 0.2);
  transform: translateY(-2px);
  box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
}

.ai-header {
  display: flex;
  align-items: center;
  margin-bottom: 12px;
}

.ai-icon {
  font-size: 20px;
  margin-right: 8px;
  color: rgba(255, 255, 255, 0.9);
}

.ai-title {
  font-size: 1rem;
  font-weight: 600;
}

.ai-content p {
  font-size: 0.875rem;
  opacity: 0.9;
  line-height: 1.5;
  margin: 0 0 16px 0;
}

.ai-features-grid {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 12px;
}

.ai-feature {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 12px;
  background: rgba(255, 255, 255, 0.1);
  border-radius: 12px;
  cursor: pointer;
  transition: all 0.3s ease;
  font-size: 0.875rem;
  font-weight: 500;
}

.ai-feature:hover {
  background: rgba(255, 255, 255, 0.2);
  transform: translateY(-2px);
}

.ai-feature .anticon {
  font-size: 16px;
  color: rgba(255, 255, 255, 0.9);
}

/* 数据洞察样式 */
.insights-widget {
  padding: 32px;
}

.insight-item {
  padding: 20px;
  border-radius: 12px;
  background: #f8fafc;
  margin-bottom: 16px;
  border: 1px solid #e2e8f0;
  transition: all 0.3s ease;
}

.insight-item:hover {
  background: white;
  box-shadow: 0 4px 15px rgba(0, 0, 0, 0.08);
}

.insight-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 16px;
}

.insight-title {
  font-size: 0.875rem;
  font-weight: 600;
  color: #1a202c;
}

.insight-trend {
  font-size: 0.75rem;
  font-weight: 500;
}

.insight-trend.up {
  color: #10b981;
}

.insight-trend.down {
  color: #ef4444;
}

.insight-chart {
  height: 40px;
  display: flex;
  align-items: end;
  justify-content: center;
}

.mini-chart {
  display: flex;
  align-items: end;
  gap: 3px;
  height: 100%;
}

.chart-bar {
  width: 6px;
  background: linear-gradient(to top, #667eea, #764ba2);
  border-radius: 3px;
  transition: all 0.3s ease;
}

.insight-value {
  text-align: center;
}

.value {
  font-size: 1.75rem;
  font-weight: 700;
  color: #1a202c;
  line-height: 1;
  margin-bottom: 4px;
}

.unit {
  font-size: 0.75rem;
  color: #64748b;
}

.insight-rating {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
}

.rating-value {
  font-size: 1.25rem;
  font-weight: 600;
  color: #1a202c;
}

/* 快速操作样式 */
.quick-actions-grid {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 12px;
  padding: 32px;
}

.quick-action {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 16px;
  background: #f8fafc;
  border-radius: 12px;
  cursor: pointer;
  transition: all 0.3s ease;
  border: 1px solid #e2e8f0;
  font-size: 0.875rem;
  font-weight: 500;
  color: #475569;
}

.quick-action:hover {
  background: white;
  transform: translateY(-2px);
  box-shadow: 0 4px 15px rgba(0, 0, 0, 0.08);
  border-color: #667eea;
  color: #667eea;
}

.quick-action .anticon {
  font-size: 18px;
  color: #667eea;
}

/* 日历样式 */
.calendar-widget {
  padding: 32px;
}

.calendar-cell {
  position: relative;
}

.event-indicators {
  position: absolute;
  bottom: 2px;
  left: 50%;
  transform: translateX(-50%);
  display: flex;
  gap: 2px;
}

.event-dot {
  width: 4px;
  height: 4px;
  border-radius: 50%;
  background: #667eea;
}

.event-dot.course {
  background: #667eea;
}

.event-dot.grading {
  background: #f093fb;
}

.event-dot.consultation {
  background: #4facfe;
}

.calendar-events {
  margin-top: 24px;
  padding-top: 24px;
  border-top: 1px solid #e2e8f0;
}

.calendar-events h4 {
  font-size: 1rem;
  font-weight: 600;
  color: #1a202c;
  margin: 0 0 16px 0;
}

.event-list {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.event-item {
  display: flex;
  gap: 12px;
  padding: 12px;
  background: #f8fafc;
  border-radius: 8px;
  border-left: 3px solid #667eea;
}

.event-time {
  font-size: 0.75rem;
  font-weight: 600;
  color: #667eea;
  min-width: 40px;
}

.event-content {
  flex: 1;
}

.event-title {
  font-size: 0.875rem;
  font-weight: 500;
  color: #1a202c;
  margin: 0 0 4px 0;
}

.event-desc {
  font-size: 0.75rem;
  color: #64748b;
  margin: 0;
}

.no-events {
  text-align: center;
  padding: 40px 20px;
  color: #94a3b8;
}

.no-events .anticon {
  font-size: 2rem;
  margin-bottom: 8px;
  opacity: 0.5;
}

.no-events p {
  margin: 0;
  font-size: 0.875rem;
}

/* 空状态样式 */
.empty-state {
  text-align: center;
  padding: 60px 20px;
  color: #94a3b8;
}

.empty-state .anticon {
  font-size: 3rem;
  margin-bottom: 16px;
  opacity: 0.5;
}

.empty-state p {
  margin: 0 0 20px 0;
  font-size: 0.875rem;
}

/* 响应式设计 */
@media (max-width: 1024px) {
  .dashboard-content {
    padding: 0 16px;
  }
  
  .stats-grid {
    grid-template-columns: repeat(2, 1fr);
    gap: 16px;
  }
  
  .course-grid {
    grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
    gap: 16px;
    padding: 0 16px 24px 16px;
  }
  
  .section-header {
    padding: 16px 16px 0 16px;
  }
  
  .assignment-list {
    padding: 0 16px 24px 16px;
  }
  
  .ai-recommendations,
  .insights-widget,
  .quick-actions-grid,
  .calendar-widget {
    padding: 16px;
  }
}

@media (max-width: 768px) {
  .header-content {
    flex-direction: column;
    text-align: center;
    gap: 24px;
  }
  
  .avatar-section {
    flex-direction: column;
    text-align: center;
    gap: 16px;
  }
  
  .stats-grid {
    grid-template-columns: 1fr;
  }
  
  .course-grid {
    grid-template-columns: 1fr;
  }
  
  .course-stats {
    grid-template-columns: repeat(2, 1fr);
  }
  
  .progress-stats {
    gap: 16px;
  }
  
  .ai-features-grid {
    grid-template-columns: 1fr;
  }
  
  .quick-actions-grid {
    grid-template-columns: 1fr;
  }
}

@media (max-width: 480px) {
  .welcome-title {
    font-size: 1.5rem;
  }
  
  .stat-number {
    font-size: 2rem;
  }
  
  .course-stats {
    grid-template-columns: 1fr;
  }
  
  .progress-stats {
    flex-direction: column;
    gap: 8px;
  }
  
  .assignment-actions {
    flex-direction: column;
    gap: 8px;
  }
}

/* 自定义滚动条 */
::-webkit-scrollbar {
  width: 6px;
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

/* 动画效果 */
@keyframes fadeInUp {
  from {
    opacity: 0;
    transform: translateY(20px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

.content-section {
  animation: fadeInUp 0.6s ease-out;
}

.stat-card {
  animation: fadeInUp 0.6s ease-out;
}

.course-card {
  animation: fadeInUp 0.6s ease-out;
}

.assignment-card {
  animation: fadeInUp 0.6s ease-out;
}

/* 加载状态 */
.ant-spin-container {
  min-height: 200px;
}

/* 模态框样式优化 */
:deep(.ant-modal-content) {
  border-radius: 16px;
  overflow: hidden;
}

:deep(.ant-modal-header) {
  background: linear-gradient(135deg, #f8fafc 0%, #ffffff 100%);
  border-bottom: 1px solid #e2e8f0;
  padding: 20px 24px;
}

:deep(.ant-modal-title) {
  font-size: 1.125rem;
  font-weight: 600;
  color: #1a202c;
}

:deep(.ant-modal-body) {
  padding: 24px;
}

:deep(.ant-form-item-label > label) {
  font-weight: 500;
  color: #374151;
}

:deep(.ant-input),
:deep(.ant-select-selector),
:deep(.ant-picker) {
  border-radius: 8px;
  border: 1px solid #d1d5db;
  transition: all 0.3s ease;
}

:deep(.ant-input:focus),
:deep(.ant-select-focused .ant-select-selector),
:deep(.ant-picker-focused) {
  border-color: #667eea;
  box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
}

:deep(.ant-btn-primary) {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  border: none;
  border-radius: 8px;
  font-weight: 500;
  transition: all 0.3s ease;
}

:deep(.ant-btn-primary:hover) {
  background: linear-gradient(135deg, #5a67d8 0%, #6b46c1 100%);
  transform: translateY(-1px);
  box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
}

:deep(.ant-progress-bg) {
  background: linear-gradient(90deg, #667eea, #764ba2);
}

:deep(.ant-tag) {
  border-radius: 6px;
  font-size: 0.75rem;
  font-weight: 500;
  padding: 2px 8px;
}
</style>