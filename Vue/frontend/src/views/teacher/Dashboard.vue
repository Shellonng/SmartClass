<template>
  <div class="teacher-dashboard">
    <a-spin :spinning="loading" tip="加载中...">
      <!-- 顶部欢迎区域 -->
      <div class="dashboard-header">
        <div class="header-content">
          <div class="welcome-section">
            <div class="avatar-section">
              <a-avatar :size="64" :src="authStore.user?.avatar" class="teacher-avatar">
                {{ authStore.user?.name?.charAt(0) || 'T' }}
              </a-avatar>
              <div class="welcome-info">
                <h1 class="welcome-title">
                  {{ greeting }}，{{ authStore.user?.name || '老师' }}
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
                  <a-button type="link" @click="$router.push('/teacher/courses')">
                    查看全部 <ArrowRightOutlined />
                  </a-button>
                </div>
              </div>
              
              <div class="course-grid">
                <div 
                  v-for="course in displayCourses" 
                  :key="course.id"
                  class="course-card"
                  @click="$router.push(`/teacher/courses/${course.id}`)"
                >
                  <div class="course-header">
                    <div class="course-cover" :style="{ background: getCourseGradient(course.subject) }">
                      <div class="course-category">{{ course.subject }}</div>
                      <div class="course-level">{{ course.difficulty || '中级' }}</div>
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
                        <span>{{ course.rating || 4.8 }}分</span>
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
                  <a-button type="link" @click="$router.push('/teacher/assignments')">
                    查看全部 <ArrowRightOutlined />
                  </a-button>
                </div>
              </div>
              
              <div class="assignment-list">
                <div 
                  v-for="assignment in displayAssignments" 
                  :key="assignment.id"
                  class="assignment-card"
                  @click="$router.push(`/teacher/assignments/${assignment.id}`)"
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
                        <span class="number">{{ assignment.gradedCount || 0 }}</span>
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
                
                <div v-if="displayAssignments.length === 0" class="empty-state">
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
                  @select="onDateSelect"
                >
                  <template #dateCellRender="{ current }">
                    <div class="calendar-cell">
                      <div v-if="getDateEvents(current).length > 0" class="event-indicators">
                        <div 
                          v-for="event in getDateEvents(current).slice(0, 2)" 
                          :key="event.id"
                          class="event-dot"
                          :class="event.type"
                        ></div>
                      </div>
                    </div>
                  </template>
                </a-calendar>
                
                <div class="calendar-events">
                  <h4>今日安排</h4>
                  <div v-if="todayEvents.length > 0" class="event-list">
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
                  <div v-else class="no-events">
                    <CalendarOutlined />
                    <p>今日暂无安排</p>
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
                <div class="quick-action" @click="$router.push('/teacher/grading')">
                  <EditOutlined />
                  <span>批改作业</span>
                </div>
                <div class="quick-action" @click="$router.push('/teacher/students')">
                  <UserOutlined />
                  <span>学生管理</span>
                </div>
                <div class="quick-action" @click="$router.push('/teacher/analytics')">
                  <BarChartOutlined />
                  <span>数据分析</span>
                </div>
                <div class="quick-action" @click="$router.push('/teacher/resources')">
                  <FolderOutlined />
                  <span>资源库</span>
                </div>
                <div class="quick-action" @click="$router.push('/teacher/settings')">
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
          
          <a-row :gutter="16">
            <a-col :span="12">
              <a-form-item label="学分">
                <a-input-number 
                  v-model:value="courseForm.credits" 
                  :min="1" 
                  :max="10" 
                  placeholder="学分"
                  style="width: 100%"
                />
              </a-form-item>
            </a-col>
            <a-col :span="12">
              <a-form-item label="预计课时">
                <a-input-number 
                  v-model:value="courseForm.hours" 
                  :min="1" 
                  placeholder="课时"
                  style="width: 100%"
                />
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
          
          <a-form-item label="课程标签">
            <a-select 
              v-model:value="courseForm.tags" 
              mode="tags" 
              placeholder="输入标签后按回车添加"
              style="width: 100%"
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
          <a-form-item label="最大人数">
            <a-input-number 
              v-model:value="classForm.maxStudents" 
              :min="1" 
              :max="200" 
              placeholder="最大学生人数"
              style="width: 100%"
            />
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
                    v-for="course in courses" 
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
          
          <a-form-item label="高级设置">
            <a-space direction="vertical" style="width: 100%">
              <a-checkbox v-model:checked="taskForm.allowLateSubmit">
                允许迟交（会扣分）
              </a-checkbox>
              <a-checkbox v-model:checked="taskForm.autoGrade">
                启用AI智能批改
              </a-checkbox>
            </a-space>
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
import {
  getDashboardData,
  getCourses,
  getAssignments,
  createCourse,
  createAssignment
} from '@/api/teacher'
import type {
  Course,
  Assignment
} from '@/api/teacher'

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
  courseCount: 0,
  studentCount: 0,
  assignmentCount: 0,
  pendingCount: 0,
  courseGrowth: 0,
  studentGrowth: 0,
  weeklyAssignments: 0
})

// 课程和作业数据
const courses = ref<Course[]>([])
const assignments = ref<Assignment[]>([])

// 显示的课程和作业（根据筛选）
const displayCourses = computed(() => {
  let filtered = courses.value
  if (courseFilter.value === 'active') {
    filtered = filtered.filter(course => course.status === 'ACTIVE')
  } else if (courseFilter.value === 'completed') {
    filtered = filtered.filter(course => course.status === 'COMPLETED')
  }
  return filtered.slice(0, 6) // 显示前6个
})

const displayAssignments = computed(() => {
  let filtered = assignments.value
  if (assignmentFilter.value === 'pending') {
    filtered = filtered.filter(assignment => assignment.status === 'PUBLISHED')
  } else if (assignmentFilter.value === 'graded') {
    filtered = filtered.filter(assignment => assignment.status === 'GRADED')
  }
  return filtered.slice(0, 5) // 显示前5个
})

// 今日事件
const todayEvents = ref([
  {
    id: 1,
    time: '09:00',
    title: '高等数学课程',
    description: '第三章 导数与微分',
    type: 'course'
  },
  {
    id: 2,
    time: '14:30',
    title: '作业批改',
    description: '线性代数作业批改',
    type: 'grading'
  },
  {
    id: 3,
    time: '16:00',
    title: '学生答疑',
    description: '在线答疑时间',
    type: 'consultation'
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
  credits: 3,
  hours: 48,
  description: '',
  tags: []
})

const classForm = reactive({
  name: '',
  code: '',
  major: '',
  grade: '',
  maxStudents: 50,
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
  submitType: 'BOTH',
  allowLateSubmit: false,
  autoGrade: false
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
  const gradients = {
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
  const colors = {
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

const getDateEvents = (date: Dayjs) => {
  // 模拟获取日期事件
  return []
}

const onDateSelect = (date: Dayjs) => {
  selectedDate.value = date
}

// 事件处理
const handleCreateCourse = async () => {
  try {
    loading.value = true
    await createCourse(courseForm)
    message.success('课程创建成功')
    showCreateCourseModal.value = false
    Object.assign(courseForm, {
      name: '',
      code: '',
      subject: '',
      difficulty: '中级',
      credits: 3,
      hours: 48,
      description: '',
      tags: []
    })
    await loadData()
  } catch (error) {
    message.error('创建课程失败')
  } finally {
    loading.value = false
  }
}

const handleCreateClass = async () => {
  try {
    loading.value = true
    // TODO: 调用创建班级API
    message.success('班级创建成功')
    showCreateClassModal.value = false
    Object.assign(classForm, {
      name: '',
      code: '',
      major: '',
      grade: '',
      maxStudents: 50,
      description: ''
    })
  } catch (error) {
    message.error('创建班级失败')
  } finally {
    loading.value = false
  }
}

const handleCreateTask = async () => {
  try {
    loading.value = true
    await createAssignment(taskForm)
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
      submitType: 'BOTH',
      allowLateSubmit: false,
      autoGrade: false
    })
    await loadData()
  } catch (error) {
    message.error('布置作业失败')
  } finally {
    loading.value = false
  }
}

const handleAIRecommendation = () => {
  message.info('AI推荐功能开发中...')
}

const handleAIFeature = (type: string) => {
  message.info(`AI${type}功能开发中...`)
}

// 数据加载
const loadData = async () => {
  try {
    loading.value = true
    
    // 并行加载数据
    const [dashboardResponse, coursesResponse, assignmentsResponse] = await Promise.all([
      getDashboardData(),
      getCourses({ page: 1, size: 10 }),
      getAssignments({ page: 1, size: 10 })
    ])
    
    // 更新统计数据
    if (dashboardResponse.data) {
      Object.assign(stats, dashboardResponse.data.stats || {})
    }
    
    // 更新课程数据
    if (coursesResponse.data) {
      courses.value = coursesResponse.data.map(course => ({
        ...course,
        progress: Math.floor(Math.random() * 100),
        weeklyActive: Math.floor(Math.random() * 50) + 10,
        completionRate: Math.floor(Math.random() * 100),
        avgScore: (Math.random() * 40 + 60).toFixed(1),
        chapterCount: Math.floor(Math.random() * 20) + 5,
        rating: (Math.random() * 1 + 4).toFixed(1)
      }))
      stats.courseCount = courses.value.length
    }
    
    // 更新作业数据
    if (assignmentsResponse.data) {
      assignments.value = assignmentsResponse.data.map(assignment => ({
        ...assignment,
        priority: ['high', 'medium', 'low'][Math.floor(Math.random() * 3)],
        gradedCount: Math.floor(Math.random() * assignment.submittedCount)
      }))
      stats.assignmentCount = assignments.value.length
      stats.pendingCount = assignments.value.filter(a => a.status === 'PUBLISHED').length
    }
    
  } catch (error) {
    console.error('加载数据失败:', error)
    message.error('加载数据失败')
  } finally {
    loading.value = false
  }
}

// 初始化
onMounted(() => {
  loadData()
})
</script>

<style scoped>
.teacher-dashboard {
  background: #f8fafc;
  min-height: 100vh;
}

/* 顶部欢迎区域样式 */
.dashboard-header {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  padding: 32px 0;
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
  background: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><defs><pattern id="grain" width="100" height="100" patternUnits="userSpaceOnUse"><circle cx="50" cy="50" r="1" fill="%23ffffff" opacity="0.1"/></pattern></defs><rect width="100" height="100" fill="url(%23grain)"/></svg>') repeat;
  pointer-events: none;
}

.header-content {
  max-width: 1200px;
  margin: 0 auto;
  padding: 0 24px;
  position: relative;
  z-index: 1;
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 48px;
}

.welcome-section {
  flex: 1;
  display: flex;
  align-items: center;
  gap: 24px;
}

.avatar-section {
  display: flex;
  align-items: center;
  gap: 16px;
}

.welcome-info {
  flex: 1;
}

.welcome-title {
  font-size: 2rem;
  font-weight: 700;
  margin: 0 0 8px 0;
  background: linear-gradient(45deg, #ffffff, #e3f2fd);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}

.welcome-subtitle {
  font-size: 1rem;
  margin: 0;
  opacity: 0.9;
}

.quick-actions {
  display: flex;
  gap: 12px;
}

.quick-actions .ant-btn {
  height: 48px;
  border-radius: 12px;
  font-weight: 500;
  display: flex;
  align-items: center;
  gap: 8px;
}

/* 核心统计卡片样式 */
.stats-overview {
  background: white;
  padding: 24px;
  border-radius: 16px;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
  border: 1px solid rgba(0, 0, 0, 0.04);
  margin-bottom: 32px;
}

.stats-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 24px;
}

.stat-card {
  background: rgba(255, 255, 255, 0.15);
  backdrop-filter: blur(10px);
  border-radius: 16px;
  padding: 20px;
  text-align: center;
  border: 1px solid rgba(255, 255, 255, 0.2);
  min-width: 120px;
}

.stat-icon {
  font-size: 24px;
  margin-bottom: 8px;
  opacity: 0.9;
}

.stat-content {
  flex: 1;
}

.stat-number {
  font-size: 2rem;
  font-weight: 700;
  color: #1a1a1a;
  line-height: 1;
  margin-bottom: 4px;
}

.stat-label {
  font-size: 14px;
  color: #666;
  font-weight: 500;
}

.stat-trend {
  display: flex;
  align-items: center;
  gap: 4px;
  font-size: 12px;
  font-weight: 500;
}

.trend-up {
  color: #52c41a;
}

.trend-down {
  color: #ff4d4f;
}

.trend-urgent {
  color: #ff4d4f;
  font-weight: 500;
}

/* 主要内容区域样式 */
.dashboard-content {
  max-width: 1200px;
  margin: 0 auto;
  padding: 0 24px;
}

.content-section {
  background: white;
  border-radius: 16px;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
  border: 1px solid rgba(0, 0, 0, 0.04);
  margin-bottom: 24px;
  overflow: hidden;
}

.section-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 24px;
  border-bottom: 1px solid #f0f0f0;
}

.section-title {
  font-size: 1.25rem;
  font-weight: 600;
  color: #1a1a1a;
  margin: 0;
  display: flex;
  align-items: center;
  gap: 8px;
}

.section-actions {
  display: flex;
  align-items: center;
  gap: 12px;
}

.course-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
  gap: 20px;
  padding: 24px;
}

.course-card {
  background: white;
  border-radius: 16px;
  box-shadow: 0 4px 15px rgba(0, 0, 0, 0.08);
  border: 1px solid rgba(0, 0, 0, 0.04);
  overflow: hidden;
  cursor: pointer;
  transition: all 0.3s ease;
  position: relative;
}

.course-card:hover {
  transform: translateY(-4px);
  box-shadow: 0 8px 30px rgba(0, 0, 0, 0.12);
}

.course-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 20px 20px 0 20px;
}

.course-cover {
  width: 48px;
  height: 48px;
  border-radius: 12px;
  display: flex;
  align-items: center;
  justify-content: center;
  color: white;
  font-weight: 600;
  font-size: 18px;
}

.course-content {
  padding: 16px 20px;
}

.course-title {
  font-size: 1.125rem;
  font-weight: 600;
  color: #1a1a1a;
  margin: 0 0 4px 0;
}

.course-description {
  font-size: 0.875rem;
  color: #666;
  margin: 0 0 16px 0;
}

.course-meta {
  display: flex;
  gap: 16px;
  margin-bottom: 16px;
}

.meta-item {
  display: flex;
  align-items: center;
  gap: 4px;
  font-size: 0.75rem;
  color: #666;
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
  color: #666;
}

.course-stats {
  display: flex;
  gap: 16px;
  margin-bottom: 16px;
}

.stat-item {
  display: flex;
  align-items: center;
  gap: 4px;
  font-size: 0.75rem;
  color: #666;
}

.course-footer {
  padding: 0 20px 20px 20px;
}

.create-card {
  display: flex;
  align-items: center;
  justify-content: center;
  min-height: 280px;
  background: linear-gradient(135deg, #f8fafc 0%, #e3f2fd 100%);
  border: 2px dashed #d0d7de;
  transition: all 0.3s ease;
}

.create-card:hover {
  border-color: #667eea;
  background: linear-gradient(135deg, #e3f2fd 0%, #f0f8ff 100%);
}

.create-content {
  text-align: center;
  color: #666;
}

.create-icon {
  font-size: 2rem;
  margin-bottom: 12px;
  color: #667eea;
}

.create-content h4 {
  font-size: 1rem;
  font-weight: 600;
  margin: 0 0 4px 0;
  color: #333;
}

.create-content p {
  font-size: 0.875rem;
  margin: 0;
  color: #666;
}

/* 作业列表样式 */
.assignment-list {
  padding: 24px;
}

.assignment-card {
  background: #f8fafc;
  border-radius: 12px;
  padding: 20px;
  margin-bottom: 16px;
  border-left: 4px solid #667eea;
  transition: all 0.3s ease;
  cursor: pointer;
}

.assignment-card:hover {
  transform: translateX(4px);
  box-shadow: 0 4px 15px rgba(102, 126, 234, 0.15);
}

.assignment-header {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
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
  background: #52c41a;
}

.priority-indicator.high {
  background: #ff4d4f;
}

.priority-indicator.medium {
  background: #faad14;
}

.priority-indicator.low {
  background: #52c41a;
}

.assignment-course {
  font-size: 0.75rem;
  color: #666;
  background: #f0f0f0;
  padding: 2px 8px;
  border-radius: 4px;
}

.assignment-deadline {
  display: flex;
  align-items: center;
  gap: 4px;
  font-size: 0.75rem;
  color: #666;
}

.assignment-deadline.urgent {
  color: #ff4d4f;
  font-weight: 500;
}

.assignment-title {
  font-size: 1rem;
  font-weight: 600;
  color: #1a1a1a;
  margin: 0 0 8px 0;
}

.assignment-description {
  font-size: 0.875rem;
  color: #666;
  margin: 0 0 16px 0;
  line-height: 1.4;
}

.assignment-progress {
  margin-bottom: 16px;
}

.progress-stats {
  display: flex;
  gap: 16px;
  margin-bottom: 8px;
}

.stat {
  display: flex;
  align-items: center;
  gap: 4px;
  font-size: 0.75rem;
  color: #666;
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
  color: #666;
}

.assignment-actions {
  display: flex;
  justify-content: flex-end;
}

/* AI教学助手样式 */
.ai-section {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
}

.ai-section .section-header {
  border-bottom-color: rgba(255, 255, 255, 0.2);
}

.ai-section .section-title {
  color: white;
}

.ai-recommendations {
  padding: 0 24px;
  display: grid;
  gap: 12px;
}

.ai-card {
  background: rgba(255, 255, 255, 0.1);
  border-radius: 12px;
  padding: 16px;
  cursor: pointer;
  transition: all 0.3s ease;
  border: 1px solid rgba(255, 255, 255, 0.2);
  display: flex;
  gap: 12px;
}

.ai-card:hover {
  background: rgba(255, 255, 255, 0.2);
  transform: translateY(-2px);
}

.ai-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 8px;
}

.ai-icon {
  font-size: 24px;
  margin-right: 8px;
}

.ai-title {
  font-size: 0.875rem;
  font-weight: 600;
}

.ai-content {
  flex: 1;
}

.ai-features-grid {
  display: grid;
  gap: 12px;
}

.ai-feature {
  display: flex;
  align-items: center;
  gap: 8px;
  cursor: pointer;
  transition: all 0.3s ease;
}

.ai-feature:hover {
  transform: translateY(-2px);
}

/* 数据洞察样式 */
.insights-widget {
  padding: 24px;
}

.insight-item {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 16px;
}

.insight-header {
  display: flex;
  flex-direction: column;
}

.insight-title {
  font-size: 1rem;
  font-weight: 600;
  color: #1a1a1a;
  margin-bottom: 4px;
}

.insight-trend {
  font-size: 0.75rem;
  color: #666;
}

.insight-value {
  display: flex;
  flex-direction: column;
}

.value {
  font-size: 1.5rem;
  font-weight: 700;
  color: #1a1a1a;
  line-height: 1;
  margin-bottom: 4px;
}

.unit {
  font-size: 0.875rem;
  color: #666;
}

.insight-rating {
  display: flex;
  align-items: center;
}

.rating-value {
  font-size: 1.25rem;
  font-weight: 700;
  color: #1a1a1a;
  margin-left: 8px;
}

/* 快速操作样式 */
.quick-actions-grid {
  display: grid;
  gap: 12px;
}

.quick-action {
  display: flex;
  align-items: center;
  gap: 8px;
  cursor: pointer;
  transition: all 0.3s ease;
}

.quick-action:hover {
  transform: translateY(-2px);
}

/* 日历样式 */
.calendar-widget {
  padding: 24px;
}

.calendar-cell {
  position: relative;
}

.event-indicators {
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  display: flex;
  gap: 4px;
}

.event-dot {
  width: 8px;
  height: 8px;
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
}

.event-list {
  margin-bottom: 24px;
}

.event-item {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 12px 0;
  border-bottom: 1px solid #f0f0f0;
}

.event-item:last-child {
  border-bottom: none;
}

.event-time {
  font-size: 12px;
  color: #999;
}

.event-content {
  flex: 1;
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.event-title {
  font-size: 14px;
  color: #333;
}

.event-desc {
  font-size: 12px;
  color: #666;
}

.no-events {
  text-align: center;
  padding: 40px 20px;
  color: #999;
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

@media (max-width: 1200px) {
  .content-grid {
    grid-template-columns: 1fr;
  }
  
  .course-grid {
    grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
  }
}

@media (max-width: 768px) {
  .header-content {
    flex-direction: column;
    text-align: center;
    gap: 24px;
  }
  
  .quick-actions {
    width: 100%;
    justify-content: center;
  }
  
  .dashboard-content {
    padding: 0 16px;
  }
  
  .course-grid {
    grid-template-columns: 1fr;
  }
}

@media (max-width: 480px) {
  .stat-card {
    min-width: 100px;
    padding: 16px;
  }
  
  .stat-number {
    font-size: 1.25rem;
  }
}
</style>