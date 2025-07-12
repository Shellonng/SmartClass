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
        <a-row :gutter="12" type="flex" justify="space-between">
          <!-- 左侧主要内容 -->
          <a-col :xs="24" :sm="24" :md="17" :lg="17" :xl="18" class="left-content">
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
                    <div class="course-cover" :style="getCourseCoverStyle(course)">
                      <div class="course-status">{{ getCourseStatusText(course.status) }}</div>
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
                    <h4 class="course-title">{{ course.title || course.courseName }}</h4>
                    <p class="course-description">{{ course.description || '暂无描述' }}</p>
                    
                    <div class="course-meta">
                      <div class="meta-item">
                        <UserOutlined />
                        <span>{{ course.studentCount || 0 }}名学生</span>
                      </div>
                      <div class="meta-item">
                        <PlayCircleOutlined />
                        <span>{{ course.chapterCount || 0 }}个章节</span>
                      </div>
                      <div class="meta-item">
                        <StarOutlined />
                        <span>{{ course.credit || course.credits || 3 }}学分</span>
                      </div>
                    </div>
                    
                    <div class="course-progress">
                      <div class="progress-header">
                        <span>教学进度</span>
                        <span>{{ course.progress || calculateProgress(course.startTime, course.endTime) }}%</span>
                      </div>
                      <a-progress 
                        :percent="course.progress || calculateProgress(course.startTime, course.endTime)" 
                        size="small" 
                        :show-info="false"
                        :stroke-color="getCourseStatusColor(course.status)"
                      />
                    </div>
                    
                    <div class="course-stats">
                      <div class="stat-item">
                        <span class="stat-value">{{ course.term || course.semester || '-' }}</span>
                        <span class="stat-label">学期</span>
                      </div>
                      <div class="stat-item">
                        <span class="stat-value">{{ formatSimpleDate(course.startTime) }}</span>
                        <span class="stat-label">开始</span>
                      </div>
                      <div class="stat-item">
                        <span class="stat-value">{{ formatSimpleDate(course.endTime) }}</span>
                        <span class="stat-label">结束</span>
                      </div>
                    </div>
                  </div>
                  
                  <div class="course-footer">
                    <a-space>
                      <a-button type="primary" size="small" @click="viewCourse(course)">
                        进入课程
                      </a-button>
                      <a-button size="small">
                        查看数据
                      </a-button>
                    </a-space>
                    <div class="course-type">{{ course.courseType || course.category || "必修课" }}</div>
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
                  <a-button type="link" @click="router.push('/teacher/assignments')">
                    查看全部 <ArrowRightOutlined />
                  </a-button>
                </div>
              </div>
              
              <div class="assignment-list">
                <div 
                  v-for="assignment in assignments" 
                  :key="assignment.id"
                  class="assignment-card"
                >
                  <div class="assignment-header">
                    <div class="assignment-priority">
                      <div class="priority-indicator" :class="getPriorityClass(assignment)"></div>
                      <span class="assignment-course">{{ assignment.courseName || '未知课程' }}</span>
                    </div>
                    <div class="assignment-deadline" :class="{ urgent: isUrgent(assignment.endTime) }">
                      <ClockCircleOutlined />
                      {{ formatDeadline(assignment.endTime) }}
                    </div>
                  </div>
                  
                  <h4 class="assignment-title">{{ assignment.title }}</h4>
                  <p class="assignment-description">{{ assignment.description || '无作业描述' }}</p>
                  
                  <div class="assignment-progress">
                    <div class="progress-stats">
                      <div class="stat">
                        <span class="number">{{ assignment.submittedCount || 0 }}</span>
                        <span class="label">已提交</span>
                      </div>
                      <div class="stat">
                        <span class="number">{{ assignment.totalStudents || 0 }}</span>
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
                        <span>{{ calculateSubmissionRate(assignment) }}%</span>
                      </div>
                      <a-progress 
                        :percent="calculateSubmissionRate(assignment)" 
                        size="small" 
                        :show-info="false"
                      />
                    </div>
                  </div>
                  
                  <div class="assignment-actions">
                    <a-space>
                      <a-button size="small" type="primary" @click="viewAssignment(assignment.id)">
                        查看提交
                      </a-button>
                      <a-button size="small" v-if="assignment.submittedCount > 0" @click="gradeAssignment(assignment.id)">
                        开始批改
                      </a-button>
                      <a-button size="small" type="text" @click="analyzeAssignment(assignment.id)">
                        统计分析
                      </a-button>
                    </a-space>
                  </div>
                </div>
                
                <div v-if="assignments.length === 0" class="empty-state">
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
          <a-col :xs="24" :sm="24" :md="7" :lg="7" :xl="6" class="right-content">
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
        title="新建课程"
        width="600px"
        @ok="handleCreateOrUpdateCourse"
        @cancel="resetCreateForm"
      >
        <a-form
          ref="createFormRef"
          :model="createForm"
          :rules="createRules"
          layout="vertical"
        >
          <a-form-item label="课程名称" name="courseName">
            <a-input 
              v-model:value="createForm.courseName" 
              placeholder="请输入课程名称" 
              @input="(value: string) => { createForm.title = value }"
            />
          </a-form-item>
          
          <a-form-item label="课程描述" name="description">
            <a-textarea 
              v-model:value="createForm.description" 
              placeholder="请输入课程描述"
              :rows="3"
            />
          </a-form-item>
          
          <a-form-item label="课程封面" name="coverImage">
            <div class="course-cover-upload">
              <a-upload
                v-model:file-list="coverFileList"
                list-type="picture-card"
                :show-upload-list="true"
                :before-upload="beforeCoverUpload"
                :customRequest="handleCoverUpload"
                :maxCount="1"
              >
                <div v-if="!createForm.coverImage">
                  <upload-outlined />
                  <div style="margin-top: 8px">上传封面</div>
                </div>
              </a-upload>
              <div class="cover-preview" v-if="createForm.coverImage">
                <img :src="createForm.coverImage" alt="课程封面预览" />
              </div>
            </div>
            <div class="upload-hint">建议上传16:9比例的图片，大小不超过2MB</div>
          </a-form-item>
          
          <a-row :gutter="16">
            <a-col :span="12">
              <a-form-item label="学分" name="credit">
                <a-input-number 
                  v-model:value="createForm.credit" 
                  :min="1" 
                  :max="10" 
                  style="width: 100%"
                  placeholder="学分"
                />
              </a-form-item>
            </a-col>
            <a-col :span="12">
              <a-form-item label="课程类型" name="category">
                <a-select v-model:value="createForm.category" placeholder="选择课程类型">
                  <a-select-option value="REQUIRED">必修课</a-select-option>
                  <a-select-option value="ELECTIVE">选修课</a-select-option>
                  <a-select-option value="PUBLIC">公共课</a-select-option>
                </a-select>
              </a-form-item>
            </a-col>
          </a-row>
          
          <a-row :gutter="16">
            <a-col :span="12">
              <a-form-item label="开始时间" name="startTime">
                <a-date-picker 
                  v-model:value="createForm.startTime" 
                  style="width: 100%"
                  placeholder="选择开始时间"
                />
              </a-form-item>
            </a-col>
            <a-col :span="12">
              <a-form-item label="结束时间" name="endTime">
                <a-date-picker 
                  v-model:value="createForm.endTime" 
                  style="width: 100%"
                  placeholder="选择结束时间"
                />
              </a-form-item>
            </a-col>
          </a-row>

          <a-row :gutter="16">
            <a-col :span="12">
              <a-form-item label="学年" name="year">
                <a-select
                  v-model:value="selectedYear"
                  placeholder="选择年份"
                  style="width: 100%"
                  @change="updateSemester"
                >
                  <a-select-option v-for="year in yearOptions" :key="year" :value="year">
                    {{ year }}
                  </a-select-option>
                </a-select>
              </a-form-item>
            </a-col>
            <a-col :span="12">
              <a-form-item label="季度" name="term">
                <a-select
                  v-model:value="selectedTerm"
                  placeholder="选择季度"
                  style="width: 100%"
                  @change="updateSemester"
                >
                  <a-select-option v-for="term in termOptions" :key="term.value" :value="term.value">
                    {{ term.label }}
                  </a-select-option>
                </a-select>
              </a-form-item>
            </a-col>
          </a-row>
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
import { ref, reactive, onMounted, computed, watch } from 'vue'
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
  SettingOutlined,
  UploadOutlined
} from '@ant-design/icons-vue'
import { useAuthStore } from '@/stores/auth'
import { getCourses, createCourse } from '@/api/teacher'
import { useRouter } from 'vue-router'
import axios from 'axios'

const authStore = useAuthStore()
const router = useRouter()

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
const mockCourses = ref<any[]>([])

// 加载课程列表
const loadCourses = async () => {
  try {
    const authStore = useAuthStore();
    
    // 确保已经有token
    let token = authStore.token;
    if (!token) {
      console.warn('加载课程列表: 未找到token，尝试从localStorage获取')
      token = localStorage.getItem('token') || localStorage.getItem('user-token')
      if (token && authStore.setToken) {
        authStore.setToken(token)
        console.log('用户已认证，但token未设置，已重新设置token')
      } else {
        console.error('用户未认证，无法加载课程列表')
        return
      }
    }
    
    const params = {
      page: 1,
      size: 7 // 修改为显示7个课程
    }
    
    console.log('开始请求课程列表API')
    const response = await axios.get('/api/teacher/courses', {
      params,
      headers: {
        'Authorization': `Bearer ${token}`
      }
    })
    console.log('课程列表API响应:', response.status)
    
    if (response.data && response.data.code === 200) {
      const result = response.data.data
      mockCourses.value = result.records || result.list || []
      
      // 更新统计数据
      stats.courseCount = mockCourses.value.length
      
      console.log('获取到的课程列表:', mockCourses.value.length, '条记录')
    } else {
      console.warn('获取课程列表返回异常:', response.data)
      // 不在初始加载时显示错误提示
      if (!loading.value) {
        message.error(response.data?.message || '获取课程列表失败')
      }
      mockCourses.value = []
    }
  } catch (error) {
    console.error('获取课程列表失败:', error)
    // 不在初始加载时显示错误提示
    if (!loading.value) {
      message.error('获取课程列表失败，请检查网络连接')
    }
    mockCourses.value = []
  }
}

// 模拟作业数据 - 替换为真实数据变量
const assignments = ref<any[]>([])

// 添加获取作业列表的方法
const loadAssignments = async () => {
  try {
    const authStore = useAuthStore();
    
    // 确保已经有token
    let token = authStore.token;
    if (!token) {
      console.warn('加载作业列表: 未找到token，尝试从localStorage获取')
      token = localStorage.getItem('token') || localStorage.getItem('user-token')
      if (token && authStore.setToken) {
        authStore.setToken(token)
        console.log('用户已认证，但token未设置，已重新设置token')
      } else {
        console.error('用户未认证，无法加载作业列表')
        return
      }
    }
    
    const params: Record<string, any> = {
      page: 1,
      size: 10,
      sort: 'createTime,desc'
    }
    
    // 根据筛选条件过滤
    if (assignmentFilter.value !== 'all') {
      if (assignmentFilter.value === 'pending') {
        params.status = '1' // 发布状态 - 进行中
      } else if (assignmentFilter.value === 'graded') {
        params.status = '2' // 已结束
      }
    }
    
    console.log('开始请求作业列表API')
    const response = await axios.get('/api/teacher/assignments', { 
      params,
      headers: {
        'Authorization': `Bearer ${token}`
      }
    })
    
    if (response && response.data && response.data.code === 200) {
      assignments.value = response.data.data.records || []
      
      // 更新统计数据
      if (assignments.value.length > 0) {
        stats.assignmentCount = response.data.data.total || assignments.value.length
        stats.pendingCount = assignments.value.filter(a => 
          (a.submittedCount || 0) > (a.gradedCount || 0)
        ).length
      }
      
      console.log('获取到的作业列表:', assignments.value.length, '条记录')
    } else {
      console.warn('获取作业列表返回异常:', response)
      // 不在初始加载时显示错误提示
      if (!loading.value) {
        message.error(response?.data?.message || '获取作业列表失败')
      }
    }
  } catch (error) {
    console.error('获取作业列表失败:', error)
    // 不再清除认证数据，只显示错误
    if (!loading.value) {
      message.error('获取作业列表失败，请检查网络连接')
    }
  }
}

// 计算提交率
const calculateSubmissionRate = (assignment: any) => {
  if (!assignment.totalStudents || assignment.totalStudents <= 0) return 0
  return Math.round(((assignment.submittedCount || 0) / assignment.totalStudents) * 100)
}

// 获取优先级样式类
const getPriorityClass = (assignment: any) => {
  // 根据截止时间判断优先级
  const now = new Date()
  const dueDate = new Date(assignment.endTime)
  const diffDays = Math.ceil((dueDate.getTime() - now.getTime()) / (1000 * 60 * 60 * 24))
  
  if (diffDays <= 1) return 'high' // 紧急
  if (diffDays <= 3) return 'medium' // 中等
  return 'low' // 普通
}

// 作业操作方法
const viewAssignment = (id: number) => {
  router.push(`/teacher/assignments/${id}`)
}

const gradeAssignment = (id: number) => {
  router.push(`/teacher/assignments/${id}/grade`)
}

const analyzeAssignment = (id: number) => {
  router.push(`/teacher/assignments/${id}/analysis`)
}

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

// 新增 - 创建课程相关数据和方法
// 封面上传相关
const uploadLoading = ref(false)
const coverFileList = ref<any[]>([])

// 创建表单
interface CreateForm {
  title: string;
  courseName: string;
  description: string;
  coverImage: string;
  credit: number | string;
  category: string;
  courseType: string;
  startTime: string | Dayjs;
  endTime: string | Dayjs;
  term: string;
  semester: string;
}

// 表单数据
const createForm = reactive({
  courseName: '',
  title: '',
  description: '',
  coverImage: '', // 添加课程封面属性
  credit: 3,
  category: 'REQUIRED',
  courseType: '必修课', // 添加课程类型属性
  startTime: null,
  endTime: null,
  semester: '',
  status: 0
})

// 表单验证规则
const createRules = {
  courseName: [
    { required: true, message: '请输入课程名称', trigger: 'blur' }
  ],
  category: [
    { required: true, message: '请选择课程类型', trigger: 'change' }
  ],
  semester: [
    { required: true, message: '请选择学期', trigger: 'change' }
  ]
}

// 学期相关数据
const yearOptions = ref([
  '2023-2024', '2024-2025', '2025-2026', '2026-2027', '2027-2028', '2028-2029'
])
const termOptions = ref([
  { label: '秋季', value: '1' },
  { label: '春季', value: '2' },
  { label: '夏季', value: '3' }
])
const selectedYear = ref('2024-2025') // 设置默认值
const selectedTerm = ref('1') // 设置默认值

// 监听学期选择变化
const updateSemester = () => {
  if (selectedYear.value && selectedTerm.value) {
    createForm.semester = `${selectedYear.value}-${selectedTerm.value}`
    console.log('学期已更新:', createForm.semester)
  } else {
    createForm.semester = ''
  }
}

// 重置创建表单
const resetCreateForm = () => {
  createForm.courseName = ''
  createForm.title = ''
  createForm.description = ''
  createForm.coverImage = ''
  createForm.credit = 3
  createForm.category = 'REQUIRED'
  createForm.startTime = null
  createForm.endTime = null
  selectedYear.value = '2024-2025'
  selectedTerm.value = '1'
  coverFileList.value = []
}

// 上传前检查文件
const beforeCoverUpload = (file: File) => {
  // 检查文件类型
  const isImage = file.type.startsWith('image/')
  if (!isImage) {
    message.error('只能上传图片文件!')
    return false
  }
  
  // 检查文件大小 (2MB)
  const isLt2M = file.size / 1024 / 1024 < 2
  if (!isLt2M) {
    message.error('图片大小不能超过2MB!')
    return false
  }
  
  return true
}

// 自定义上传方法
const handleCoverUpload = async (options: any) => {
  const { file, onSuccess, onError } = options
  uploadLoading.value = true
  
  try {
    console.log('开始上传课程封面图片:', file.name)
    
    // 创建FormData对象
    const formData = new FormData()
    formData.append('file', file)
    
    // 发送上传请求
    const response = await axios.post('http://localhost:8080/api/common/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
        'Authorization': `Bearer ${localStorage.getItem('token')}`
      }
    })
    
    console.log('封面上传响应:', response)
    
    if (response.data && response.data.code === 200) {
      // 获取上传后的文件URL
      const fileUrl = response.data.data
      console.log('文件上传成功，原始URL:', fileUrl)
      
      // 确保URL是完整的
      let fullUrl = fileUrl
      if (!fileUrl.startsWith('http')) {
        if (fileUrl.startsWith('/')) {
          fullUrl = `http://localhost:8080${fileUrl}`
        } else {
          fullUrl = `http://localhost:8080/${fileUrl}`
        }
      }
      
      console.log('处理后的完整URL:', fullUrl)
      
      // 设置课程封面
      createForm.coverImage = fullUrl
      
      message.success('封面上传成功')
      onSuccess(response)
    } else {
      console.error('封面上传失败:', response)
      message.error('封面上传失败')
      onError(new Error('Upload failed'))
    }
  } catch (error) {
    console.error('封面上传出错:', error)
    message.error('封面上传出错')
    onError(new Error('Upload error'))
  } finally {
    uploadLoading.value = false
  }
}

// 根据category获取courseType
const getCourseTypeFromCategory = (category?: string) => {
  if (!category) return '必修课'
  
  const categoryMap: Record<string, string> = {
    'REQUIRED': '必修课',
    'ELECTIVE': '选修课',
    'PUBLIC': '公共课'
  }
  
  return categoryMap[category] || '必修课'
}

// 创建或更新课程
const handleCreateOrUpdateCourse = async () => {
  try {
    loading.value = true;
    
    // 表单验证
    if (!createForm.courseName || createForm.courseName.trim() === '') {
      message.error('请输入课程名称');
      loading.value = false;
      return;
    }
    
    // 确保学分是数字类型
    let creditValue = createForm.credit;
    if (typeof creditValue === 'string') {
      creditValue = parseFloat(creditValue);
    }
    
    // 格式化开始和结束时间
    const startTime = createForm.startTime 
      ? (typeof createForm.startTime === 'string' 
        ? createForm.startTime 
        : dayjs(createForm.startTime).format('YYYY-MM-DD HH:mm:ss')) 
      : undefined;
    
    const endTime = createForm.endTime 
      ? (typeof createForm.endTime === 'string' 
        ? createForm.endTime 
        : dayjs(createForm.endTime).format('YYYY-MM-DD HH:mm:ss')) 
      : undefined;
    
    // 如果没有选择开始时间，使用当前时间
    const formattedStartTime = startTime || dayjs().format('YYYY-MM-DD HH:mm:ss');
    // 如果没有选择结束时间，使用开始时间后3个月
    const formattedEndTime = endTime || dayjs(formattedStartTime).add(3, 'month').format('YYYY-MM-DD HH:mm:ss');
    
    // 确保学期格式正确
    let term = createForm.semester || '2024-2025-1';
    if (selectedYear.value && selectedTerm.value) {
      term = `${selectedYear.value}-${selectedTerm.value}`;
    }
    
    const formData = {
      title: createForm.courseName, // 使用courseName作为title
      courseName: createForm.courseName,
      description: createForm.description || '',
      coverImage: createForm.coverImage || '',
      credit: creditValue,
      category: createForm.category || 'REQUIRED',
      courseType: createForm.courseType || getCourseTypeFromCategory(createForm.category),
      startTime: formattedStartTime,
      endTime: formattedEndTime,
      term: term,
      semester: term,
      status: '未开始'
    };
    
    console.log('提交的课程数据:', JSON.stringify(formData));
    const response = await createCourse(formData);
    
    if (response.data.code === 200) {
      message.success('课程创建成功');
      showCreateCourseModal.value = false;
      resetCreateForm();
      await loadCourses(); // 重新加载课程列表
    } else {
      message.error(response.data.message || '课程创建失败');
    }
  } catch (error: any) {
    console.error('创建课程失败:', error);
    
    // 处理不同类型的错误
    if (error.response) {
      // 服务器返回了错误响应
      if (error.response.status === 401) {
        message.error('请先登录后再操作课程');
        setTimeout(() => {
          router.push('/login');
        }, 1500);
      } else if (error.response.data && error.response.data.message) {
        message.error(error.response.data.message);
      } else {
        message.error(`操作失败 (${error.response.status})`);
      }
    } else if (error.request) {
      // 请求已经发出，但没有收到响应
      message.error('服务器无响应，请检查网络连接');
    } else {
      // 请求设置时发生错误
      message.error('请求错误: ' + error.message);
    }
  } finally {
    loading.value = false;
  }
}

// 查看课程详情
const viewCourse = (course: any) => {
  if (course && course.id) {
    router.push(`/teacher/courses/${course.id}`)
  } else {
    message.error('无效的课程数据')
  }
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
onMounted(async () => {
  // 加载数据
  loading.value = true
  
  // 确保认证信息已加载
  const initializeData = async () => {
    try {
      // 确保token已经设置
      const authStore = useAuthStore();
      if (!authStore.token) {
        console.log('Dashboard初始化: 尝试恢复token')
        const savedToken = localStorage.getItem('token')
        if (savedToken) {
          authStore.setToken(savedToken)
        }
      }
      
      // 等待一小段时间确保token已经设置到axios头部
      await new Promise(resolve => setTimeout(resolve, 100))
      
      // 先加载课程数据
      await loadCourses()
      
      // 再加载作业列表
      await loadAssignments()
    } catch (error) {
      console.error('初始化数据失败:', error)
    } finally {
  setTimeout(() => {
    loading.value = false
      }, 500)
    }
  }
  
  initializeData()
})

// 监听作业筛选条件变化
watch(assignmentFilter, () => {
  loadAssignments()
})

// 获取课程封面样式
const getCourseCoverStyle = (course: any) => {
  console.log('处理课程封面:', course.id, course.title, '封面图片:', course.coverImage)
  
  // 判断课程是否有有效的封面图片
  if (course.coverImage && course.coverImage.trim() && !course.coverImage.includes('undefined')) {
    // 检查URL是否已经是完整URL
    let imageUrl = course.coverImage
    
    // 确保URL是完整的
    if (!imageUrl.startsWith('http')) {
      // 如果以/开头，直接添加基础URL
      if (imageUrl.startsWith('/')) {
        imageUrl = `http://localhost:8080${imageUrl}`
      } else {
        // 否则添加斜杠再添加基础URL
        imageUrl = `http://localhost:8080/${imageUrl}`
      }
    }
    
    console.log('处理后的课程封面URL:', imageUrl)
    
    return {
      background: `linear-gradient(rgba(0, 0, 0, 0.3), rgba(0, 0, 0, 0.3)), url("${imageUrl}") no-repeat center center / cover`
    }
  }
  
  // 如果没有封面图片，使用渐变色背景
  // 创建多种五彩缤纷的渐变色
  const gradients = [
    'linear-gradient(135deg, #ff9a9e 0%, #fad0c4 100%)',
    'linear-gradient(135deg, #a18cd1 0%, #fbc2eb 100%)',
    'linear-gradient(135deg, #fad0c4 0%, #ffd1ff 100%)',
    'linear-gradient(135deg, #fbc2eb 0%, #a6c1ee 100%)',
    'linear-gradient(135deg, #fdcbf1 0%, #fdcbf1 1%, #e6dee9 100%)',
    'linear-gradient(135deg, #a1c4fd 0%, #c2e9fb 100%)',
    'linear-gradient(135deg, #d4fc79 0%, #96e6a1 100%)',
    'linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%)',
    'linear-gradient(135deg, #f6d365 0%, #fda085 100%)'
  ]
  
  // 使用课程ID或名称的哈希值来选择渐变色
  const courseId = course.id || 0
  const courseName = course.title || course.courseName || ''
  const hash = Math.abs(courseId + courseName.length)
  const index = hash % gradients.length
  
  console.log('使用渐变色背景:', gradients[index])
  
  return {
    background: gradients[index]
  }
}

// 简单的字符串哈希函数，用于将课程名称转换为数字
const hashCode = (str: string): number => {
  let hash = 0
  for (let i = 0; i < str.length; i++) {
    hash = ((hash << 5) - hash) + str.charCodeAt(i)
    hash |= 0 // Convert to 32bit integer
  }
  return Math.abs(hash)
}

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

// 简化的日期格式化
const formatSimpleDate = (dateStr?: string) => {
  if (!dateStr) return '-'
  return dayjs(dateStr).format('MM-DD')
}

const isUrgent = (dateStr: string) => {
  const date = dayjs(dateStr)
  const now = dayjs()
  return date.diff(now, 'day') <= 2
}

// 计算课程进度
const calculateProgress = (startTime?: string, endTime?: string): number => {
  if (!startTime || !endTime) return 0
  
  const start = dayjs(startTime)
  const end = dayjs(endTime)
  const now = dayjs()
  
  // 如果未开始，进度为0
  if (now.isBefore(start)) return 0
  // 如果已结束，进度为100
  if (now.isAfter(end)) return 100
  
  // 计算总时长和已过时长
  const totalDuration = end.diff(start, 'day')
  const passedDuration = now.diff(start, 'day')
  
  // 计算百分比，限制在0-100之间
  return Math.min(100, Math.max(0, Math.round((passedDuration / totalDuration) * 100)))
}

// 根据课程状态获取文本描述
const getCourseStatusText = (status?: string | number) => {
  if (!status) return '未开始'
  
  // 处理数字状态（0: 未开始, 1: 进行中, 2: 已结束）
  if (typeof status === 'number') {
    if (status === 0) return '未开始'
    if (status === 1) return '进行中'
    if (status === 2) return '已结束'
    return '未知'
  }
  
  // 处理字符串状态
  const statusMap: Record<string, string> = {
    '0': '未开始',
    '1': '进行中',
    '2': '已结束',
    'not_started': '未开始',
    'in_progress': '进行中',
    'ended': '已结束',
    '未开始': '未开始',
    '进行中': '进行中',
    '已结束': '已结束'
  }
  
  return statusMap[status] || '未开始'
}

// 根据课程状态获取颜色
const getCourseStatusColor = (status?: string | number) => {
  const statusText = getCourseStatusText(status)
  
  const colors: Record<string, string> = {
    '未开始': '#2F80ED',
    '进行中': '#764ba2',
    '已结束': '#f9748f'
  }
  
  return colors[statusText] || '#764ba2'
}

// 课程封面上传
const handleUploadCover = async (info: any) => {
  if (info.file.status === 'uploading') {
    uploadLoading.value = true
    return
  }
  
  if (info.file.status === 'done') {
    uploadLoading.value = false
    
    // 检查响应数据
    if (info.file.response && info.file.response.code === 200) {
      // 获取上传后的文件URL
      const fileUrl = info.file.response.data
      console.log('文件上传成功，URL:', fileUrl)
      
      // 确保URL是完整的
      createForm.coverImage = fileUrl.startsWith('http') 
        ? fileUrl 
        : `http://localhost:8080${fileUrl}`
        
      console.log('设置课程封面:', createForm.coverImage)
      message.success('封面上传成功')
    } else {
      console.error('文件上传失败:', info.file.response)
      message.error('封面上传失败')
    }
  } else if (info.file.status === 'error') {
    uploadLoading.value = false
    console.error('文件上传错误:', info.file)
    message.error('封面上传失败')
  }
}
</script>

<style scoped>
/* 课程封面上传样式 */
.course-cover-upload {
  display: flex;
  align-items: center;
  gap: 16px;
}

.upload-hint {
  font-size: 12px;
  color: #8c8c8c;
  margin-top: 4px;
}

.cover-preview {
  width: 160px;
  height: 90px;
  border-radius: 8px;
  overflow: hidden;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
}

.cover-preview img {
  width: 100%;
  height: 100%;
  object-fit: cover;
}

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
  width: 100%;
  max-width: 100%;
  margin: 0 auto;
  padding: 0;
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
  padding: 24px 24px 0 24px;
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 20px;
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
  grid-template-columns: repeat(auto-fill, minmax(240px, 1fr));
  gap: 12px;
  padding: 0 20px 20px 20px;
}

.course-card {
  background: white;
  border-radius: 14px;
  border: none;
  overflow: hidden;
  transition: all 0.3s ease;
  cursor: pointer;
  position: relative;
  box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
  display: flex;
  flex-direction: column;
  height: 100%;
}

.course-card:hover {
  transform: translateY(-5px);
  box-shadow: 0 12px 30px rgba(0, 0, 0, 0.15);
  border-color: transparent;
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
  padding: 14px;
  color: white;
  position: relative;
  transition: all 0.3s ease;
}

.course-card:hover .course-cover {
  transform: scale(1.05);
}

.course-cover::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: rgba(0, 0, 0, 0.2);
  z-index: 0;
}

.course-status {
  font-size: 11px;
  font-weight: 600;
  background: rgba(255, 255, 255, 0.3);
  backdrop-filter: blur(5px);
  padding: 3px 8px;
  border-radius: 4px;
  z-index: 1;
  position: relative;
  display: inline-block;
  text-shadow: 0 1px 2px rgba(0, 0, 0, 0.3);
  align-self: flex-start;
}

.course-type {
  font-size: 11px;
  font-weight: 500;
  background: rgba(102, 126, 234, 0.1);
  padding: 3px 8px;
  border-radius: 4px;
  color: #667eea;
  margin-left: auto;
}

.course-actions {
  position: absolute;
  top: 10px;
  right: 10px;
  z-index: 2;
}

.course-content {
  padding: 16px;
}

.course-title {
  font-size: 16px;
  font-weight: 600;
  color: #1a202c;
  margin: 0 0 6px 0;
  line-height: 1.3;
  overflow: hidden;
  text-overflow: ellipsis;
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
}

.course-description {
  font-size: 13px;
  color: #64748b;
  margin: 0 0 12px 0;
  line-height: 1.4;
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
  overflow: hidden;
}

.course-meta {
  display: flex;
  gap: 12px;
  margin-bottom: 12px;
  flex-wrap: wrap;
}

.meta-item {
  display: flex;
  align-items: center;
  gap: 3px;
  font-size: 12px;
  color: #64748b;
}

.course-progress {
  margin-bottom: 12px;
}

.progress-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 4px;
  font-size: 12px;
  color: #64748b;
  font-weight: 500;
}

.course-stats {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 10px;
  margin-bottom: 16px;
  padding: 10px;
  background: #f8fafc;
  border-radius: 10px;
}

.course-stats .stat-item {
  text-align: center;
}

.stat-value {
  display: block;
  font-size: 13px;
  font-weight: 600;
  color: #1a202c;
  line-height: 1;
}

.stat-label {
  display: block;
  font-size: 11px;
  color: #64748b;
}

.course-footer {
  padding: 0 16px 16px 16px;
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-top: auto;
}

.create-card {
  background: #f8f9fa;
  border: 2px dashed #d0d7de;
  display: flex;
  align-items: center;
  justify-content: center;
  min-height: 280px;
  transition: all 0.3s ease;
}

.create-card:hover {
  background: #f0f2f5;
  border-color: #a0a8b0;
  transform: translateY(-5px);
  box-shadow: 0 8px 20px rgba(0, 0, 0, 0.05);
}

.create-content {
  text-align: center;
  color: #6c757d;
}

.create-icon {
  font-size: 2.5rem;
  margin-bottom: 16px;
  color: #6c757d;
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
  padding: 20px;
}

/* 调整ant-design日历组件的样式 */
:deep(.calendar-widget .ant-picker-calendar) {
  font-size: 0.875rem;
}

:deep(.calendar-widget .ant-picker-calendar-header) {
  padding: 8px 0;
}

:deep(.calendar-widget .ant-picker-panel) {
  border-top: none;
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
  margin-top: 16px;
  padding-top: 16px;
  border-top: 1px solid #e2e8f0;
}

.calendar-events h4 {
  font-size: 0.95rem;
  font-weight: 600;
  color: #1a202c;
  margin: 0 0 12px 0;
}

.event-list {
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.event-item {
  display: flex;
  gap: 10px;
  padding: 10px;
  background: #f8fafc;
  border-radius: 6px;
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

/* 课程封面上传样式 */
.course-cover-upload {
  display: flex;
  align-items: center;
  gap: 16px;
}

.upload-hint {
  font-size: 12px;
  color: #8c8c8c;
  margin-top: 4px;
}

.cover-preview {
  width: 160px;
  height: 90px;
  border-radius: 8px;
  overflow: hidden;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
}

.cover-preview img {
  width: 100%;
  height: 100%;
  object-fit: cover;
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

/* 修正Ant Design Row的默认样式 */
:deep(.dashboard-content .ant-row) {
  margin-right: 0 !important;
  margin-left: 0 !important;
  width: 100%;
}

/* 左右内容区域定位 */
.left-content {
  padding-left: 0 !important;
  padding-right: 6px !important;
}

.right-content {
  padding-right: 0 !important;
  padding-left: 6px !important;
}

@media (max-width: 768px) {
  .left-content,
  .right-content {
    padding-left: 0 !important;
    padding-right: 0 !important;
    margin-bottom: 24px;
  }
}
</style>