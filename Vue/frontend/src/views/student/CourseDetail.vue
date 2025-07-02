<template>
  <div class="fixed-width-container course-detail">
    <a-spin :spinning="loading">
      <div class="content-wrapper">
        <!-- 页面头部 -->
        <div class="page-header">
          <div class="header-left">
            <a-button type="text" @click="goBack" class="back-btn">
              <ArrowLeftOutlined />
              返回课程列表
            </a-button>
          </div>
          
          <div class="header-right">
            <a-button v-if="course?.status !== '已结束'" type="primary" @click="startLearning">
              <PlayCircleOutlined />
              开始学习
            </a-button>
            <a-button @click="toggleFavorite" :type="isFavorite ? 'default' : 'dashed'">
              <HeartFilled v-if="isFavorite" style="color: #ff4d4f" />
              <HeartOutlined v-else />
              {{ isFavorite ? '取消收藏' : '收藏课程' }}
            </a-button>
          </div>
        </div>

        <!-- 课程信息 -->
        <div class="course-info-section">
          <div class="course-header">
            <div class="course-cover">
              <img v-if="course?.coverImage" :src="course.coverImage" :alt="course.title" />
              <div v-else class="cover-placeholder">
                <BookOutlined style="font-size: 64px" />
              </div>
              <div class="course-status" :class="getStatusClass(course?.status)">
                {{ getStatusText(course?.status) }}
              </div>
            </div>
            
            <div class="course-info">
              <h1 class="course-title">{{ course?.title || '加载中...' }}</h1>
              
              <div class="course-meta">
                <div class="meta-item">
                  <ClockCircleOutlined />
                  <span>{{ course?.credit || 0 }} 学分</span>
                </div>
                <div class="meta-item">
                  <CalendarOutlined />
                  <span>{{ course?.term || '未知学期' }}</span>
                </div>
                <div class="meta-item">
                  <TeamOutlined />
                  <span>{{ course?.studentCount || 0 }} 名学生</span>
                </div>
                <div class="meta-item">
                  <UserOutlined />
                  <span>{{ teacherName || '未知教师' }}</span>
                </div>
              </div>
              
              <div class="course-description">
                <p>{{ course?.description || '暂无课程描述' }}</p>
              </div>
              
              <div class="course-progress">
                <div class="progress-header">
                  <span>学习进度</span>
                </div>
                <a-progress :percent="progress" :stroke-color="getProgressColor(progress)" />
                <div class="progress-info">
                  已学习 {{ completedSections }}/{{ totalSections }} 小节
                </div>
              </div>
            </div>
          </div>
        </div>

        <!-- 课程内容 -->
        <div class="course-content-section">
          <a-tabs v-model:activeKey="activeTab">
            <a-tab-pane key="chapters" tab="课程章节">
              <div class="chapter-list">
                <!-- 有章节数据时显示 -->
                <div v-if="chapters.length > 0">
                  <div v-for="(chapter, index) in chapters" :key="chapter.id || index" class="chapter-item">
                    <div class="chapter-info">
                      <div class="chapter-number">{{ index + 1 }}</div>
                      <div class="chapter-content">
                        <h3 class="chapter-title">{{ chapter.title }}</h3>
                        <p class="chapter-description" v-if="chapter.description">{{ chapter.description }}</p>
                        
                        <div class="chapter-sections">
                          <div v-if="chapter.sections && chapter.sections.length > 0">
                            <div v-for="(section, sectionIndex) in chapter.sections" :key="section.id || sectionIndex" class="section-item">
                              <div class="section-icon">
                                <FileTextOutlined v-if="section.type === 'text'" />
                                <PlayCircleOutlined v-else-if="section.type === 'video'" />
                                <FileOutlined v-else />
                              </div>
                              <div class="section-content" @click="viewSection(chapter, section)">
                                <div class="section-title">{{ section.title }}</div>
                                <div class="section-meta">
                                  <span v-if="section.duration" class="section-duration">{{ section.duration }}分钟</span>
                                  <a-tag v-if="section.completed" color="success">已完成</a-tag>
                                  <a-tag v-else color="processing">未学习</a-tag>
                                </div>
                              </div>
                            </div>
                          </div>
                          <div v-else class="empty-sections">
                            <a-empty description="暂无小节" :image="Empty.PRESENTED_IMAGE_SIMPLE" />
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
                
                <!-- 无章节数据时显示 -->
                <div v-else class="empty-chapters">
                  <a-empty description="暂无章节数据">
                    <template #description>
                      <span>该课程暂无章节内容</span>
                    </template>
                  </a-empty>
                </div>
              </div>
            </a-tab-pane>
            
            <a-tab-pane key="resources" tab="课程资源">
              <div class="resources-list">
                <a-empty v-if="resources.length === 0" description="暂无课程资源" />
                <a-list v-else :data-source="resources" :grid="{ gutter: 16, column: 3 }">
                  <template #renderItem="{ item }">
                    <a-list-item>
                      <a-card hoverable @click="downloadResource(item)">
                        <template #cover>
                          <div class="resource-icon">
                            <FileExcelOutlined v-if="item.fileType === 'xlsx' || item.fileType === 'xls'" style="color: #52c41a" />
                            <FilePdfOutlined v-else-if="item.fileType === 'pdf'" style="color: #f5222d" />
                            <FileWordOutlined v-else-if="item.fileType === 'doc' || item.fileType === 'docx'" style="color: #1890ff" />
                            <FilePptOutlined v-else-if="item.fileType === 'ppt' || item.fileType === 'pptx'" style="color: #fa8c16" />
                            <FileImageOutlined v-else-if="['jpg', 'jpeg', 'png', 'gif'].includes(item.fileType)" style="color: #722ed1" />
                            <FileOutlined v-else />
                          </div>
                        </template>
                        <a-card-meta :title="item.name">
                          <template #description>
                            <div class="resource-meta">
                              <div>{{ item.formattedSize || formatFileSize(item.fileSize) }}</div>
                              <div>下载次数: {{ item.downloadCount || 0 }}</div>
                            </div>
                          </template>
                        </a-card-meta>
                      </a-card>
                    </a-list-item>
                  </template>
                </a-list>
              </div>
            </a-tab-pane>
            
            <a-tab-pane key="discussions" tab="讨论区">
              <div class="discussions-section">
                <a-empty description="讨论区功能开发中" />
              </div>
            </a-tab-pane>

            <a-tab-pane key="tasks" tab="任务">
              <div class="tasks-section">
                <div class="task-list">
                  <a-empty v-if="tasks.length === 0" description="暂无课程任务" />
                  <div v-else>
                    <a-list :data-source="tasks" item-layout="horizontal">
                      <template #renderItem="{ item }">
                        <a-list-item>
                          <a-list-item-meta>
                            <template #avatar>
                              <a-avatar :style="{ backgroundColor: getTaskStatusColor(item.status) }">
                                <template #icon><CheckSquareOutlined /></template>
                              </a-avatar>
                            </template>
                            <template #title>
                              <div class="task-title">
                                <span>{{ item.title }}</span>
                                <a-tag :color="getTaskStatusColor(item.status)">{{ getTaskStatusText(item.status) }}</a-tag>
                              </div>
                            </template>
                            <template #description>
                              <div class="task-description">
                                <div>{{ item.description || '暂无描述' }}</div>
                                <div class="task-meta">
                                  <span>截止日期: {{ formatDate(item.endTime) }}</span>
                                  <span>类型: {{ item.type === 'homework' ? '作业' : '考试' }}</span>
                                  <span>模式: {{ item.mode === 'question' ? '答题型' : '文件上传型' }}</span>
                                </div>
                              </div>
                            </template>
                          </a-list-item-meta>
                          <template #actions>
                            <a key="view" @click="viewTask(item)">查看详情</a>
                          </template>
                        </a-list-item>
                      </template>
                    </a-list>
                  </div>
                </div>
              </div>
            </a-tab-pane>

            <a-tab-pane key="wrong-questions" tab="错题集">
              <div class="wrong-questions-section">
                <div class="filter-section">
                  <a-space>
                    <a-select v-model:value="questionFilter.type" placeholder="题型筛选" style="width: 120px">
                      <a-select-option value="all">全部题型</a-select-option>
                      <a-select-option value="single">单选题</a-select-option>
                      <a-select-option value="multiple">多选题</a-select-option>
                      <a-select-option value="judge">判断题</a-select-option>
                      <a-select-option value="fill">填空题</a-select-option>
                    </a-select>
                    <a-select v-model:value="questionFilter.difficulty" placeholder="难度筛选" style="width: 120px">
                      <a-select-option value="all">全部难度</a-select-option>
                      <a-select-option value="easy">简单</a-select-option>
                      <a-select-option value="medium">中等</a-select-option>
                      <a-select-option value="hard">困难</a-select-option>
                    </a-select>
                    <a-input-search placeholder="搜索题目关键词" style="width: 200px" />
                  </a-space>
                </div>
                
                <div class="wrong-questions-list">
                  <a-empty v-if="wrongQuestions.length === 0" description="暂无错题记录" />
                  <div v-else>
                    <div v-for="(question, index) in wrongQuestions" :key="question.id || index" class="question-card">
                      <div class="question-header">
                        <div class="question-type">
                          <a-tag :color="getQuestionTypeColor(question.type)">{{ getQuestionTypeText(question.type) }}</a-tag>
                        </div>
                        <div class="question-meta">
                          <a-tag color="orange" v-if="question.difficulty === 'hard'">困难</a-tag>
                          <a-tag color="blue" v-else-if="question.difficulty === 'medium'">中等</a-tag>
                          <a-tag color="green" v-else>简单</a-tag>
                          <span class="wrong-count">错误次数: {{ question.wrongCount }}</span>
                        </div>
                      </div>
                      
                      <div class="question-content">
                        <div class="question-title">{{ index + 1 }}. {{ question.title }}</div>
                        
                        <div class="question-options" v-if="['single', 'multiple'].includes(question.type)">
                          <div v-for="option in question.options" :key="option.key" class="option-item">
                            <span :class="{ 'option-correct': option.key === question.correctAnswer, 'option-wrong': question.userAnswer.includes(option.key) && option.key !== question.correctAnswer }">
                              {{ option.key }}. {{ option.content }}
                            </span>
                          </div>
                        </div>
                        
                        <div class="question-answer">
                          <div class="answer-label">你的答案:</div>
                          <div class="user-answer wrong">{{ question.userAnswer.join(', ') }}</div>
                          <div class="answer-label">正确答案:</div>
                          <div class="correct-answer">{{ Array.isArray(question.correctAnswer) ? question.correctAnswer.join(', ') : question.correctAnswer }}</div>
                        </div>
                        
                        <div class="question-analysis" v-if="question.analysis">
                          <div class="analysis-label">解析:</div>
                          <div class="analysis-content">{{ question.analysis }}</div>
                        </div>
                      </div>
                      
                      <div class="question-actions">
                        <a-button type="primary" size="small" @click="practiceQuestion(question)">
                          <FormOutlined />
                          练习
                        </a-button>
                        <a-button size="small" @click="markAsResolved(question)">
                          <CheckOutlined />
                          标记已掌握
                        </a-button>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </a-tab-pane>

            <a-tab-pane key="learning-records" tab="学习记录">
              <div class="learning-records-section">
                <div class="statistics-cards">
                  <a-row :gutter="16">
                    <a-col :span="8">
                      <a-card>
                        <template #title>
                          <span><ClockCircleOutlined /> 总学习时长</span>
                        </template>
                        <div class="statistic-value">{{ totalLearningHours }} 小时</div>
                        <div class="statistic-description">本周学习 {{ weeklyLearningHours }} 小时</div>
                      </a-card>
                    </a-col>
                    <a-col :span="8">
                      <a-card>
                        <template #title>
                          <span><BookOutlined /> 课程完成率</span>
                        </template>
                        <div class="statistic-value">{{ courseCompletionRate }}%</div>
                        <a-progress :percent="courseCompletionRate" :stroke-color="getProgressColor(courseCompletionRate)" />
                      </a-card>
                    </a-col>
                    <a-col :span="8">
                      <a-card>
                        <template #title>
                          <span><TrophyOutlined /> 学习成就</span>
                        </template>
                        <div class="statistic-value">{{ achievementPoints }} 分</div>
                        <div class="statistic-description">班级排名: 第 {{ classRank }} 名</div>
                      </a-card>
                    </a-col>
                  </a-row>
                </div>
                
                <div class="learning-chart">
                  <div class="chart-header">
                    <h3>学习趋势</h3>
                    <a-radio-group v-model:value="chartTimeRange" button-style="solid" size="small">
                      <a-radio-button value="week">本周</a-radio-button>
                      <a-radio-button value="month">本月</a-radio-button>
                      <a-radio-button value="semester">本学期</a-radio-button>
                    </a-radio-group>
                  </div>
                  <div class="chart-placeholder">
                    <a-empty description="图表加载中" />
                  </div>
                </div>
                
                <div class="learning-activities">
                  <div class="activities-header">
                    <h3>学习活动记录</h3>
                  </div>
                  <a-timeline>
                    <a-timeline-item v-for="(activity, index) in learningActivities" :key="index" :color="getActivityColor(activity.type)">
                      <div class="activity-item">
                        <div class="activity-title">{{ activity.title }}</div>
                        <div class="activity-time">{{ activity.time }}</div>
                        <div class="activity-description">{{ activity.description }}</div>
                      </div>
                    </a-timeline-item>
                  </a-timeline>
                </div>
              </div>
            </a-tab-pane>

            <a-tab-pane key="knowledge-graph" tab="知识图谱">
              <div class="knowledge-graph-section">
                <div class="graph-controls">
                  <a-space>
                    <a-select v-model:value="graphViewMode" style="width: 150px">
                      <a-select-option value="tree">树形视图</a-select-option>
                      <a-select-option value="force">力导向图</a-select-option>
                      <a-select-option value="cluster">聚类视图</a-select-option>
                    </a-select>
                    <a-button><ZoomInOutlined /> 放大</a-button>
                    <a-button><ZoomOutOutlined /> 缩小</a-button>
                    <a-button><FullscreenOutlined /> 全屏</a-button>
                  </a-space>
                </div>
                
                <div class="graph-container">
                  <a-empty description="知识图谱加载中" />
                </div>
                
                <div class="knowledge-details">
                  <h3>知识点详情</h3>
                  <p class="empty-tip">点击图谱中的节点查看详情</p>
                </div>
              </div>
            </a-tab-pane>
          </a-tabs>
        </div>
      </div>
    </a-spin>
  </div>
</template>

<script setup lang="ts">
import { ref, reactive, onMounted } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { message, Empty } from 'ant-design-vue'
import { 
  ArrowLeftOutlined, 
  BookOutlined,
  PlayCircleOutlined,
  HeartOutlined,
  HeartFilled,
  ClockCircleOutlined,
  CalendarOutlined,
  TeamOutlined,
  UserOutlined,
  FileTextOutlined,
  FileOutlined,
  FileExcelOutlined,
  FilePdfOutlined,
  FileWordOutlined,
  FilePptOutlined,
  FileImageOutlined,
  CheckSquareOutlined,
  FormOutlined,
  CheckOutlined,
  ZoomInOutlined,
  ZoomOutOutlined,
  FullscreenOutlined,
  TrophyOutlined
} from '@ant-design/icons-vue'
import dayjs from 'dayjs'
import { 
  getCourseDetail, 
  getCourseChapters, 
  getChapterDetail, 
  getCourseResources,
  getCourseInstructor,
  getSectionsByChapterId,
  getStudentCourseDetail,
  getStudentCourseResources,
  getStudentCourseTasks
} from '@/api/course'
import axios from 'axios'

const route = useRoute()
const router = useRouter()

// 基本数据
const courseId = ref<number>(Number(route.params.id) || 0)
const loading = ref<boolean>(true)
const course = ref<any>(null)
const teacherName = ref<string>('')
const isFavorite = ref<boolean>(false)
const activeTab = ref<string>('chapters')

// 章节数据
const chapters = ref<any[]>([])
const resources = ref<any[]>([])
const progress = ref<number>(0)
const completedSections = ref<number>(0)
const totalSections = ref<number>(0)

// 任务数据
const tasks = ref<any[]>([])

// 错题集数据
const wrongQuestions = ref<any[]>([])
const questionFilter = reactive({
  type: 'all',
  difficulty: 'all'
})

// 学习记录数据
const totalLearningHours = ref<number>(0)
const weeklyLearningHours = ref<number>(0)
const courseCompletionRate = ref<number>(0)
const achievementPoints = ref<number>(0)
const classRank = ref<number>(0)
const chartTimeRange = ref<string>('week')
const learningActivities = ref<any[]>([])

// 知识图谱数据
const graphViewMode = ref<string>('tree')

// 加载课程详情
const loadCourseDetail = async () => {
  try {
    loading.value = true
    console.log('开始加载课程详情，ID:', courseId.value)
    
    // 获取课程详情
    try {
      console.log('开始获取课程详情，ID:', courseId.value)
      const response = await getStudentCourseDetail(courseId.value)
      console.log('课程详情API响应:', response)
      
      // 新的API同时返回了课程信息和章节信息，我们需要处理这两部分数据
      course.value = response.course
      chapters.value = response.chapters || []
      
      console.log('课程详情获取成功:', course.value)
      console.log('课程章节获取成功，数量:', chapters.value.length)
      
      // 从课程数据中提取教师姓名
      teacherName.value = course.value?.teacherName || '未知教师'
      
      // 计算学习进度
      let completed = 0
      let total = 0
      
      chapters.value.forEach(chapter => {
        if (chapter.sections) {
          total += chapter.sections.length
          completed += chapter.sections.filter((section: any) => section.completed).length
        }
      })
      
      totalSections.value = total
      completedSections.value = completed
      progress.value = total > 0 ? Math.round((completed / total) * 100) : 0
    } catch (error) {
      console.error('获取课程详情API错误:', error)
      message.error('获取课程详情失败: ' + ((error as Error).message || '未知错误'))
      return
    }
    
    // 获取课程资源
    try {
      console.log('开始获取课程资源，课程ID:', courseId.value)
      resources.value = await getStudentCourseResources(courseId.value)
      console.log('课程资源获取成功，数量:', resources.value.length)
    } catch (error) {
      console.error('获取课程资源API错误:', error)
      resources.value = []
    }
    
    // 获取课程任务（作业和考试）
    try {
      console.log('开始获取课程任务，课程ID:', courseId.value)
      tasks.value = await getStudentCourseTasks(courseId.value)
      console.log('课程任务获取成功，数量:', tasks.value.length)
    } catch (error) {
      console.error('获取课程任务API错误:', error)
      tasks.value = []
    }
    
    // 模拟错题数据 - 这部分可以在后续实现
    wrongQuestions.value = [
      {
        id: 1,
        type: 'single',
        difficulty: 'medium',
        title: '以下关于计算机系统层次结构的描述，错误的是：',
        options: [
          { key: 'A', content: '微程序级是计算机系统的最低层次' },
          { key: 'B', content: '汇编语言级是机器语言级的高级抽象' },
          { key: 'C', content: '操作系统级提供了对硬件资源的管理' },
          { key: 'D', content: '高级语言级是最接近硬件的编程层次' }
        ],
        userAnswer: ['A'],
        correctAnswer: 'D',
        wrongCount: 2,
        analysis: '高级语言级是对底层硬件的高度抽象，而不是最接近硬件的编程层次。最接近硬件的是机器语言级。'
      },
      {
        id: 2,
        type: 'multiple',
        difficulty: 'hard',
        title: '下列关于CPU性能评价指标的说法中，正确的有：',
        options: [
          { key: 'A', content: 'CPI表示执行一条指令所需的时钟周期数' },
          { key: 'B', content: 'MIPS值越大表示CPU性能越高' },
          { key: 'C', content: 'CPU主频是衡量CPU性能的唯一指标' },
          { key: 'D', content: 'IPC表示每个时钟周期内可执行的指令数' }
        ],
        userAnswer: ['A', 'C'],
        correctAnswer: ['A', 'B', 'D'],
        wrongCount: 3,
        analysis: 'CPU主频不是衡量CPU性能的唯一指标，还需要考虑IPC、缓存大小等因素。'
      }
    ]
    
    // 模拟学习记录数据 - 这部分可以在后续实现
    totalLearningHours.value = 12.5
    weeklyLearningHours.value = 4.2
    courseCompletionRate.value = 35
    achievementPoints.value = 120
    classRank.value = 8
    
    // 模拟学习活动记录 - 这部分可以在后续实现
    learningActivities.value = [
      {
        type: 'video',
        title: '观看视频：计算机组成原理 - 第3章 存储系统',
        time: '2025-07-01 14:30',
        description: '完成度：100%，观看时长：45分钟'
      },
      {
        type: 'quiz',
        title: '完成测验：第3章课后习题',
        time: '2025-07-01 15:20',
        description: '得分：85分，用时：25分钟'
      },
      {
        type: 'resource',
        title: '下载资源：第3章PPT',
        time: '2025-07-01 15:45',
        description: '资源类型：PPT，大小：2.5MB'
      },
      {
        type: 'assignment',
        title: '提交作业：第3章课后作业',
        time: '2025-07-02 10:15',
        description: '状态：已批改，得分：92分'
      }
    ]
    
    console.log('课程详情加载完成')
  } catch (error) {
    console.error('加载课程详情失败:', error)
    message.error('加载课程详情失败')
  } finally {
    loading.value = false
  }
}

// 返回课程列表
const goBack = () => {
  router.push('/student/courses')
}

// 开始学习
const startLearning = () => {
  // 查找第一个未完成的小节
  let targetSection = null
  let targetChapter = null
  
  for (const chapter of chapters.value) {
    if (chapter.sections && chapter.sections.length > 0) {
      for (const section of chapter.sections) {
        if (!section.completed) {
          targetSection = section
          targetChapter = chapter
          break
        }
      }
      if (targetSection) break
    }
  }
  
  // 如果没有未完成的小节，则从第一个小节开始
  if (!targetSection && chapters.value.length > 0 && chapters.value[0].sections && chapters.value[0].sections.length > 0) {
    targetSection = chapters.value[0].sections[0]
    targetChapter = chapters.value[0]
  }
  
  if (targetSection) {
    viewSection(targetChapter, targetSection)
  } else {
    message.info('该课程暂无可学习内容')
  }
}

// 查看小节
const viewSection = (chapter: any, section: any) => {
  if (section.type === 'video') {
    router.push(`/student/courses/${courseId.value}/video/${section.id}`)
  } else {
    message.info(`正在查看: ${chapter.title} - ${section.title}`)
    // 这里可以跳转到文本内容页面或者打开一个模态框显示内容
  }
}

// 下载资源
const downloadResource = (resource: any) => {
  try {
    console.log('开始下载资源:', resource)
    // 构建下载URL
    const downloadUrl = `/api/teacher/courses/resources/${resource.id}/download`
    
    // 创建一个隐藏的a标签来触发下载
    const link = document.createElement('a')
    link.href = downloadUrl
    link.target = '_blank'
    link.download = resource.name || 'download'
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
    
    message.success(`正在下载: ${resource.name}`)
  } catch (error) {
    console.error('下载资源失败:', error)
    message.error('下载资源失败')
  }
}

// 切换收藏状态
const toggleFavorite = () => {
  isFavorite.value = !isFavorite.value
  message.success(isFavorite.value ? '已收藏课程' : '已取消收藏')
  // 实际收藏API调用
}

// 格式化文件大小
const formatFileSize = (size: number): string => {
  if (size < 1024) {
    return size + ' B'
  } else if (size < 1024 * 1024) {
    return (size / 1024).toFixed(1) + ' KB'
  } else if (size < 1024 * 1024 * 1024) {
    return (size / (1024 * 1024)).toFixed(1) + ' MB'
  } else {
    return (size / (1024 * 1024 * 1024)).toFixed(1) + ' GB'
  }
}

// 获取状态文本
const getStatusText = (status?: string): string => {
  if (!status) return '未知'
  
  const statusMap: Record<string, string> = {
    'NOT_STARTED': '未开始',
    'IN_PROGRESS': '进行中',
    'FINISHED': '已结束',
    '未开始': '未开始',
    '进行中': '进行中',
    '已结束': '已结束'
  }
  
  return statusMap[status] || status
}

// 获取状态样式类
const getStatusClass = (status?: string): string => {
  if (!status) return 'status-unknown'
  
  const normalizedStatus = getStatusText(status).toLowerCase()
  
  if (normalizedStatus.includes('进行')) return 'status-active'
  if (normalizedStatus.includes('结束')) return 'status-completed'
  if (normalizedStatus.includes('未开')) return 'status-pending'
  
  return 'status-unknown'
}

// 获取进度条颜色
const getProgressColor = (progress: number): string => {
  if (progress >= 80) return '#52c41a'
  if (progress >= 60) return '#1890ff'
  if (progress >= 40) return '#faad14'
  return '#ff4d4f'
}

// 获取任务状态颜色
const getTaskStatusColor = (status: string): string => {
  const statusColorMap: Record<string, string> = {
    'pending': '#faad14',        // 未开始
    'in_progress': '#1890ff',    // 进行中
    'completed': '#ff4d4f'       // 已截止
  }
  return statusColorMap[status] || '#d9d9d9'
}

// 获取任务状态文本
const getTaskStatusText = (status: string): string => {
  const statusTextMap: Record<string, string> = {
    'pending': '未开始',
    'in_progress': '进行中',
    'completed': '已截止'
  }
  return statusTextMap[status] || '未知状态'
}

// 查看任务详情
const viewTask = (task: any) => {
  console.log('查看任务详情:', task)
  
  // 如果任务未发布，则提示
  if (task.status === 'not_published') {
    message.info('该任务尚未发布，请等待教师发布后再查看')
    return
  }
  
  // 根据任务类型和模式跳转到不同页面
  if (task.type === 'homework') {
    // 作业
    if (task.mode === 'question') {
      // 答题型作业
      router.push(`/student/assignments/${task.id}`)
    } else {
      // 文件上传型作业
      router.push(`/student/assignments/file/${task.id}`)
    }
  } else {
    // 考试
    router.push(`/student/exams/${task.id}`)
  }
}

// 获取题目类型颜色
const getQuestionTypeColor = (type: string): string => {
  const typeColorMap: Record<string, string> = {
    'single': '#1890ff',
    'multiple': '#722ed1',
    'judge': '#52c41a',
    'fill': '#fa8c16'
  }
  return typeColorMap[type] || '#d9d9d9'
}

// 获取题目类型文本
const getQuestionTypeText = (type: string): string => {
  const typeTextMap: Record<string, string> = {
    'single': '单选题',
    'multiple': '多选题',
    'judge': '判断题',
    'fill': '填空题'
  }
  return typeTextMap[type] || '其他题型'
}

// 练习题目
const practiceQuestion = (question: any) => {
  message.info(`开始练习题目：${question.title}`)
  // 实际练习逻辑
}

// 标记题目为已掌握
const markAsResolved = (question: any) => {
  message.success(`已将题目标记为已掌握`)
  // 实际标记逻辑
}

// 获取活动颜色
const getActivityColor = (type: string): string => {
  const typeColorMap: Record<string, string> = {
    'video': '#1890ff',
    'quiz': '#722ed1',
    'assignment': '#fa8c16',
    'resource': '#52c41a'
  }
  return typeColorMap[type] || '#d9d9d9'
}

// 格式化日期
const formatDate = (date: string): string => {
  return dayjs(date).format('YYYY-MM-DD HH:mm')
}

onMounted(() => {
  loadCourseDetail()
})
</script>

<style scoped>
.fixed-width-container {
  max-width: 1200px;
  margin: 0 auto;
  padding: 24px;
}

.course-detail {
  background-color: #fff;
  border-radius: 8px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);
}

.content-wrapper {
  padding: 24px;
}

/* 页面头部 */
.page-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 24px;
}

.back-btn {
  font-size: 16px;
  padding: 0;
}

.header-right {
  display: flex;
  gap: 12px;
}

/* 课程信息部分 */
.course-info-section {
  margin-bottom: 32px;
}

.course-header {
  display: flex;
  gap: 24px;
}

.course-cover {
  width: 280px;
  height: 180px;
  border-radius: 8px;
  overflow: hidden;
  position: relative;
  flex-shrink: 0;
}

.course-cover img {
  width: 100%;
  height: 100%;
  object-fit: cover;
}

.cover-placeholder {
  width: 100%;
  height: 100%;
  display: flex;
  align-items: center;
  justify-content: center;
  background-color: #f5f5f5;
  color: #999;
}

.course-status {
  position: absolute;
  top: 12px;
  right: 12px;
  padding: 4px 12px;
  border-radius: 4px;
  font-size: 12px;
  font-weight: 500;
  color: white;
}

.status-active {
  background-color: #1890ff;
}

.status-completed {
  background-color: #52c41a;
}

.status-pending {
  background-color: #faad14;
}

.status-unknown {
  background-color: #d9d9d9;
  color: #666;
}

.course-info {
  flex: 1;
}

.course-title {
  font-size: 24px;
  font-weight: 600;
  margin: 0 0 16px 0;
}

.course-meta {
  display: flex;
  flex-wrap: wrap;
  gap: 16px;
  margin-bottom: 16px;
}

.meta-item {
  display: flex;
  align-items: center;
  gap: 8px;
  color: #666;
}

.course-description {
  margin-bottom: 24px;
  color: #666;
  line-height: 1.6;
}

.course-progress {
  margin-top: 16px;
}

.progress-header {
  display: flex;
  justify-content: space-between;
  margin-bottom: 8px;
  font-weight: 500;
}

.progress-info {
  margin-top: 8px;
  color: #666;
  font-size: 12px;
}

/* 章节列表 */
.chapter-item {
  margin-bottom: 24px;
  border: 1px solid #f0f0f0;
  border-radius: 8px;
  overflow: hidden;
}

.chapter-info {
  display: flex;
  background-color: #fafafa;
}

.chapter-number {
  width: 48px;
  height: 48px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 18px;
  font-weight: 600;
  color: #1890ff;
  background-color: rgba(24, 144, 255, 0.1);
}

.chapter-content {
  flex: 1;
  padding: 16px;
}

.chapter-title {
  font-size: 16px;
  font-weight: 600;
  margin: 0 0 8px 0;
}

.chapter-description {
  font-size: 14px;
  color: #666;
  margin: 0 0 16px 0;
}

.chapter-sections {
  background-color: white;
  border-radius: 4px;
}

.section-item {
  display: flex;
  align-items: center;
  padding: 12px 16px;
  border-bottom: 1px solid #f0f0f0;
  cursor: pointer;
  transition: background-color 0.3s;
}

.section-item:last-child {
  border-bottom: none;
}

.section-item:hover {
  background-color: #f5f5f5;
}

.section-icon {
  margin-right: 12px;
  font-size: 18px;
  color: #1890ff;
}

.section-content {
  flex: 1;
}

.section-title {
  font-weight: 500;
}

.section-meta {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-top: 4px;
  font-size: 12px;
  color: #999;
}

.section-duration {
  color: #666;
}

.empty-sections, .empty-chapters {
  padding: 32px 0;
  text-align: center;
}

/* 资源列表 */
.resources-list {
  padding: 16px 0;
}

.resource-icon {
  height: 120px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 48px;
  background-color: #f5f5f5;
}

.resource-meta {
  display: flex;
  justify-content: space-between;
  color: #666;
  font-size: 12px;
}

/* 任务列表样式 */
.tasks-section {
  padding: 16px 0;
}

.task-title {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.task-description {
  color: #666;
}

.task-meta {
  display: flex;
  justify-content: space-between;
  margin-top: 8px;
  font-size: 12px;
  color: #999;
}

/* 错题集样式 */
.wrong-questions-section {
  padding: 16px 0;
}

.filter-section {
  margin-bottom: 24px;
  padding: 16px;
  background-color: #f5f5f5;
  border-radius: 8px;
}

.question-card {
  margin-bottom: 24px;
  padding: 16px;
  border: 1px solid #f0f0f0;
  border-radius: 8px;
  background-color: #fff;
}

.question-header {
  display: flex;
  justify-content: space-between;
  margin-bottom: 16px;
}

.question-meta {
  display: flex;
  align-items: center;
  gap: 8px;
}

.wrong-count {
  color: #ff4d4f;
  font-size: 12px;
}

.question-title {
  font-weight: 500;
  margin-bottom: 16px;
}

.question-options {
  margin-bottom: 16px;
}

.option-item {
  margin-bottom: 8px;
  padding: 8px 12px;
  background-color: #fafafa;
  border-radius: 4px;
}

.option-correct {
  color: #52c41a;
  font-weight: 500;
}

.option-wrong {
  color: #ff4d4f;
  text-decoration: line-through;
}

.question-answer {
  margin-bottom: 16px;
  padding: 12px;
  background-color: #f6f6f6;
  border-radius: 4px;
}

.answer-label {
  font-weight: 500;
  margin-bottom: 4px;
}

.user-answer.wrong {
  color: #ff4d4f;
}

.correct-answer {
  color: #52c41a;
  font-weight: 500;
}

.question-analysis {
  margin-bottom: 16px;
  padding: 12px;
  background-color: #e6f7ff;
  border-radius: 4px;
}

.analysis-label {
  font-weight: 500;
  margin-bottom: 4px;
}

.question-actions {
  display: flex;
  justify-content: flex-end;
  gap: 8px;
}

/* 学习记录样式 */
.learning-records-section {
  padding: 16px 0;
}

.statistics-cards {
  margin-bottom: 24px;
}

.statistic-value {
  font-size: 24px;
  font-weight: 600;
  margin-bottom: 8px;
}

.statistic-description {
  color: #666;
  font-size: 12px;
}

.learning-chart {
  margin-bottom: 32px;
  padding: 16px;
  background-color: #fff;
  border: 1px solid #f0f0f0;
  border-radius: 8px;
}

.chart-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 16px;
}

.chart-header h3 {
  margin: 0;
}

.chart-placeholder {
  height: 300px;
  display: flex;
  align-items: center;
  justify-content: center;
}

.learning-activities {
  padding: 16px;
  background-color: #fff;
  border: 1px solid #f0f0f0;
  border-radius: 8px;
}

.activities-header {
  margin-bottom: 16px;
}

.activities-header h3 {
  margin: 0;
}

.activity-item {
  margin-bottom: 8px;
}

.activity-title {
  font-weight: 500;
}

.activity-time {
  font-size: 12px;
  color: #999;
  margin: 4px 0;
}

.activity-description {
  color: #666;
}

/* 知识图谱样式 */
.knowledge-graph-section {
  padding: 16px 0;
}

.graph-controls {
  margin-bottom: 16px;
  padding: 12px;
  background-color: #f5f5f5;
  border-radius: 8px;
}

.graph-container {
  height: 500px;
  margin-bottom: 24px;
  border: 1px solid #f0f0f0;
  border-radius: 8px;
  display: flex;
  align-items: center;
  justify-content: center;
}

.knowledge-details {
  padding: 16px;
  background-color: #fff;
  border: 1px solid #f0f0f0;
  border-radius: 8px;
}

.empty-tip {
  color: #999;
  text-align: center;
  padding: 32px 0;
}
</style> 