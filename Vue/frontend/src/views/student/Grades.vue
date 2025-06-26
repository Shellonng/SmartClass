<template>
  <div class="student-grades">
    <!-- 页面头部 -->
    <div class="page-header">
      <div class="header-content">
        <h1 class="page-title">
          <TrophyOutlined />
          成绩查询
        </h1>
        <p class="page-description">全面了解学习成果，持续改进提升</p>
      </div>
      <div class="header-actions">
        <a-button type="primary" @click="exportGrades">
          <DownloadOutlined />
          导出成绩单
        </a-button>
      </div>
    </div>

    <!-- 成绩概览 -->
    <div class="grades-overview">
      <div class="overview-card gpa">
        <div class="card-icon">
          <TrophyOutlined />
        </div>
        <div class="card-content">
          <div class="card-title">总体GPA</div>
          <div class="card-value">{{ overallGPA }}</div>
          <div class="card-trend" :class="{ positive: gpaChange > 0, negative: gpaChange < 0 }">
            <ArrowUpOutlined v-if="gpaChange > 0" />
            <ArrowDownOutlined v-else-if="gpaChange < 0" />
            <MinusOutlined v-else />
            {{ gpaChange > 0 ? '+' : '' }}{{ gpaChange.toFixed(2) }}
          </div>
        </div>
      </div>

      <div class="overview-card average">
        <div class="card-icon">
          <BarChartOutlined />
        </div>
        <div class="card-content">
          <div class="card-title">平均分</div>
          <div class="card-value">{{ averageScore }}</div>
          <div class="card-subtitle">满分100分</div>
        </div>
      </div>

      <div class="overview-card rank">
        <div class="card-icon">
          <CrownOutlined />
        </div>
        <div class="card-content">
          <div class="card-title">班级排名</div>
          <div class="card-value">{{ classRank }}/{{ totalStudents }}</div>
          <div class="card-subtitle">年级排名 {{ gradeRank }}</div>
        </div>
      </div>

      <div class="overview-card credits">
        <div class="card-icon">
          <BookOutlined />
        </div>
        <div class="card-content">
          <div class="card-title">已获学分</div>
          <div class="card-value">{{ earnedCredits }}/{{ totalCredits }}</div>
          <div class="card-subtitle">完成度 {{ Math.round(earnedCredits / totalCredits * 100) }}%</div>
        </div>
      </div>
    </div>

    <!-- 成绩趋势图表 -->
    <div class="grades-charts">
      <div class="chart-section">
        <div class="section-header">
          <h2 class="section-title">
            <LineChartOutlined />
            成绩趋势分析
          </h2>
          <div class="section-controls">
            <a-radio-group v-model:value="chartPeriod" @change="updateChart">
              <a-radio-button value="semester">学期</a-radio-button>
              <a-radio-button value="month">月度</a-radio-button>
              <a-radio-button value="exam">考试</a-radio-button>
            </a-radio-group>
          </div>
        </div>
        <div class="chart-container">
          <div id="gradesChart" class="chart"></div>
        </div>
      </div>

      <div class="chart-section">
        <div class="section-header">
          <h2 class="section-title">
            <PieChartOutlined />
            学科成绩分布
          </h2>
        </div>
        <div class="chart-container">
          <div id="subjectChart" class="chart"></div>
        </div>
      </div>
    </div>

    <!-- 筛选和搜索 -->
    <div class="filter-section">
      <div class="filter-controls">
        <a-select
          v-model:value="semesterFilter"
          placeholder="选择学期"
          style="width: 150px"
          @change="handleFilter"
        >
          <a-select-option value="">全部学期</a-select-option>
          <a-select-option v-for="semester in semesters" :key="semester" :value="semester">
            {{ semester }}
          </a-select-option>
        </a-select>

        <a-select
          v-model:value="courseFilter"
          placeholder="选择课程"
          style="width: 150px"
          @change="handleFilter"
        >
          <a-select-option value="">全部课程</a-select-option>
          <a-select-option v-for="course in courses" :key="course.id" :value="course.id">
            {{ course.name }}
          </a-select-option>
        </a-select>

        <a-select
          v-model:value="typeFilter"
          placeholder="成绩类型"
          style="width: 120px"
          @change="handleFilter"
        >
          <a-select-option value="">全部类型</a-select-option>
          <a-select-option value="exam">期末考试</a-select-option>
          <a-select-option value="midterm">期中考试</a-select-option>
          <a-select-option value="quiz">随堂测验</a-select-option>
          <a-select-option value="homework">平时作业</a-select-option>
        </a-select>

        <a-input-search
          v-model:value="searchKeyword"
          placeholder="搜索课程或考试名称..."
          style="width: 250px"
          @search="handleSearch"
        />
      </div>

      <div class="view-controls">
        <a-radio-group v-model:value="viewMode" @change="handleViewChange">
          <a-radio-button value="table">
            <TableOutlined />
            表格视图
          </a-radio-button>
          <a-radio-button value="card">
            <AppstoreOutlined />
            卡片视图
          </a-radio-button>
        </a-radio-group>
      </div>
    </div>

    <!-- 成绩列表 -->
    <div class="grades-content">
      <!-- 表格视图 -->
      <div v-if="viewMode === 'table'" class="table-view">
        <a-table
          :columns="columns"
          :data-source="filteredGrades"
          :loading="loading"
          :pagination="pagination"
          :scroll="{ x: 1200 }"
          row-key="id"
          @change="handleTableChange"
        >
          <template #bodyCell="{ column, record }">
            <template v-if="column.key === 'courseName'">
              <div class="course-info">
                <div class="course-name">{{ record.courseName }}</div>
                <div class="course-meta">{{ record.courseCode }} · {{ record.credits }}学分</div>
              </div>
            </template>

            <template v-else-if="column.key === 'score'">
              <div class="score-cell">
                <div class="score-value" :class="getScoreLevel(record.score)">
                  {{ record.score }}
                </div>
                <div class="score-level">{{ getScoreText(record.score) }}</div>
              </div>
            </template>

            <template v-else-if="column.key === 'rank'">
              <div class="rank-cell">
                <div class="rank-value">{{ record.rank }}/{{ record.totalStudents }}</div>
                <a-progress 
                  :percent="Math.round((record.totalStudents - record.rank + 1) / record.totalStudents * 100)"
                  :stroke-width="4"
                  :show-info="false"
                  :stroke-color="getRankColor(record.rank, record.totalStudents)"
                />
              </div>
            </template>

            <template v-else-if="column.key === 'gpa'">
              <div class="gpa-cell">
                <div class="gpa-value">{{ record.gpa.toFixed(2) }}</div>
                <div class="gpa-points">{{ getGpaPoints(record.gpa) }}</div>
              </div>
            </template>

            <template v-else-if="column.key === 'trend'">
              <div class="trend-cell">
                <div class="trend-icon" :class="{ 
                  positive: record.trend > 0, 
                  negative: record.trend < 0,
                  neutral: record.trend === 0
                }">
                  <ArrowUpOutlined v-if="record.trend > 0" />
                  <ArrowDownOutlined v-else-if="record.trend < 0" />
                  <MinusOutlined v-else />
                </div>
                <div class="trend-value">{{ record.trend > 0 ? '+' : '' }}{{ record.trend }}</div>
              </div>
            </template>

            <template v-else-if="column.key === 'action'">
              <a-button-group size="small">
                <a-button @click="viewGradeDetail(record)">
                  <EyeOutlined />
                  详情
                </a-button>
                <a-button @click="downloadReport(record)">
                  <DownloadOutlined />
                  报告
                </a-button>
              </a-button-group>
            </template>
          </template>
        </a-table>
      </div>

      <!-- 卡片视图 -->
      <div v-else class="card-view">
        <div class="grades-grid">
          <div 
            v-for="grade in filteredGrades" 
            :key="grade.id"
            class="grade-card"
            :class="getScoreLevel(grade.score)"
          >
            <div class="grade-header">
              <div class="course-info">
                <h3 class="course-name">{{ grade.courseName }}</h3>
                <div class="course-meta">
                  <span class="course-code">{{ grade.courseCode }}</span>
                  <span class="credits">{{ grade.credits }}学分</span>
                  <span class="semester">{{ grade.semester }}</span>
                </div>
              </div>
              <div class="grade-score">
                <div class="score-value">{{ grade.score }}</div>
                <div class="score-level">{{ getScoreText(grade.score) }}</div>
              </div>
            </div>

            <div class="grade-details">
              <div class="detail-row">
                <span class="detail-label">考试类型：</span>
                <span class="detail-value">{{ getTypeText(grade.type) }}</span>
              </div>
              <div class="detail-row">
                <span class="detail-label">考试时间：</span>
                <span class="detail-value">{{ formatDate(grade.examDate) }}</span>
              </div>
              <div class="detail-row">
                <span class="detail-label">班级排名：</span>
                <span class="detail-value">{{ grade.rank }}/{{ grade.totalStudents }}</span>
                <div class="rank-progress">
                  <a-progress 
                    :percent="Math.round((grade.totalStudents - grade.rank + 1) / grade.totalStudents * 100)"
                    :stroke-width="4"
                    :show-info="false"
                    :stroke-color="getRankColor(grade.rank, grade.totalStudents)"
                  />
                </div>
              </div>
              <div class="detail-row">
                <span class="detail-label">绩点：</span>
                <span class="detail-value gpa">{{ grade.gpa.toFixed(2) }}</span>
                <span class="gpa-points">{{ getGpaPoints(grade.gpa) }}</span>
              </div>
            </div>

            <div class="grade-footer">
              <div class="grade-trend">
                <div class="trend-icon" :class="{ 
                  positive: grade.trend > 0, 
                  negative: grade.trend < 0,
                  neutral: grade.trend === 0
                }">
                  <ArrowUpOutlined v-if="grade.trend > 0" />
                  <ArrowDownOutlined v-else-if="grade.trend < 0" />
                  <MinusOutlined v-else />
                </div>
                <span class="trend-text">
                  较上次{{ grade.trend > 0 ? '提高' : grade.trend < 0 ? '下降' : '持平' }}
                  {{ Math.abs(grade.trend) }}分
                </span>
              </div>
              <div class="grade-actions">
                <a-button size="small" @click="viewGradeDetail(grade)">
                  <EyeOutlined />
                  详情
                </a-button>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- 成绩详情弹窗 -->
    <a-modal
      v-model:open="detailModalVisible"
      title="成绩详情"
      :width="800"
      :footer="null"
    >
      <div class="grade-detail" v-if="selectedGrade">
        <div class="detail-header">
          <div class="course-info">
            <h2>{{ selectedGrade.courseName }}</h2>
            <div class="course-meta">
              <span>{{ selectedGrade.courseCode }}</span>
              <span>{{ selectedGrade.credits }}学分</span>
              <span>{{ selectedGrade.semester }}</span>
            </div>
          </div>
          <div class="score-display">
            <div class="main-score">{{ selectedGrade.score }}</div>
            <div class="score-level">{{ getScoreText(selectedGrade.score) }}</div>
          </div>
        </div>

        <div class="detail-sections">
          <div class="section">
            <h3>成绩组成</h3>
            <div class="score-breakdown">
              <div class="breakdown-item" v-for="item in selectedGrade.breakdown" :key="item.name">
                <div class="breakdown-info">
                  <span class="breakdown-name">{{ item.name }}</span>
                  <span class="breakdown-weight">{{ item.weight }}%</span>
                </div>
                <div class="breakdown-score">
                  <span class="score">{{ item.score }}</span>
                  <div class="score-bar">
                    <div class="score-progress" :style="{ width: item.score + '%' }"></div>
                  </div>
                </div>
              </div>
            </div>
          </div>

          <div class="section">
            <h3>统计信息</h3>
            <div class="stats-grid">
              <div class="stat-item">
                <div class="stat-label">班级平均分</div>
                <div class="stat-value">{{ selectedGrade.classAverage }}</div>
              </div>
              <div class="stat-item">
                <div class="stat-label">最高分</div>
                <div class="stat-value">{{ selectedGrade.maxScore }}</div>
              </div>
              <div class="stat-item">
                <div class="stat-label">最低分</div>
                <div class="stat-value">{{ selectedGrade.minScore }}</div>
              </div>
              <div class="stat-item">
                <div class="stat-label">标准差</div>
                <div class="stat-value">{{ selectedGrade.standardDeviation }}</div>
              </div>
            </div>
          </div>

          <div class="section" v-if="selectedGrade.feedback">
            <h3>教师评价</h3>
            <div class="feedback">
              {{ selectedGrade.feedback }}
            </div>
          </div>
        </div>
      </div>
    </a-modal>
  </div>
</template>

<script setup lang="ts">
import { ref, reactive, computed, onMounted, nextTick } from 'vue'
import { message } from 'ant-design-vue'
import dayjs from 'dayjs'
import {
  TrophyOutlined,
  DownloadOutlined,
  BarChartOutlined,
  CrownOutlined,
  BookOutlined,
  LineChartOutlined,
  PieChartOutlined,
  TableOutlined,
  AppstoreOutlined,
  EyeOutlined,
  ArrowUpOutlined,
  ArrowDownOutlined,
  MinusOutlined
} from '@ant-design/icons-vue'

interface Grade {
  id: number
  courseName: string
  courseCode: string
  credits: number
  semester: string
  type: 'exam' | 'midterm' | 'quiz' | 'homework'
  score: number
  rank: number
  totalStudents: number
  gpa: number
  trend: number
  examDate: string
  teacherName: string
  classAverage: number
  maxScore: number
  minScore: number
  standardDeviation: number
  feedback?: string
  breakdown: {
    name: string
    score: number
    weight: number
  }[]
}

interface Course {
  id: number
  name: string
}

// 页面状态
const loading = ref(false)
const viewMode = ref('table')
const chartPeriod = ref('semester')
const semesterFilter = ref('')
const courseFilter = ref('')
const typeFilter = ref('')
const searchKeyword = ref('')
const detailModalVisible = ref(false)
const selectedGrade = ref<Grade | null>(null)

// 统计数据
const overallGPA = ref(3.72)
const gpaChange = ref(0.15)
const averageScore = ref(87.3)
const classRank = ref(8)
const gradeRank = ref(42)
const totalStudents = ref(156)
const earnedCredits = ref(48)
const totalCredits = ref(60)

// 下拉选项
const semesters = ref(['2023-2024-1', '2023-2024-2', '2024-2025-1'])
const courses = ref<Course[]>([
  { id: 1, name: '高等数学' },
  { id: 2, name: '大学英语' },
  { id: 3, name: 'C++程序设计' },
  { id: 4, name: '线性代数' }
])

// 成绩数据
const grades = ref<Grade[]>([
  {
    id: 1,
    courseName: '高等数学（上）',
    courseCode: 'MATH101',
    credits: 4,
    semester: '2024-2025-1',
    type: 'exam',
    score: 92,
    rank: 5,
    totalStudents: 156,
    gpa: 4.0,
    trend: 8,
    examDate: '2024-01-15',
    teacherName: '张教授',
    classAverage: 82.5,
    maxScore: 98,
    minScore: 45,
    standardDeviation: 12.3,
    feedback: '学习态度认真，解题思路清晰，建议继续保持。',
    breakdown: [
      { name: '平时作业', score: 95, weight: 20 },
      { name: '期中考试', score: 88, weight: 30 },
      { name: '期末考试', score: 92, weight: 50 }
    ]
  },
  {
    id: 2,
    courseName: '大学英语四级',
    courseCode: 'ENG201',
    credits: 3,
    semester: '2024-2025-1',
    type: 'exam',
    score: 85,
    rank: 12,
    totalStudents: 156,
    gpa: 3.7,
    trend: -2,
    examDate: '2024-01-12',
    teacherName: '李老师',
    classAverage: 79.2,
    maxScore: 94,
    minScore: 52,
    standardDeviation: 11.8,
    breakdown: [
      { name: '听力', score: 82, weight: 25 },
      { name: '阅读', score: 88, weight: 35 },
      { name: '写作', score: 85, weight: 25 },
      { name: '翻译', score: 83, weight: 15 }
    ]
  }
])

// 分页配置
const pagination = reactive({
  current: 1,
  pageSize: 10,
  total: 0,
  showSizeChanger: true,
  showQuickJumper: true,
  showTotal: (total: number, range: number[]) => `共 ${total} 条记录，当前显示 ${range[0]}-${range[1]} 条`
})

// 表格列配置
const columns = [
  {
    title: '课程信息',
    key: 'courseName',
    width: 200,
    fixed: 'left'
  },
  {
    title: '学期',
    dataIndex: 'semester',
    width: 120
  },
  {
    title: '类型',
    dataIndex: 'type',
    width: 100,
    customRender: ({ text }: { text: string }) => getTypeText(text)
  },
  {
    title: '成绩',
    key: 'score',
    width: 120,
    sorter: (a: Grade, b: Grade) => a.score - b.score
  },
  {
    title: '排名',
    key: 'rank',
    width: 150
  },
  {
    title: '绩点',
    key: 'gpa',
    width: 100,
    sorter: (a: Grade, b: Grade) => a.gpa - b.gpa
  },
  {
    title: '趋势',
    key: 'trend',
    width: 100
  },
  {
    title: '考试时间',
    dataIndex: 'examDate',
    width: 120,
    customRender: ({ text }: { text: string }) => formatDate(text)
  },
  {
    title: '操作',
    key: 'action',
    width: 150,
    fixed: 'right'
  }
]

// 计算属性
const filteredGrades = computed(() => {
  let filtered = grades.value

  if (semesterFilter.value) {
    filtered = filtered.filter(g => g.semester === semesterFilter.value)
  }

  if (courseFilter.value) {
    const course = courses.value.find(c => c.id === courseFilter.value)
    if (course) {
      filtered = filtered.filter(g => g.courseName.includes(course.name))
    }
  }

  if (typeFilter.value) {
    filtered = filtered.filter(g => g.type === typeFilter.value)
  }

  if (searchKeyword.value) {
    const keyword = searchKeyword.value.toLowerCase()
    filtered = filtered.filter(g => 
      g.courseName.toLowerCase().includes(keyword) ||
      g.courseCode.toLowerCase().includes(keyword)
    )
  }

  pagination.total = filtered.length
  const start = (pagination.current - 1) * pagination.pageSize
  const end = start + pagination.pageSize
  return filtered.slice(start, end)
})

// 方法
const handleFilter = () => {
  pagination.current = 1
}

const handleSearch = () => {
  pagination.current = 1
}

const handleViewChange = () => {
  pagination.current = 1
}

const handleTableChange = (pag: any) => {
  pagination.current = pag.current
  pagination.pageSize = pag.pageSize
}

const exportGrades = () => {
  message.success('成绩单导出成功')
}

const updateChart = () => {
  // 更新图表逻辑
  message.info('图表已更新')
}

const viewGradeDetail = (grade: Grade) => {
  selectedGrade.value = grade
  detailModalVisible.value = true
}

const downloadReport = (grade: Grade) => {
  message.success(`${grade.courseName} 成绩报告下载成功`)
}

// 工具方法
const getScoreLevel = (score: number) => {
  if (score >= 90) return 'excellent'
  if (score >= 80) return 'good'
  if (score >= 70) return 'fair'
  if (score >= 60) return 'pass'
  return 'fail'
}

const getScoreText = (score: number) => {
  if (score >= 90) return '优秀'
  if (score >= 80) return '良好'
  if (score >= 70) return '中等'
  if (score >= 60) return '及格'
  return '不及格'
}

const getTypeText = (type: string) => {
  const typeMap = {
    exam: '期末考试',
    midterm: '期中考试',
    quiz: '随堂测验',
    homework: '平时作业'
  }
  return typeMap[type as keyof typeof typeMap] || '未知'
}

const getRankColor = (rank: number, total: number) => {
  const percentage = (total - rank + 1) / total
  if (percentage >= 0.8) return '#52c41a'
  if (percentage >= 0.6) return '#1890ff'
  if (percentage >= 0.4) return '#faad14'
  return '#ff4d4f'
}

const getGpaPoints = (gpa: number) => {
  if (gpa >= 3.7) return 'A'
  if (gpa >= 3.3) return 'B+'
  if (gpa >= 3.0) return 'B'
  if (gpa >= 2.7) return 'C+'
  if (gpa >= 2.0) return 'C'
  return 'D'
}

const formatDate = (date: string) => {
  return dayjs(date).format('YYYY-MM-DD')
}

// 页面初始化
onMounted(() => {
  console.log('成绩查询页面初始化完成')
  loading.value = true
  setTimeout(() => {
    loading.value = false
  }, 1000)
})
</script>

<style scoped>
.student-grades {
  padding: 24px;
  min-height: 100vh;
  background: #f5f7fa;
}

/* 页面头部 */
.page-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 32px;
  padding: 32px;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  border-radius: 20px;
  color: white;
  box-shadow: 0 12px 40px rgba(102, 126, 234, 0.3);
}

.header-content {
  flex: 1;
}

.page-title {
  font-size: 28px;
  font-weight: 700;
  margin: 0 0 8px 0;
  display: flex;
  align-items: center;
  gap: 12px;
}

.page-description {
  font-size: 16px;
  margin: 0;
  opacity: 0.9;
}

.header-actions .ant-btn {
  background: rgba(255, 255, 255, 0.2);
  border-color: rgba(255, 255, 255, 0.3);
  color: white;
  height: 40px;
  padding: 0 20px;
}

/* 成绩概览 */
.grades-overview {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 24px;
  margin-bottom: 32px;
}

.overview-card {
  background: white;
  border-radius: 16px;
  padding: 24px;
  display: flex;
  align-items: center;
  gap: 16px;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.06);
  transition: all 0.3s ease;
}

.overview-card:hover {
  transform: translateY(-4px);
  box-shadow: 0 8px 30px rgba(0, 0, 0, 0.12);
}

.card-icon {
  width: 60px;
  height: 60px;
  border-radius: 16px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 24px;
  color: white;
}

.overview-card.gpa .card-icon {
  background: linear-gradient(135deg, #667eea, #764ba2);
}

.overview-card.average .card-icon {
  background: linear-gradient(135deg, #1890ff, #36cfc9);
}

.overview-card.rank .card-icon {
  background: linear-gradient(135deg, #faad14, #ffc53d);
}

.overview-card.credits .card-icon {
  background: linear-gradient(135deg, #52c41a, #73d13d);
}

.card-content {
  flex: 1;
}

.card-title {
  font-size: 14px;
  color: #666;
  margin-bottom: 8px;
}

.card-value {
  font-size: 28px;
  font-weight: 700;
  color: #333;
  margin-bottom: 4px;
}

.card-subtitle {
  font-size: 12px;
  color: #999;
}

.card-trend {
  display: flex;
  align-items: center;
  gap: 4px;
  font-size: 12px;
  font-weight: 500;
}

.card-trend.positive {
  color: #52c41a;
}

.card-trend.negative {
  color: #ff4d4f;
}

/* 图表区域 */
.grades-charts {
  display: grid;
  grid-template-columns: 2fr 1fr;
  gap: 24px;
  margin-bottom: 32px;
}

.chart-section {
  background: white;
  border-radius: 16px;
  padding: 24px;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.06);
}

.section-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 20px;
}

.section-title {
  font-size: 18px;
  font-weight: 600;
  color: #333;
  margin: 0;
  display: flex;
  align-items: center;
  gap: 8px;
}

.chart-container {
  height: 300px;
  width: 100%;
}

.chart {
  width: 100%;
  height: 100%;
  background: #fafafa;
  border-radius: 8px;
  display: flex;
  align-items: center;
  justify-content: center;
  color: #999;
}

/* 筛选区域 */
.filter-section {
  background: white;
  border-radius: 16px;
  padding: 20px 24px;
  margin-bottom: 24px;
  display: flex;
  justify-content: space-between;
  align-items: center;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.06);
}

.filter-controls {
  display: flex;
  gap: 12px;
  align-items: center;
}

.view-controls {
  display: flex;
  gap: 12px;
  align-items: center;
}

/* 成绩内容 */
.grades-content {
  background: white;
  border-radius: 16px;
  padding: 24px;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.06);
}

/* 表格视图样式 */
.course-info .course-name {
  font-weight: 600;
  color: #333;
  margin-bottom: 4px;
}

.course-info .course-meta {
  font-size: 12px;
  color: #999;
}

.score-cell {
  text-align: center;
}

.score-value {
  font-size: 18px;
  font-weight: 700;
  margin-bottom: 4px;
}

.score-value.excellent {
  color: #52c41a;
}

.score-value.good {
  color: #1890ff;
}

.score-value.fair {
  color: #faad14;
}

.score-value.pass {
  color: #fa8c16;
}

.score-value.fail {
  color: #ff4d4f;
}

.score-level {
  font-size: 12px;
  color: #666;
}

.rank-cell {
  text-align: center;
}

.rank-value {
  font-size: 14px;
  font-weight: 500;
  margin-bottom: 8px;
}

.gpa-cell {
  text-align: center;
}

.gpa-value {
  font-size: 16px;
  font-weight: 600;
  color: #333;
}

.gpa-points {
  font-size: 12px;
  color: #666;
}

.trend-cell {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 4px;
}

.trend-icon.positive {
  color: #52c41a;
}

.trend-icon.negative {
  color: #ff4d4f;
}

.trend-icon.neutral {
  color: #999;
}

/* 卡片视图样式 */
.grades-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(400px, 1fr));
  gap: 20px;
}

.grade-card {
  border: 1px solid #f0f0f0;
  border-radius: 12px;
  padding: 20px;
  transition: all 0.3s ease;
}

.grade-card:hover {
  box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1);
  transform: translateY(-2px);
}

.grade-card.excellent {
  border-left: 4px solid #52c41a;
}

.grade-card.good {
  border-left: 4px solid #1890ff;
}

.grade-card.fair {
  border-left: 4px solid #faad14;
}

.grade-card.pass {
  border-left: 4px solid #fa8c16;
}

.grade-card.fail {
  border-left: 4px solid #ff4d4f;
}

.grade-header {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  margin-bottom: 16px;
}

.grade-header .course-name {
  font-size: 18px;
  font-weight: 600;
  color: #333;
  margin: 0 0 8px 0;
}

.grade-header .course-meta {
  display: flex;
  gap: 12px;
  font-size: 12px;
  color: #666;
}

.grade-score {
  text-align: right;
}

.grade-score .score-value {
  font-size: 32px;
  font-weight: 700;
  line-height: 1;
}

.grade-details {
  margin-bottom: 16px;
}

.detail-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 8px;
  font-size: 14px;
}

.detail-label {
  color: #666;
}

.detail-value {
  color: #333;
  font-weight: 500;
}

.detail-value.gpa {
  color: #1890ff;
  font-weight: 600;
}

.rank-progress {
  width: 100px;
  margin-left: 12px;
}

.grade-footer {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding-top: 16px;
  border-top: 1px solid #f0f0f0;
}

.grade-trend {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 12px;
}

/* 成绩详情弹窗 */
.grade-detail {
  padding: 20px 0;
}

.detail-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 32px;
  padding-bottom: 20px;
  border-bottom: 1px solid #f0f0f0;
}

.detail-header h2 {
  font-size: 24px;
  font-weight: 600;
  margin: 0 0 8px 0;
}

.score-display {
  text-align: right;
}

.main-score {
  font-size: 48px;
  font-weight: 700;
  color: #1890ff;
  line-height: 1;
}

.detail-sections .section {
  margin-bottom: 32px;
}

.detail-sections .section h3 {
  font-size: 16px;
  font-weight: 600;
  margin-bottom: 16px;
  color: #333;
}

.score-breakdown {
  background: #fafafa;
  border-radius: 8px;
  padding: 16px;
}

.breakdown-item {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 12px;
}

.breakdown-item:last-child {
  margin-bottom: 0;
}

.breakdown-info {
  display: flex;
  gap: 12px;
  align-items: center;
}

.breakdown-name {
  font-weight: 500;
}

.breakdown-weight {
  font-size: 12px;
  color: #666;
}

.breakdown-score {
  display: flex;
  align-items: center;
  gap: 12px;
}

.breakdown-score .score {
  font-weight: 600;
  color: #1890ff;
  min-width: 40px;
  text-align: right;
}

.score-bar {
  width: 100px;
  height: 6px;
  background: #f0f0f0;
  border-radius: 3px;
  overflow: hidden;
}

.score-progress {
  height: 100%;
  background: linear-gradient(90deg, #1890ff, #36cfc9);
  border-radius: 3px;
  transition: width 0.3s ease;
}

.stats-grid {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 16px;
}

.stat-item {
  background: #fafafa;
  border-radius: 8px;
  padding: 16px;
  text-align: center;
}

.stat-label {
  font-size: 12px;
  color: #666;
  margin-bottom: 8px;
}

.stat-value {
  font-size: 20px;
  font-weight: 600;
  color: #333;
}

.feedback {
  background: #f6ffed;
  border: 1px solid #b7eb8f;
  border-radius: 8px;
  padding: 16px;
  color: #333;
  line-height: 1.6;
}

/* 响应式设计 */
@media (max-width: 1200px) {
  .grades-overview {
    grid-template-columns: repeat(2, 1fr);
  }
  
  .grades-charts {
    grid-template-columns: 1fr;
  }
  
  .grades-grid {
    grid-template-columns: repeat(auto-fill, minmax(350px, 1fr));
  }
}

@media (max-width: 768px) {
  .student-grades {
    padding: 16px;
  }
  
  .page-header {
    flex-direction: column;
    gap: 16px;
    text-align: center;
    padding: 24px;
  }
  
  .grades-overview {
    grid-template-columns: 1fr;
    gap: 16px;
  }
  
  .filter-section {
    flex-direction: column;
    gap: 16px;
    align-items: stretch;
  }
  
  .filter-controls {
    flex-wrap: wrap;
    gap: 8px;
  }
  
  .grades-grid {
    grid-template-columns: 1fr;
  }
  
  .stats-grid {
    grid-template-columns: repeat(2, 1fr);
  }
  
  .detail-header {
    flex-direction: column;
    text-align: center;
    gap: 16px;
  }
}
</style> 