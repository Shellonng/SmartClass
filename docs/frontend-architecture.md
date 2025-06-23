# 前端架构设计文档

## 一、页面路由设计

### 1.1 路由结构总览

```javascript
// router/index.js
const routes = [
  {
    path: '/login',
    name: 'Login',
    component: () => import('@/views/auth/Login.vue'),
    meta: { requiresAuth: false }
  },
  {
    path: '/',
    name: 'Home',
    component: () => import('@/views/Home.vue'),
    meta: { requiresAuth: true }
  },
  {
    path: '/teacher',
    component: () => import('@/layouts/TeacherLayout.vue'),
    meta: { requiresAuth: true, role: 'teacher' },
    children: [
      {
        path: 'dashboard',
        name: 'TeacherDashboard',
        component: () => import('@/views/teacher/Dashboard.vue')
      },
      {
        path: 'classes',
        name: 'ClassManagement',
        component: () => import('@/views/teacher/ClassManagement.vue')
      },
      {
        path: 'students',
        name: 'StudentManagement',
        component: () => import('@/views/teacher/StudentManagement.vue')
      },
      {
        path: 'courses',
        name: 'CourseManagement',
        component: () => import('@/views/teacher/CourseManagement.vue')
      },
      {
        path: 'tasks',
        name: 'TaskManagement',
        component: () => import('@/views/teacher/TaskManagement.vue')
      },
      {
        path: 'grades',
        name: 'GradeManagement',
        component: () => import('@/views/teacher/GradeManagement.vue')
      },
      {
        path: 'resources',
        name: 'ResourceManagement',
        component: () => import('@/views/teacher/ResourceManagement.vue')
      },
      {
        path: 'knowledge',
        name: 'KnowledgeGraph',
        component: () => import('@/views/teacher/KnowledgeGraph.vue')
      },
      {
        path: 'ai-innovation',
        name: 'AIInnovation',
        component: () => import('@/views/teacher/AIInnovation.vue')
      }
    ]
  },
  {
    path: '/student',
    component: () => import('@/layouts/StudentLayout.vue'),
    meta: { requiresAuth: true, role: 'student' },
    children: [
      {
        path: 'dashboard',
        name: 'StudentDashboard',
        component: () => import('@/views/student/Dashboard.vue')
      },
      {
        path: 'courses',
        name: 'StudentCourses',
        component: () => import('@/views/student/Courses.vue')
      },
      {
        path: 'tasks',
        name: 'StudentTasks',
        component: () => import('@/views/student/Tasks.vue')
      },
      {
        path: 'grades',
        name: 'StudentGrades',
        component: () => import('@/views/student/Grades.vue')
      },
      {
        path: 'resources',
        name: 'StudentResources',
        component: () => import('@/views/student/Resources.vue')
      },
      {
        path: 'knowledge',
        name: 'StudentKnowledge',
        component: () => import('@/views/student/KnowledgeGraph.vue')
      },
      {
        path: 'ai-learning',
        name: 'AILearning',
        component: () => import('@/views/student/AILearning.vue')
      }
    ]
  },
  {
    path: '/404',
    name: 'NotFound',
    component: () => import('@/views/error/404.vue')
  },
  {
    path: '/:pathMatch(.*)*',
    redirect: '/404'
  }
]
```

## 二、页面详细设计

### 2.1 登录页面 (/login)

**功能目标：** 用户身份验证，支持教师和学生登录

**页面主流程：**
1. 用户选择角色（教师/学生）
2. 输入账号密码/验证码
3. 登录成功后跳转到对应角色主页

**组件树结构：**
```
Login.vue
├── LoginForm.vue
│   ├── RoleSelector.vue
│   ├── AccountInput.vue
│   ├── PasswordInput.vue
│   ├── CaptchaInput.vue
│   └── LoginButton.vue
├── RegisterModal.vue
└── ForgetPasswordModal.vue
```

**组件详细说明：**
- `LoginForm.vue`: 主登录表单
  - Props: { mode: 'login' | 'register' }
  - Events: @submit, @switchMode
  - 交互: 表单验证、提交登录请求

- `RoleSelector.vue`: 角色选择器
  - Props: { value: string }
  - Events: @change
  - 交互: 单选按钮，切换教师/学生模式

**API调用链：**
```javascript
// 登录流程
POST /api/auth/login
→ 成功: 存储token，跳转到角色主页
→ 失败: 显示错误信息

// 获取验证码
GET /api/auth/captcha
```

**AI功能占位：**
- 登录页底部预留"智能登录助手"按钮
- 支持人脸识别登录（占位）

### 2.2 教师端 - 班级管理 (/teacher/classes)

**功能目标：** 管理班级信息，查看班级学生列表

**页面主流程：**
1. 查看班级列表（表格形式）
2. 点击"新建班级"弹出表单
3. 点击班级名称查看详情（抽屉形式）

**组件树结构：**
```
ClassManagement.vue
├── PageHeader.vue
├── ClassTable.vue
│   ├── TableToolbar.vue
│   │   ├── SearchInput.vue
│   │   ├── FilterSelect.vue
│   │   └── ActionButtons.vue
│   └── ClassRow.vue
├── ClassModal.vue
│   └── ClassForm.vue
├── ClassDrawer.vue
│   ├── ClassInfo.vue
│   ├── StudentList.vue
│   └── StudentActions.vue
└── AIFeatureCard.vue (占位)
```

**组件详细说明：**
- `ClassTable.vue`: 班级列表表格
  - Props: { data: Array, loading: Boolean }
  - Events: @edit, @delete, @view
  - 交互: 分页、排序、筛选

- `ClassModal.vue`: 新建/编辑班级弹窗
  - Props: { visible: Boolean, editData: Object }
  - Events: @submit, @cancel
  - 交互: 表单验证、提交保存

**API调用链：**
```javascript
// 获取班级列表
GET /api/teacher/classes?page=1&size=10&keyword=

// 创建班级
POST /api/teacher/classes

// 获取班级详情
GET /api/teacher/classes/{id}

// 获取班级学生列表
GET /api/teacher/classes/{id}/students
```

**AI功能占位：**
- 右侧固定"智能班级分析"卡片
- 支持智能分班建议（占位按钮）

### 2.3 教师端 - 成绩管理 (/teacher/grades)

**功能目标：** 查看和分析学生成绩，生成个性化反馈

**页面主流程：**
1. 选择课程/班级筛选成绩
2. 查看成绩列表和统计图表
3. 点击学生查看详细分析

**组件树结构：**
```
GradeManagement.vue
├── GradeFilter.vue
│   ├── CourseSelect.vue
│   ├── ClassSelect.vue
│   └── TimeRangePicker.vue
├── GradeOverview.vue
│   ├── StatisticCards.vue
│   └── GradeChart.vue
├── GradeTable.vue
├── GradeDetailModal.vue
│   ├── StudentInfo.vue
│   ├── GradeTrend.vue
│   └── FeedbackGenerator.vue
└── AIAnalysisPanel.vue (占位)
```

**组件详细说明：**
- `GradeChart.vue`: 成绩可视化图表
  - Props: { data: Array, type: 'bar' | 'line' | 'pie' }
  - 使用ECharts渲染图表
  - 支持柱状图、折线图、饼图切换

- `FeedbackGenerator.vue`: 个性化反馈生成器
  - Props: { studentId: String, gradeData: Object }
  - Events: @generate, @save
  - 交互: AI辅助生成反馈内容（占位）

**API调用链：**
```javascript
// 获取成绩统计
GET /api/teacher/grades/statistics?courseId=&classId=

// 获取成绩列表
GET /api/teacher/grades?courseId=&classId=&page=1&size=10

// 获取学生成绩详情
GET /api/teacher/grades/student/{studentId}

// 生成个性化反馈
POST /api/teacher/grades/feedback
```

**AI功能占位：**
- 右侧"AI成绩分析"面板
- "智能反馈生成"按钮
- "学习能力预测"图表区域

### 2.4 学生端 - 我的作业 (/student/tasks)

**功能目标：** 查看和提交作业任务

**页面主流程：**
1. 查看作业列表（按状态分类）
2. 点击作业查看详情
3. 在线提交作业或上传文件

**组件树结构：**
```
Tasks.vue
├── TaskFilter.vue
├── TaskTabs.vue
│   ├── PendingTasks.vue
│   ├── SubmittedTasks.vue
│   └── CompletedTasks.vue
├── TaskCard.vue
├── TaskDetailModal.vue
│   ├── TaskDescription.vue
│   ├── TaskResources.vue
│   ├── SubmissionForm.vue
│   └── FileUpload.vue
└── AIAssistant.vue (占位)
```

**组件详细说明：**
- `TaskCard.vue`: 作业卡片
  - Props: { task: Object }
  - Events: @view, @submit
  - 交互: 显示作业基本信息、状态、截止时间

- `SubmissionForm.vue`: 作业提交表单
  - Props: { taskId: String }
  - Events: @submit
  - 交互: 文本输入、文件上传、提交验证

**API调用链：**
```javascript
// 获取学生作业列表
GET /api/student/tasks?status=&courseId=

// 获取作业详情
GET /api/student/tasks/{id}

// 提交作业
POST /api/student/tasks/{id}/submit

// 上传作业文件
POST /api/student/tasks/{id}/upload
```

**AI功能占位：**
- 底部固定"AI学习助手"按钮
- 作业智能提醒功能
- 作业完成度预测

## 三、公共组件设计

### 3.1 布局组件

```
layouts/
├── TeacherLayout.vue    # 教师端布局
├── StudentLayout.vue    # 学生端布局
└── AuthLayout.vue       # 认证页面布局
```

### 3.2 通用组件

```
components/
├── common/
│   ├── AppHeader.vue        # 顶部导航
│   ├── AppSidebar.vue       # 侧边栏
│   ├── AppFooter.vue        # 页脚
│   ├── LoadingSpinner.vue   # 加载动画
│   └── EmptyState.vue       # 空状态
├── form/
│   ├── FormModal.vue        # 表单弹窗
│   ├── SearchInput.vue      # 搜索输入框
│   ├── DateRangePicker.vue  # 日期范围选择
│   └── FileUpload.vue       # 文件上传
├── table/
│   ├── DataTable.vue        # 数据表格
│   ├── TableToolbar.vue     # 表格工具栏
│   └── TablePagination.vue  # 分页组件
├── chart/
│   ├── BarChart.vue         # 柱状图
│   ├── LineChart.vue        # 折线图
│   ├── PieChart.vue         # 饼图
│   └── ChartContainer.vue   # 图表容器
└── ai/
    ├── AIFeatureCard.vue    # AI功能卡片
    ├── AIAssistant.vue      # AI助手
    └── AIPlaceholder.vue    # AI占位组件
```

## 四、状态管理设计 (Pinia)

### 4.1 Store结构

```javascript
// store/index.js
import { createPinia } from 'pinia'
import { useAuthStore } from './modules/auth'
import { useUserStore } from './modules/user'
import { useClassStore } from './modules/class'
import { useCourseStore } from './modules/course'
import { useTaskStore } from './modules/task'
import { useGradeStore } from './modules/grade'
import { useResourceStore } from './modules/resource'
import { useAIStore } from './modules/ai'

const pinia = createPinia()

export {
  useAuthStore,
  useUserStore,
  useClassStore,
  useCourseStore,
  useTaskStore,
  useGradeStore,
  useResourceStore,
  useAIStore
}

export default pinia
```

### 4.2 认证Store示例

```javascript
// store/modules/auth.js
import { defineStore } from 'pinia'
import { login, logout, refreshToken } from '@/api/auth'

export const useAuthStore = defineStore('auth', {
  state: () => ({
    token: localStorage.getItem('token') || '',
    user: null,
    isAuthenticated: false,
    role: null
  }),
  
  getters: {
    isTeacher: (state) => state.role === 'teacher',
    isStudent: (state) => state.role === 'student'
  },
  
  actions: {
    async login(credentials) {
      try {
        const response = await login(credentials)
        this.token = response.data.token
        this.user = response.data.user
        this.role = response.data.user.role
        this.isAuthenticated = true
        localStorage.setItem('token', this.token)
        return response
      } catch (error) {
        throw error
      }
    },
    
    async logout() {
      try {
        await logout()
        this.token = ''
        this.user = null
        this.isAuthenticated = false
        this.role = null
        localStorage.removeItem('token')
      } catch (error) {
        console.error('Logout error:', error)
      }
    }
  }
})
```

## 五、API接口设计

### 5.1 API模块结构

```
api/
├── index.js           # API配置和拦截器
├── auth.js           # 认证相关接口
├── user.js           # 用户管理接口
├── class.js          # 班级管理接口
├── course.js         # 课程管理接口
├── task.js           # 任务管理接口
├── grade.js          # 成绩管理接口
├── resource.js       # 资源管理接口
├── knowledge.js      # 知识图谱接口
└── ai.js             # AI功能接口
```

### 5.2 API配置示例

```javascript
// api/index.js
import axios from 'axios'
import { useAuthStore } from '@/store'
import { ElMessage } from 'element-plus'

const api = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL,
  timeout: 10000
})

// 请求拦截器
api.interceptors.request.use(
  (config) => {
    const authStore = useAuthStore()
    if (authStore.token) {
      config.headers.Authorization = `Bearer ${authStore.token}`
    }
    return config
  },
  (error) => {
    return Promise.reject(error)
  }
)

// 响应拦截器
api.interceptors.response.use(
  (response) => {
    return response.data
  },
  (error) => {
    if (error.response?.status === 401) {
      const authStore = useAuthStore()
      authStore.logout()
      router.push('/login')
    }
    ElMessage.error(error.response?.data?.message || '请求失败')
    return Promise.reject(error)
  }
)

export default api
```

## 六、响应式设计

### 6.1 断点设计

```scss
// styles/variables.scss
$breakpoints: (
  xs: 0,
  sm: 576px,
  md: 768px,
  lg: 992px,
  xl: 1200px,
  xxl: 1600px
);

// 响应式混入
@mixin respond-to($breakpoint) {
  @media (min-width: map-get($breakpoints, $breakpoint)) {
    @content;
  }
}
```

### 6.2 移动端适配

- 使用Ant Design Vue的栅格系统
- 表格在移动端转换为卡片布局
- 侧边栏在移动端改为抽屉模式
- 表单在移动端使用全屏弹窗

## 七、性能优化

### 7.1 代码分割

```javascript
// 路由懒加载
const routes = [
  {
    path: '/teacher/classes',
    component: () => import('@/views/teacher/ClassManagement.vue')
  }
]

// 组件懒加载
const LazyComponent = defineAsyncComponent(() =>
  import('@/components/HeavyComponent.vue')
)
```

### 7.2 缓存策略

- 使用keep-alive缓存页面组件
- API响应数据缓存
- 图片懒加载
- 虚拟滚动处理大列表

### 7.3 打包优化

```javascript
// vite.config.js
export default defineConfig({
  build: {
    rollupOptions: {
      output: {
        manualChunks: {
          vendor: ['vue', 'vue-router', 'pinia'],
          antd: ['ant-design-vue'],
          charts: ['echarts']
        }
      }
    }
  }
})
```

## 八、AI功能占位设计

### 8.1 占位组件

```vue
<!-- components/ai/AIPlaceholder.vue -->
<template>
  <div class="ai-placeholder">
    <div class="ai-icon">
      <Icon type="robot" />
    </div>
    <div class="ai-content">
      <h4>{{ title }}</h4>
      <p>{{ description }}</p>
      <Button type="primary" @click="handleClick">
        {{ buttonText }}
      </Button>
    </div>
  </div>
</template>
```

### 8.2 占位位置

- **页面右侧固定区域**: AI功能卡片
- **表格工具栏**: AI分析按钮
- **表单底部**: AI辅助填写
- **图表区域**: AI数据洞察
- **浮动按钮**: AI助手入口

### 8.3 预留接口

```javascript
// api/ai.js
export const aiAPI = {
  // 智能分析
  analyze: (data) => api.post('/api/ai/analyze', data),
  
  // 内容推荐
  recommend: (userId, type) => api.get(`/api/ai/recommend/${userId}/${type}`),
  
  // 智能批改
  autoGrade: (submissionId) => api.post(`/api/ai/grade/${submissionId}`),
  
  // 知识图谱生成
  generateKnowledgeGraph: (courseId) => api.post(`/api/ai/knowledge-graph/${courseId}`)
}
```

这个前端架构设计为团队开发和AI功能扩展提供了完整的基础框架，所有组件都采用模块化设计，便于维护和扩展。