import { createRouter, createWebHistory, type RouteRecordRaw, type NavigationGuardNext, type RouteLocationNormalized } from 'vue-router'
import { useAuthStore } from '@/stores/auth'
import '@/types/router.d.ts'

// 确保类型声明被正确应用
declare module 'vue-router' {
  interface RouteMeta {
    requiresAuth?: boolean
    role?: string
    mode?: string
  }
}

// 布局组件
const AuthLayout = () => import('@/components/layout/AuthLayout.vue')
const TeacherLayout = () => import('@/components/layout/TeacherLayout.vue')
const StudentLayout = () => import('@/components/layout/StudentLayout.vue')
const CourseLayout = () => import('@/components/layout/CourseLayout.vue')

// 认证相关页面
const Login = () => import('@/views/auth/Login.vue')

// 主页
const HomePage = () => import('@/components/HomePage.vue')

// 公共页面
const CourseList = () => import('@/views/CourseList.vue')

// 教师端页面
const TeacherDashboard = () => import('@/views/teacher/Dashboard.vue')

// 教师端 - 个人资料
const TeacherProfile = () => import('../views/teacher/Profile.vue')

// 教师端 - 班级管理
const TeacherClasses = () => import('@/views/teacher/Classes.vue')
const TeacherClassDetail = () => import('@/views/teacher/ClassDetail.vue')

// 教师端 - 学生管理
const TeacherStudents = () => import('@/views/teacher/Students.vue')
const TeacherStudentDetail = () => import('@/views/teacher/StudentDetail.vue')

// 教师端 - 课程管理
const TeacherCourses = () => import('@/views/teacher/Courses.vue')
const TeacherCourseDetail = () => import('@/views/teacher/CourseDetail.vue')
const TeacherSectionDetail = () => import('@/views/teacher/SectionDetail.vue')

// 教师端 - 任务管理
const TeacherTasks = () => import('@/views/teacher/Tasks.vue')
const TeacherTaskDetail = () => import('@/views/teacher/TaskDetail.vue')

// 教师端 - 考试管理
const TeacherExams = () => import('@/views/teacher/Exams.vue')
const TeacherExamDetail = () => import('@/views/teacher/ExamDetail.vue')

// 教师端 - 作业管理
const TeacherAssignments = () => import('@/views/teacher/Assignments.vue')
const TeacherAssignmentDetail = () => import('@/views/teacher/AssignmentDetail.vue')

// 教师端 - 成绩管理
const TeacherGrades = () => import('@/views/teacher/Grades.vue')

// 教师端 - 资源管理
const TeacherResources = () => import('@/views/teacher/Resources.vue')
const TeacherResourceDetail = () => import('@/views/teacher/ResourceDetail.vue')

// 教师端 - 知识图谱
const TeacherKnowledgeGraph = () => import('@/views/teacher/KnowledgeGraph.vue')
const TeacherKnowledgeGraphGenerator = () => import('@/views/teacher/KnowledgeGraphGenerator.vue')

// 教师端 - 智能组卷
const TeacherSmartPaperGeneration = () => import('@/views/teacher/SmartPaperGeneration.vue')

// 教师端 - 智能批改
const TeacherSmartGrading = () => import('@/views/teacher/SmartGrading.vue')

// 教师端 - 题库管理
const TeacherQuestionBank = () => import('@/views/teacher/QuestionBank.vue')
const TeacherQuestionDetail = () => import('@/views/teacher/QuestionDetail.vue')

// 教师端 - 数据分析
const TeacherDataAnalysis = () => import('@/views/teacher/DataAnalysis.vue')

// 学生端页面
const StudentDashboard = () => import('@/views/student/Dashboard.vue')

// 学生端 - 课程管理
const StudentCourses = () => import('@/views/student/Courses.vue')
const StudentCourseDetail = () => import('@/views/student/CourseDetail.vue')
const StudentVideoLearning = () => import('@/views/student/VideoLearning.vue')

// 学生端 - 作业管理
const StudentAssignments = () => import('@/views/student/Assignments.vue')
const StudentAssignmentDetail = () => import('@/views/student/AssignmentDetail.vue')
const StudentFileSubmission = () => import('@/views/student/FileSubmission.vue')

// 学生端 - 错题集
const StudentWrongQuestions = () => import('@/views/student/WrongQuestions.vue')

// 学生端 - 学习记录
const StudentLearningRecords = () => import('@/views/student/LearningRecords.vue')

// 学生端 - 考试管理
const StudentExamDetail = () => import('@/views/student/ExamDetail.vue')
const StudentExamDo = () => import('@/views/student/ExamDo.vue')

// 学生端 - 成绩查看
const StudentGrades = () => import('@/views/student/Grades.vue')

// 学生端 - 资源管理
const StudentResources = () => import('@/views/student/Resources.vue')
const StudentResourceDetail = () => import('@/views/student/ResourceDetail.vue')

// 学生端 - 知识图谱
const StudentKnowledgeGraph = () => import('@/views/student/KnowledgeGraph.vue')
const StudentKnowledgeGraphViewer = () => import('@/views/student/KnowledgeGraphViewer.vue')

// 学生端 - 能力图谱
const StudentAbilityGraph = () => import('@/views/student/AbilityGraph.vue')

// 学生端 - 个性化练习
const StudentPersonalizedPractice = () => import('@/views/student/PersonalizedPractice.vue')

// 学生端 - 个性化学习路径
const StudentLearningPathway = () => import('@/views/student/LearningPathway.vue')

// 学生端 - AI学习助手
const StudentAITutor = () => import('@/views/student/AITutor.vue')

// 学生端 - 班级管理
const StudentClasses = () => import('@/views/student/Classes.vue')
const StudentProfile = () => import('@/views/student/Profile.vue')
const StudentSettings = () => import('@/views/student/Settings.vue')
const StudentSchedule = () => import('@/views/student/Schedule.vue')

// 子页面组件
const AllAssignments = () => import('@/views/student/assignments/AllAssignments.vue')
const TeacherAllAssignments = () => import('../views/teacher/assignments/AllAssignments.vue')
const TodoAssignments = () => import('@/views/student/assignments/TodoAssignments.vue')
const CompletedAssignments = () => import('@/views/student/assignments/CompletedAssignments.vue')
const ClassInfo = () => import('@/views/student/classes/ClassInfo.vue')
const ClassMembers = () => import('@/views/student/classes/ClassMembers.vue')
const ResourceLibrary = () => import('@/views/student/resources/ResourceLibrary.vue')

// 添加路由引用
const StudentSectionDetail = () => import('@/views/teacher/SectionDetail.vue')

const routes: RouteRecordRaw[] = [
  {
    path: '/',
    redirect: '/home'
  },
  {
    path: '/home',
    name: 'Home',
    component: HomePage,
    meta: { requiresAuth: false }
  },
  
  // 公共课程列表
  {
    path: '/courses',
    name: 'CourseList',
    component: CourseList,
    meta: { requiresAuth: false }
  },
  
  // 课程详情页面
  {
    path: '/courses/:id',
    name: 'CourseDetail',
    component: CourseLayout,
    meta: { requiresAuth: false },
    children: [
      {
        path: '',
        component: StudentCourseDetail,
        props: true
      }
    ]
  },
  
  // 认证相关路由
  {
    path: '/login',
    name: 'Login',
    component: Login,
    meta: { requiresAuth: false, mode: 'login' }
  },
  {
    path: '/register',
    name: 'Register',
    component: Login,
    meta: { requiresAuth: false, mode: 'register' }
  },

  // 课程章节页面 - 使用独立布局
  {
    path: '/teacher/courses/:courseId/sections/:sectionId',
    name: 'TeacherSectionDetail',
    component: CourseLayout,
    meta: { requiresAuth: true, role: 'TEACHER' },
    children: [
      {
        path: '',
        component: TeacherSectionDetail,
        props: true
      }
    ]
  },

  // 学生章节页面
  {
    path: '/student/courses/:courseId/sections/:sectionId',
    name: 'StudentSectionDetail',
    component: CourseLayout,
    meta: { requiresAuth: true, role: 'STUDENT', viewOnly: true },
    children: [
      {
        path: '',
        component: StudentSectionDetail,
        props: true
      }
    ]
  },

  // 教师端路由
  {
    path: '/teacher',
    component: TeacherLayout,
    meta: { requiresAuth: true, role: 'TEACHER' },
    children: [
      {
        path: '',
        redirect: '/teacher/dashboard'
      },
      {
        path: 'dashboard',
        name: 'TeacherDashboard',
        component: TeacherDashboard
      },
      
      // 个人资料
      {
        path: 'profile',
        name: 'TeacherProfile',
        component: TeacherProfile
      },
      
      // 班级管理
      {
        path: 'classes',
        name: 'TeacherClasses',
        component: TeacherClasses
      },
      {
        path: 'classes/:id',
        name: 'TeacherClassDetail',
        component: TeacherClassDetail,
        props: true
      },
      
      // 学生管理
      {
        path: 'students',
        name: 'TeacherStudents',
        component: TeacherStudents
      },
      {
        path: 'students/:id',
        name: 'TeacherStudentDetail',
        component: TeacherStudentDetail,
        props: true
      },
      
      // 课程管理
      {
        path: 'courses',
        name: 'TeacherCourses',
        component: TeacherCourses
      },
      {
        path: 'courses/:id',
        name: 'TeacherCourseDetail',
        component: TeacherCourseDetail,
        props: true
      },
      
      // 任务管理
      {
        path: 'tasks',
        name: 'TeacherTasks',
        component: TeacherTasks
      },
      {
        path: 'tasks/:id',
        name: 'TeacherTaskDetail',
        component: TeacherTaskDetail,
        props: true
      },
      
      // 考试管理
      {
        path: 'exams',
        name: 'TeacherExams',
        component: TeacherExams
      },
      {
        path: 'exams/:id',
        name: 'TeacherExamDetail',
        component: TeacherExamDetail,
        props: true
      },
      
      // 作业管理
      {
        path: 'assignments/:id(\\d+)',
        name: 'TeacherAssignmentDetail',
        component: TeacherAssignmentDetail,
        props: true
      },
      {
        path: 'assignments',
        name: 'TeacherAssignments',
        component: TeacherAssignments,
        children: [
          {
            path: '',
            name: 'TeacherAssignmentsDefault',
            component: TeacherAllAssignments
          },
          {
            path: 'all',
            name: 'AllAssignments',
            component: TeacherAllAssignments
          },
          {
            path: 'todo',
            name: 'TodoAssignments',
            component: TodoAssignments
          },
          {
            path: 'completed',
            name: 'CompletedAssignments',
            component: CompletedAssignments
          },
          {
            path: 'create',
            name: 'CreateAssignment',
            component: () => import('@/views/teacher/CreateAssignment.vue'),
            meta: {
              title: '创建作业',
              requiresAuth: true,
              roles: ['TEACHER']
            }
          },
          {
            path: ':id/edit',
            name: 'EditAssignment',
            component: () => import('@/views/teacher/CreateAssignment.vue'),
            meta: {
              title: '编辑作业',
              requiresAuth: true,
              roles: ['TEACHER']
            }
          }
        ]
      },
      
      // 成绩管理
      {
        path: 'grades',
        name: 'TeacherGrades',
        component: TeacherGrades
      },
      
      // 资源管理
      {
        path: 'resources',
        name: 'TeacherResources',
        component: TeacherResources
      },
      {
        path: 'resources/:id',
        name: 'TeacherResourceDetail',
        component: TeacherResourceDetail,
        props: true
      },
      
      // 知识图谱
      {
        path: 'knowledge-graph',
        name: 'TeacherKnowledgeGraph',
        component: TeacherKnowledgeGraph
      },
      {
        path: 'knowledge-graph/generator',
        name: 'TeacherKnowledgeGraphGenerator',
        component: TeacherKnowledgeGraphGenerator
      },
      
      // 智能组卷
      {
        path: 'smart-paper-generation',
        name: 'TeacherSmartPaperGeneration',
        component: TeacherSmartPaperGeneration
      },
      
      // 智能批改
      {
        path: 'smart-grading',
        name: 'TeacherSmartGrading',
        component: TeacherSmartGrading
      },
      
      // 题库管理
      {
        path: 'question-bank',
        name: 'TeacherQuestionBank',
        component: TeacherQuestionBank
      },
      {
        path: 'question-bank/:id',
        name: 'TeacherQuestionDetail',
        component: TeacherQuestionDetail,
        props: true
      },
      
      // 数据分析
      {
        path: 'data-analysis',
        name: 'TeacherDataAnalysis',
        component: TeacherDataAnalysis
      }
    ]
  },

  // 学生端路由
  {
    path: '/student',
    component: StudentLayout,
    meta: { requiresAuth: true, role: 'STUDENT' },
    children: [
      {
        path: '',
        redirect: '/student/dashboard'
      },
      {
        path: 'dashboard',
        name: 'StudentDashboard',
        component: StudentDashboard
      },
      
      // 课程相关
      {
        path: 'courses',
        name: 'StudentCourses',
        component: StudentCourses
      },
      {
        path: 'courses/:id',
        name: 'StudentCourseDetail',
        component: StudentCourseDetail,
        props: true
      },
      {
        path: 'courses/:courseId/videos/:videoId',
        name: 'StudentVideoLearning',
        component: StudentVideoLearning,
        props: true
      },
      
      // 作业管理
      {
        path: 'assignments/:id(\\d+)',
        name: 'StudentAssignmentDetail',
        component: StudentAssignmentDetail,
        props: true
      },
      {
        path: 'assignments',
        name: 'StudentAssignments',
        component: StudentAssignments,
        children: [
          {
            path: '',
            name: 'StudentAssignmentsDefault',
            redirect: '/student/assignments/all'
          },
          {
            path: 'all',
            name: 'AllAssignments',
            component: AllAssignments
          },
          {
            path: 'todo',
            name: 'TodoAssignments',
            component: TodoAssignments
          },
          {
            path: 'completed',
            name: 'CompletedAssignments',
            component: CompletedAssignments
          }
        ]
      },
      
      // 文件提交作业
      {
        path: 'assignments/file/:id/submit',
        name: 'StudentFileSubmission',
        component: StudentFileSubmission,
        props: true,
        alias: '/student/assignments/file/:id/submit'
      },
      
      // 错题集
      {
        path: 'wrong-questions',
        name: 'StudentWrongQuestions',
        component: StudentWrongQuestions
      },
      
      // 考试
      {
        path: 'exams/:id',
        name: 'StudentExamDetail',
        component: StudentExamDetail,
        props: true
      },
      {
        path: 'exams/:id/do',
        name: 'StudentExamDo',
        component: StudentExamDo,
        props: true
      },
      
      // 学习记录
      {
        path: 'learning-records',
        name: 'StudentLearningRecords',
        component: StudentDashboard,  // 临时使用Dashboard作为占位符
      },
      
      // 成绩查看
      {
        path: 'grades',
        name: 'StudentGrades',
        component: StudentGrades
      },
      
      // 资源管理
      {
        path: 'resources',
        name: 'StudentResources',
        component: StudentResources,
        children: [
          {
            path: '',
            name: 'StudentResourcesDefault',
            redirect: '/student/resources/library'
          },
          {
            path: 'library',
            name: 'ResourceLibrary',
            component: ResourceLibrary
          }
        ]
      },
      {
        path: 'resources/:id',
        name: 'StudentResourceDetail',
        component: StudentResourceDetail,
        props: true
      },
      
      // 知识图谱
      {
        path: 'knowledge-graph',
        name: 'StudentKnowledgeGraph',
        component: StudentKnowledgeGraph
      },
      {
        path: 'knowledge-graph/viewer',
        name: 'StudentKnowledgeGraphViewer',
        component: StudentKnowledgeGraphViewer
      },
      
      // 能力图谱
      {
        path: 'ability-graph',
        name: 'StudentAbilityGraph',
        component: StudentAbilityGraph
      },
      
      // AI学习助手
      {
        path: 'ai-tutor',
        name: 'StudentAITutor',
        component: StudentAITutor
      },
      
      // 个性化练习
      {
        path: 'personalized-practice',
        name: 'StudentPersonalizedPractice',
        component: StudentPersonalizedPractice
      },
      
      // 个性化学习路径
      {
        path: 'learning-pathway',
        name: 'StudentLearningPathway',
        component: StudentLearningPathway
      },
      
      // 班级管理
      {
        path: 'classes',
        name: 'StudentClasses',
        component: StudentClasses,
        children: [
          {
            path: '',
            name: 'StudentClassesDefault',
            redirect: '/student/classes/info'
          },
          {
            path: 'info',
            name: 'ClassInfo',
            component: ClassInfo
          },
          {
            path: 'members',
            name: 'ClassMembers',
            component: ClassMembers
          }
        ]
      },
      
      // 其他功能
      {
        path: 'schedule',
        name: 'StudentSchedule',
        component: StudentSchedule
      },
      {
        path: 'profile',
        name: 'StudentProfile',
        component: StudentProfile
      },
      {
        path: 'settings',
        name: 'StudentSettings',
        component: StudentSettings
      }
    ]
  },

  // 404页面
  {
    path: '/:pathMatch(.*)*',
    name: 'NotFound',
    redirect: (to) => {
      console.error('路由未找到:', to.path)
      console.log('未匹配路由的完整信息:', to)
      // 如果URL中包含student或teacher，则重定向到相应的首页
      if (to.path.includes('/student')) {
        return '/student/dashboard'
      } else if (to.path.includes('/teacher')) {
        return '/teacher/dashboard'
      } else {
        // 否则重定向到通用首页
        return '/home'
      }
    }
  },
  
  // 直接路径映射 - 文件提交页面
  {
    path: '/student/assignments/file/:id/submit',
    component: StudentFileSubmission,
    props: true,
    meta: { requiresAuth: true, role: 'STUDENT' }
  }
]

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes
})

// 路由守卫 - 优化版本，减少不必要的API调用
router.beforeEach(async (to, from, next) => {
  console.log('🚦 路由守卫触发:', to)
  
  const authStore = useAuthStore()
  console.log('🔐 认证状态:', authStore.user)
  
  // 如果目标路由不需要认证，直接放行
  if (!to.meta.requiresAuth) {
    console.log('✅ 路由不需要认证，直接放行')
    return next()
  }
  
  // 检查认证状态
  if (authStore.isAuthenticated) {
    // 已登录状态，检查角色权限
    if (to.meta.role && authStore.user?.role.toUpperCase() !== to.meta.role) {
      console.log('⛔ 用户角色不匹配，无权访问')
      return next('/login')
    }
    console.log('✅ 路由守卫放行')
    return next()
  } else if (authStore.hasStoredAuth()) {
    // 如果本地有认证信息但状态未同步，恢复状态并放行
    await authStore.init()
    // 二次检查认证状态
    if (authStore.isAuthenticated) {
      // 已恢复登录状态，检查角色权限
      if (to.meta.role && authStore.user?.role.toUpperCase() !== to.meta.role) {
        console.log('⛔ 用户角色不匹配，无权访问')
        return next('/login')
      }
      console.log('✅ 路由守卫放行')
      return next()
    } else {
      console.log('⛔ 认证信息无效，重定向到登录页')
      return next('/login')
    }
  } else {
    // 未登录状态，重定向到登录页
    console.log('⛔ 用户未登录，重定向到登录页')
    return next('/login')
  }
})

export default router
