import { createRouter, createWebHistory } from 'vue-router'
import { useAuthStore } from '@/stores/auth'

// 布局组件
const AuthLayout = () => import('@/components/layout/AuthLayout.vue')
const TeacherLayout = () => import('@/components/layout/TeacherLayout.vue')
const StudentLayout = () => import('@/components/layout/StudentLayout.vue')
const CourseLayout = () => import('@/components/layout/CourseLayout.vue')

// 认证相关页面
const Login = () => import('@/views/auth/Login.vue')

// 主页
const HomePage = () => import('@/components/HomePage.vue')

// 教师端页面
const TeacherDashboard = () => import('@/views/teacher/Dashboard.vue')

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

// 教师端 - 成绩管理
const TeacherGrades = () => import('@/views/teacher/Grades.vue')

// 教师端 - 资源管理
const TeacherResources = () => import('@/views/teacher/Resources.vue')
const TeacherResourceDetail = () => import('@/views/teacher/ResourceDetail.vue')

// 教师端 - 知识图谱
const TeacherKnowledgeGraph = () => import('@/views/teacher/KnowledgeGraph.vue')

// 教师端 - 题库管理
const TeacherQuestionBank = () => import('@/views/teacher/QuestionBank.vue')
const TeacherQuestionDetail = () => import('@/views/teacher/QuestionDetail.vue')

// 教师端 - AI工具
const TeacherAITools = () => import('@/views/teacher/AITools.vue')

// 学生端页面
const StudentDashboard = () => import('@/views/student/Dashboard.vue')

// 学生端 - 课程管理
const StudentCourses = () => import('@/views/student/Courses.vue')
const StudentCourseDetail = () => import('@/views/student/CourseDetail.vue')
const StudentVideoLearning = () => import('@/views/student/VideoLearning.vue')

// 学生端 - 作业管理
const StudentAssignments = () => import('@/views/student/Assignments.vue')
const StudentAssignmentDetail = () => import('@/views/student/AssignmentDetail.vue')

// 学生端 - 成绩查看
const StudentGrades = () => import('@/views/student/Grades.vue')

// 学生端 - 资源管理
const StudentResources = () => import('@/views/student/Resources.vue')
const StudentResourceDetail = () => import('@/views/student/ResourceDetail.vue')

// 学生端 - 知识图谱
const StudentKnowledgeGraph = () => import('@/views/student/KnowledgeGraph.vue')

// 学生端 - 能力图谱
const StudentAbilityGraph = () => import('@/views/student/AbilityGraph.vue')

// 学生端 - AI学习助手
const StudentAITutor = () => import('@/views/student/AITutor.vue')

// 学生端 - 其他页面
const StudentClasses = () => import('@/views/student/Classes.vue')
const StudentProfile = () => import('@/views/student/Profile.vue')
const StudentSettings = () => import('@/views/student/Settings.vue')
const StudentSchedule = () => import('@/views/student/Schedule.vue')

// 子页面组件
const AllAssignments = () => import('@/views/student/assignments/AllAssignments.vue')
const TodoAssignments = () => import('@/views/student/assignments/TodoAssignments.vue')
const CompletedAssignments = () => import('@/views/student/assignments/CompletedAssignments.vue')
const ClassInfo = () => import('@/views/student/classes/ClassInfo.vue')
const ClassMembers = () => import('@/views/student/classes/ClassMembers.vue')
const ResourceLibrary = () => import('@/views/student/resources/ResourceLibrary.vue')
const Favorites = () => import('@/views/student/resources/Favorites.vue')

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes: [
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
          path: 'courses/create',
          name: 'TeacherCourseCreate',
          component: TeacherCourseDetail,
          props: { mode: 'create' }
        },
        {
          path: 'courses/:id',
          name: 'TeacherCourseDetail',
          component: TeacherCourseDetail,
          props: true
        },
        {
          path: 'courses/chapters',
          name: 'TeacherCourseChapters',
          component: TeacherCourses,
          props: { mode: 'chapters' }
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
        
        // AI工具
        {
          path: 'ai-tools',
          name: 'TeacherAITools',
          component: TeacherAITools
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
        
        // 课程管理
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
          path: 'courses/:id/video/:videoId',
          name: 'StudentVideoLearning',
          component: StudentVideoLearning,
          props: true
        },
        
        // 作业管理
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
        {
          path: 'assignments/:id',
          name: 'StudentAssignmentDetail',
          component: StudentAssignmentDetail,
          props: true
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
            },
            {
              path: 'favorites',
              name: 'Favorites',
              component: Favorites
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
      redirect: '/home'
    }
  ]
})

// 路由守卫
router.beforeEach(async (to, from, next) => {
  const authStore = useAuthStore()
  
  console.log('路由守卫:', { 
    to: to.path, 
    from: from.path, 
    requiresAuth: to.meta.requiresAuth,
    role: to.meta.role,
    userRole: authStore.user?.role,
    isAuthenticated: authStore.isAuthenticated
  })
  
  // 如果有token或sessionId但没有用户信息，尝试获取用户信息
  if ((localStorage.getItem('token') || localStorage.getItem('sessionId')) && !authStore.user) {
    try {
      console.log('尝试获取用户信息...')
      await authStore.fetchUserInfo()
    } catch (error) {
      console.error('路由守卫中获取用户信息失败:', error)
    }
  }
  
  // 如果已登录用户访问登录页，重定向到对应首页
  if ((to.path === '/login' || to.path === '/register') && authStore.isAuthenticated) {
    const redirectPath = authStore.user?.role === 'TEACHER' ? '/teacher/dashboard' : '/student/dashboard'
    console.log('已登录用户访问登录页，重定向到:', redirectPath)
    next(redirectPath)
    return
  }
  
  // 检查是否需要认证
  if (to.meta.requiresAuth && !authStore.isAuthenticated) {
    console.log('需要认证但未登录，重定向到登录页')
    next('/login')
    return
  }
  
  // 检查角色权限
  if (to.meta.role && authStore.user?.role && to.meta.role !== authStore.user.role) {
    // 如果角色不匹配，重定向到对应角色的首页
    const redirectPath = authStore.user.role === 'TEACHER' ? '/teacher/dashboard' : '/student/dashboard'
    console.log('角色不匹配，重定向到:', redirectPath)
    next(redirectPath)
    return
  }
  
  console.log('路由守卫通过，继续导航')
  next()
})

export default router
