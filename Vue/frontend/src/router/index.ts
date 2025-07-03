import { createRouter, createWebHistory } from 'vue-router'
import { useAuthStore } from '@/stores/auth'

// 为路由元数据添加类型声明
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

// 学生端 - 错题集
const StudentWrongQuestions = () => import('@/views/student/WrongQuestions.vue')

// 学生端 - 学习记录
const StudentLearningRecords = () => import('@/views/student/LearningRecords.vue')

// 学生端 - 考试管理
const StudentExamDetail = () => import('@/views/student/ExamDetail.vue')

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

// 添加路由引用
const StudentSectionDetail = () => import('@/views/teacher/SectionDetail.vue')

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
    
    // 课程题库页面 - 使用独立布局
    {
      path: '/teacher/courses/:courseId/question-bank',
      name: 'TeacherCourseQuestionBank',
      component: CourseLayout,
      meta: { requiresAuth: true, role: 'TEACHER' },
      children: [
        {
          path: '',
          component: TeacherQuestionBank,
          props: route => ({ courseId: Number(route.params.courseId) })
        }
      ]
    },
    
    // 课程题目详情页面
    {
      path: '/teacher/courses/:courseId/question-bank/:id',
      name: 'TeacherCourseQuestionDetail',
      component: CourseLayout,
      meta: { requiresAuth: true, role: 'TEACHER' },
      children: [
        {
          path: '',
          component: TeacherQuestionDetail,
          props: route => ({ 
            id: Number(route.params.id),
            courseId: Number(route.params.courseId)
          })
        }
      ]
    },

    // 教师考试详情页面 - 使用独立布局
    {
      path: '/teacher/exams/:id',
      name: 'TeacherExamDetail',
      component: CourseLayout,
      meta: { requiresAuth: true, role: 'TEACHER' },
      children: [
        {
          path: '',
          component: TeacherExamDetail,
          props: route => ({ id: Number(route.params.id) })
        }
      ]
    },

    // 教师作业详情页面 - 使用独立布局
    {
      path: '/teacher/assignments/:id',
      name: 'TeacherAssignmentDetail',
      component: CourseLayout,
      meta: { requiresAuth: true, role: 'TEACHER' },
      children: [
        {
          path: '',
          component: TeacherAssignmentDetail,
          props: route => ({ id: Number(route.params.id) })
        },
        {
          path: 'edit',
          component: TeacherExamDetail, // 复用考试编辑组件
          props: route => ({ 
            id: Number(route.params.id),
            isAssignment: true // 标记为作业模式
          })
        },
        {
          path: 'detail',
          component: TeacherExamDetail, // 复用考试详情组件
          props: route => ({ 
            id: Number(route.params.id),
            isAssignment: true, // 标记为作业模式
            viewOnly: true // 标记为查看模式
          })
        }
      ]
    },

    // 文件提交型作业(学生端) - 使用独立布局
    {
      path: '/student/assignments/file/:id',
      name: 'StudentFileAssignmentDetail',
      component: CourseLayout,
      meta: { requiresAuth: true, role: 'STUDENT' },
      children: [
        {
          path: '',
          component: StudentAssignmentDetail,
          props: route => ({ id: Number(route.params.id), isFileMode: true })
        },
        {
          path: 'submit',
          component: () => import('@/views/student/FileSubmit.vue'),
          props: route => ({ id: Number(route.params.id) })
        }
      ]
    },

    // 作业详情页面(学生端) - 使用独立布局
    {
      path: '/student/assignments/:id',
      name: 'StudentAssignmentDetail',
      component: CourseLayout,
      meta: { requiresAuth: true, role: 'STUDENT' },
      children: [
        {
          path: '',
          component: StudentAssignmentDetail,
          props: route => ({ id: Number(route.params.id) })
        },
        {
          path: 'do',
          component: () => import('@/views/student/AssignmentDo.vue'),
          props: route => ({ id: Number(route.params.id) })
        }
      ]
    },

    // 考试详情页面(学生端) - 使用独立布局
    {
      path: '/student/exams/:id',
      name: 'StudentExamDetail',
      component: CourseLayout,
      meta: { requiresAuth: true, role: 'STUDENT' },
      children: [
        {
          path: '',
          component: StudentExamDetail,
          props: route => ({ id: Number(route.params.id) })
        },
        {
          path: 'do',
          component: () => import('@/views/student/ExamDo.vue'),
          props: route => ({ id: Number(route.params.id) })
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
        
        // 考试管理
        {
          path: 'exams',
          name: 'TeacherExams',
          component: TeacherExams
        },
        
        // 作业管理
        {
          path: 'assignments',
          name: 'TeacherAssignments',
          component: TeacherAssignments
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
          component: StudentCourses // 使用学生端课程组件
        },
        {
          path: 'courses/:id',
          name: 'StudentCourseDetail',
          component: StudentCourseDetail,
          props: true
        },
        {
          path: 'courses/:courseId/sections/:sectionId',
          name: 'StudentSectionDetail',
          component: StudentSectionDetail,
          props: true,
          meta: { requiresAuth: true, role: 'STUDENT', viewOnly: true }
        },
        {
          path: 'courses/:id/video/:videoId',
          name: 'StudentVideoLearning',
          component: StudentVideoLearning,
          props: true
        },
        
        // 任务管理
        {
          path: 'tasks',
          name: 'StudentTasks',
          component: StudentDashboard, // 临时使用Dashboard作为占位符
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
        // 作业答题页面重定向到考试路径，统一使用/student/exams/:id/do
        {
          path: 'assignments/:id/do',
          redirect: to => `/student/exams/${to.params.id}/do`
        },
        {
          path: 'assignments/file/:id/submit',
          name: 'StudentFileSubmit',
          component: () => import('@/views/student/FileSubmit.vue'),
          props: true,
          meta: { requiresAuth: true, role: 'STUDENT' }
        },
        
        // 考试列表
        {
          path: 'exams',
          name: 'StudentExams',
          component: StudentDashboard,  // 临时使用Dashboard作为占位符
        },
        
        // 添加考试答题页面路由，使用相同的AssignmentDo组件
        {
          path: 'exams/:id/do',
          name: 'StudentExamDo',
          component: () => import('@/views/student/AssignmentDo.vue'),
          props: true,
          meta: { requiresAuth: true, role: 'STUDENT' }
        },
        
        // 错题集
        {
          path: 'wrong-questions',
          name: 'StudentWrongQuestions',
          component: StudentDashboard,  // 临时使用Dashboard作为占位符
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
      console.log('获取用户信息成功，用户角色:', authStore.user?.role)
    } catch (error) {
      console.error('路由守卫中获取用户信息失败:', error)
    }
  }
  
  // 如果已登录用户访问登录页，重定向到对应首页
  if ((to.path === '/login' || to.path === '/register') && authStore.isAuthenticated) {
    const userRole = authStore.user?.role?.toUpperCase() || '';
    const redirectPath = userRole === 'TEACHER' ? '/teacher/dashboard' : '/student/dashboard'
    console.log('已登录用户访问登录页，重定向到:', redirectPath, '用户角色:', userRole)
    next(redirectPath)
    return
  }
  
  // 检查是否需要认证
  if (to.meta.requiresAuth && !authStore.isAuthenticated) {
    console.log('需要认证但未登录，重定向到登录页')
    next('/login')
    return
  }
  
  // 检查角色权限 - 不区分大小写比较角色
  if (to.meta.role && authStore.user?.role) {
    const metaRole = String(to.meta.role).toUpperCase();
    const userRole = authStore.user.role.toUpperCase();
    
    console.log('角色检查:', {
      路径: to.path,
      需要角色: metaRole,
      用户角色: userRole,
      匹配结果: metaRole === userRole
    });
    
    if (metaRole !== userRole) {
    // 如果角色不匹配，重定向到对应角色的首页
      const redirectPath = userRole === 'TEACHER' ? '/teacher/dashboard' : '/student/dashboard'
      console.log('角色不匹配，重定向到:', redirectPath, '用户角色:', userRole, '路由要求角色:', metaRole)
    next(redirectPath)
    return
    }
  }
  
  console.log('路由守卫通过，继续导航到:', to.path)
  next()
})

export default router
