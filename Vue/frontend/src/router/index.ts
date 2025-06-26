import { createRouter, createWebHistory } from 'vue-router'
import { useAuthStore } from '@/stores/auth'
import HomeView from '../views/HomeView.vue'

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes: [
    {
      path: '/login',
      name: 'login',
      component: () => import('../views/auth/Login.vue'),
      meta: {
        requiresGuest: true,
        title: '登录'
      }
    },
    {
      path: '/',
      name: 'home',
      component: HomeView,
      meta: {
        title: '智慧课堂 - 精品在线课程学习平台'
      }
    },
    {
      path: '/courses',
      name: 'courses',
      component: () => import('../views/CourseList.vue')
    },
    {
      path: '/course/:id',
      name: 'course-detail',
      component: () => import('../views/CourseDetail.vue')
    },
    {
      path: '/teacher',
      name: 'teacher',
      redirect: '/teacher/dashboard',
      meta: {
        requiresAuth: true,
        requiresRole: 'teacher'
      },
      children: [
        {
          path: 'dashboard',
          name: 'teacher-dashboard',
          component: () => import('../views/teacher/Dashboard.vue'),
          meta: {
            title: '教师工作台'
          }
        }
      ]
    },
    {
      path: '/student',
      name: 'student',
      redirect: '/student/dashboard',
      meta: {
        requiresAuth: true,
        requiresRole: 'student'
      },
      children: [
        {
          path: 'dashboard',
          name: 'student-dashboard',
          component: () => import('../views/student/Dashboard.vue'),
          meta: {
            title: '学生工作台'
          }
        }
      ]
    },
    {
      path: '/about',
      name: 'about',
      component: () => import('../views/AboutView.vue'),
    },
    {
      path: '/:pathMatch(.*)*',
      name: 'not-found',
      component: () => import('../views/NotFound.vue')
    }
  ],
})

// 路由守卫
router.beforeEach(async (to, from, next) => {
  const authStore = useAuthStore()
  
  // 设置页面标题
  if (to.meta.title) {
    document.title = `${to.meta.title} - 智慧教育平台`
  } else {
    document.title = '智慧教育平台'
  }
  
  // 如果需要认证
  if (to.meta.requiresAuth) {
    if (!authStore.isAuthenticated) {
      next({ name: 'login', query: { redirect: to.fullPath } })
      return
    }
    
    // 检查角色权限
    if (to.meta.requiresRole && authStore.user?.role !== to.meta.requiresRole) {
      next({ name: 'home' })
      return
    }
  }
  
  // 如果是访客页面（如登录页），已登录用户不能访问
  if (to.meta.requiresGuest && authStore.isAuthenticated) {
    const redirectPath = (to.query.redirect as string) || '/'
    next(redirectPath)
    return
  }
  
  next()
})

export default router
