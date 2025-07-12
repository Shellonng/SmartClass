// 测试router类型文件 - 验证类型声明是否正确
import type { RouteLocationNormalized } from 'vue-router'

// 测试函数
function testRouterTypes(to: RouteLocationNormalized) {
  // 这些访问应该不会产生类型错误
  console.log('requiresAuth:', to.meta?.requiresAuth)
  console.log('role:', to.meta?.role)
  console.log('mode:', to.meta?.mode)
  
  // 如果这个文件没有类型错误，说明类型声明工作正常
  return {
    hasAuth: to.meta?.requiresAuth || false,
    userRole: to.meta?.role || 'guest',
    pageMode: to.meta?.mode || 'normal'
  }
}

export { testRouterTypes } 