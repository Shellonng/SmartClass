/**
 * 登录会话持久化修复方案
 * 
 * 本文件记录了解决"登录后仍提示请先登录"问题的完整方案
 */

// 问题描述：
// 用户登录后，刷新页面或重新打开应用时，会丢失登录状态，需要重新登录

// 解决方案：

// 1. 前端Token存储和传递:
// - 使用localStorage同时存储两个key ('token' 和 'user-token')，增加兼容性
// - 每次请求自动从localStorage获取token并添加到请求头
// - 使用Bearer认证格式

// 示例代码 - 请求拦截器:
/*
axios.interceptors.request.use(
  (config) => {
    // 在请求发送之前添加token
    const token = localStorage.getItem('user-token') || localStorage.getItem('token');
    if (token) {
      // 使用Bearer认证格式
      config.headers['Authorization'] = `Bearer ${token}`;
    }
    
    // 确保请求包含cookie
    config.withCredentials = true
    
    return config
  },
  (error) => {
    return Promise.reject(error)
  }
)
*/

// 2. 响应拦截器处理401错误:
// - 检测到401未授权错误时，清除本地token
// - 重定向到登录页面

// 示例代码 - 响应拦截器:
/*
axios.interceptors.response.use(
  (response) => {
    return response
  },
  (error) => {
    // 处理401未授权错误
    if (error.response && error.response.status === 401) {
      // 删除本地存储的token和用户信息
      localStorage.removeItem('token');
      localStorage.removeItem('user-token');
      localStorage.removeItem('userInfo');
      
      // 重定向到登录页
      window.location.href = '/login';
    }
    return Promise.reject(error)
  }
)
*/

// 3. 用户信息持久化:
// - 登录成功后将用户信息存储到localStorage
// - 应用初始化时优先从localStorage加载用户信息

// 示例代码 - 登录成功后:
/*
if (result.success) {
  // 将用户信息存储到localStorage，确保持久化
  if (result.userInfo) {
    localStorage.setItem('userInfo', JSON.stringify(result.userInfo))
  }
  
  message.success('登录成功')
}
*/

// 4. 后端Token生成和验证:
// - 登录时生成简化的token (格式: "token-{userId}")
// - 认证过滤器支持解析简化token格式

// 示例代码 - 后端生成token:
/*
// 添加简化的token，格式为"token-{userId}"
String simpleToken = "token-" + authResponse.getUserId();
data.put("token", simpleToken);
*/

// 5. 后端认证过滤器增强:
// - 支持解析简化的token格式
// - 从token中提取userId并设置到请求属性中

// 示例代码 - 解析token:
/*
// 处理简化的token格式: token-{userId}
if (jwtToken.startsWith("token-")) {
    String userId = jwtToken.substring(6); // 提取userId部分
    logger.debug("使用简化token格式，用户ID: {}", userId);
    
    // 设置到请求属性中，供后续使用
    request.setAttribute("userId", userId);
}
*/

// 6. 会话超时时间延长:
// - 设置会话超时时间为24小时
// session.setMaxInactiveInterval(86400);

// 通过以上修改，确保用户登录状态能够持久化保存，解决了"登录后仍提示请先登录"的问题。

// 修复登录数据处理的示例代码
// 正确的数据访问方式：

const handleLoginFixed = async () => {
  try {
    const response = await login(loginData)
    
    if (response.data.code === 200) {
      console.log('完整响应数据:', response.data)
      
      // 后端返回的数据结构：
      // {
      //   code: 200,
      //   data: {
      //     userInfo: { id, username, realName, email, role, avatar },
      //     sessionId: "..."
      //   }
      // }
      
      const data = response.data.data
      const userInfo = data.userInfo
      
      // 安全访问，避免undefined错误
      if (userInfo && userInfo.role) {
        authStore.user = {
          id: userInfo.id,
          username: userInfo.username,
          realName: userInfo.realName,
          email: userInfo.email,
          role: userInfo.role.toLowerCase(),
          avatar: userInfo.avatar || undefined
        }
        
        // 跳转逻辑
        if (userInfo.role === 'STUDENT') {
          router.push('/student/dashboard')
        } else if (userInfo.role === 'TEACHER') {
          router.push('/teacher/dashboard')
        }
      } else {
        console.error('用户信息格式错误:', userInfo)
        message.error('登录数据格式错误')
      }
    }
  } catch (error) {
    console.error('登录失败:', error)
    message.error(error.response?.data?.message || '登录失败')
  }
} 