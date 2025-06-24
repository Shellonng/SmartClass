<template>
  <div class="auth-layout">
    <!-- 背景装饰 -->
    <div class="background-decoration">
      <div class="decoration-circle circle-1"></div>
      <div class="decoration-circle circle-2"></div>
      <div class="decoration-circle circle-3"></div>
      <div class="floating-shapes">
        <div class="shape shape-1"></div>
        <div class="shape shape-2"></div>
        <div class="shape shape-3"></div>
        <div class="shape shape-4"></div>
      </div>
    </div>
    
    <!-- 主要内容区域 -->
    <div class="auth-container">
      <!-- 左侧信息区域 -->
      <div class="info-section">
        <div class="brand-info">
          <div class="logo-container">
            <img src="/logo.svg" alt="Logo" class="logo" />
            <h1 class="brand-title">智慧教学平台</h1>
          </div>
          <p class="brand-description">
            基于AI技术的现代化教学管理系统，为师生提供智能化的教学体验
          </p>
        </div>
        
        <div class="features-list">
          <div class="feature-item">
            <div class="feature-icon">
              <RobotOutlined />
            </div>
            <div class="feature-content">
              <h3>AI智能助手</h3>
              <p>个性化学习建议和智能批改</p>
            </div>
          </div>
          
          <div class="feature-item">
            <div class="feature-icon">
              <CloudOutlined />
            </div>
            <div class="feature-content">
              <h3>云端同步</h3>
              <p>随时随地访问学习资源</p>
            </div>
          </div>
          
          <div class="feature-item">
            <div class="feature-icon">
              <SafetyOutlined />
            </div>
            <div class="feature-content">
              <h3>安全可靠</h3>
              <p>企业级安全保障</p>
            </div>
          </div>
        </div>
        
        <!-- 统计信息 -->
        <div class="stats-section">
          <div class="stat-item">
            <div class="stat-number">10,000+</div>
            <div class="stat-label">活跃用户</div>
          </div>
          <div class="stat-item">
            <div class="stat-number">500+</div>
            <div class="stat-label">合作学校</div>
          </div>
          <div class="stat-item">
            <div class="stat-number">98%</div>
            <div class="stat-label">满意度</div>
          </div>
        </div>
      </div>
      
      <!-- 右侧表单区域 -->
      <div class="form-section">
        <div class="form-container">
          <div class="form-header">
            <h2 class="form-title">{{ formTitle }}</h2>
            <p class="form-subtitle">{{ formSubtitle }}</p>
          </div>
          
          <!-- 表单内容插槽 -->
          <div class="form-content">
            <slot></slot>
          </div>
          
          <!-- 底部链接 -->
          <div class="form-footer">
            <slot name="footer">
              <div class="footer-links">
                <a href="#" class="footer-link">使用条款</a>
                <span class="divider">|</span>
                <a href="#" class="footer-link">隐私政策</a>
                <span class="divider">|</span>
                <a href="#" class="footer-link">帮助中心</a>
              </div>
            </slot>
          </div>
        </div>
      </div>
    </div>
    
    <!-- 版权信息 -->
    <div class="copyright">
      <p>&copy; 2024 智慧教学平台. All rights reserved.</p>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import { useRoute } from 'vue-router'
import {
  RobotOutlined,
  CloudOutlined,
  SafetyOutlined
} from '@ant-design/icons-vue'

const route = useRoute()

// 根据路由动态设置标题
const formTitle = computed(() => {
  switch (route.name) {
    case 'login':
      return '欢迎回来'
    case 'register':
      return '创建账户'
    case 'forgot-password':
      return '找回密码'
    case 'reset-password':
      return '重置密码'
    default:
      return '用户认证'
  }
})

const formSubtitle = computed(() => {
  switch (route.name) {
    case 'login':
      return '登录您的账户以继续使用'
    case 'register':
      return '注册新账户开始学习之旅'
    case 'forgot-password':
      return '输入您的邮箱地址找回密码'
    case 'reset-password':
      return '设置新的登录密码'
    default:
      return '请完成身份验证'
  }
})
</script>

<style scoped>
.auth-layout {
  min-height: 100vh;
  position: relative;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  overflow: hidden;
}

.background-decoration {
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  pointer-events: none;
  z-index: 1;
}

.decoration-circle {
  position: absolute;
  border-radius: 50%;
  background: rgba(255, 255, 255, 0.1);
  animation: float 6s ease-in-out infinite;
}

.circle-1 {
  width: 200px;
  height: 200px;
  top: 10%;
  left: 10%;
  animation-delay: 0s;
}

.circle-2 {
  width: 150px;
  height: 150px;
  top: 60%;
  right: 15%;
  animation-delay: 2s;
}

.circle-3 {
  width: 100px;
  height: 100px;
  bottom: 20%;
  left: 20%;
  animation-delay: 4s;
}

.floating-shapes {
  position: absolute;
  width: 100%;
  height: 100%;
}

.shape {
  position: absolute;
  background: rgba(255, 255, 255, 0.05);
  animation: drift 8s linear infinite;
}

.shape-1 {
  width: 60px;
  height: 60px;
  top: 20%;
  left: 30%;
  border-radius: 30% 70% 70% 30% / 30% 30% 70% 70%;
  animation-delay: 0s;
}

.shape-2 {
  width: 80px;
  height: 80px;
  top: 70%;
  right: 30%;
  border-radius: 50% 50% 50% 50% / 60% 60% 40% 40%;
  animation-delay: 3s;
}

.shape-3 {
  width: 40px;
  height: 40px;
  top: 40%;
  right: 20%;
  border-radius: 40% 60% 60% 40% / 60% 30% 70% 40%;
  animation-delay: 1s;
}

.shape-4 {
  width: 70px;
  height: 70px;
  bottom: 30%;
  left: 40%;
  border-radius: 60% 40% 30% 70% / 60% 30% 70% 40%;
  animation-delay: 5s;
}

@keyframes float {
  0%, 100% {
    transform: translateY(0px) rotate(0deg);
  }
  50% {
    transform: translateY(-20px) rotate(180deg);
  }
}

@keyframes drift {
  0% {
    transform: translateX(-10px) translateY(-10px) rotate(0deg);
  }
  25% {
    transform: translateX(10px) translateY(-20px) rotate(90deg);
  }
  50% {
    transform: translateX(20px) translateY(10px) rotate(180deg);
  }
  75% {
    transform: translateX(-5px) translateY(20px) rotate(270deg);
  }
  100% {
    transform: translateX(-10px) translateY(-10px) rotate(360deg);
  }
}

.auth-container {
  position: relative;
  z-index: 2;
  min-height: 100vh;
  display: flex;
  align-items: center;
  max-width: 1200px;
  margin: 0 auto;
  padding: 40px 20px;
  gap: 60px;
}

.info-section {
  flex: 1;
  color: white;
  padding: 40px;
}

.brand-info {
  margin-bottom: 60px;
}

.logo-container {
  display: flex;
  align-items: center;
  gap: 16px;
  margin-bottom: 24px;
}

.logo {
  width: 48px;
  height: 48px;
  filter: brightness(0) invert(1);
}

.brand-title {
  font-size: 32px;
  font-weight: 700;
  margin: 0;
  background: linear-gradient(45deg, #fff, #e0e7ff);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}

.brand-description {
  font-size: 18px;
  line-height: 1.6;
  opacity: 0.9;
  margin: 0;
}

.features-list {
  margin-bottom: 60px;
}

.feature-item {
  display: flex;
  align-items: flex-start;
  gap: 16px;
  margin-bottom: 32px;
  padding: 20px;
  background: rgba(255, 255, 255, 0.1);
  border-radius: 16px;
  backdrop-filter: blur(10px);
  border: 1px solid rgba(255, 255, 255, 0.2);
  transition: transform 0.3s ease, background 0.3s ease;
}

.feature-item:hover {
  transform: translateY(-2px);
  background: rgba(255, 255, 255, 0.15);
}

.feature-icon {
  width: 48px;
  height: 48px;
  background: rgba(255, 255, 255, 0.2);
  border-radius: 12px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 20px;
  flex-shrink: 0;
}

.feature-content h3 {
  font-size: 18px;
  font-weight: 600;
  margin: 0 0 8px 0;
}

.feature-content p {
  font-size: 14px;
  opacity: 0.8;
  margin: 0;
  line-height: 1.5;
}

.stats-section {
  display: flex;
  gap: 40px;
}

.stat-item {
  text-align: center;
}

.stat-number {
  font-size: 28px;
  font-weight: 700;
  margin-bottom: 8px;
  background: linear-gradient(45deg, #fff, #e0e7ff);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}

.stat-label {
  font-size: 14px;
  opacity: 0.8;
}

.form-section {
  flex: 0 0 480px;
  display: flex;
  align-items: center;
  justify-content: center;
}

.form-container {
  width: 100%;
  max-width: 400px;
  background: rgba(255, 255, 255, 0.95);
  backdrop-filter: blur(20px);
  border-radius: 24px;
  padding: 40px;
  box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
  border: 1px solid rgba(255, 255, 255, 0.3);
}

.form-header {
  text-align: center;
  margin-bottom: 32px;
}

.form-title {
  font-size: 28px;
  font-weight: 700;
  color: #1a1a1a;
  margin: 0 0 8px 0;
}

.form-subtitle {
  font-size: 16px;
  color: #666;
  margin: 0;
  line-height: 1.5;
}

.form-content {
  margin-bottom: 24px;
}

.form-footer {
  text-align: center;
  padding-top: 24px;
  border-top: 1px solid #f0f0f0;
}

.footer-links {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
  flex-wrap: wrap;
}

.footer-link {
  color: #666;
  text-decoration: none;
  font-size: 14px;
  transition: color 0.3s ease;
}

.footer-link:hover {
  color: #1890ff;
}

.divider {
  color: #d9d9d9;
  font-size: 12px;
}

.copyright {
  position: absolute;
  bottom: 20px;
  left: 50%;
  transform: translateX(-50%);
  color: rgba(255, 255, 255, 0.7);
  font-size: 14px;
  z-index: 2;
}

/* 响应式设计 */
@media (max-width: 1024px) {
  .auth-container {
    flex-direction: column;
    gap: 40px;
    padding: 20px;
  }
  
  .info-section {
    padding: 20px;
    text-align: center;
  }
  
  .features-list {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
    gap: 20px;
    margin-bottom: 40px;
  }
  
  .stats-section {
    justify-content: center;
    gap: 30px;
  }
  
  .form-section {
    flex: none;
    width: 100%;
    max-width: 400px;
  }
}


</style>