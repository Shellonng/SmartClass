<script setup lang="ts">
import { ref, reactive, computed } from 'vue'
import { useRouter } from 'vue-router'

const router = useRouter()
const activeTab = ref('student')
const owlElement = ref<HTMLElement | null>(null)

const studentForm = reactive({
  username: '',
  password: ''
})

const teacherForm = reactive({
  username: '',
  password: ''
})

const handleLogin = (userType: string) => {
  // Here you would normally handle authentication with backend
  console.log(`${userType} login attempt:`, userType === 'student' ? studentForm : teacherForm)
  
  // Navigate to home page
  router.push('/home')
}

// Handle owl animation on password focus/blur
const handlePasswordFocus = () => {
  if (owlElement.value) {
    owlElement.value.classList.add('password')
  }
}

const handlePasswordBlur = () => {
  if (owlElement.value) {
    owlElement.value.classList.remove('password')
  }
}

// 计算属性，根据当前选择的标签返回对应的用户名
const username = computed(() => {
  return activeTab.value === 'student' ? studentForm.username : teacherForm.username
})

// 计算属性，根据当前选择的标签返回对应的密码
const password = computed(() => {
  return activeTab.value === 'student' ? studentForm.password : teacherForm.password
})

// 更新用户名的方法
const updateUsername = (value: string) => {
  if (activeTab.value === 'student') {
    studentForm.username = value
  } else {
    teacherForm.username = value
  }
}

// 更新密码的方法
const updatePassword = (value: string) => {
  if (activeTab.value === 'student') {
    studentForm.password = value
  } else {
    teacherForm.password = value
  }
}
</script>

<template>
  <div class="login-container">
    <div class="login-box">
      <div class="owl" ref="owlElement">
        <div class="hand"></div>
        <div class="hand hand-r"></div>
        <div class="arms">
          <div class="arm"></div>
          <div class="arm arm-r"></div>
        </div>
      </div>
      
      <div class="input-box">
        <input 
          :value="username"
          @input="event => updateUsername((event.target as HTMLInputElement).value)"
          type="text" 
          placeholder="账号"
        />
        <input 
          :value="password"
          @input="event => updatePassword((event.target as HTMLInputElement).value)"
          type="password" 
          placeholder="密码"
          @focus="handlePasswordFocus"
          @blur="handlePasswordBlur"
        />
        <button @click="handleLogin(activeTab)">登录</button>
      </div>
    </div>
  </div>
</template>

<style scoped>
* {
  /* 初始化 */
  margin: 0;
  padding: 0;
}

.login-container {
  /* 100%窗口高度 */
  height: 100vh;
  /* 弹性布局 居中 */
  display: flex;
  justify-content: center;
  align-items: center;
  /* 渐变背景 */
  background: linear-gradient(200deg, #72afd3, #96fbc4);
}

.login-box {
  /* 相对定位 */
  position: relative;
  width: 320px;
}

.input-box {
  /* 弹性布局 垂直排列 */
  display: flex;
  flex-direction: column;
}

.input-box input {
  height: 40px;
  border-radius: 3px;
  /* 缩进15像素 */
  text-indent: 15px;
  outline: none;
  border: none;
  margin-bottom: 15px;
}

.input-box input:focus {
  outline: 1px solid lightseagreen;
}

.input-box button {
  border: none;
  height: 45px;
  background-color: lightseagreen;
  color: #fff;
  border-radius: 3px;
  cursor: pointer;
}

/* 猫头鹰样式 */
.owl {
  width: 211px;
  height: 108px;
  /* 背景图片 */
  background: url("../image/owl-login.png") no-repeat;
  /* 绝对定位 */
  position: absolute;
  top: -100px;
  /* 水平居中 */
  left: 50%;
  transform: translateX(-50%);
}

.owl .hand {
  width: 34px;
  height: 34px;
  border-radius: 40px;
  background-color: #472d20;
  /* 绝对定位 */
  position: absolute;
  left: 12px;
  bottom: -8px;
  /* 沿Y轴缩放0.6倍（压扁） */
  transform: scaleY(0.6);
  /* 动画过渡 */
  transition: 0.3s ease-out;
}

.owl .hand.hand-r {
  left: 170px;
}

.owl.password .hand {
  transform: translateX(42px) translateY(-15px) scale(0.7);
}

.owl.password .hand.hand-r {
  transform: translateX(-42px) translateY(-15px) scale(0.7);
}

.owl .arms {
  position: absolute;
  top: 58px;
  width: 100%;
  height: 41px;
  overflow: hidden;
}

.owl .arms .arm {
  width: 40px;
  height: 65px;
  position: absolute;
  left: 20px;
  top: 40px;
  background: url("../image/owl-login-arm.png") no-repeat;
  transform: rotate(-20deg);
  transition: 0.3s ease-out;
}

.owl .arms .arm.arm-r {
  transform: rotate(20deg) scaleX(-1);
  left: 158px;
}

.owl.password .arms .arm {
  transform: translateY(-40px) translateX(40px);
}

.owl.password .arms .arm.arm-r {
  transform: translateY(-40px) translateX(-40px) scaleX(-1);
}
</style> 