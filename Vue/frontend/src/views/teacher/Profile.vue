<template>
  <div class="teacher-profile">
    <div class="profile-header">
      <h1>个人资料</h1>
      <p class="subtitle">查看和管理您的个人信息</p>
    </div>

    <a-row :gutter="24">
      <a-col :span="8">
        <div class="profile-card">
          <div class="avatar-container">
            <a-avatar :size="120" :src="userInfo.avatar || defaultAvatar" />
            <div class="upload-btn">
              <a-upload
                name="avatar"
                :showUploadList="false"
                :customRequest="handleAvatarUpload"
                accept="image/*"
              >
                <a-button type="primary" size="small">
                  <UploadOutlined />
                  更换头像
                </a-button>
              </a-upload>
            </div>
          </div>
          <div class="user-info">
            <h2>{{ userInfo.realName || userInfo.username }}</h2>
            <p class="user-role">{{ userInfo.title || '教师' }}</p>
            <p class="user-dept">{{ userInfo.department || '未设置院系' }}</p>
          </div>
          <a-divider />
          <div class="contact-info">
            <p>
              <MailOutlined /> {{ userInfo.email || '未设置邮箱' }}
            </p>
            <p>
              <PhoneOutlined /> {{ userInfo.contactPhone || '未设置联系电话' }}
            </p>
            <p>
              <EnvironmentOutlined /> {{ userInfo.officeLocation || '未设置办公地点' }}
            </p>
          </div>
        </div>
      </a-col>

      <a-col :span="16">
        <div class="profile-tabs">
          <a-tabs v-model:activeKey="activeTab">
            <a-tab-pane key="basic" tab="基本信息">
              <a-form
                :model="formState"
                name="basicForm"
                :label-col="{ span: 6 }"
                :wrapper-col="{ span: 16 }"
                @finish="onFinish"
              >
                <a-form-item label="用户名" name="username">
                  <a-input v-model:value="formState.username" disabled />
                </a-form-item>

                <a-form-item label="真实姓名" name="realName">
                  <a-input v-model:value="formState.realName" />
                </a-form-item>

                <a-form-item label="邮箱" name="email">
                  <a-input v-model:value="formState.email" />
                </a-form-item>

                <a-form-item label="联系电话" name="contactPhone">
                  <a-input v-model:value="formState.contactPhone" />
                </a-form-item>

                <a-form-item label="职称" name="title">
                  <a-select v-model:value="formState.title">
                    <a-select-option value="教授">教授</a-select-option>
                    <a-select-option value="副教授">副教授</a-select-option>
                    <a-select-option value="讲师">讲师</a-select-option>
                    <a-select-option value="助教">助教</a-select-option>
                  </a-select>
                </a-form-item>

                <a-form-item label="所属院系" name="department">
                  <a-input v-model:value="formState.department" />
                </a-form-item>

                <a-form-item label="办公地点" name="officeLocation">
                  <a-input v-model:value="formState.officeLocation" />
                </a-form-item>

                <a-form-item label="办公时间" name="officeHours">
                  <a-input v-model:value="formState.officeHours" />
                </a-form-item>

                <a-form-item label="专业领域" name="specialty">
                  <a-textarea v-model:value="formState.specialty" :rows="3" />
                </a-form-item>

                <a-form-item label="个人简介" name="introduction">
                  <a-textarea v-model:value="formState.introduction" :rows="5" />
                </a-form-item>

                <a-form-item :wrapper-col="{ offset: 6, span: 16 }">
                  <a-button type="primary" html-type="submit" :loading="loading">保存更改</a-button>
                </a-form-item>
              </a-form>
            </a-tab-pane>

            <a-tab-pane key="security" tab="安全设置">
              <a-form
                :model="passwordForm"
                name="passwordForm"
                :label-col="{ span: 6 }"
                :wrapper-col="{ span: 16 }"
                @finish="onPasswordChange"
              >
                <a-form-item
                  label="当前密码"
                  name="currentPassword"
                  :rules="[{ required: true, message: '请输入当前密码' }]"
                >
                  <a-input-password v-model:value="passwordForm.currentPassword" />
                </a-form-item>

                <a-form-item
                  label="新密码"
                  name="newPassword"
                  :rules="[
                    { required: true, message: '请输入新密码' },
                    { min: 6, message: '密码长度不能少于6个字符' }
                  ]"
                >
                  <a-input-password v-model:value="passwordForm.newPassword" />
                </a-form-item>

                <a-form-item
                  label="确认新密码"
                  name="confirmPassword"
                  :rules="[
                    { required: true, message: '请确认新密码' },
                    { validator: validateConfirmPassword }
                  ]"
                >
                  <a-input-password v-model:value="passwordForm.confirmPassword" />
                </a-form-item>

                <a-form-item :wrapper-col="{ offset: 6, span: 16 }">
                  <a-button type="primary" html-type="submit" :loading="passwordLoading">
                    修改密码
                  </a-button>
                </a-form-item>
              </a-form>
            </a-tab-pane>
          </a-tabs>
        </div>
      </a-col>
    </a-row>
  </div>
</template>

<script setup lang="ts">
import { ref, reactive, onMounted } from 'vue'
import { message } from 'ant-design-vue'
import axios from 'axios'
import {
  MailOutlined,
  PhoneOutlined,
  EnvironmentOutlined,
  UploadOutlined
} from '@ant-design/icons-vue'

// 默认头像
const defaultAvatar = 'https://gw.alipayobjects.com/zos/rmsportal/BiazfanxmamNRoxxVxka.png'

// 状态
const activeTab = ref('basic')
const loading = ref(false)
const passwordLoading = ref(false)
const userInfo = ref({
  id: null,
  username: '',
  realName: '',
  email: '',
  avatar: '',
  department: '',
  title: '',
  education: '',
  specialty: '',
  introduction: '',
  officeLocation: '',
  officeHours: '',
  contactEmail: '',
  contactPhone: ''
})

// 表单状态
const formState = reactive({
  username: '',
  realName: '',
  email: '',
  contactPhone: '',
  title: '',
  department: '',
  officeLocation: '',
  officeHours: '',
  specialty: '',
  introduction: ''
})

// 密码表单
const passwordForm = reactive({
  currentPassword: '',
  newPassword: '',
  confirmPassword: ''
})

// 验证确认密码
const validateConfirmPassword = async (_rule: any, value: string) => {
  if (value !== passwordForm.newPassword) {
    return Promise.reject('两次输入的密码不一致')
  }
  return Promise.resolve()
}

// 生命周期钩子
onMounted(() => {
  fetchUserInfo()
})

// 获取用户信息
const fetchUserInfo = async () => {
  loading.value = true
  try {
    const response = await axios.get('/api/teacher/profile')
    if (response.data && response.data.code === 200) {
      const data = response.data.data
      userInfo.value = data
      
      // 填充表单
      formState.username = data.username || ''
      formState.realName = data.realName || ''
      formState.email = data.email || ''
      formState.contactPhone = data.contactPhone || ''
      formState.title = data.title || ''
      formState.department = data.department || ''
      formState.officeLocation = data.officeLocation || ''
      formState.officeHours = data.officeHours || ''
      formState.specialty = data.specialty || ''
      formState.introduction = data.introduction || ''
    } else {
      message.error(response.data?.message || '获取用户信息失败')
    }
  } catch (error) {
    console.error('获取用户信息失败:', error)
    message.error('获取用户信息失败，请重试')
  } finally {
    loading.value = false
  }
}

// 提交表单
const onFinish = async () => {
  loading.value = true
  try {
    const response = await axios.put('/api/teacher/profile', formState)
    if (response.data && response.data.code === 200) {
      message.success('个人信息更新成功')
      fetchUserInfo() // 刷新数据
    } else {
      message.error(response.data?.message || '更新个人信息失败')
    }
  } catch (error) {
    console.error('更新个人信息失败:', error)
    message.error('更新个人信息失败，请重试')
  } finally {
    loading.value = false
  }
}

// 修改密码
const onPasswordChange = async () => {
  passwordLoading.value = true
  try {
    const response = await axios.put('/api/user/password', {
      oldPassword: passwordForm.currentPassword,
      newPassword: passwordForm.newPassword
    })
    if (response.data && response.data.code === 200) {
      message.success('密码修改成功，请重新登录')
      passwordForm.currentPassword = ''
      passwordForm.newPassword = ''
      passwordForm.confirmPassword = ''
      
      // 延迟2秒后退出登录
      setTimeout(() => {
        // 清除token并跳转到登录页
        localStorage.removeItem('token')
        window.location.href = '/login'
      }, 2000)
    } else {
      message.error(response.data?.message || '密码修改失败')
    }
  } catch (error) {
    console.error('密码修改失败:', error)
    message.error('密码修改失败，请检查当前密码是否正确')
  } finally {
    passwordLoading.value = false
  }
}

// 上传头像
const handleAvatarUpload = async (options: any) => {
  const { file, onSuccess, onError } = options
  const formData = new FormData()
  formData.append('file', file)
  
  try {
    const response = await axios.post('/api/upload/avatar', formData, {
      headers: {
        'Content-Type': 'multipart/form-data'
      }
    })
    
    if (response.data && response.data.code === 200) {
      const avatarUrl = response.data.data
      userInfo.value.avatar = avatarUrl
      
      // 更新用户头像
      await axios.put('/api/teacher/profile', { avatar: avatarUrl })
      message.success('头像更新成功')
      onSuccess(response, file)
    } else {
      message.error(response.data?.message || '头像上传失败')
      onError(new Error('上传失败'))
    }
  } catch (error) {
    console.error('头像上传失败:', error)
    message.error('头像上传失败，请重试')
    onError(new Error('上传失败'))
  }
}
</script>

<style scoped>
.teacher-profile {
  padding: 24px;
}

.profile-header {
  margin-bottom: 24px;
}

.profile-header h1 {
  font-size: 24px;
  margin-bottom: 8px;
}

.subtitle {
  color: rgba(0, 0, 0, 0.45);
}

.profile-card {
  background: #fff;
  padding: 24px;
  border-radius: 4px;
  box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);
  text-align: center;
}

.avatar-container {
  position: relative;
  display: inline-block;
  margin-bottom: 16px;
}

.upload-btn {
  margin-top: 12px;
}

.user-info {
  margin-bottom: 16px;
}

.user-info h2 {
  margin-bottom: 4px;
  font-size: 18px;
}

.user-role {
  color: rgba(0, 0, 0, 0.65);
  margin-bottom: 4px;
}

.user-dept {
  color: rgba(0, 0, 0, 0.45);
}

.contact-info {
  text-align: left;
}

.contact-info p {
  margin-bottom: 8px;
  display: flex;
  align-items: center;
}

.contact-info .anticon {
  margin-right: 8px;
  color: rgba(0, 0, 0, 0.45);
}

.profile-tabs {
  background: #fff;
  padding: 24px;
  border-radius: 4px;
  box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);
  min-height: 500px;
}
</style> 