<template>
  <div class="fixed-width-container course-detail">
    <a-spin :spinning="loading">
      <div class="content-wrapper">
        <!-- 页面头部 -->
        <div class="page-header">
          <div class="header-left">
            <a-button type="text" @click="goBack" class="back-btn">
              <ArrowLeftOutlined />
              返回课程列表
            </a-button>
          </div>
        </div>

        <!-- 章节管理内容 -->
        <div class="chapter-management">
          <div class="chapter-header">
            <h2 class="section-title">章节管理</h2>
            <a-button type="primary" @click="showAddChapterModal">
              <PlusOutlined />
              添加章节
            </a-button>
          </div>
          
          <div class="chapter-list">
            <!-- 有章节数据时显示 -->
            <div v-if="chapters.length > 0">
              <div v-for="(chapter, index) in chapters" :key="chapter.id || index" class="chapter-item">
                <div class="chapter-info">
                  <div class="chapter-number">{{ index + 1 }}</div>
                  <div class="chapter-content">
                    <h3 class="chapter-title">{{ chapter.title }}</h3>
                    <p class="chapter-description" v-if="chapter.description">{{ chapter.description }}</p>
                    
                    <div class="chapter-sections">
                      <div v-if="chapter.sections && chapter.sections.length > 0">
                        <div v-for="(section, sectionIndex) in chapter.sections" :key="section.id || sectionIndex" class="section-item">
                          <div class="section-icon">
                            <FileTextOutlined />
                          </div>
                          <div class="section-content" @click="viewSection(chapter, section)">
                            <div class="section-title">{{ section.title }}</div>
                            <div class="section-duration" v-if="section.duration">{{ section.duration }}分钟</div>
                          </div>
                          <div class="section-actions">
                            <a-button type="text" size="small" @click="editSection(chapter, section)">
                              <EditOutlined />
                            </a-button>
                            <a-button type="text" size="small" @click="deleteSectionItem(chapter, section)">
                              <DeleteOutlined />
                            </a-button>
                          </div>
                        </div>
                      </div>
                      <div v-else class="empty-sections">
                        <a-empty description="暂无小节" :image="Empty.PRESENTED_IMAGE_SIMPLE" />
                      </div>
                      
                      <div class="add-section">
                        <a-button type="dashed" block @click="showAddSectionModal(chapter)">
                          <PlusOutlined />
                          添加小节
                        </a-button>
                      </div>
                    </div>
                  </div>
                  <div class="chapter-actions">
                    <a-button type="text" @click="editChapter(chapter)">
                      <EditOutlined />
                    </a-button>
                    <a-button type="text" danger @click="deleteChapterItem(chapter)">
                      <DeleteOutlined />
                    </a-button>
                  </div>
                </div>
              </div>
            </div>
            
            <!-- 无章节数据时显示 -->
            <div v-else class="empty-chapters">
              <a-empty description="暂无章节数据">
                <template #description>
                  <span>当前课程暂无章节，请点击"添加章节"按钮创建</span>
                </template>
              </a-empty>
            </div>
          </div>
        </div>
        
        <!-- 添加章节弹窗 -->
        <a-modal
          v-model:open="addChapterModalVisible"
          :title="isEditingChapter ? '编辑章节' : '添加章节'"
          @ok="handleAddChapter"
          @cancel="cancelAddChapter"
        >
          <a-form :model="chapterForm" layout="vertical">
            <a-form-item label="章节标题" required>
              <a-input v-model:value="chapterForm.title" placeholder="请输入章节标题" />
            </a-form-item>
            <a-form-item label="章节描述">
              <a-textarea v-model:value="chapterForm.description" placeholder="请输入章节描述" :rows="4" />
            </a-form-item>
          </a-form>
        </a-modal>
        
        <!-- 添加小节弹窗 -->
        <a-modal
          v-model:open="addSectionModalVisible"
          :title="isEditingSection ? '编辑小节' : '添加小节'"
          @ok="handleAddSection"
          @cancel="cancelAddSection"
        >
          <a-form :model="sectionForm" layout="vertical">
            <a-form-item label="小节标题" required>
              <a-input v-model:value="sectionForm.title" placeholder="请输入小节标题" />
            </a-form-item>
            <a-form-item label="预计时长">
              <a-input-number v-model:value="sectionForm.duration" :min="1" :max="300" addonAfter="分钟" style="width: 100%" />
            </a-form-item>
            <a-form-item label="内容描述">
              <a-textarea v-model:value="sectionForm.description" placeholder="请输入内容描述" :rows="4" />
            </a-form-item>
          </a-form>
        </a-modal>
      </div>
    </a-spin>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { message, Empty } from 'ant-design-vue'
import {
  BookOutlined,
  UserOutlined,
  CalendarOutlined,
  ClockCircleOutlined,
  TagOutlined,
  FileTextOutlined,
  TeamOutlined,
  BarChartOutlined,
  SettingOutlined,
  EditOutlined,
  DeleteOutlined,
  ShareAltOutlined,
  DownloadOutlined,
  ArrowLeftOutlined,
  PlusOutlined,
  VideoCameraOutlined,
  FormOutlined,
  FileOutlined
} from '@ant-design/icons-vue'
import { 
  getChaptersByCourseId, 
  createChapter, 
  updateChapter, 
  deleteChapter, 
  createSection, 
  updateSection, 
  deleteSection,
  type Course,
  type Chapter,
  type Section
} from '@/api/teacher'
import axios from 'axios'

const route = useRoute()
const router = useRouter()

// 状态变量
const loading = ref(false)
const chapters = ref<Chapter[]>([])

// 章节表单相关
const addChapterModalVisible = ref(false)
const isEditingChapter = ref(false)
const editingChapterId = ref<number | null>(null)
const chapterForm = ref({
  title: '',
  description: ''
})

// 小节表单相关
const addSectionModalVisible = ref(false)
const isEditingSection = ref(false)
const editingSectionId = ref<number | null>(null)
const currentChapter = ref<Chapter | null>(null)
const sectionForm = ref({
  title: '',
  duration: 30,
  description: ''
})

// 计算属性
const courseId = computed(() => {
  return parseInt(route.params.id as string)
})

// 加载章节列表
const loadChapters = async () => {
  try {
    console.log('开始加载章节列表，课程ID:', courseId.value);
    
    // 使用API函数获取章节列表
    const response = await getChaptersByCourseId(courseId.value);
    console.log('章节列表响应:', response);
    
    if (response.data.code === 200) {
      chapters.value = response.data.data || [];
      console.log('成功加载章节:', chapters.value);
      
      // 如果没有章节，显示空状态
      if (chapters.value.length === 0) {
        console.log('没有找到章节数据');
      }
    } else {
      message.error(response.data.message || '获取章节列表失败');
      console.error('获取章节列表失败:', response.data);
    }
  } catch (error: any) {
    console.error('获取章节列表异常:', error);
    if (error.response) {
      console.error('错误响应:', error.response.data);
      console.error('状态码:', error.response.status);
      console.error('响应头:', error.response.headers);
      message.error(`获取章节列表失败: ${error.response.status} - ${error.response.data?.message || '未知错误'}`);
    } else if (error.request) {
      console.error('请求未收到响应:', error.request);
      message.error('获取章节列表失败: 服务器未响应');
    } else {
      console.error('请求配置错误:', error.message);
      message.error(`获取章节列表失败: ${error.message}`);
    }
  }
}

// 返回课程列表
const goBack = () => {
  router.push('/teacher/courses')
}

// 编辑课程
const editCourse = () => {
  router.push(`/teacher/courses/${courseId.value}/edit`)
}

// 删除课程
const deleteCourse = () => {
  message.info('删除功能待实现')
}

// 分享课程
const shareCourse = () => {
  message.info('分享功能待实现')
}

// 导出课程
const exportCourse = () => {
  message.info('导出功能待实现')
}

// 显示添加章节弹窗
const showAddChapterModal = () => {
  isEditingChapter.value = false
  editingChapterId.value = null
  chapterForm.value = {
    title: '',
    description: ''
  }
  addChapterModalVisible.value = true
}

// 取消添加章节
const cancelAddChapter = () => {
  addChapterModalVisible.value = false
  chapterForm.value = {
    title: '',
    description: ''
  }
}

// 编辑章节
const editChapter = (chapter: Chapter) => {
  isEditingChapter.value = true
  editingChapterId.value = chapter.id || null
  chapterForm.value = {
    title: chapter.title || '',
    description: chapter.description || ''
  }
  addChapterModalVisible.value = true
}

// 处理添加或更新章节
const handleAddChapter = async () => {
  if (!chapterForm.value.title) {
    message.error('请输入章节标题')
    return
  }
  
  try {
    const chapterData: Chapter = {
      courseId: courseId.value,
      title: chapterForm.value.title,
      description: chapterForm.value.description || ''
    }
    
    let response
    
    if (isEditingChapter.value && editingChapterId.value) {
      // 更新章节
      console.log('发送更新章节请求，数据:', chapterData, '章节ID:', editingChapterId.value)
      response = await updateChapter(editingChapterId.value, chapterData)
      console.log('更新章节响应:', response)
      
      if (response.data.code === 200) {
        message.success('更新章节成功')
      } else {
        message.error(response.data.message || '更新章节失败')
        console.error('更新章节失败:', response.data)
      }
    } else {
      // 创建新章节
      console.log('发送创建章节请求，数据:', chapterData)
      response = await createChapter(chapterData)
      console.log('创建章节响应:', response)
      
      if (response.data.code === 200) {
        message.success('添加章节成功')
      } else {
        message.error(response.data.message || '添加章节失败')
        console.error('添加章节失败:', response.data)
      }
    }
    
    if (response.data.code === 200) {
      addChapterModalVisible.value = false
      await loadChapters()
    }
  } catch (error: any) {
    console.error('操作章节异常:', error)
    if (error.response) {
      console.error('错误响应:', error.response.data)
      console.error('状态码:', error.response.status)
      console.error('响应头:', error.response.headers)
      message.error(`操作章节失败: ${error.response.status} - ${error.response.data?.message || '未知错误'}`)
    } else if (error.request) {
      console.error('请求未收到响应:', error.request)
      message.error('操作章节失败: 服务器未响应')
    } else {
      console.error('请求配置错误:', error.message)
      message.error(`操作章节失败: ${error.message}`)
    }
  }
}

// 删除章节
const deleteChapterItem = async (chapter: Chapter) => {
  if (!chapter.id) {
    message.error('章节ID不存在')
    return
  }
  
  try {
    console.log('发送删除章节请求，章节ID:', chapter.id)
    
    const response = await deleteChapter(chapter.id)
    console.log('删除章节响应:', response)
    
    if (response.data.code === 200) {
      message.success('删除章节成功')
      await loadChapters()
    } else {
      message.error(response.data.message || '删除章节失败')
      console.error('删除章节失败:', response.data)
    }
  } catch (error: any) {
    console.error('删除章节异常:', error)
    if (error.response) {
      console.error('错误响应:', error.response.data)
      console.error('状态码:', error.response.status)
      console.error('响应头:', error.response.headers)
      message.error(`删除章节失败: ${error.response.status} - ${error.response.data?.message || '未知错误'}`)
    } else if (error.request) {
      console.error('请求未收到响应:', error.request)
      message.error('删除章节失败: 服务器未响应')
    } else {
      console.error('请求配置错误:', error.message)
      message.error(`删除章节失败: ${error.message}`)
    }
  }
}

// 查看小节详情
const viewSection = (chapter: Chapter, section: Section) => {
  if (!section.id) {
    message.error('小节ID不存在')
    return
  }
  
  router.push(`/teacher/courses/${courseId.value}/sections/${section.id}`)
}

// 显示添加小节弹窗
const showAddSectionModal = (chapter: Chapter) => {
  currentChapter.value = chapter
  isEditingSection.value = false
  editingSectionId.value = null
  sectionForm.value = {
    title: '',
    duration: 30,
    description: ''
  }
  addSectionModalVisible.value = true
}

// 取消添加小节
const cancelAddSection = () => {
  addSectionModalVisible.value = false
  currentChapter.value = null
  sectionForm.value = {
    title: '',
    duration: 30,
    description: ''
  }
}

// 编辑小节
const editSection = (chapter: Chapter, section: Section) => {
  currentChapter.value = chapter
  isEditingSection.value = true
  editingSectionId.value = section.id || null
  sectionForm.value = {
    title: section.title || '',
    duration: section.duration || 30,
    description: section.description || ''
  }
  addSectionModalVisible.value = true
}

// 处理添加或更新小节
const handleAddSection = async () => {
  if (!currentChapter.value || !currentChapter.value.id) {
    message.error('章节信息不完整')
    return
  }
  
  if (!sectionForm.value.title) {
    message.error('请输入小节标题')
    return
  }
  
  try {
    const sectionData: Section = {
      chapterId: currentChapter.value.id,
      title: sectionForm.value.title,
      duration: sectionForm.value.duration,
      description: sectionForm.value.description || ''
    }
    
    let response
    
    if (isEditingSection.value && editingSectionId.value) {
      // 更新小节
      console.log('发送更新小节请求，数据:', sectionData, '小节ID:', editingSectionId.value)
      response = await updateSection(editingSectionId.value, sectionData)
      console.log('更新小节响应:', response)
      
      if (response.data.code === 200) {
        message.success('更新小节成功')
      } else {
        message.error(response.data.message || '更新小节失败')
        console.error('更新小节失败:', response.data)
      }
    } else {
      // 创建新小节
      console.log('发送创建小节请求，数据:', sectionData)
      response = await createSection(sectionData)
      console.log('创建小节响应:', response)
      
      if (response.data.code === 200) {
        message.success('添加小节成功')
      } else {
        message.error(response.data.message || '添加小节失败')
        console.error('添加小节失败:', response.data)
      }
    }
    
    if (response.data.code === 200) {
      addSectionModalVisible.value = false
      currentChapter.value = null
      await loadChapters()
    }
  } catch (error: any) {
    console.error('操作小节异常:', error)
    if (error.response) {
      console.error('错误响应:', error.response.data)
      console.error('状态码:', error.response.status)
      console.error('响应头:', error.response.headers)
      message.error(`操作小节失败: ${error.response.status} - ${error.response.data?.message || '未知错误'}`)
    } else if (error.request) {
      console.error('请求未收到响应:', error.request)
      message.error('操作小节失败: 服务器未响应')
    } else {
      console.error('请求配置错误:', error.message)
      message.error(`操作小节失败: ${error.message}`)
    }
  }
}

// 删除小节
const deleteSectionItem = async (chapter: Chapter, section: Section) => {
  if (!section.id) {
    message.error('小节ID不存在')
    return
  }
  
  try {
    const response = await deleteSection(section.id)
    
    if (response.data.code === 200) {
      message.success('删除小节成功')
      await loadChapters()
    } else {
      message.error(response.data.message || '删除小节失败')
    }
  } catch (error) {
    console.error('删除小节失败:', error)
    message.error('删除小节失败')
  }
}

// 页面加载时直接获取章节列表
onMounted(async () => {
  console.log('页面加载，课程ID:', courseId.value);
  loading.value = true;
  try {
    await loadChapters();
  } catch (error) {
    console.error('页面初始化失败:', error);
  } finally {
    loading.value = false;
  }
})
</script>

<style scoped>
.fixed-width-container {
  width: 98%;
  min-width: 1200px;
  max-width: 2000px;
  margin: 0 auto;
  padding: 20px;
  box-sizing: border-box;
  border-radius: 8px;
}

.course-detail {
  padding: 0;
  background: #f0f2f5;
  min-height: 100vh;
}

.content-wrapper {
  padding: 20px;
  background: #f0f2f5;
  border-radius: 16px;
}

.page-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 32px;
  padding: 20px 28px;
  background: white;
  border-radius: 12px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
}

.back-btn {
  display: flex;
  align-items: center;
  gap: 8px;
  color: #666;
}

.back-btn:hover {
  color: #1890ff;
}

.course-info-card {
  display: flex;
  gap: 28px;
  padding: 28px;
  background: white;
  border-radius: 12px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
  margin-bottom: 32px;
}

.course-cover {
  width: 200px;
  height: 150px;
  border-radius: 8px;
  overflow: hidden;
}

.course-cover img {
  width: 100%;
  height: 100%;
  object-fit: cover;
}

.default-cover {
  width: 100%;
  height: 100%;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  display: flex;
  align-items: center;
  justify-content: center;
  color: white;
  font-size: 48px;
}

.course-details {
  flex: 1;
}

.course-title {
  font-size: 28px;
  font-weight: 600;
  margin-bottom: 16px;
  color: #333;
}

.course-meta {
  display: flex;
  flex-wrap: wrap;
  gap: 16px;
  margin-bottom: 24px;
}

.meta-item {
  display: flex;
  align-items: center;
  gap: 8px;
  color: #666;
}

.course-description,
.course-objectives,
.course-requirements {
  margin-bottom: 16px;
}

.course-description h3,
.course-objectives h3,
.course-requirements h3 {
  font-size: 16px;
  font-weight: 600;
  margin-bottom: 8px;
  color: #333;
}

.course-stats {
  margin-bottom: 32px;
}

.course-time-info {
  margin-bottom: 32px;
}

.chapter-management {
  background: white;
  border-radius: 12px;
  padding: 28px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
}

.chapter-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 28px;
}

.section-title {
  font-size: 20px;
  font-weight: 600;
  margin: 0;
}

.chapter-list {
  display: flex;
  flex-direction: column;
  gap: 20px;
}

.chapter-item {
  border: 1px solid #e8e8e8;
  border-radius: 8px;
  overflow: hidden;
}

.chapter-info {
  display: flex;
  padding: 20px;
}

.chapter-number {
  width: 32px;
  height: 32px;
  border-radius: 50%;
  background-color: #1890ff;
  color: white;
  display: flex;
  align-items: center;
  justify-content: center;
  font-weight: 600;
  margin-right: 16px;
}

.chapter-content {
  flex: 1;
}

.chapter-title {
  font-size: 18px;
  font-weight: 600;
  margin: 0 0 8px 0;
}

.chapter-description {
  color: #666;
  margin-bottom: 16px;
}

.chapter-actions {
  display: flex;
  gap: 8px;
}

.chapter-sections {
  background-color: #f9f9f9;
  border-radius: 4px;
  padding: 8px;
}

.section-item {
  display: flex;
  align-items: center;
  padding: 12px;
  border-bottom: 1px solid #eee;
}

.section-item:last-child {
  border-bottom: none;
}

.section-icon {
  margin-right: 12px;
  font-size: 18px;
  color: #1890ff;
}

.section-content {
  flex: 1;
  cursor: pointer;
}

.section-title {
  font-weight: 500;
}

.section-duration {
  font-size: 12px;
  color: #999;
}

.section-status {
  margin-right: 16px;
}

.section-actions {
  display: flex;
  gap: 8px;
}

.add-section {
  margin-top: 12px;
}

.empty-state {
  display: flex;
  justify-content: center;
  align-items: center;
  height: 400px;
  background: white;
  border-radius: 8px;
}
</style>
