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
          
          <!-- 删除导航菜单 -->
        </div>

        <!-- 章节管理内容 -->
        <div v-if="currentView === 'chapters'" class="chapter-management">
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

        <!-- 讨论区管理内容 -->
        <div v-if="currentView === 'discussions'" class="management-content">
          <div class="content-header">
            <h2 class="section-title">讨论区管理</h2>
          </div>
          <div class="content-body">
            <a-spin :spinning="commentsLoading">
              <div v-if="courseComments.length === 0" class="empty-comments">
                <a-empty description="暂无讨论内容" />
              </div>
              <div v-else class="comments-list">
                <a-list
                  :data-source="courseComments"
                  item-layout="horizontal"
                >
                  <template #renderItem="{ item }">
                    <a-list-item>
                      <a-list-item-meta>
                        <template #avatar>
                          <template v-if="item.userAvatar">
                            <a-avatar :src="item.userAvatar" />
                          </template>
                          <template v-else>
                            <a-avatar>{{ item.userName?.charAt(0) }}</a-avatar>
                          </template>
                        </template>
                        <template #title>
                          <div class="comment-header">
                            <span class="comment-user">{{ item.userName }}</span>
                            <span class="comment-role">{{ getUserRole(item.userRole) }}</span>
                            <span class="comment-section" @click="navigateToSection(item.sectionId)">
                              <a-tag color="blue">{{ item.sectionTitle || '未知小节' }}</a-tag>
                            </span>
                          </div>
                        </template>
                        <template #description>
                          <div class="comment-content">{{ item.content }}</div>
                          <div class="comment-footer">
                            <span class="comment-time">{{ formatDate(item.createTime) }}</span>
                            <span 
                              v-if="item.replyCount > 0" 
                              class="comment-replies-link" 
                              @click.stop="fetchAndShowReplies(item)"
                            >
                              <span v-if="!expandedComments.has(item.id)">
                                <DownOutlined /> 查看{{ item.replyCount }}条回复
                              </span>
                              <span v-else>
                                <UpOutlined /> 收起回复
                              </span>
                            </span>
                          </div>
                          
                          <!-- 显示回复列表 -->
                          <div v-if="expandedComments.has(item.id) && item.replies && item.replies.length > 0" class="replies-list">
                            <div v-for="reply in item.replies" :key="reply.id" class="reply-item">
                              <div class="reply-avatar">
                                <template v-if="reply.userAvatar">
                                  <a-avatar :src="reply.userAvatar" size="small" />
                                </template>
                                <template v-else>
                                  <a-avatar size="small">{{ reply.userName?.charAt(0) }}</a-avatar>
                                </template>
                              </div>
                              <div class="reply-content">
                                <div>
                                  <span class="reply-username">{{ reply.userName }}</span>
                                  <span class="reply-role">({{ getUserRole(reply.userRole) }})</span>
                                  <span class="reply-text">：{{ reply.content }}</span>
                                </div>
                                <div class="reply-footer">
                                  <span class="reply-time">{{ formatDate(reply.createTime) }}</span>
                                  <a-popconfirm
                                    title="确定要删除这条回复吗？"
                                    @confirm="deleteComment(reply)"
                                    ok-text="确定"
                                    cancel-text="取消"
                                  >
                                    <a class="delete-link">删除</a>
                                  </a-popconfirm>
                                </div>
                              </div>
                            </div>
                          </div>
                        </template>
                      </a-list-item-meta>
                      <template #actions>
                        <a-button type="link" @click="navigateToSection(item.sectionId)">查看</a-button>
                        <a-popconfirm
                          title="确定要删除这条评论吗？"
                          @confirm="deleteComment(item)"
                          ok-text="确定"
                          cancel-text="取消"
                        >
                          <a-button type="link" danger>删除</a-button>
                        </a-popconfirm>
                      </template>
                    </a-list-item>
                  </template>
                </a-list>
                
                <div class="pagination">
                  <a-pagination
                    v-model:current="commentPagination.current"
                    :total="commentPagination.total"
                    :page-size="commentPagination.pageSize"
                    @change="handleCommentPageChange"
                    show-quick-jumper
                    show-size-changer
                    :page-size-options="['10', '20', '50', '100']"
                    @showSizeChange="handleCommentSizeChange"
                  />
                </div>
              </div>
            </a-spin>
          </div>
        </div>

        <!-- 资料管理内容 -->
        <div v-if="currentView === 'resources'" class="management-content">
          <CourseResources :courseId="courseId" />
        </div>

        <!-- 题库管理内容 -->
        <div v-if="currentView === 'question-bank'" class="management-content">
          <QuestionBank :courseId="courseId" />
        </div>

        <!-- 考试管理内容 -->
        <div v-if="currentView === 'exams'" class="management-content">
          <CourseExams :courseId="courseId" />
        </div>

        <!-- 作业管理内容 -->
        <div v-if="currentView === 'assignments'" class="management-content">
          <CourseAssignments :courseId="courseId" />
        </div>

        <!-- 错题集管理内容 -->
        <div v-if="currentView === 'wrongbook'" class="management-content">
          <div class="content-header">
            <h2 class="section-title">错题集管理</h2>
            <a-button type="primary">
              <PlusOutlined />
              添加错题
            </a-button>
          </div>
          <div class="content-body">
            <a-empty description="暂无错题内容" />
          </div>
        </div>

        <!-- 学习记录管理内容 -->
        <div v-if="currentView === 'records'" class="management-content">
          <div class="content-header">
            <h2 class="section-title">学习记录管理</h2>
          </div>
          <div class="content-body">
            <a-empty description="暂无学习记录" />
          </div>
        </div>

        <!-- 知识图谱管理内容 -->
        <div v-if="currentView === 'knowledge-map'" class="management-content">
          <div class="content-header">
            <h2 class="section-title">知识图谱管理</h2>
            <a-button type="primary">
              <PlusOutlined />
              添加知识点
            </a-button>
          </div>
          <div class="content-body">
            <a-empty description="暂无知识图谱内容" />
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
import { ref, computed, onMounted, watch } from 'vue'
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
  FileOutlined,
  DownOutlined,
  UpOutlined
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
import { getCourseComments, deleteSectionComment, getSectionCommentReplies } from '@/api/course'
import CourseResources from './CourseResources.vue'
import axios from 'axios'
import dayjs from 'dayjs'
import QuestionBank from './QuestionBank.vue'
import CourseExams from './CourseExams.vue'
import CourseAssignments from './CourseAssignments.vue'

const route = useRoute()
const router = useRouter()

// 当前视图状态
const currentView = ref('chapters')

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

// 评论相关数据
const commentsLoading = ref(false)
const courseComments = ref<any[]>([])
const commentPagination = ref({
  current: 1,
  pageSize: 10,
  total: 0
})

// 展开评论相关
const expandedComments = ref(new Set<number>())

// 计算属性
const courseId = computed(() => {
  // 先尝试从路由参数中获取id
  const idParam = route.params.id;
  // 确保idParam是字符串
  const idStr = typeof idParam === 'string' ? idParam : Array.isArray(idParam) ? idParam[0] : '';
  // 转换为数字
  const id = parseInt(idStr);
  
  console.log('CourseDetail - 计算courseId:', { 
    routeParams: route.params,
    idParam,
    idStr,
    id,
    isNaN: isNaN(id),
    final: isNaN(id) ? -1 : id
  });
  
  return isNaN(id) ? -1 : id;
})

// 监听路由变化，更新当前视图
watch(() => route.query.view, (newView) => {
  if (newView) {
    currentView.value = newView as string
  } else {
    currentView.value = 'chapters'
  }
}, { immediate: true })

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

// 加载课程评论
const loadCourseComments = async () => {
  commentsLoading.value = true
  try {
    const response = await getCourseComments(
      courseId.value,
      commentPagination.value.current,
      commentPagination.value.pageSize
    )
    
    if (response.data.code === 200) {
      courseComments.value = response.data.data.records || []
      commentPagination.value.total = response.data.data.total || 0
    } else {
      message.error(response.data.message || '获取评论失败')
      courseComments.value = []
    }
  } catch (error) {
    console.error('获取课程评论失败:', error)
    message.error('获取评论失败')
    courseComments.value = []
  } finally {
    commentsLoading.value = false
  }
}

// 处理评论分页变化
const handleCommentPageChange = (page: number) => {
  commentPagination.value.current = page
  loadCourseComments()
}

// 处理评论每页条数变化
const handleCommentSizeChange = (current: number, size: number) => {
  commentPagination.value.current = 1
  commentPagination.value.pageSize = size
  loadCourseComments()
}

// 删除评论
const deleteComment = async (comment: any) => {
  try {
    const response = await deleteSectionComment(comment.sectionId, comment.id)
    
    if (response.data.code === 200) {
      message.success('删除评论成功')
      // 重新加载评论列表，确保前后端数据一致
      await loadCourseComments()
    } else {
      message.error(response.data.message || '删除评论失败')
    }
  } catch (error) {
    console.error('删除评论失败:', error)
    message.error('删除评论失败')
  }
}

// 跳转到小节详情页
const navigateToSection = (sectionId: number) => {
  router.push(`/teacher/courses/${courseId.value}/sections/${sectionId}`)
}

// 格式化日期
const formatDate = (dateStr: string) => {
  return dayjs(dateStr).format('YYYY-MM-DD HH:mm')
}

// 获取用户角色显示文本
const getUserRole = (role: string) => {
  if (!role) return '用户'
  
  const roleMap: Record<string, string> = {
    'TEACHER': '教师',
    'STUDENT': '学员',
    'ADMIN': '管理员'
  }
  
  return roleMap[role.toUpperCase()] || '用户'
}

// 监听视图变化，加载相应数据
watch(() => currentView.value, (newView) => {
  if (newView === 'discussions') {
    loadCourseComments()
  }
})

// 组件挂载时初始化数据
onMounted(() => {
  loadChapters()
  if (currentView.value === 'discussions') {
    loadCourseComments()
  }
})

// 展开评论
const fetchAndShowReplies = async (comment: any) => {
  if (expandedComments.value.has(comment.id)) {
    expandedComments.value.delete(comment.id)
  } else {
    expandedComments.value.add(comment.id)
    await loadReplies(comment)
  }
}

// 加载回复
const loadReplies = async (comment: any) => {
  try {
    const response = await getSectionCommentReplies(comment.sectionId, comment.id)
    
    if (response.data.code === 200) {
      comment.replies = response.data.data.records || []
    } else {
      message.error(response.data.message || '获取回复失败')
      comment.replies = []
    }
  } catch (error) {
    console.error('获取回复失败:', error)
    message.error('获取回复失败')
    comment.replies = []
  }
}
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
  flex-direction: column;
  margin-bottom: 24px;
}

.header-left {
  display: flex;
  align-items: center;
  margin-bottom: 16px;
}

.back-btn {
  padding-left: 0;
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

.management-content {
  background: white;
  border-radius: 12px;
  padding: 28px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
}

.content-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 28px;
}

.content-body {
  min-height: 400px;
  display: flex;
  flex-direction: column;
}

.comments-list {
  background: #fff;
  border-radius: 8px;
}

.comment-header {
  display: flex;
  align-items: center;
  gap: 8px;
}

.comment-user {
  font-weight: 500;
  color: #333;
}

.comment-role {
  font-size: 12px;
  color: #666;
  background-color: #f0f0f0;
  padding: 2px 6px;
  border-radius: 10px;
}

.comment-section {
  cursor: pointer;
}

.comment-content {
  margin: 8px 0;
  color: #333;
}

.comment-footer {
  display: flex;
  align-items: center;
  gap: 16px;
  font-size: 12px;
  color: #999;
}

.pagination {
  margin-top: 24px;
  display: flex;
  justify-content: center;
}

.comment-replies-link {
  cursor: pointer;
  color: #1890ff;
}

.replies-list {
  margin-top: 12px;
  padding-left: 20px;
}

.reply-item {
  display: flex;
  align-items: center;
  margin-bottom: 8px;
}

.reply-avatar {
  margin-right: 8px;
}

.reply-content {
  flex: 1;
}

.reply-username {
  font-weight: 500;
  color: #333;
}

.reply-role {
  font-size: 12px;
  color: #666;
  background-color: #f0f0f0;
  padding: 2px 6px;
  border-radius: 10px;
}

.reply-text {
  color: #333;
}

.reply-footer {
  display: flex;
  align-items: center;
  gap: 16px;
  font-size: 12px;
  color: #999;
}

.delete-link {
  cursor: pointer;
  color: #1890ff;
}
</style>
