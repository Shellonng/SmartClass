<template>
  <div class="section-detail-container">
    <!-- 顶部导航栏 -->
    <nav class="top-nav">
      <div class="nav-left">
        <div class="logo">
          <img src="/logo-mini.svg" alt="SmartClass" />
          <span>SmartClass</span>
        </div>
          <a-button type="text" @click="goBack" class="back-btn">
            <ArrowLeftOutlined />
            返回课程
          </a-button>
        </div>
      <div class="nav-right">
        <a-dropdown>
          <a-button type="text">
            <UserOutlined />
            {{ userInfo?.realName || userInfo?.username || '测试教师' }}
            <DownOutlined />
          </a-button>
          <template #overlay>
            <a-menu>
              <a-menu-item key="profile">个人信息</a-menu-item>
              <a-menu-item key="settings">设置</a-menu-item>
              <a-menu-divider />
              <a-menu-item key="logout" @click="logout">退出登录</a-menu-item>
            </a-menu>
          </template>
        </a-dropdown>
      </div>
    </nav>

    <!-- 主要内容区域 -->
    <div class="main-container">
      <!-- 左侧目录 -->
      <aside class="sidebar">
        <div class="sidebar-header">
          <h3>目录</h3>
          <span class="course-info">{{ courseTitle }}</span>
        </div>
        
        <div class="chapter-list">
          <a-collapse 
            v-model:activeKey="activeChapters"
            :bordered="false"
            expand-icon-position="end"
          >
            <a-collapse-panel 
              v-for="chapter in chapters" 
              :key="chapter.id" 
              :header="chapter.title"
            >
              <div class="section-list">
                <div
                  v-for="(section, index) in chapter.sections"
                  :key="section.id"
                >
                  <div
                    class="section-item"
                    :class="{ active: currentSectionId === section.id }"
                    @click="handleSectionClick(section)"
                  >
                    <div class="section-title">{{ section.title }}</div>
                    <div class="section-duration">{{ section.duration }}分钟</div>
                    <div class="section-actions">
                      <EditOutlined @click.stop="handleEdit(section)" />
                      <DeleteOutlined class="delete" @click.stop="handleDelete(section)" />
                    </div>
                  </div>
                </div>
              </div>
            </a-collapse-panel>
          </a-collapse>
        </div>
      </aside>

      <!-- 右侧内容区 -->
      <div class="content">
        <a-spin :spinning="loading">
          <div class="content-header">
            <h1>{{ section?.title || '加载中...' }}</h1>
            <a-button type="primary" @click="editSection">
                <EditOutlined />
                编辑
              </a-button>
            </div>
            
          <div class="content-body">
            <!-- 视频区域 -->
            <div class="video-section">
              <div v-if="section?.videoUrl" class="video-container">
                <video 
                  ref="videoPlayer"
                  class="video-player"
                  controls
                  preload="auto"
                  :src="getVideoUrl"
                >
                  您的浏览器不支持视频播放
                </video>
              </div>
              <div v-else class="empty-video">
                <PlayCircleOutlined class="empty-icon" />
                <p>暂无视频内容</p>
                <a-upload
                  accept="video/*"
                  :show-upload-list="false"
                  :before-upload="handleBeforeUpload"
                >
                  <a-button type="primary">
                    <UploadOutlined />
                    上传视频
                  </a-button>
                </a-upload>
              </div>
            </div>
            
            <!-- 小节简介 -->
            <div class="section-info">
              <h2>小节简介</h2>
              <p>{{ section?.description || '暂无简介' }}</p>
              </div>

            <!-- 评论区 -->
            <div class="section-comments">
              <h3>讨论区</h3>
              
              <!-- 评论列表 -->
              <div class="comment-list">
                <a-spin :spinning="commentLoading">
                  <a-empty v-if="!comments.length" description="暂无评论" />
                  <div v-else class="comments-container">
                    <!-- 遍历根评论 -->
                    <div v-for="comment in comments" :key="comment.id" class="comment-thread">
                      <!-- 主评论 -->
                      <div class="comment-main">
                        <div class="comment-avatar">
                          <template v-if="comment.userAvatar">
                            <img :src="comment.userAvatar" :alt="comment.userName" />
                          </template>
                          <div v-else class="avatar-placeholder">
                            {{ getFirstChar(comment.userName) }}
                          </div>
                        </div>
                        <div class="comment-content">
                          <div class="comment-header">
                            <span class="username">{{ comment.userName }}</span>
                            <span class="user-title">{{ getUserTitle(comment.userRole) }}</span>
                          </div>
                          <div class="comment-text">{{ comment.content }}</div>
                          <div class="comment-footer">
                            <span class="comment-time">{{ formatDate(comment.createTime) }}</span>
                            <div class="comment-actions">
                              <span class="action-item" @click="showReplyInput(comment)">回复</span>
                              <template v-if="isCurrentUser(comment.userId)">
                                <span class="action-item" @click="showEditInput(comment)">编辑</span>
                                <a-popconfirm
                                  title="确定要删除这条评论吗？"
                                  @confirm="deleteComment(comment)"
                                >
                                  <span class="action-item delete">删除</span>
                                </a-popconfirm>
                              </template>
            </div>
          </div>
          
                          <!-- 追评 -->
                          <div v-if="comment.additionalComment" class="additional-comment">
                            <div class="additional-header">{{ comment.additionalDays || 14 }}天后追评：</div>
                            <div class="additional-content">{{ comment.additionalComment }}</div>
            </div>
            
                          <!-- 显示回复数量，点击展开/收起回复 -->
                          <div v-if="comment.replyCount > 0" class="reply-toggle" @click="toggleReplies(comment)">
                            <span v-if="!expandedComments.has(comment.id)">
                              <DownOutlined /> 查看{{ comment.replyCount }}条回复
                            </span>
                            <span v-else>
                              <UpOutlined /> 收起回复
                            </span>
                          </div>
                          
                          <!-- 回复列表 -->
                          <div v-if="expandedComments.has(comment.id) && comment.replies && comment.replies.length > 0" class="replies-container">
                            <div v-for="reply in comment.replies" :key="reply.id" class="reply-item">
                              <div class="reply-avatar">
                                <template v-if="reply.userAvatar">
                                  <img :src="reply.userAvatar" :alt="reply.userName" />
                                </template>
                                <div v-else class="avatar-placeholder small">
                                  {{ getFirstChar(reply.userName) }}
                                </div>
                              </div>
                              <div class="reply-content">
                                <span class="reply-username">{{ reply.userName }}</span>
                                <span class="reply-role">({{ getUserTitle(reply.userRole) }})</span>
                                <span class="reply-text">：{{ reply.content }}</span>
                                <div class="reply-footer">
                                  <span class="reply-time">{{ formatDate(reply.createTime) }}</span>
                                  <div class="reply-actions" v-if="isCurrentUser(reply.userId)">
                                    <span class="action-item" @click="showEditInput(reply)">编辑</span>
                                    <a-popconfirm
                                      title="确定要删除这条回复吗？"
                                      @confirm="deleteComment(reply)"
                                    >
                                      <span class="action-item delete">删除</span>
                                    </a-popconfirm>
                    </div>
                                </div>
                              </div>
                            </div>
                          </div>

                          <!-- 回复输入框 -->
                          <div v-if="replyingTo?.id === comment.id" class="reply-input">
                            <div class="input-wrapper">
                              <a-textarea
                                v-model:value="replyContent"
                                placeholder="回复评论..."
                                :rows="2"
                                :maxlength="500"
                              />
                              <div class="comment-tip">
                                还可以输入{{ 500 - replyContent.length }}字
                              </div>
                            </div>
                            <div class="reply-actions">
                              <a-button @click="cancelReply">取消</a-button>
                              <a-button type="primary" :loading="submitting" @click="submitReply(comment)">
                                提交回复
                      </a-button>
                            </div>
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                </a-spin>
              </div>

              <!-- 分页器 - 调整位置，确保有足够的空间 -->
              <div class="pagination-container">
                <a-pagination
                  v-model:current="pagination.current"
                  :total="pagination.total"
                  :pageSize="pagination.pageSize"
                  @change="pagination.onChange"
                  show-size-changer
                  :pageSizeOptions="['10', '20', '50']"
                  @showSizeChange="handleSizeChange"
                />
              </div>
            </div>

            <!-- 评论输入框 -->
            <div class="comment-input">
              <div class="comment-input-container">
                <div class="input-area">
                  <div class="input-wrapper">
                    <a-textarea
                      v-model:value="newComment"
                      placeholder="发表你的看法..."
                      :rows="4"
                      :maxlength="500"
                    />
                    <div class="comment-tip">
                      还可以输入{{ 500 - newComment.length }}字
                    </div>
                  </div>
                </div>
                <div class="button-area">
                  <a-button type="primary" :loading="submitting" @click="submitComment" class="submit-btn">
                    发表评论
                      </a-button>
                    </div>
                  </div>
                </div>
          </div>
        </a-spin>
          </div>
        </div>
        
    <!-- 编辑弹窗 -->
        <a-modal
          v-model:open="editModalVisible"
          :title="isCreating ? '添加小节' : '编辑小节'"
          @ok="handleSaveSection"
          @cancel="cancelEditSection"
      width="600px"
        >
          <a-form :model="sectionForm" layout="vertical">
            <a-form-item label="小节标题" required>
              <a-input v-model:value="sectionForm.title" placeholder="请输入小节标题" />
            </a-form-item>
            <a-form-item label="预计时长">
          <a-input-number 
            v-model:value="sectionForm.duration" 
            :min="1" 
            :max="300" 
            addonAfter="分钟"
            style="width: 100%"
          />
            </a-form-item>
            <a-form-item label="内容描述">
          <a-textarea 
            v-model:value="sectionForm.description" 
            placeholder="请输入内容描述"
            :rows="4"
          />
            </a-form-item>
            <a-form-item label="视频链接">
          <div class="video-url-input">
            <a-input 
              v-model:value="sectionForm.videoUrl" 
              placeholder="请输入视频链接" 
              :disabled="true"
              style="flex: 1;"
            />
            <a-button 
              v-if="sectionForm.videoUrl" 
              type="primary" 
              danger 
              @click="removeVideo"
            >
              <DeleteOutlined />
              移除视频
            </a-button>
          </div>
            </a-form-item>
          </a-form>
        </a-modal>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted, onBeforeUnmount, watch, nextTick } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { message, Modal } from 'ant-design-vue'
import { useAuthStore } from '@/stores/auth'
import {
  ArrowLeftOutlined,
  EditOutlined,
  DeleteOutlined,
  PlayCircleOutlined,
  ClockCircleOutlined,
  CheckCircleOutlined,
  UserOutlined,
  DownOutlined,
  UploadOutlined,
  UpOutlined
} from '@ant-design/icons-vue'
import {
  getChaptersByCourseId,
  getSectionById,
  updateSection,
  deleteSection,
  type Chapter,
  type Section,
  uploadSectionVideo
} from '@/api/teacher'
import {
  getSectionComments,
  createSectionComment,
  updateSectionComment,
  deleteSectionComment,
  getSectionCommentReplies
} from '@/api/course'
import videojs from 'video.js'
import 'video.js/dist/video-js.css'
import axios from 'axios'
import { formatDate } from '@/utils/date'

const route = useRoute()
const router = useRouter()
const authStore = useAuthStore()
const loading = ref(false)
const chapters = ref<Chapter[]>([])
const section = ref<Section | null>(null)
const activeChapters = ref<string[]>([])
const courseTitle = ref('')

// 用户信息
const userInfo = computed(() => authStore.user)

// 编辑相关状态
const editModalVisible = ref(false)
const isCreating = ref(false)
const sectionForm = ref({
  title: '',
  duration: 30,
  description: '',
  videoUrl: ''
})

// 计算当前小节ID
const currentSectionId = ref(Number(route.params.sectionId))

// 计算当前课程ID
const courseId = computed(() => {
  return parseInt(route.params.courseId as string)
})

// 处理视频URL
const getVideoUrl = computed(() => {
  if (!section.value?.videoUrl) return ''
  return `http://localhost:8080/resource/video/${section.value.videoUrl}`
})

let player: any = null

// 初始化视频播放器
const initializePlayer = () => {
  if (player) {
    player.dispose()
  }
  
  const videoElement = document.querySelector('.video-js')
  if (videoElement) {
    player = videojs(videoElement, {
      controls: true,
      fluid: true,
      preload: 'auto',
      playbackRates: [0.5, 1, 1.5, 2],
      sources: [{
        src: getVideoUrl.value,
        type: 'video/mp4'
      }]
    })
  }
}

// 监听视频URL变化
watch(() => getVideoUrl.value, (newUrl) => {
  if (newUrl) {
    nextTick(() => {
      initializePlayer()
    })
  }
})

// 初始化数据
const initializeData = async () => {
  const sectionId = Number(route.params.sectionId)
  const courseId = Number(route.params.courseId)
  
  if (!sectionId || !courseId) {
    message.error('参数错误')
    return
  }

  loading.value = true
  try {
    // 并行加载数据
    const [chaptersRes, sectionRes] = await Promise.all([
      getChaptersByCourseId(courseId),
      getSectionById(sectionId)
    ])

    // 处理章节数据
    if (chaptersRes.data.code === 200) {
      chapters.value = chaptersRes.data.data || []
      
      // 找到当前小节所在的章节，并设置为展开状态
      const currentChapter = chapters.value.find(chapter => 
        chapter.sections?.some(s => s.id === sectionId)
      )
      if (currentChapter) {
        activeChapters.value = [currentChapter.id!.toString()]
        courseTitle.value = currentChapter.title || '课程'
      }
    }

    // 处理小节数据
    if (sectionRes.data.code === 200 && sectionRes.data.data) {
      section.value = sectionRes.data.data
      
      // 更新表单数据
      if (section.value) {
        sectionForm.value = {
          title: section.value.title || '',
          duration: section.value.duration || 30,
          description: section.value.description || '',
          videoUrl: section.value.videoUrl || ''
        }
      }
      
      // 加载评论数据
      await fetchComments()
      } else {
      // 如果找不到小节数据，可能是已被删除
      message.warning('找不到小节数据，可能已被删除')
      // 跳转回课程页面
      router.push(`/teacher/courses/${courseId}`)
    }
  } catch (error) {
    console.error('加载数据失败:', error)
    message.error('加载数据失败')
    // 发生错误时也跳转回课程页面
    router.push(`/teacher/courses/${courseId}`)
  } finally {
    loading.value = false
  }
}

// 监听路由参数变化
watch(
  () => route.params,
  (newParams) => {
    if (newParams.sectionId) {
      currentSectionId.value = Number(newParams.sectionId)
      initializeData()
    }
  },
  { deep: true, immediate: true }
)

// 组件挂载时初始化数据
onMounted(() => {
  // 初始化数据已经在watch中通过immediate: true触发
  console.log('组件挂载，初始化数据');
})

// 组件卸载时清理
onBeforeUnmount(() => {
  if (player) {
    player.dispose()
    player = null
  }
  // 清理展开状态，避免内存泄漏
  expandedComments.value.clear()
})

// 返回课程详情页
const goBack = () => {
  router.push(`/teacher/courses/${courseId.value}`)
}

// 跳转到指定小节
const navigateToSection = (sectionId: number) => {
  if (sectionId === currentSectionId.value) return
    router.push(`/teacher/courses/${courseId.value}/sections/${sectionId}`)
}

// 编辑小节
const editSection = () => {
  if (section.value) {
  sectionForm.value = {
    title: section.value.title || '',
    duration: section.value.duration || 30,
    description: section.value.description || '',
    videoUrl: section.value.videoUrl || ''
  }
    isCreating.value = false
  editModalVisible.value = true
  }
}

// 编辑小节项
const editSectionItem = (item: Section) => {
  sectionForm.value = {
    title: item.title || '',
    duration: item.duration || 30,
    description: item.description || '',
    videoUrl: item.videoUrl || ''
  }
  isCreating.value = false
  editModalVisible.value = true
}

// 保存小节
const handleSaveSection = async () => {
  try {
    if (!sectionForm.value.title.trim()) {
      message.error('请输入小节标题')
      return
    }

    if (!section.value?.id) {
    message.error('小节ID不存在')
    return
  }
  
    const sectionData = {
      ...sectionForm.value,
      chapterId: section.value.chapterId || 0
    }

    const response = await updateSection(section.value.id, sectionData)
    if (response.data.code === 200) {
      message.success('保存成功')
      editModalVisible.value = false
      initializeData() // 重新加载数据
      } else {
      message.error(response.data.message || '保存失败')
    }
  } catch (error) {
    console.error('保存失败:', error)
    message.error('保存失败，请重试')
  }
}

// 取消编辑
const cancelEditSection = () => {
  editModalVisible.value = false
  sectionForm.value = {
    title: '',
    duration: 30,
    description: '',
    videoUrl: ''
  }
}

// 退出登录
const logout = async () => {
  try {
    await authStore.logout()
    router.push('/login')
  } catch (error) {
    console.error('退出登录失败:', error)
    message.error('退出登录失败')
  }
}

// 处理视频上传
const handleBeforeUpload = async (file: File) => {
  try {
    // 检查文件类型
    if (!file.type.startsWith('video/')) {
      message.error('只能上传视频文件！');
      return false;
    }
    
    // 检查文件大小（限制为500MB）
    const maxSize = 500 * 1024 * 1024;
    if (file.size > maxSize) {
      message.error('视频文件大小不能超过500MB！');
      return false;
    }

    // 显示上传中提示
    const hide = message.loading('视频上传中...', 0);
    
    try {
      const response = await uploadSectionVideo(Number(currentSectionId.value), file);
    if (response.data.code === 200) {
        message.success('视频上传成功');
        // 重新加载小节数据
        initializeData();
    } else {
        message.error(response.data.message || '视频上传失败');
    }
    } finally {
      hide();
    }

    return false; // 阻止默认上传行为
  } catch (error) {
    console.error('视频上传失败:', error);
    message.error('视频上传失败，请重试');
    return false;
  }
};

// 移除视频
const removeVideo = async () => {
  try {
    if (!section.value?.chapterId) {
      throw new Error('章节ID不存在');
    }
    const updatedSection = {
      ...sectionForm.value,
      chapterId: section.value.chapterId,
      videoUrl: undefined
    } as Section;
    const response = await updateSection(currentSectionId.value, updatedSection);
    if (response.data.code === 200) {
      message.success('视频已移除');
      sectionForm.value.videoUrl = '';
      await initializeData(); // 重新加载数据
    } else {
      message.error('移除视频失败');
    }
  } catch (error) {
    console.error('移除视频时发生错误:', error);
    message.error('移除视频失败');
  }
};

// 处理小节点击
const handleSectionClick = (section: any) => {
  // 如果已经在当前小节，不需要跳转
  if (currentSectionId.value === section.id) {
    return
  }
  
  // 使用路由跳转到新的小节
  router.push({
    path: `/teacher/courses/${courseId.value}/sections/${section.id}`
  }).catch((err) => {
    console.error('路由跳转失败:', err)
    message.error('页面跳转失败')
  })
}

const handleEdit = (section: Section) => {
  editSectionItem(section);
};

const handleDelete = async (section: Section) => {
  Modal.confirm({
    title: '确认删除',
    content: `确定要删除小节"${section.title}"吗？`,
    okText: '确定',
    cancelText: '取消',
    okType: 'danger',
    async onOk() {
      try {
        const response = await deleteSection(section.id!);
        if (response.data.code === 200) {
          message.success('删除成功');
          
          // 获取当前章节的所有小节（在删除前）
          const currentChapter = chapters.value?.find(c => c.sections?.some(s => s.id === section.id));
          
          // 如果是当前正在查看的小节被删除，需要跳转
          if (section.id === currentSectionId.value) {
            if (currentChapter?.sections) {
              // 查找除了当前小节外的其他小节
              const otherSections = currentChapter.sections.filter(s => s.id !== section.id);
              
              if (otherSections.length > 0) {
                // 如果还有其他小节，跳转到第一个小节
                router.push(`/teacher/courses/${courseId.value}/sections/${otherSections[0].id}`);
              } else {
                // 如果没有其他小节了，返回到课程页面
                router.push(`/teacher/courses/${courseId.value}`);
              }
            } else {
              // 如果找不到章节信息，直接返回课程页面
              router.push(`/teacher/courses/${courseId.value}`);
            }
          } else {
            // 如果删除的不是当前查看的小节，只需重新加载数据
            await initializeData();
          }
        } else {
          message.error(response.data.message || '删除失败');
        }
      } catch (error) {
        message.error('删除失败');
        console.error('Delete section error:', error);
      }
    }
  });
};

// 评论相关数据
const comments = ref<any[]>([])
const submitting = ref(false)
const newComment = ref('')
const commentLoading = ref(false)
const pagination = ref({
  current: 1,
  pageSize: 10,
  total: 0,
  onChange: (page: number) => {
    pagination.value.current = page
    fetchComments()
  }
})

// 获取评论列表并处理父子关系
const fetchComments = async () => {
  commentLoading.value = true
  try {
    const sectionId = Number(route.params.sectionId)
    console.log('正在获取评论，小节ID:', sectionId)
    const res = await getSectionComments(
      sectionId,
      pagination.value.current,
      pagination.value.pageSize
    )
    console.log('获取评论响应:', res)
    if (res.data && res.data.code === 200) {
      // 清空已展开评论的集合
      expandedComments.value.clear()
      
      // 处理评论数据，构建父子关系
      const commentMap = new Map()
      const rootComments: any[] = []
      
      // 先将所有评论放入 Map 中
      res.data.data.records.forEach((comment: any) => {
        comment.replies = []
        // 确保userRole字段存在
        if (!comment.userRole && comment.userId) {
          // 如果后端没有提供userRole，根据用户ID判断是否为当前用户，并使用当前用户的角色
          if (comment.userId === authStore.user?.id) {
            comment.userRole = authStore.user?.role?.toLowerCase();
          } else {
            // 默认设置为student，后续可以根据实际情况调整
            comment.userRole = 'student';
          }
        }
        commentMap.set(comment.id, comment)
      })
      
      // 构建父子关系
      res.data.data.records.forEach((comment: any) => {
        if (comment.parentId) {
          const parentComment = commentMap.get(comment.parentId)
          if (parentComment) {
            // 确保子评论也有userRole
            if (!comment.userRole && comment.userId) {
              if (comment.userId === authStore.user?.id) {
                comment.userRole = authStore.user?.role?.toLowerCase();
              } else {
                comment.userRole = 'student';
              }
            }
            parentComment.replies.push(comment)
            
            // 更新父评论的回复计数
            parentComment.replyCount = (parentComment.replyCount || 0) + 1
          }
        } else {
          rootComments.push(comment)
        }
      })
      
      comments.value = rootComments
      pagination.value.total = res.data.data.total || 0
      console.log('处理后的评论列表:', comments.value)
      
      // 自动展开有回复的评论
      comments.value.forEach(comment => {
        if (comment.replies && comment.replies.length > 0) {
          expandedComments.value.add(comment.id)
        }
      })
    } else {
      console.warn('获取评论响应异常:', res.data)
      comments.value = []
      pagination.value.total = 0
    }
  } catch (error) {
    console.error('获取评论失败:', error)
    message.error('获取评论失败')
    comments.value = []
    pagination.value.total = 0
  } finally {
    commentLoading.value = false
  }
}

// 回复相关的状态
const replyingTo = ref<any>(null)
const replyContent = ref('')

// 显示回复输入框
const showReplyInput = (comment: any) => {
  replyingTo.value = comment
  replyContent.value = ''
}

// 取消回复
const cancelReply = () => {
  replyingTo.value = null
  replyContent.value = ''
}

// 提交回复
const submitReply = async (parentComment: any) => {
  if (!replyContent.value.trim()) {
    message.warning('请输入回复内容')
    return
  }

  submitting.value = true
  try {
    const sectionId = Number(route.params.sectionId)
    const res = await createSectionComment(sectionId, {
      content: replyContent.value.trim(),
      parentId: parentComment.id
    })
    
    if (res.data && res.data.code === 200) {
      // 如果后端返回了完整的评论对象，包括用户角色
      if (res.data.data && res.data.data.id) {
        const newReply = res.data.data;
        
        // 确保有用户角色
        if (!newReply.userRole && authStore.user?.role) {
          newReply.userRole = authStore.user.role.toLowerCase();
        }
        
        // 立即更新界面显示
        if (!parentComment.replies) {
          parentComment.replies = [];
        }
        parentComment.replies.push(newReply);
        
        // 更新回复计数
        parentComment.replyCount = (parentComment.replyCount || 0) + 1;
        
        // 自动展开回复
        expandedComments.value.add(parentComment.id);
      } else {
        // 后端没有返回完整对象，手动构造
        const newReply = {
          id: res.data.data || Date.now(), // 如果没有返回id，使用时间戳作为临时id
          content: replyContent.value.trim(),
          userId: authStore.user?.id,
          userName: authStore.user?.realName || authStore.user?.username || '用户',
          userAvatar: authStore.user?.avatar,
          userRole: authStore.user?.role?.toLowerCase(), // 添加用户角色
          createTime: new Date().toISOString(),
          parentId: parentComment.id
        };
        
        // 立即更新界面显示
        if (!parentComment.replies) {
          parentComment.replies = [];
        }
        parentComment.replies.push(newReply);
        
        // 更新回复计数
        parentComment.replyCount = (parentComment.replyCount || 0) + 1;
        
        // 自动展开回复
        expandedComments.value.add(parentComment.id);
      }
      
      message.success('回复成功')
      replyContent.value = ''
      replyingTo.value = null
    } else {
      console.warn('提交回复响应异常:', res.data)
      message.error(res.data?.message || '回复失败')
    }
  } catch (error) {
    console.error('提交回复失败:', error)
    message.error('回复失败')
  } finally {
    submitting.value = false
  }
}

// 获取用户名的第一个字符
const getFirstChar = (name: string | undefined | null): string => {
  if (!name) return '用';
  return name.charAt(0);
}

// 检查是否是当前用户的评论
const isCurrentUser = (userId: number) => {
  return userId === authStore.user?.id
}

// 评论编辑和回复
const editingComment = ref<any>(null)

const showEditInput = (comment: any) => {
  editingComment.value = { ...comment }
  newComment.value = comment.content
}

// 提交评论
const submitComment = async () => {
  if (!newComment.value.trim()) {
    message.warning('请输入评论内容')
    return
  }

  submitting.value = true
  try {
    const sectionId = Number(route.params.sectionId)
    console.log('正在提交评论，小节ID:', sectionId)
    const res = await createSectionComment(sectionId, {
      content: newComment.value.trim()
    })
    console.log('提交评论响应:', res)
    if (res.data && res.data.code === 200) {
      message.success('评论成功')
      newComment.value = ''
      await fetchComments()
    } else {
      console.warn('提交评论响应异常:', res.data)
      message.error(res.data?.message || '评论失败')
    }
  } catch (error) {
    console.error('提交评论失败:', error)
    message.error('评论失败')
  } finally {
    submitting.value = false
  }
}

// 删除评论
const deleteComment = async (comment: any) => {
  try {
    const sectionId = Number(route.params.sectionId)
    console.log('正在删除评论，小节ID:', sectionId, '评论ID:', comment.id)
    const res = await deleteSectionComment(sectionId, comment.id)
    console.log('删除评论响应:', res)
    if (res.data && res.data.code === 200) {
      message.success('删除成功')
      
      // 删除后重新获取评论列表，确保前后端数据一致
      await fetchComments()
    } else {
      console.warn('删除评论响应异常:', res.data)
      message.error(res.data?.message || '删除失败')
    }
  } catch (error) {
    console.error('删除评论失败:', error)
    message.error('删除失败')
  }
}

// 获取用户标题
const getUserTitle = (userRole: string | undefined): string => {
  if (!userRole) return '用户';
  
  // 统一转为小写处理
  const role = userRole.toLowerCase();
  
  switch (role) {
    case 'teacher':
      return '教师';
    case 'student':
      return '学员';
    case 'admin':
      return '管理员';
    default:
      return '用户';
  }
}

// 回复相关的状态
const expandedComments = ref<Set<number>>(new Set())

const toggleReplies = async (comment: any) => {
  if (expandedComments.value.has(comment.id)) {
    // 收起回复
    expandedComments.value.delete(comment.id)
  } else {
    // 展开回复
    expandedComments.value.add(comment.id)
    
    // 如果回复数大于0但回复数组为空或不存在，则加载回复
    if ((comment.replyCount > 0) && (!comment.replies || comment.replies.length === 0)) {
      try {
        const sectionId = Number(route.params.sectionId)
        console.log('加载评论回复，小节ID:', sectionId, '评论ID:', comment.id)
        const res = await getSectionCommentReplies(sectionId, comment.id)
        
        if (res.data && res.data.code === 200) {
          // 更新回复数据
          comment.replies = res.data.data.records || []
          console.log('成功加载回复:', comment.replies)
        } else {
          console.warn('加载回复响应异常:', res.data)
          message.error(res.data?.message || '加载回复失败')
        }
      } catch (error) {
        console.error('加载回复失败:', error)
        message.error('加载回复失败')
      }
    }
  }
}

const handleSizeChange = (pageSize: number) => {
  pagination.value.pageSize = pageSize
  fetchComments()
}
</script>

<style scoped>
.section-detail-container {
  min-height: 100vh;
  display: flex;
  flex-direction: column;
  background-color: #fff;
}

.top-nav {
  height: 60px;
  padding: 0 24px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  background-color: #fff;
  border-bottom: 1px solid #f0f0f0;
  position: sticky;
  top: 0;
  z-index: 100;
}

.nav-left {
  display: flex;
  align-items: center;
  gap: 24px;
}

.logo {
  display: flex;
  align-items: center;
  gap: 8px;
}

.logo img {
  height: 32px;
}

.logo span {
  font-size: 18px;
  font-weight: 600;
  color: #1890ff;
}

.back-btn {
  display: flex;
  align-items: center;
  gap: 4px;
}

.main-container {
  flex: 1;
  display: flex;
  height: calc(100vh - 60px);
}

.sidebar {
  width: 280px;
  border-right: 1px solid #f0f0f0;
  overflow-y: auto;
  background-color: #f9f9f9;
  box-shadow: 2px 0 8px rgba(0, 0, 0, 0.05);
}

.sidebar-header {
  padding: 16px;
  border-bottom: 1px solid #f0f0f0;
  background-color: #fff;
}

.sidebar-header h3 {
  margin: 0 0 8px 0;
  font-size: 18px;
  font-weight: 600;
  color: #333;
}

.course-info {
  font-size: 14px;
  color: #666;
}

.chapter-list {
  flex: 1;
  overflow-y: auto;
  padding: 0;
}

.section-item {
  display: flex;
  align-items: center;
  padding: 12px 16px 12px 32px;
  cursor: pointer;
  transition: all 0.2s;
  position: relative;
  background: #fff;
  font-size: 14px;
  border-bottom: 1px solid #f5f5f5;
  margin-bottom: 2px;
}

.section-item:hover {
  background: #f6f6f6;
}

.section-item.active {
  background: #e6f7ff;
  border-left: 3px solid #1890ff;
}

.section-item.active .section-title {
  color: #1890ff;
  font-weight: 500;
}

.section-title {
  flex: 1;
  color: #555;
  margin-right: 8px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.section-duration {
  font-size: 12px;
  color: #999;
  margin-right: 8px;
  flex-shrink: 0;
  background-color: #f5f5f5;
  padding: 2px 6px;
  border-radius: 10px;
}

.section-actions {
  display: flex;
  align-items: center;
  gap: 8px;
  flex-shrink: 0;
}

.section-actions .anticon {
  font-size: 14px;
  color: #666;
  cursor: pointer;
}

.section-actions .anticon:hover {
  color: #1890ff;
}

.section-actions .delete:hover {
  color: #ff4d4f;
}

.video-url-input {
  display: flex;
  gap: 8px;
  align-items: center;
}

:deep(.ant-collapse) {
  border: none;
  background: transparent;
}

:deep(.ant-collapse-item) {
  border: none;
}

:deep(.ant-collapse-header) {
  padding: 12px 16px !important;
  font-size: 15px !important;
  color: #333 !important;
  font-weight: 500 !important;
}

:deep(.ant-collapse-content-box) {
  padding: 0 !important;
}

:deep(.ant-collapse-arrow) {
  font-size: 12px !important;
  color: #999 !important;
}

.section-title-area {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 24px;
}

.section-title-area h1 {
  font-size: 24px;
  margin: 0;
  color: #333;
}

.section-description {
  color: #666;
  font-size: 14px;
  line-height: 1.6;
}

.section-comments {
  margin-top: 24px;
}

.comment-list {
  margin-bottom: 24px;
}

.comments-container {
  display: flex;
  flex-direction: column;
  gap: 24px;
}

.comment-thread {
  border-bottom: 1px solid #f0f0f0;
  padding-bottom: 24px;
}

.comment-thread:last-child {
  border-bottom: none;
}

.comment-main {
  display: flex;
  gap: 16px;
}

.comment-avatar {
  flex-shrink: 0;
}

.comment-avatar img {
  width: 40px;
  height: 40px;
  border-radius: 50%;
  object-fit: cover;
}

.avatar-placeholder {
  width: 40px;
  height: 40px;
  border-radius: 50%;
  background-color: #1890ff;
  color: white;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 18px;
  font-weight: bold;
}

.avatar-placeholder.small {
  width: 30px;
  height: 30px;
  font-size: 14px;
}

.comment-content {
  flex: 1;
}

.comment-header {
  margin-bottom: 8px;
}

.username {
  font-weight: 500;
  margin-right: 8px;
}

.user-title {
  font-size: 12px;
  color: #999;
  background-color: #f5f5f5;
  padding: 2px 6px;
  border-radius: 10px;
}

.comment-text {
  margin-bottom: 8px;
  line-height: 1.6;
}

.comment-footer {
  display: flex;
  justify-content: space-between;
  align-items: center;
  font-size: 12px;
  color: #999;
}

.comment-time {
  margin-right: 16px;
}

.comment-actions {
  display: flex;
  gap: 16px;
}

.action-item {
  cursor: pointer;
  color: #1890ff;
}

.action-item:hover {
  color: #40a9ff;
}

.action-item.delete {
  color: #ff4d4f;
}

.action-item.delete:hover {
  color: #ff7875;
}

.reply-toggle {
  margin-top: 12px;
  color: #1890ff;
  cursor: pointer;
  font-size: 13px;
  display: inline-block;
  padding: 4px 8px;
  background-color: #f0f8ff;
  border-radius: 4px;
}

.reply-toggle:hover {
  background-color: #e6f7ff;
}

.replies-container {
  margin-top: 16px;
  padding: 12px;
  background-color: #f9f9f9;
  border-radius: 4px;
}

.reply-item {
  display: flex;
  gap: 8px;
  margin-bottom: 8px;
  padding: 8px 0;
  border-bottom: 1px solid #f0f0f0;
}

.reply-item:last-child {
  border-bottom: none;
  margin-bottom: 0;
}

.reply-avatar {
  flex-shrink: 0;
}

.reply-avatar img {
  width: 30px;
  height: 30px;
  border-radius: 50%;
  object-fit: cover;
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
  color: #999;
  margin-left: 4px;
}

.reply-text {
  margin-left: 4px;
}

.reply-footer {
  display: flex;
  justify-content: space-between;
  align-items: center;
  font-size: 12px;
  color: #999;
  margin-top: 4px;
}

.reply-input {
  margin: 16px 0 8px 0;
  padding: 12px;
  background-color: #f9f9f9;
  border-radius: 4px;
}

.input-wrapper {
  position: relative;
  margin-bottom: 16px;
}

.input-wrapper :deep(.ant-input) {
  resize: none;
  padding-bottom: 24px;
}

.comment-tip {
  position: absolute;
  right: 8px;
  bottom: 8px;
  font-size: 12px;
  color: #999;
}

.reply-actions {
  display: flex;
  justify-content: flex-end;
  gap: 8px;
}

.comment-input {
  background: #fff;
  border: 1px solid #e8e8e8;
  border-radius: 4px;
  padding: 16px;
  margin-top: 24px;
}

.comment-input-container {
  display: flex;
  align-items: flex-start;
  gap: 16px;
}

.input-area {
  flex: 1;
}

.input-wrapper {
  position: relative;
}

.input-wrapper :deep(.ant-input) {
  resize: none;
  padding-bottom: 24px;
}

.comment-tip {
  position: absolute;
  right: 8px;
  bottom: 8px;
  font-size: 12px;
  color: #999;
}

.button-area {
  padding-top: 8px;
}

.submit-btn {
  min-width: 100px;
  height: 40px;
}

.pagination-container {
  margin: 32px 0;
  display: flex;
  justify-content: center;
}

.content {
  flex: 1;
  overflow-y: auto;
  padding: 24px;
  background-color: #f5f5f5;
}

.content-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 24px;
  padding-bottom: 16px;
  border-bottom: 1px solid #f0f0f0;
}

.content-header h1 {
  margin: 0;
  font-size: 24px;
  font-weight: 600;
  color: #333;
}

.video-section {
  margin-bottom: 24px;
  background: #fff;
  border-radius: 8px;
  overflow: hidden;
}

.video-container {
  width: 100%;
  max-width: 1200px;
  margin: 0 auto;
  aspect-ratio: 16 / 9;
  background: #000;
  border-radius: 8px;
  overflow: hidden;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
}

.video-player {
  width: 100%;
  height: 100%;
  object-fit: contain;
}

.empty-video {
  aspect-ratio: 16 / 9;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  background: #fafafa;
  border: 2px dashed #e8e8e8;
  border-radius: 8px;
  padding: 24px;
}

.empty-icon {
  font-size: 48px;
  color: #d9d9d9;
  margin-bottom: 16px;
}

.section-info {
  background: #fff;
  padding: 24px;
  border-radius: 8px;
  margin-top: 24px;
}

.section-info h2 {
  font-size: 18px;
  font-weight: 600;
  margin-bottom: 16px;
  color: #333;
}

.section-info p {
  color: #666;
  line-height: 1.6;
}
</style> 