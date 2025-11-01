<template>
  <div class="video-learning">
    <a-spin :spinning="loading">
      <div class="video-header">
        <a-button type="text" @click="goBack" class="back-btn">
          <ArrowLeftOutlined />
          返回课程详情
        </a-button>
        <h1>{{ videoTitle }}</h1>
        <div class="video-meta">
          <span><BookOutlined /> 课程：{{ courseName }}</span>
          <span><ClockCircleOutlined /> 时长：{{ duration }}</span>
          <span><BarChartOutlined /> 进度：{{ progress }}%</span>
        </div>
      </div>

      <div class="video-container">
        <div class="video-player">
          <video
            v-if="videoUrl"
            ref="videoRef"
            controls
            autoplay
            class="video-element"
            @timeupdate="handleTimeUpdate"
            @ended="handleVideoEnded"
          >
            <source :src="videoUrl" type="video/mp4">
            您的浏览器不支持 HTML5 视频播放。
          </video>
          <div v-else class="video-placeholder">
            <PlayCircleOutlined />
            <p>视频加载中或暂无视频资源</p>
          </div>
        </div>
        
        <div class="progress-bar">
          <a-progress :percent="progress" :stroke-color="progressColor" />
        </div>

        <div class="video-notes">
          <h3>学习笔记</h3>
          <a-textarea
            v-model:value="notes"
            placeholder="在这里记录学习笔记..."
            :rows="4"
            @change="saveNotes"
          />
          <div class="notes-actions">
            <a-button type="primary" @click="saveNotes">
              <SaveOutlined />
              保存笔记
            </a-button>
          </div>
        </div>
      </div>

      <div class="video-stats">
        <div class="stat-item">
          <h4>观看次数</h4>
          <span>{{ viewCount }}</span>
        </div>
        <div class="stat-item">
          <h4>完成率</h4>
          <span>{{ completionRate }}%</span>
        </div>
        <div class="stat-item">
          <h4>平均观看时长</h4>
          <span>{{ avgWatchTime }}</span>
        </div>
      </div>
      
      <div class="video-navigation">
        <a-button 
          v-if="prevVideoId" 
          @click="navigateToVideo(prevVideoId)"
        >
          <LeftOutlined />
          上一节
        </a-button>
        <a-button 
          type="primary" 
          v-if="nextVideoId" 
          @click="navigateToVideo(nextVideoId)"
        >
          下一节
          <RightOutlined />
        </a-button>
      </div>
    </a-spin>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted, onUnmounted, watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { message } from 'ant-design-vue'
import { 
  ArrowLeftOutlined,
  BookOutlined,
  ClockCircleOutlined,
  BarChartOutlined,
  PlayCircleOutlined,
  SaveOutlined,
  LeftOutlined,
  RightOutlined
} from '@ant-design/icons-vue'
import { startLearningRecord, endLearningRecord } from '@/api/learningRecord'

const route = useRoute()
const router = useRouter()

// 从路由参数获取课程ID和视频ID
const courseId = ref<number>(Number(route.params.id) || 0)
const videoId = ref<number>(Number(route.params.videoId) || 0)

// 基本数据
const loading = ref<boolean>(true)
const videoTitle = ref<string>('')
const courseName = ref<string>('')
const duration = ref<string>('')
const progress = ref<number>(0)
const viewCount = ref<number>(0)
const completionRate = ref<number>(0)
const avgWatchTime = ref<string>('')
const notes = ref<string>('')
const videoUrl = ref<string>('')
const videoRef = ref<HTMLVideoElement | null>(null)

// 学习记录相关
const learningRecordId = ref<number | null>(null)
const lastProgressUpdate = ref<number>(0)
const progressUpdateInterval = 15 // 每15秒更新一次进度

// 导航数据
const prevVideoId = ref<number | null>(null)
const nextVideoId = ref<number | null>(null)

// 加载视频数据
const loadVideoData = async () => {
  try {
    loading.value = true
    
    // 模拟API调用
    await new Promise(resolve => setTimeout(resolve, 1000))
    
    // 模拟数据
    videoTitle.value = '计算机系统的层次结构'
    courseName.value = '计算机组成原理'
    duration.value = '30:00'
    progress.value = 0
    viewCount.value = 1
    completionRate.value = 0
    avgWatchTime.value = '0:00'
    notes.value = ''
    
    // 模拟视频URL (使用一个示例视频)
    videoUrl.value = 'https://www.w3schools.com/html/mov_bbb.mp4'
    
    // 模拟导航数据
    prevVideoId.value = videoId.value > 101 ? videoId.value - 1 : null
    nextVideoId.value = videoId.value < 304 ? videoId.value + 1 : null
    
    // 创建学习记录
    await createLearningRecord()
    
  } catch (error) {
    console.error('加载视频数据失败:', error)
    message.error('加载视频数据失败')
  } finally {
    loading.value = false
  }
}

// 创建学习记录
const createLearningRecord = async () => {
  try {
    const response = await startLearningRecord({
      courseId: courseId.value,
      sectionId: videoId.value,
      resourceType: 'video'
    })
    
    if (response?.data?.success) {
      learningRecordId.value = response.data.recordId
      console.log('创建学习记录成功, ID:', learningRecordId.value)
    }
  } catch (error) {
    console.error('创建学习记录失败:', error)
  }
}

// 更新学习记录
const updateLearningRecord = async (isCompleted: boolean = false) => {
  if (!learningRecordId.value) return
  
  try {
    await endLearningRecord(
      learningRecordId.value,
      progress.value,
      isCompleted
    )
    console.log('更新学习记录成功')
  } catch (error) {
    console.error('更新学习记录失败:', error)
  }
}

// 定期更新学习进度
const updateProgress = () => {
  if (!videoRef.value) return
  
  const currentProgress = Math.floor((videoRef.value.currentTime / videoRef.value.duration) * 100)
  
  // 如果进度变化超过5%或者距离上次更新时间超过一定间隔，则更新记录
  if (Math.abs(currentProgress - lastProgressUpdate.value) >= 5) {
    progress.value = currentProgress
    lastProgressUpdate.value = currentProgress
    updateLearningRecord(false)
  }
}

// 返回课程详情
const goBack = async () => {
  // 在离开页面前更新学习记录
  await updateLearningRecord()
  router.push(`/student/courses/${courseId.value}`)
}

// 处理视频进度更新
const handleTimeUpdate = () => {
  if (!videoRef.value) return
  
  const video = videoRef.value
  const currentProgress = Math.floor((video.currentTime / video.duration) * 100)
  progress.value = currentProgress
  
  // 如果进度超过90%，标记为已完成
  if (currentProgress > 90 && completionRate.value === 0) {
    completionRate.value = 100
    updateLearningRecord(true)
    message.success('恭喜您完成本节学习！')
  }
  
  // 定期更新学习记录
  updateProgress()
}

// 处理视频播放结束
const handleVideoEnded = () => {
  completionRate.value = 100
  updateLearningRecord(true)
  message.success('恭喜您完成本节学习！')
  
  // 自动跳转到下一节（如果有）
  if (nextVideoId.value) {
    setTimeout(() => {
      message.info('即将跳转到下一节...')
      navigateToVideo(nextVideoId.value as number)
    }, 2000)
  }
}

// 保存笔记
const saveNotes = () => {
  message.success('笔记保存成功')
  // 实际应该调用API保存笔记
}

// 导航到其他视频
const navigateToVideo = async (id: number) => {
  // 在切换视频前更新当前视频的学习记录
  await updateLearningRecord()
  router.push(`/student/courses/${courseId.value}/video/${id}`)
}

// 进度条颜色
const progressColor = computed(() => {
  if (progress.value >= 80) return '#52c41a'
  if (progress.value >= 60) return '#1890ff'
  return '#faad14'
})

// 监听路由参数变化
watch(
  () => route.params,
  (newParams) => {
    const newVideoId = Number(newParams.videoId)
    if (newVideoId !== videoId.value) {
      videoId.value = newVideoId
      loadVideoData()
    }
  }
)

// 在组件卸载前更新学习记录
onUnmounted(async () => {
  await updateLearningRecord()
})

onMounted(() => {
  loadVideoData()
})
</script>

<style scoped>
.video-learning {
  padding: 24px;
  max-width: 1200px;
  margin: 0 auto;
}

.video-header {
  margin-bottom: 24px;
}

.back-btn {
  display: block;
  margin-bottom: 16px;
  font-size: 16px;
  padding: 0;
}

.video-header h1 {
  margin-bottom: 8px;
  font-size: 24px;
}

.video-meta {
  display: flex;
  gap: 24px;
  color: #666;
}

.video-container {
  background: white;
  border-radius: 8px;
  padding: 24px;
  margin-bottom: 24px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
}

.video-player {
  height: 400px;
  background: #f5f5f5;
  border-radius: 8px;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  margin-bottom: 16px;
  overflow: hidden;
}

.video-element {
  width: 100%;
  height: 100%;
  background: #000;
}

.video-placeholder {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  height: 100%;
  width: 100%;
  color: #999;
}

.video-placeholder .anticon {
  font-size: 48px;
  margin-bottom: 16px;
}

.progress-bar {
  margin-bottom: 24px;
}

.video-notes {
  margin-top: 24px;
}

.video-notes h3 {
  margin-bottom: 16px;
}

.notes-actions {
  margin-top: 16px;
  display: flex;
  justify-content: flex-end;
}

.video-stats {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 16px;
  margin-bottom: 24px;
}

.stat-item {
  background: white;
  padding: 20px;
  border-radius: 8px;
  text-align: center;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
}

.stat-item h4 {
  margin-bottom: 8px;
  color: #666;
}

.stat-item span {
  font-size: 24px;
  font-weight: 600;
  color: #1890ff;
}

.video-navigation {
  display: flex;
  justify-content: space-between;
}
</style> 