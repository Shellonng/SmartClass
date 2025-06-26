<template>
  <div class="video-learning">
    <div class="video-header">
      <h1>{{ videoTitle }}</h1>
      <div class="video-meta">
        <span>课程：{{ courseName }}</span>
        <span>时长：{{ duration }}</span>
        <span>进度：{{ progress }}%</span>
      </div>
    </div>

    <div class="video-container">
      <div class="video-player">
        <p>视频播放器区域</p>
        <p>视频ID: {{ videoId }}</p>
      </div>
      
      <div class="progress-bar">
        <a-progress :percent="progress" :stroke-color="progressColor" />
      </div>

      <div class="video-heatmap">
        <h3>学习热力图</h3>
        <div class="heatmap-container">
          <p>显示重复观看片段和跳过片段</p>
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
  </div>
</template>

<script setup lang="ts">
import { ref, computed } from 'vue'

interface Props {
  id: string
  videoId: string
}

const props = defineProps<Props>()

const videoTitle = ref('第一章：函数基础')
const courseName = ref('高等数学')
const duration = ref('45:30')
const progress = ref(68)
const viewCount = ref(3)
const completionRate = ref(85)
const avgWatchTime = ref('38:20')

const progressColor = computed(() => {
  if (progress.value >= 80) return '#52c41a'
  if (progress.value >= 60) return '#1890ff'
  return '#faad14'
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

.video-header h1 {
  margin-bottom: 8px;
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
  border: 2px dashed #d9d9d9;
}

.progress-bar {
  margin-bottom: 24px;
}

.video-heatmap h3 {
  margin-bottom: 16px;
}

.heatmap-container {
  height: 100px;
  background: #f5f5f5;
  border-radius: 8px;
  display: flex;
  align-items: center;
  justify-content: center;
  border: 1px solid #d9d9d9;
}

.video-stats {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 16px;
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
</style> 