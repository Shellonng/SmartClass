<template>
  <div class="student-ability-graph">
    <div class="page-header">
      <h1>能力图谱</h1>
      <p class="description">全面评估学习能力，发现优势与不足</p>
    </div>

    <div class="ability-overview">
      <div class="overall-score">
        <div class="score-circle">
          <div class="score-value">{{ overallScore }}</div>
          <div class="score-label">综合能力</div>
        </div>
      </div>
      
      <div class="score-breakdown">
        <div class="breakdown-item" v-for="item in scoreBreakdown" :key="item.name">
          <div class="breakdown-name">{{ item.name }}</div>
          <div class="breakdown-progress">
            <a-progress 
              :percent="item.score" 
              :stroke-color="getScoreColor(item.score)"
              :stroke-width="8"
            />
          </div>
          <div class="breakdown-score">{{ item.score }}分</div>
        </div>
      </div>
    </div>

    <div class="ability-details">
      <h2>能力详细分析</h2>
      <div class="abilities-grid">
        <div 
          v-for="ability in abilities" 
          :key="ability.id" 
          class="ability-card"
          :class="getAbilityLevel(ability.score)"
        >
          <div class="ability-header">
            <div class="ability-icon" :style="{ background: ability.color }">
              <component :is="ability.icon" />
            </div>
            <div class="ability-info">
              <h3>{{ ability.name }}</h3>
              <p>{{ ability.description }}</p>
            </div>
          </div>
          
          <div class="ability-score">
            <div class="score-display">
              <span class="current-score">{{ ability.score }}</span>
              <span class="max-score">/100</span>
            </div>
            <div class="score-level">{{ getScoreLevel(ability.score) }}</div>
          </div>

          <div class="ability-progress">
            <a-progress 
              :percent="ability.score" 
              :stroke-color="getScoreColor(ability.score)"
              :show-info="false"
              :stroke-width="6"
            />
          </div>

          <div class="ability-details-btn">
            <a-button size="small" @click="viewAbilityDetail(ability)">
              查看详情
            </a-button>
          </div>
        </div>
      </div>
    </div>

    <div class="improvement-suggestions">
      <h2>提升建议</h2>
      <div class="suggestions-list">
        <div v-for="suggestion in suggestions" :key="suggestion.id" class="suggestion-item">
          <div class="suggestion-icon">
            <BulbOutlined />
          </div>
          <div class="suggestion-content">
            <h4>{{ suggestion.title }}</h4>
            <p>{{ suggestion.description }}</p>
            <div class="suggestion-resources">
              <a-tag v-for="resource in suggestion.resources" :key="resource" color="blue">
                {{ resource }}
              </a-tag>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- 能力详情弹窗 -->
    <a-modal v-model:open="detailModalVisible" title="能力详情" width="600px">
      <div v-if="selectedAbility" class="ability-detail-modal">
        <div class="detail-header">
          <div class="detail-icon" :style="{ background: selectedAbility.color }">
            <component :is="selectedAbility.icon" />
          </div>
          <div class="detail-info">
            <h3>{{ selectedAbility.name }}</h3>
            <p>{{ selectedAbility.description }}</p>
          </div>
        </div>

        <div class="detail-metrics">
          <h4>评估维度</h4>
          <div class="metrics-list">
            <div v-for="metric in selectedAbility.metrics" :key="metric.name" class="metric-item">
              <span class="metric-name">{{ metric.name }}</span>
              <div class="metric-progress">
                <a-progress 
                  :percent="metric.score" 
                  :stroke-color="getScoreColor(metric.score)"
                  size="small"
                />
              </div>
              <span class="metric-score">{{ metric.score }}分</span>
            </div>
          </div>
        </div>

        <div class="detail-history">
          <h4>能力变化趋势</h4>
          <div class="history-chart">
            <p>能力变化图表区域</p>
          </div>
        </div>
      </div>
    </a-modal>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue'
import { BulbOutlined, BookOutlined, CodeOutlined, TeamOutlined, ThunderboltOutlined } from '@ant-design/icons-vue'

interface Ability {
  id: number
  name: string
  description: string
  score: number
  color: string
  icon: any
  metrics: Array<{ name: string; score: number }>
}

interface Suggestion {
  id: number
  title: string
  description: string
  resources: string[]
}

const detailModalVisible = ref(false)
const selectedAbility = ref<Ability | null>(null)

const overallScore = ref(82)

const scoreBreakdown = ref([
  { name: '理论基础', score: 85 },
  { name: '实践能力', score: 78 },
  { name: '创新思维', score: 80 },
  { name: '协作能力', score: 88 }
])

const abilities = ref<Ability[]>([
  {
    id: 1,
    name: '数学基础',
    description: '数学概念理解和运算能力',
    score: 85,
    color: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
    icon: BookOutlined,
    metrics: [
      { name: '概念理解', score: 88 },
      { name: '计算能力', score: 82 },
      { name: '逻辑推理', score: 86 }
    ]
  },
  {
    id: 2,
    name: '编程能力',
    description: '代码编写和问题解决能力',
    score: 78,
    color: 'linear-gradient(135deg, #f093fb 0%, #f5576c 100%)',
    icon: CodeOutlined,
    metrics: [
      { name: '语法掌握', score: 85 },
      { name: '算法思维', score: 75 },
      { name: '调试能力', score: 80 }
    ]
  },
  {
    id: 3,
    name: '团队协作',
    description: '团队合作和沟通交流能力',
    score: 88,
    color: 'linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)',
    icon: TeamOutlined,
    metrics: [
      { name: '沟通表达', score: 90 },
      { name: '协调配合', score: 86 },
      { name: '冲突处理', score: 88 }
    ]
  },
  {
    id: 4,
    name: '创新思维',
    description: '创造性思考和问题解决能力',
    score: 80,
    color: 'linear-gradient(135deg, #43e97b 0%, #38f9d7 100%)',
    icon: ThunderboltOutlined,
    metrics: [
      { name: '发散思维', score: 82 },
      { name: '批判思维', score: 78 },
      { name: '解决方案', score: 80 }
    ]
  }
])

const suggestions = ref<Suggestion[]>([
  {
    id: 1,
    title: '加强算法思维训练',
    description: '编程能力中算法思维相对薄弱，建议多做算法练习题，培养逻辑思维能力',
    resources: ['算法导论', 'LeetCode练习', '数据结构课程']
  },
  {
    id: 2,
    title: '提升批判思维能力',
    description: '在创新思维方面，批判性思考能力有提升空间，建议多参与讨论和辩论',
    resources: ['逻辑学基础', '批判性思维训练', '案例分析']
  }
])

const getScoreColor = (score: number) => {
  if (score >= 80) return '#52c41a'
  if (score >= 60) return '#1890ff'
  if (score >= 40) return '#faad14'
  return '#ff4d4f'
}

const getScoreLevel = (score: number) => {
  if (score >= 90) return '优秀'
  if (score >= 80) return '良好'
  if (score >= 70) return '中等'
  if (score >= 60) return '及格'
  return '待提升'
}

const getAbilityLevel = (score: number) => {
  if (score >= 80) return 'high'
  if (score >= 60) return 'medium'
  return 'low'
}

const viewAbilityDetail = (ability: Ability) => {
  selectedAbility.value = ability
  detailModalVisible.value = true
}
</script>

<style scoped>
.student-ability-graph {
  padding: 24px;
  max-width: 1200px;
  margin: 0 auto;
}

.page-header {
  text-align: center;
  margin-bottom: 32px;
}

.page-header h1 {
  font-size: 28px;
  margin-bottom: 8px;
  color: #333;
}

.description {
  color: #666;
  font-size: 16px;
}

/* 能力概览 */
.ability-overview {
  background: white;
  border-radius: 12px;
  padding: 32px;
  margin-bottom: 32px;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
  display: flex;
  gap: 32px;
  align-items: center;
}

.overall-score {
  text-align: center;
}

.score-circle {
  width: 120px;
  height: 120px;
  border-radius: 50%;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  color: white;
}

.score-value {
  font-size: 32px;
  font-weight: 700;
}

.score-label {
  font-size: 12px;
  opacity: 0.9;
}

.score-breakdown {
  flex: 1;
}

.breakdown-item {
  display: flex;
  align-items: center;
  gap: 16px;
  margin-bottom: 16px;
}

.breakdown-item:last-child {
  margin-bottom: 0;
}

.breakdown-name {
  width: 80px;
  font-weight: 500;
  color: #333;
}

.breakdown-progress {
  flex: 1;
}

.breakdown-score {
  width: 50px;
  text-align: right;
  font-weight: 600;
  color: #1890ff;
}

/* 能力详情 */
.ability-details {
  background: white;
  border-radius: 12px;
  padding: 32px;
  margin-bottom: 32px;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
}

.ability-details h2 {
  margin-bottom: 24px;
  color: #333;
}

.abilities-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
  gap: 20px;
}

.ability-card {
  background: #fafafa;
  border-radius: 12px;
  padding: 20px;
  border: 1px solid #f0f0f0;
  transition: all 0.3s ease;
}

.ability-card:hover {
  box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1);
  transform: translateY(-2px);
}

.ability-card.high {
  border-left: 4px solid #52c41a;
}

.ability-card.medium {
  border-left: 4px solid #1890ff;
}

.ability-card.low {
  border-left: 4px solid #ff4d4f;
}

.ability-header {
  display: flex;
  gap: 12px;
  margin-bottom: 16px;
}

.ability-icon {
  width: 40px;
  height: 40px;
  border-radius: 8px;
  display: flex;
  align-items: center;
  justify-content: center;
  color: white;
  font-size: 18px;
  flex-shrink: 0;
}

.ability-info h3 {
  margin-bottom: 4px;
  color: #333;
  font-size: 16px;
}

.ability-info p {
  color: #666;
  font-size: 12px;
  margin: 0;
}

.ability-score {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 12px;
}

.score-display {
  font-size: 24px;
  font-weight: 700;
  color: #333;
}

.max-score {
  font-size: 14px;
  color: #999;
}

.score-level {
  font-size: 12px;
  padding: 2px 8px;
  background: #1890ff;
  color: white;
  border-radius: 10px;
}

.ability-progress {
  margin-bottom: 16px;
}

.ability-details-btn {
  text-align: center;
}

/* 提升建议 */
.improvement-suggestions {
  background: white;
  border-radius: 12px;
  padding: 32px;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
}

.improvement-suggestions h2 {
  margin-bottom: 24px;
  color: #333;
}

.suggestions-list {
  display: flex;
  flex-direction: column;
  gap: 20px;
}

.suggestion-item {
  display: flex;
  gap: 16px;
  padding: 20px;
  background: #f8f9fa;
  border-radius: 8px;
  border-left: 4px solid #faad14;
}

.suggestion-icon {
  width: 40px;
  height: 40px;
  background: #faad14;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  color: white;
  flex-shrink: 0;
}

.suggestion-content h4 {
  margin-bottom: 8px;
  color: #333;
}

.suggestion-content p {
  color: #666;
  margin-bottom: 12px;
  line-height: 1.5;
}

.suggestion-resources {
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
}

/* 弹窗样式 */
.ability-detail-modal {
  padding: 20px 0;
}

.detail-header {
  display: flex;
  gap: 16px;
  margin-bottom: 24px;
}

.detail-icon {
  width: 60px;
  height: 60px;
  border-radius: 12px;
  display: flex;
  align-items: center;
  justify-content: center;
  color: white;
  font-size: 24px;
}

.detail-info h3 {
  margin-bottom: 8px;
  color: #333;
}

.detail-info p {
  color: #666;
  margin: 0;
}

.detail-metrics,
.detail-history {
  margin-bottom: 24px;
}

.detail-metrics h4,
.detail-history h4 {
  margin-bottom: 16px;
  color: #333;
}

.metrics-list {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.metric-item {
  display: flex;
  align-items: center;
  gap: 16px;
}

.metric-name {
  width: 80px;
  font-size: 14px;
  color: #333;
}

.metric-progress {
  flex: 1;
}

.metric-score {
  width: 50px;
  text-align: right;
  font-size: 14px;
  font-weight: 600;
  color: #1890ff;
}

.history-chart {
  height: 200px;
  background: #f8f9fa;
  border-radius: 8px;
  display: flex;
  align-items: center;
  justify-content: center;
  border: 1px solid #d9d9d9;
  color: #666;
}
</style> 