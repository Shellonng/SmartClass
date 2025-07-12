<template>
  <div class="knowledge-graph-management">
    <div class="page-header">
      <h1>知识图谱管理</h1>
      <p>智能生成和管理课程知识图谱，帮助学生更好地理解知识结构</p>
    </div>

    <!-- 操作工具栏 -->
    <div class="toolbar">
      <div class="toolbar-left">
        <el-input
          v-model="searchKeyword"
          placeholder="搜索知识图谱..."
          style="width: 300px"
          clearable
          @input="handleSearch"
        >
          <template #prefix>
            <el-icon><Search /></el-icon>
          </template>
        </el-input>
        
        <el-select
          v-model="filterCourse"
          placeholder="选择课程"
          style="width: 200px; margin-left: 12px"
          clearable
          @change="handleFilter"
        >
          <el-option
            v-for="course in courseList"
            :key="course.id || ''"
            :label="course.title || course.name || '未命名课程'"
            :value="course.id || ''"
          />
        </el-select>
      </div>
      
      <div class="toolbar-right">
        <el-button
          type="primary"
          @click="showGenerateDialog = true"
          :icon="Plus"
        >
          生成知识图谱
        </el-button>
      </div>
    </div>

    <!-- 图谱列表 -->
    <div class="graph-grid">
      <div
        v-for="graph in filteredGraphs"
        :key="graph.id"
        class="graph-card"
        @click="viewGraph(graph)"
      >
        <div class="card-header">
          <h3>{{ graph.title }}</h3>
          <el-dropdown @command="handleCommand">
            <el-button type="text" size="small">
              <el-icon><MoreFilled /></el-icon>
            </el-button>
            <template #dropdown>
              <el-dropdown-menu>
                <el-dropdown-item :command="`edit-${graph.id}`">
                  <el-icon><Edit /></el-icon>
                  编辑
                </el-dropdown-item>
                <el-dropdown-item :command="`copy-${graph.id}`">
                  <el-icon><CopyDocument /></el-icon>
                  复制
                </el-dropdown-item>
                <el-dropdown-item :command="`share-${graph.id}`">
                  <el-icon><Share /></el-icon>
                  分享
                </el-dropdown-item>
                <el-dropdown-item :command="`delete-${graph.id}`" divided>
                  <el-icon><Delete /></el-icon>
                  删除
                </el-dropdown-item>
              </el-dropdown-menu>
            </template>
          </el-dropdown>
        </div>
        
        <div class="card-content">
          <p class="description">{{ graph.description || '暂无描述' }}</p>
          
          <div class="card-info">
            <div class="info-item">
              <el-icon><Document /></el-icon>
              <span>{{ graph.courseName }}</span>
            </div>
            <div class="info-item">
              <el-icon><View /></el-icon>
              <span>{{ graph.viewCount || 0 }} 次查看</span>
            </div>
            <div class="info-item">
              <el-icon><Clock /></el-icon>
              <span>{{ formatDate(graph.updateTime) }}</span>
            </div>
          </div>
          
          <div class="card-tags">
            <el-tag :type="getGraphTypeTagType(graph.graphType)" size="small">
              {{ getGraphTypeLabel(graph.graphType) }}
            </el-tag>
            <el-tag
              :type="graph.isPublic ? 'success' : 'info'"
              size="small"
              style="margin-left: 8px"
            >
              {{ graph.isPublic ? '公开' : '私有' }}
            </el-tag>
          </div>
        </div>
      </div>
      
      <!-- 空状态 -->
      <div v-if="filteredGraphs.length === 0" class="empty-state">
        <el-empty
          description="暂无知识图谱"
          :image-size="120"
        >
          <el-button
            type="primary"
            @click="showGenerateDialog = true"
          >
            创建第一个知识图谱
          </el-button>
        </el-empty>
      </div>
    </div>

    <!-- 分页 -->
    <div v-if="total > pageSize" class="pagination">
      <el-pagination
        v-model:current-page="currentPage"
        v-model:page-size="pageSize"
        :total="total"
        :page-sizes="[10, 20, 50]"
        layout="total, sizes, prev, pager, next, jumper"
        @size-change="handleSizeChange"
        @current-change="handleCurrentChange"
      />
    </div>

    <!-- 生成知识图谱对话框 -->
    <el-dialog
      v-model="showGenerateDialog"
      title="生成知识图谱"
      width="600px"
      :close-on-click-modal="false"
    >
      <el-form
        ref="generateFormRef"
        :model="generateForm"
        :rules="generateRules"
        label-width="120px"
      >
        <el-form-item label="选择课程" prop="courseId">
          <el-select
            v-model="generateForm.courseId"
            placeholder="请选择课程"
            style="width: 100%"
            @change="onCourseChange"
          >
            <el-option
              v-for="course in courseList"
              :key="course.id || ''"
              :label="course.title || course.name || '未命名课程'"
              :value="course.id || 0"
            />
          </el-select>
        </el-form-item>
        
        <el-form-item label="选择章节" prop="chapterIds">
          <el-select
            v-model="generateForm.chapterIds"
            placeholder="请选择要包含的章节"
            multiple
            style="width: 100%"
            :disabled="!generateForm.courseId"
          >
            <el-option
              v-for="chapter in chapterList"
              :key="chapter.id || ''"
              :label="chapter.title || '未命名章节'"
              :value="chapter.id || 0"
            />
          </el-select>
        </el-form-item>
        
        <el-form-item label="图谱类型" prop="graphType">
          <el-radio-group v-model="generateForm.graphType">
            <el-radio value="concept">概念图谱</el-radio>
            <el-radio value="skill">技能图谱</el-radio>
            <el-radio value="comprehensive">综合图谱</el-radio>
          </el-radio-group>
        </el-form-item>
        
        <el-form-item label="深度级别">
          <el-slider
            v-model="generateForm.depth"
            :min="1"
            :max="5"
            show-stops
            show-input
            style="width: 80%"
          />
          <el-text size="small" type="info" style="margin-left: 12px">
            级别越高，图谱越详细
          </el-text>
        </el-form-item>
        
        <el-form-item label="包含关系">
          <el-checkbox-group v-model="generateForm.relations">
            <el-checkbox value="prerequisites">先修关系</el-checkbox>
            <el-checkbox value="applications">应用关系</el-checkbox>
          </el-checkbox-group>
        </el-form-item>
        
        <el-form-item label="附加要求">
          <el-input
            v-model="generateForm.additionalRequirements"
            type="textarea"
            :rows="3"
            placeholder="可以描述特殊的生成要求，如重点关注某些知识点等..."
          />
        </el-form-item>
      </el-form>
      
      <template #footer>
        <el-button @click="showGenerateDialog = false">取消</el-button>
        <el-button
          type="primary"
          @click="generateGraph"
          :loading="generating"
        >
          {{ generating ? '生成中...' : '生成图谱' }}
        </el-button>
      </template>
    </el-dialog>

    <!-- 查看/编辑图谱对话框 -->
    <el-dialog
      v-model="showViewDialog"
      :title="currentGraph?.title || '知识图谱'"
      width="90%"
      top="5vh"
      :close-on-click-modal="false"
    >
      <KnowledgeGraph
        v-if="currentGraphData"
        :graph-data="currentGraphData"
        :editable="editMode"
        container-height="70vh"
        @save="handleSaveGraph"
        @node-click="handleNodeClick"
      />
      
      <template #footer>
        <el-button @click="showViewDialog = false">关闭</el-button>
        <el-button
          v-if="!editMode"
          type="primary"
          @click="editMode = true"
        >
          编辑模式
        </el-button>
        <el-button
          v-if="editMode"
          @click="editMode = false"
        >
          查看模式
        </el-button>
      </template>
    </el-dialog>
  </div>
</template>

<script setup>
import { ref, reactive, onMounted, computed } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import KnowledgeGraph from '@/components/KnowledgeGraph.vue'
import { request } from '@/utils/request'
import { formatDate } from '@/utils/date'
import {
  Search, Plus, MoreFilled, Edit, CopyDocument,
  Share, Delete, Document, View, Clock
} from '@element-plus/icons-vue'

// 响应式数据
const graphList = ref([])
const courseList = ref([])
const chapterList = ref([])
const searchKeyword = ref('')
const filterCourse = ref('')
const currentPage = ref(1)
const pageSize = ref(20)
const total = ref(0)
const loading = ref(false)

// 对话框状态
const showGenerateDialog = ref(false)
const showViewDialog = ref(false)
const generating = ref(false)
const editMode = ref(false)

// 当前操作的图谱
const currentGraph = ref(null)
const currentGraphData = ref(null)

// 生成表单
const generateFormRef = ref(null)
const generateForm = reactive({
  courseId: '',
  chapterIds: [],
  graphType: 'comprehensive',
  depth: 3,
  relations: ['prerequisites', 'applications'],
  additionalRequirements: ''
})

const generateRules = {
  courseId: [
    { required: true, message: '请选择课程', trigger: 'change' }
  ],
  chapterIds: [
    { required: true, message: '请至少选择一个章节', trigger: 'change' }
  ]
}

// 计算属性
const filteredGraphs = computed(() => {
  let graphs = graphList.value
  
  if (searchKeyword.value) {
    graphs = graphs.filter(graph =>
      graph.title.includes(searchKeyword.value) ||
      graph.description?.includes(searchKeyword.value)
    )
  }
  
  if (filterCourse.value) {
    graphs = graphs.filter(graph => graph.courseId === filterCourse.value)
  }
  
  return graphs
})

// 方法
const loadGraphList = async () => {
  loading.value = true
  try {
    const response = await request.post('/api/teacher/knowledge-graph/page', {
      page: currentPage.value,
      size: pageSize.value
    })
    
    if (response.code === 200) {
      graphList.value = response.data.records
      total.value = response.data.total
    }
  } catch (error) {
    ElMessage.error('获取图谱列表失败: ' + error.message)
  } finally {
    loading.value = false
  }
}

const loadCourseList = async () => {
  try {
    console.log('📚 开始获取课程列表...')
    const response = await request.get('/api/teacher/courses')
    console.log('📊 课程列表响应:', response)
    
    if (response && response.data && response.data.code === 200) {
      // 处理不同的响应结构
      const responseData = response.data.data
      console.log('📊 响应数据结构:', responseData)
      
      // 检查是否有content字段(分页数据)
      if (responseData.content && Array.isArray(responseData.content)) {
        courseList.value = responseData.content
        console.log('📚 从content字段获取到', courseList.value.length, '个课程')
      } 
      // 检查是否有records字段(分页数据)
      else if (responseData.records && Array.isArray(responseData.records)) {
        courseList.value = responseData.records
        console.log('📚 从records字段获取到', courseList.value.length, '个课程')
      }
      // 检查是否有list字段
      else if (responseData.list && Array.isArray(responseData.list)) {
        courseList.value = responseData.list
        console.log('📚 从list字段获取到', courseList.value.length, '个课程')
      }
      // 检查responseData本身是否为数组
      else if (Array.isArray(responseData)) {
        courseList.value = responseData
        console.log('📚 直接从data字段获取到', courseList.value.length, '个课程')
      }
      else {
        console.warn('未找到有效的课程数据结构:', responseData)
        courseList.value = []
      }
    } else {
      console.warn('获取课程列表失败:', response)
      courseList.value = []
    }
  } catch (error) {
    console.error('获取课程列表失败:', error)
    courseList.value = []
  }
}

const onCourseChange = async (courseId) => {
  if (!courseId) {
    chapterList.value = []
    return
  }
  
  try {
    const response = await request.get(`/api/teacher/chapter/course/${courseId}`)
    if (response.code === 200) {
      chapterList.value = response.data
    }
  } catch (error) {
    ElMessage.error('获取章节列表失败: ' + error.message)
  }
}

const generateGraph = async () => {
  if (!generateFormRef.value) return
  
  const valid = await generateFormRef.value.validate().catch(() => false)
  if (!valid) return
  
  generating.value = true
  try {
    const requestData = {
      courseId: generateForm.courseId,
      chapterIds: generateForm.chapterIds,
      graphType: generateForm.graphType,
      depth: generateForm.depth,
      includePrerequisites: generateForm.relations.includes('prerequisites'),
      includeApplications: generateForm.relations.includes('applications'),
      additionalRequirements: generateForm.additionalRequirements
    }
    
    const response = await request.post('/api/teacher/knowledge-graph/create', requestData)
    
    if (response.code === 200) {
      const result = response.data
      if (result.status === 'completed') {
        ElMessage.success('知识图谱生成成功！')
        showGenerateDialog.value = false
        loadGraphList()
      } else if (result.status === 'failed') {
        ElMessage.error(result.errorMessage || '生成失败')
      } else {
        ElMessage.info('图谱生成中，请稍后刷新查看结果')
        showGenerateDialog.value = false
      }
    }
  } catch (error) {
    ElMessage.error('生成失败: ' + error.message)
  } finally {
    generating.value = false
  }
}

const viewGraph = async (graph) => {
  try {
    const response = await request.get(`/api/teacher/knowledge-graph/${graph.id}`)
    if (response.code === 200) {
      currentGraph.value = graph
      currentGraphData.value = response.data
      editMode.value = false
      showViewDialog.value = true
    }
  } catch (error) {
    ElMessage.error('获取图谱详情失败: ' + error.message)
  }
}

const handleSaveGraph = async (graphData) => {
  try {
    await request.put(`/api/teacher/knowledge-graph/update/${currentGraph.value.id}`, graphData)
    ElMessage.success('图谱保存成功')
    editMode.value = false
  } catch (error) {
    ElMessage.error('保存失败: ' + error.message)
  }
}

const handleNodeClick = (node) => {
  console.log('节点点击:', node)
}

const handleCommand = async (command) => {
  const [action, graphId] = command.split('-')
  
  switch (action) {
    case 'edit':
      const graph = graphList.value.find(g => g.id == graphId)
      if (graph) {
        await viewGraph(graph)
        editMode.value = true
      }
      break
      
    case 'copy':
      // 复制图谱逻辑
      ElMessage.info('复制功能开发中')
      break
      
    case 'share':
      // 分享图谱逻辑
      ElMessage.info('分享功能开发中')
      break
      
    case 'delete':
      await deleteGraph(graphId)
      break
  }
}

const deleteGraph = async (graphId) => {
  try {
    await ElMessageBox.confirm('确定删除此知识图谱吗？', '确认删除', {
      type: 'warning'
    })
    
    await request.delete(`/api/teacher/knowledge-graph/${graphId}`)
    ElMessage.success('删除成功')
    loadGraphList()
  } catch (error) {
    if (error !== 'cancel') {
      ElMessage.error('删除失败: ' + error.message)
    }
  }
}

const handleSearch = () => {
  currentPage.value = 1
  // 由于使用了计算属性，搜索是实时的
}

const handleFilter = () => {
  currentPage.value = 1
  // 由于使用了计算属性，筛选是实时的
}

const handleSizeChange = (size) => {
  pageSize.value = size
  loadGraphList()
}

const handleCurrentChange = (page) => {
  currentPage.value = page
  loadGraphList()
}

const getGraphTypeLabel = (type) => {
  const labels = {
    concept: '概念图谱',
    skill: '技能图谱',
    comprehensive: '综合图谱'
  }
  return labels[type] || type
}

const getGraphTypeTagType = (type) => {
  const types = {
    concept: 'primary',
    skill: 'success',
    comprehensive: 'warning'
  }
  return types[type] || 'info'
}

// 生命周期
onMounted(() => {
  loadGraphList()
  loadCourseList()
})
</script>

<style scoped>
.knowledge-graph-management {
  padding: 24px;
  background: #f5f7fa;
  min-height: 100vh;
}

.page-header {
  margin-bottom: 24px;
}

.page-header h1 {
  margin: 0 0 8px 0;
  color: #2c3e50;
  font-size: 28px;
  font-weight: 600;
}

.page-header p {
  margin: 0;
  color: #7f8c8d;
  font-size: 16px;
}

.toolbar {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 24px;
  padding: 16px;
  background: #fff;
  border-radius: 8px;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}

.toolbar-left {
  display: flex;
  align-items: center;
}

.graph-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(350px, 1fr));
  gap: 24px;
  margin-bottom: 24px;
}

.graph-card {
  background: #fff;
  border-radius: 12px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
  transition: all 0.3s ease;
  cursor: pointer;
  overflow: hidden;
}

.graph-card:hover {
  transform: translateY(-4px);
  box-shadow: 0 8px 24px rgba(0, 0, 0, 0.15);
}

.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 20px 20px 12px 20px;
  border-bottom: 1px solid #f0f0f0;
}

.card-header h3 {
  margin: 0;
  color: #2c3e50;
  font-size: 18px;
  font-weight: 600;
  flex: 1;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.card-content {
  padding: 20px;
}

.description {
  margin: 0 0 16px 0;
  color: #606266;
  font-size: 14px;
  line-height: 1.6;
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
  overflow: hidden;
}

.card-info {
  display: flex;
  flex-direction: column;
  gap: 8px;
  margin-bottom: 16px;
}

.info-item {
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 13px;
  color: #909399;
}

.card-tags {
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 8px;
}

.empty-state {
  grid-column: 1 / -1;
  padding: 40px;
  text-align: center;
}

.pagination {
  display: flex;
  justify-content: center;
  margin-top: 24px;
}

/* 响应式设计 */
@media (max-width: 768px) {
  .toolbar {
    flex-direction: column;
    gap: 16px;
  }
  
  .toolbar-left,
  .toolbar-right {
    width: 100%;
  }
  
  .graph-grid {
    grid-template-columns: 1fr;
  }
}
</style> 