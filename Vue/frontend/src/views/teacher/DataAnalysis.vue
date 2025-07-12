<template>
  <div class="data-analysis">
    <div class="page-header">
      <h2>数据分析</h2>
      <p>查看课程和题目的统计数据，帮助优化教学</p>
    </div>

    <a-spin :spinning="loading" tip="加载中...">
      <div class="filter-bar">
        <a-space>
          <a-select
            v-model:value="selectedCourse"
            style="width: 200px"
            placeholder="选择课程"
            @change="loadData"
          >
            <a-select-option :value="null">全部课程</a-select-option>
            <a-select-option
              v-for="course in courses"
              :key="course.id"
              :value="course.id"
            >
              {{ course.title }}
            </a-select-option>
          </a-select>
          
          <a-radio-group v-model:value="chartType" @change="updateChartType">
            <a-radio-button value="questions">题目正确率</a-radio-button>
            <a-radio-button value="types">题型分析</a-radio-button>
            <a-radio-button value="difficulty">难度分析</a-radio-button>
          </a-radio-group>
        </a-space>
      </div>

      <div class="chart-container">
        <div v-if="noData" class="no-data">
          <a-empty description="暂无数据" />
        </div>
        <div v-else>
          <!-- 题目正确率图表 -->
          <div v-show="chartType === 'questions'" class="chart-wrapper">
            <h3>题目正确率统计</h3>
            <div id="questionChart" class="chart"></div>
            <div class="data-table">
              <a-table 
                :columns="questionColumns" 
                :data-source="questionData" 
                :pagination="{ pageSize: 5 }"
                size="small"
              >
                <template #bodyCell="{ column, record }">
                  <template v-if="column.key === 'correctRate'">
                    <a-progress 
                      :percent="record.correctRate" 
                      :format="percent => `${percent}%`"
                      :status="getProgressStatus(record.correctRate)"
                    />
                  </template>
                  <template v-if="column.key === 'questionType'">
                    {{ formatQuestionType(record.type) }}
                  </template>
                  <template v-if="column.key === 'difficulty'">
                    <a-rate 
                      :value="record.difficulty" 
                      disabled 
                      :count="5"
                    />
                  </template>
                </template>
              </a-table>
            </div>
          </div>

          <!-- 题型分析图表 -->
          <div v-show="chartType === 'types'" class="chart-wrapper">
            <h3>各题型正确率分析</h3>
            <div id="typeChart" class="chart"></div>
          </div>

          <!-- 难度分析图表 -->
          <div v-show="chartType === 'difficulty'" class="chart-wrapper">
            <h3>各难度级别正确率分析</h3>
            <div id="difficultyChart" class="chart"></div>
          </div>
        </div>
      </div>
    </a-spin>
  </div>
</template>

<script>
import { ref, onMounted, reactive, computed } from 'vue';
import { message } from 'ant-design-vue';
import axios from 'axios';
import * as echarts from 'echarts/core';
import { BarChart, PieChart, LineChart } from 'echarts/charts';
import {
  TitleComponent,
  TooltipComponent,
  LegendComponent,
  GridComponent,
} from 'echarts/components';
import { CanvasRenderer } from 'echarts/renderers';

// 注册必要的组件
echarts.use([
  BarChart,
  PieChart,
  LineChart,
  TitleComponent,
  TooltipComponent,
  LegendComponent,
  GridComponent,
  CanvasRenderer,
]);

export default {
  name: 'DataAnalysis',
  setup() {
    const loading = ref(false);
    const courses = ref([]);
    const selectedCourse = ref(null);
    const chartType = ref('questions');
    const analysisData = ref(null);
    const noData = computed(() => !analysisData.value || analysisData.value.questions.length === 0);

    const questionData = computed(() => {
      if (!analysisData.value) return [];
      return analysisData.value.questions.map((q, index) => ({
        key: index,
        id: q.questionId,
        title: q.title,
        type: q.type,
        difficulty: q.difficulty,
        totalAnswers: q.totalAnswers,
        correctAnswers: q.correctAnswers,
        correctRate: q.correctRate,
      }));
    });

    const questionColumns = [
      {
        title: '题目',
        dataIndex: 'title',
        key: 'title',
        ellipsis: true,
      },
      {
        title: '题型',
        dataIndex: 'type',
        key: 'questionType',
        width: 120,
      },
      {
        title: '难度',
        dataIndex: 'difficulty',
        key: 'difficulty',
        width: 120,
      },
      {
        title: '答题次数',
        dataIndex: 'totalAnswers',
        key: 'totalAnswers',
        width: 100,
        sorter: (a, b) => a.totalAnswers - b.totalAnswers,
      },
      {
        title: '正确率',
        dataIndex: 'correctRate',
        key: 'correctRate',
        width: 200,
        sorter: (a, b) => a.correctRate - b.correctRate,
      },
    ];

    // 获取课程列表
    const loadCourses = async () => {
      try {
        const { data } = await axios.get('/api/teacher/analytics/courses');
        if (data.code === 200) {
          courses.value = data.data || [];
        } else {
          message.error('获取课程列表失败：' + data.message);
        }
      } catch (error) {
        console.error('获取课程列表失败', error);
        message.error('获取课程列表失败：' + error.message);
      }
    };

    // 加载分析数据
    const loadData = async () => {
      loading.value = true;
      try {
        const params = {};
        if (selectedCourse.value) {
          params.courseId = selectedCourse.value;
        }

        const { data } = await axios.get('/api/teacher/analytics/question-stats', { params });
        if (data.code === 200) {
          analysisData.value = data.data;
          // 渲染图表
          setTimeout(() => {
            renderCharts();
          }, 0);
        } else {
          message.error('获取分析数据失败：' + data.message);
        }
      } catch (error) {
        console.error('获取分析数据失败', error);
        message.error('获取分析数据失败：' + error.message);
      } finally {
        loading.value = false;
      }
    };

    const updateChartType = () => {
      // 切换图表类型时重新渲染
      setTimeout(() => {
        renderCharts();
      }, 0);
    };

    // 渲染所有图表
    const renderCharts = () => {
      if (noData.value) return;

      switch (chartType.value) {
        case 'questions':
          renderQuestionChart();
          break;
        case 'types':
          renderTypeChart();
          break;
        case 'difficulty':
          renderDifficultyChart();
          break;
      }
    };

    // 渲染题目正确率图表
    const renderQuestionChart = () => {
      const chartDom = document.getElementById('questionChart');
      if (!chartDom) return;
      
      const chart = echarts.init(chartDom);
      
      // 准备数据
      const data = analysisData.value.questions.map(item => ({
        name: item.title.length > 20 ? item.title.substring(0, 20) + '...' : item.title,
        value: item.correctRate
      }));
      
      const option = {
        tooltip: {
          trigger: 'axis',
          axisPointer: {
            type: 'shadow'
          },
          formatter: '{b}: {c}%'
        },
        grid: {
          left: '3%',
          right: '4%',
          bottom: '3%',
          containLabel: true
        },
        xAxis: {
          type: 'value',
          max: 100,
          axisLabel: {
            formatter: '{value}%'
          }
        },
        yAxis: {
          type: 'category',
          data: data.map(item => item.name),
          inverse: true,
          axisLabel: {
            width: 200,
            overflow: 'truncate'
          }
        },
        series: [
          {
            name: '正确率',
            type: 'bar',
            data: data.map(item => item.value),
            itemStyle: {
              color: function(params) {
                // 根据正确率设置不同颜色
                const value = params.value;
                if (value >= 80) return '#52c41a';
                if (value >= 60) return '#1890ff';
                if (value >= 40) return '#faad14';
                return '#f5222d';
              }
            },
            label: {
              show: true,
              position: 'right',
              formatter: '{c}%'
            }
          }
        ]
      };
      
      chart.setOption(option);
      
      // 窗口大小改变时重绘
      window.addEventListener('resize', () => {
        chart.resize();
      });
    };

    // 渲染题型分析图表
    const renderTypeChart = () => {
      const chartDom = document.getElementById('typeChart');
      if (!chartDom) return;
      
      const chart = echarts.init(chartDom);
      
      // 准备数据
      const typeStats = analysisData.value.typeStats;
      const data = Object.keys(typeStats).map(key => ({
        name: formatQuestionType(key),
        value: typeStats[key]
      }));
      
      const option = {
        tooltip: {
          trigger: 'item',
          formatter: '{b}: {c}%'
        },
        legend: {
          orient: 'vertical',
          left: 'left'
        },
        series: [
          {
            name: '题型正确率',
            type: 'pie',
            radius: ['40%', '70%'],
            avoidLabelOverlap: false,
            label: {
              show: true,
              formatter: '{b}: {c}%'
            },
            emphasis: {
              label: {
                show: true,
                fontSize: '18',
                fontWeight: 'bold'
              }
            },
            labelLine: {
              show: true
            },
            data: data
          }
        ]
      };
      
      chart.setOption(option);
      
      // 窗口大小改变时重绘
      window.addEventListener('resize', () => {
        chart.resize();
      });
    };

    // 渲染难度分析图表
    const renderDifficultyChart = () => {
      const chartDom = document.getElementById('difficultyChart');
      if (!chartDom) return;
      
      const chart = echarts.init(chartDom);
      
      // 准备数据
      const difficultyStats = analysisData.value.difficultyStats;
      const categories = [];
      const data = [];
      
      // 按照难度级别排序
      const sortedKeys = Object.keys(difficultyStats).map(Number).sort((a, b) => a - b);
      
      sortedKeys.forEach(key => {
        categories.push(`难度${key}`);
        data.push(difficultyStats[key]);
      });
      
      const option = {
        tooltip: {
          trigger: 'axis',
          formatter: '{b}: {c}%'
        },
        xAxis: {
          type: 'category',
          data: categories
        },
        yAxis: {
          type: 'value',
          max: 100,
          axisLabel: {
            formatter: '{value}%'
          }
        },
        series: [
          {
            name: '正确率',
            type: 'line',
            data: data,
            markLine: {
              data: [
                {
                  type: 'average',
                  name: '平均值'
                }
              ]
            },
            smooth: true,
            lineStyle: {
              width: 3
            },
            itemStyle: {
              color: '#1890ff'
            },
            label: {
              show: true,
              formatter: '{c}%'
            }
          }
        ]
      };
      
      chart.setOption(option);
      
      // 窗口大小改变时重绘
      window.addEventListener('resize', () => {
        chart.resize();
      });
    };

    // 格式化题型显示
    const formatQuestionType = (type) => {
      const typeMap = {
        'single': '单选题',
        'multiple': '多选题',
        'true_false': '判断题',
        'blank': '填空题',
        'short': '简答题',
        'code': '编程题'
      };
      return typeMap[type] || type;
    };

    // 获取进度条状态
    const getProgressStatus = (rate) => {
      if (rate >= 80) return 'success';
      if (rate >= 60) return 'normal';
      if (rate >= 40) return 'active';
      return 'exception';
    };

    onMounted(() => {
      loadCourses();
      loadData();
    });

    return {
      loading,
      courses,
      selectedCourse,
      chartType,
      noData,
      questionData,
      questionColumns,
      formatQuestionType,
      getProgressStatus,
      loadData,
      updateChartType
    };
  }
};
</script>

<style scoped>
.data-analysis {
  padding: 20px;
}

.page-header {
  margin-bottom: 24px;
}

.page-header h2 {
  margin-bottom: 8px;
  font-size: 24px;
}

.page-header p {
  color: rgba(0, 0, 0, 0.45);
}

.filter-bar {
  margin-bottom: 16px;
  background: #fff;
  padding: 16px;
  border-radius: 4px;
  box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);
}

.chart-container {
  background: #fff;
  padding: 16px;
  border-radius: 4px;
  box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);
}

.chart-wrapper {
  margin-bottom: 24px;
}

.chart-wrapper h3 {
  margin-bottom: 16px;
  font-size: 18px;
}

.chart {
  height: 400px;
  width: 100%;
}

.data-table {
  margin-top: 24px;
}

.no-data {
  display: flex;
  justify-content: center;
  align-items: center;
  height: 400px;
}
</style> 