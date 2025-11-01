import request from '../utils/request';

export interface LearningRecord {
  courseId: number;
  sectionId?: number;
  resourceId?: number;
  resourceType?: string;
  progress?: number;
  completed?: boolean;
}

export interface LearningStatistics {
  dailyDurations: Array<{date: string, duration: number}>;
  sectionDistribution: Array<{section_id: number, section_title: string, duration: number}>;
  resourceTypeDistribution: Array<{resource_type: string, duration: number}>;
  totalLearningDays: number;
  totalDuration: number;
  avgDailyDuration: number;
  completedSections: number;
  totalSections: number;
  viewedResources: number;
}

/**
 * 开始学习记录
 * @param record 学习记录信息
 */
export function startLearningRecord(record: LearningRecord) {
  return request({
    url: '/api/student/learning-records/start',
    method: 'post',
    data: record
  });
}

/**
 * 结束学习记录
 * @param recordId 记录ID
 * @param progress 学习进度
 * @param completed 是否完成
 */
export function endLearningRecord(recordId: number, progress: number, completed: boolean = false) {
  return request({
    url: `/api/student/learning-records/${recordId}/end`,
    method: 'post',
    params: {
      progress,
      completed
    }
  });
}

/**
 * 获取学习统计数据
 * @param courseId 课程ID
 * @param startDate 开始日期（可选）
 * @param endDate 结束日期（可选）
 */
export function getLearningStatistics(courseId: number, startDate?: string, endDate?: string) {
  return request({
    url: '/api/student/learning-records/statistics',
    method: 'get',
    params: {
      courseId,
      startDate,
      endDate
    }
  });
}

/**
 * 获取学习记录列表
 * @param courseId 课程ID（可选）
 */
export function getLearningRecords(courseId?: number) {
  return request({
    url: '/api/student/learning-records',
    method: 'get',
    params: {
      courseId
    }
  });
} 