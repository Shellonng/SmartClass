-- 知识图谱表
CREATE TABLE `knowledge_graph` (
  `id` bigint(20) NOT NULL AUTO_INCREMENT COMMENT '图谱ID',
  `course_id` bigint(20) NOT NULL COMMENT '关联课程ID',
  `title` varchar(255) NOT NULL COMMENT '图谱标题',
  `description` text COMMENT '图谱描述',
  `graph_type` varchar(50) NOT NULL DEFAULT 'comprehensive' COMMENT '图谱类型(concept/skill/comprehensive)',
  `graph_data` longtext NOT NULL COMMENT '图谱数据(JSON格式)',
  `creator_id` bigint(20) NOT NULL COMMENT '创建者ID',
  `status` varchar(20) NOT NULL DEFAULT 'draft' COMMENT '图谱状态(draft/published/archived)',
  `version` int(11) NOT NULL DEFAULT 1 COMMENT '版本号',
  `is_public` tinyint(1) NOT NULL DEFAULT 0 COMMENT '是否公开',
  `view_count` int(11) NOT NULL DEFAULT 0 COMMENT '访问次数',
  `create_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  `update_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  PRIMARY KEY (`id`),
  KEY `idx_course_id` (`course_id`),
  KEY `idx_creator_id` (`creator_id`),
  KEY `idx_status` (`status`),
  KEY `idx_is_public` (`is_public`),
  KEY `idx_create_time` (`create_time`),
  CONSTRAINT `fk_knowledge_graph_course` FOREIGN KEY (`course_id`) REFERENCES `course` (`id`) ON DELETE CASCADE ON UPDATE CASCADE,
  CONSTRAINT `fk_knowledge_graph_creator` FOREIGN KEY (`creator_id`) REFERENCES `user` (`id`) ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='知识图谱表';

-- 学生学习进度表（可选，用于记录学生对知识点的掌握情况）
CREATE TABLE `student_knowledge_progress` (
  `id` bigint(20) NOT NULL AUTO_INCREMENT COMMENT '进度ID',
  `student_id` bigint(20) NOT NULL COMMENT '学生ID',
  `graph_id` bigint(20) NOT NULL COMMENT '知识图谱ID',
  `node_id` varchar(100) NOT NULL COMMENT '知识点节点ID',
  `mastery_level` int(11) NOT NULL DEFAULT 0 COMMENT '掌握程度(0-100)',
  `completed` tinyint(1) NOT NULL DEFAULT 0 COMMENT '是否完成',
  `learning_time` int(11) NOT NULL DEFAULT 0 COMMENT '学习时长(分钟)',
  `last_study_time` datetime DEFAULT NULL COMMENT '最后学习时间',
  `create_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
  `update_time` datetime NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
  PRIMARY KEY (`id`),
  UNIQUE KEY `uk_student_graph_node` (`student_id`, `graph_id`, `node_id`),
  KEY `idx_student_id` (`student_id`),
  KEY `idx_graph_id` (`graph_id`),
  KEY `idx_completed` (`completed`),
  CONSTRAINT `fk_student_progress_student` FOREIGN KEY (`student_id`) REFERENCES `user` (`id`) ON DELETE CASCADE ON UPDATE CASCADE,
  CONSTRAINT `fk_student_progress_graph` FOREIGN KEY (`graph_id`) REFERENCES `knowledge_graph` (`id`) ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='学生知识点学习进度表';

-- 插入示例数据
INSERT INTO `knowledge_graph` (`course_id`, `title`, `description`, `graph_type`, `graph_data`, `creator_id`, `status`, `version`, `is_public`, `view_count`) VALUES
(5, '计算机系统基础知识图谱', '包含计算机系统组成、层次结构、性能指标等核心概念的知识图谱', 'comprehensive', 
'{
  "title": "计算机系统基础知识图谱",
  "description": "包含计算机系统组成、层次结构、性能指标等核心概念的知识图谱",
  "nodes": [
    {
      "id": "node_1",
      "name": "计算机系统",
      "type": "concept",
      "level": 1,
      "description": "计算机系统的整体概念",
      "chapterId": 11,
      "style": {
        "color": "#3498db",
        "size": 30,
        "shape": "circle",
        "fontSize": 16
      },
      "position": {
        "x": 400,
        "y": 200,
        "fixed": false
      }
    },
    {
      "id": "node_2",
      "name": "硬件系统",
      "type": "concept",
      "level": 2,
      "description": "计算机的物理组成部分",
      "chapterId": 11,
      "style": {
        "color": "#e74c3c",
        "size": 25,
        "shape": "circle",
        "fontSize": 14
      },
      "position": {
        "x": 200,
        "y": 300,
        "fixed": false
      }
    },
    {
      "id": "node_3",
      "name": "软件系统",
      "type": "concept",
      "level": 2,
      "description": "计算机程序和相关文档",
      "chapterId": 11,
      "style": {
        "color": "#2ecc71",
        "size": 25,
        "shape": "circle",
        "fontSize": 14
      },
      "position": {
        "x": 600,
        "y": 300,
        "fixed": false
      }
    },
    {
      "id": "node_4",
      "name": "CPU",
      "type": "concept",
      "level": 3,
      "description": "中央处理器",
      "chapterId": 11,
      "style": {
        "color": "#f39c12",
        "size": 20,
        "shape": "circle",
        "fontSize": 12
      },
      "position": {
        "x": 100,
        "y": 400,
        "fixed": false
      }
    },
    {
      "id": "node_5",
      "name": "内存",
      "type": "concept",
      "level": 3,
      "description": "主存储器",
      "chapterId": 11,
      "style": {
        "color": "#f39c12",
        "size": 20,
        "shape": "circle",
        "fontSize": 12
      },
      "position": {
        "x": 200,
        "y": 400,
        "fixed": false
      }
    },
    {
      "id": "node_6",
      "name": "I/O设备",
      "type": "concept",
      "level": 3,
      "description": "输入输出设备",
      "chapterId": 11,
      "style": {
        "color": "#f39c12",
        "size": 20,
        "shape": "circle",
        "fontSize": 12
      },
      "position": {
        "x": 300,
        "y": 400,
        "fixed": false
      }
    }
  ],
  "edges": [
    {
      "id": "edge_1",
      "source": "node_1",
      "target": "node_2",
      "type": "contains",
      "description": "包含",
      "weight": 1.0,
      "style": {
        "color": "#7f8c8d",
        "width": 2,
        "lineType": "solid",
        "showArrow": true
      }
    },
    {
      "id": "edge_2",
      "source": "node_1",
      "target": "node_3",
      "type": "contains",
      "description": "包含",
      "weight": 1.0,
      "style": {
        "color": "#7f8c8d",
        "width": 2,
        "lineType": "solid",
        "showArrow": true
      }
    },
    {
      "id": "edge_3",
      "source": "node_2",
      "target": "node_4",
      "type": "contains",
      "description": "包含",
      "weight": 1.0,
      "style": {
        "color": "#95a5a6",
        "width": 1,
        "lineType": "solid",
        "showArrow": true
      }
    },
    {
      "id": "edge_4",
      "source": "node_2",
      "target": "node_5",
      "type": "contains",
      "description": "包含",
      "weight": 1.0,
      "style": {
        "color": "#95a5a6",
        "width": 1,
        "lineType": "solid",
        "showArrow": true
      }
    },
    {
      "id": "edge_5",
      "source": "node_2",
      "target": "node_6",
      "type": "contains",
      "description": "包含",
      "weight": 1.0,
      "style": {
        "color": "#95a5a6",
        "width": 1,
        "lineType": "solid",
        "showArrow": true
      }
    }
  ],
  "metadata": {
    "nodeCount": 6,
    "edgeCount": 5,
    "generatedAt": "2024-12-28",
    "aiGenerated": true
  }
}', 2, 'published', 1, 1, 15); 