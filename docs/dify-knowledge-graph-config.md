# Dify 知识图谱生成智能体配置指南

## 概述

本文档详细说明如何在Dify平台配置知识图谱生成智能体，用于基于数据库中的课程结构化数据智能生成知识图谱。

## 工作流创建

### 1. 基础配置

- **工作流名称**: `knowledge-graph`
- **工作流类型**: `Workflow`
- **描述**: 基于课程结构化数据生成知识图谱

### 2. 输入变量配置

在Dify工作流中配置以下输入变量：

| 变量名 | 类型 | 必填 | 描述 | 示例值 |
|--------|------|------|------|--------|
| `course_id` | Number | 是 | 课程ID | `5` |
| `chapter_ids` | Array | 是 | 章节ID列表 | `[11, 12, 13]` |
| `graph_type` | String | 否 | 图谱类型 | `comprehensive` |
| `depth_level` | Number | 否 | 深度级别(1-5) | `3` |
| `include_prerequisites` | Boolean | 否 | 是否包含先修关系 | `true` |
| `include_applications` | Boolean | 否 | 是否包含应用关系 | `true` |
| `course_content` | String | 是 | 结构化课程内容 | JSON格式的课程数据 |
| `user_id` | String | 是 | 用户ID | `123` |
| `task_type` | String | 否 | 任务类型 | `knowledge_graph_generation` |
| `response_format` | String | 否 | 响应格式 | `json` |
| `min_nodes` | Number | 否 | 最少节点数 | `5` |
| `max_nodes` | Number | 否 | 最大节点数 | `35` |
| `require_validation` | Boolean | 否 | 是否需要验证 | `true` |

### 3. 提示词模板

在Dify的LLM节点中使用以下提示词模板：

```text
你是一个专业的教育知识图谱生成专家。请根据提供的课程结构化数据，生成一个高质量的知识图谱。

## 输入数据分析

课程信息:
- 课程ID: {{course_id}}
- 选择章节: {{chapter_ids}}
- 图谱类型: {{graph_type}}
- 深度级别: {{depth_level}}/5
- 包含先修关系: {{include_prerequisites}}
- 包含应用关系: {{include_applications}}

## 课程结构化内容

{{course_content}}

## 生成要求

### 图谱类型处理
{% if graph_type == "concept" %}
- 重点提取和连接概念性知识点
- 突出概念间的层次关系和逻辑关系
- 节点类型以concept和topic为主
{% elif graph_type == "skill" %}
- 重点提取技能型知识点
- 突出技能的递进关系和依赖关系
- 节点类型以skill为主，包含相关的concept
{% else %}
- 全面提取概念、技能、应用等各类知识点
- 构建完整的知识体系结构
- 平衡各种类型的节点
{% endif %}

### 深度级别控制
{% if depth_level <= 2 %}
- 提取{{min_nodes}}-{{max_nodes}}个主要的核心知识点
- 保持结构简洁清晰
- 关注最重要的概念和关系
{% elif depth_level == 3 %}
- 提取{{min_nodes}}-{{max_nodes}}个核心和重要的知识点
- 包含适当的细节和子概念
- 构建合理的层次结构
{% else %}
- 提取{{min_nodes}}-{{max_nodes}}个详细的知识点
- 包含深层次的关联关系
- 建立复杂但清晰的知识网络
{% endif %}

### 关系类型处理
{% if include_prerequisites %}
- 必须识别和标注先修关系(prerequisite)
- 建立知识点的学习顺序
{% endif %}
{% if include_applications %}
- 必须识别和标注应用关系(application)
- 连接理论与实践应用
{% endif %}

## 输出格式要求

请严格按照以下JSON格式输出，不要添加任何markdown标记或其他格式：

```json
{
  "title": "知识图谱标题",
  "description": "图谱描述",
  "nodes": [
    {
      "id": "node_1",
      "name": "节点名称",
      "type": "concept|skill|topic|chapter",
      "level": 1,
      "description": "节点详细描述",
      "chapterId": 章节ID(如果关联),
      "sectionId": 小节ID(如果关联),
      "style": {
        "color": "#3498db",
        "size": 25,
        "shape": "circle",
        "fontSize": 14
      },
      "position": {
        "x": 200,
        "y": 300,
        "fixed": false
      },
      "properties": {}
    }
  ],
  "edges": [
    {
      "id": "edge_1",
      "source": "node_1",
      "target": "node_2",
      "type": "contains|prerequisite|application|similar|extends",
      "description": "关系描述",
      "weight": 1.0,
      "style": {
        "color": "#7f8c8d",
        "width": 2,
        "lineType": "solid",
        "showArrow": true
      },
      "properties": {}
    }
  ],
  "metadata": {
    "nodeCount": 节点数量,
    "edgeCount": 边数量,
    "generatedAt": "生成时间",
    "aiGenerated": true,
    "complexity": "low|medium|high"
  },
  "suggestions": "改进建议或学习建议"
}
```

## 注意事项

1. **节点ID唯一性**: 确保每个节点的ID在图谱中唯一
2. **边的有效性**: 确保每条边的source和target都指向存在的节点
3. **颜色编码**: 使用不同颜色区分节点类型
   - concept: #3498db (蓝色)
   - skill: #e74c3c (红色)  
   - topic: #2ecc71 (绿色)
   - chapter: #f39c12 (橙色)
4. **位置布局**: 提供合理的初始位置，便于可视化展示
5. **数据完整性**: 确保所有必填字段都有有效值

请开始生成知识图谱：
```

### 4. 输出处理

在工作流的最后添加一个Code节点，用于验证和清理输出：

```python
import json
import re

def main(arg1: str) -> dict:
    """
    验证和清理知识图谱生成结果
    """
    try:
        # 尝试解析JSON
        if arg1.startswith('```json'):
            # 移除markdown格式
            content = re.search(r'```json\s*(.*?)\s*```', arg1, re.DOTALL)
            if content:
                arg1 = content.group(1)
        
        graph_data = json.loads(arg1)
        
        # 基础验证
        if not isinstance(graph_data, dict):
            return {"error": "输出必须是JSON对象"}
        
        if "nodes" not in graph_data or "edges" not in graph_data:
            return {"error": "缺少必要的nodes或edges字段"}
        
        nodes = graph_data["nodes"]
        edges = graph_data["edges"]
        
        if not isinstance(nodes, list) or len(nodes) == 0:
            return {"error": "节点列表不能为空"}
        
        # 验证节点ID唯一性
        node_ids = set()
        for node in nodes:
            if "id" not in node:
                return {"error": "节点缺少id字段"}
            if node["id"] in node_ids:
                return {"error": f"节点ID重复: {node['id']}"}
            node_ids.add(node["id"])
        
        # 验证边的有效性
        if isinstance(edges, list):
            for edge in edges:
                if "source" not in edge or "target" not in edge:
                    return {"error": "边缺少source或target字段"}
                if edge["source"] not in node_ids:
                    return {"error": f"边的源节点不存在: {edge['source']}"}
                if edge["target"] not in node_ids:
                    return {"error": f"边的目标节点不存在: {edge['target']}"}
        
        # 返回验证通过的数据
        return {
            "status": "success",
            "data": graph_data,
            "node_count": len(nodes),
            "edge_count": len(edges) if isinstance(edges, list) else 0
        }
        
    except json.JSONDecodeError as e:
        return {"error": f"JSON解析失败: {str(e)}"}
    except Exception as e:
        return {"error": f"处理失败: {str(e)}"}
```

### 5. 工作流节点配置

推荐的工作流节点顺序：

1. **开始节点** - 接收输入参数
2. **LLM节点** - 使用上述提示词模板生成图谱
3. **Code节点** - 验证和清理输出
4. **条件判断节点** - 检查生成是否成功
5. **结束节点** - 返回最终结果

## 测试数据示例

### 输入示例

```json
{
  "course_id": 5,
  "chapter_ids": [11, 12],
  "graph_type": "comprehensive",
  "depth_level": 3,
  "include_prerequisites": true,
  "include_applications": true,
  "course_content": "请根据以下课程结构化数据生成知识图谱：\n\n=== 课程数据 ===\n{\"course\":{\"id\":5,\"title\":\"计算机系统基础\",\"description\":\"介绍计算机系统的基本组成和原理\"},\"chapters\":[{\"id\":11,\"title\":\"计算机系统概述\",\"description\":\"计算机系统的基本概念和组成\",\"sections\":[{\"id\":21,\"title\":\"计算机发展历史\",\"description\":\"计算机的发展历程\"},{\"id\":22,\"title\":\"计算机系统组成\",\"description\":\"硬件和软件系统\"}]}]}",
  "user_id": "123",
  "min_nodes": 5,
  "max_nodes": 35
}
```

### 期望输出示例

```json
{
  "title": "计算机系统基础知识图谱",
  "description": "包含计算机系统组成、发展历史等核心概念的知识图谱",
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
    }
  ],
  "metadata": {
    "nodeCount": 6,
    "edgeCount": 5,
    "generatedAt": "2024-12-28",
    "aiGenerated": true
  }
}
```

## 调试和优化

### 常见问题

1. **JSON格式错误**: 确保LLM输出严格的JSON格式
2. **节点ID重复**: 使用唯一的ID生成策略
3. **边引用错误**: 验证所有边的source和target都存在
4. **数据结构不完整**: 确保包含所有必需字段

### 性能优化

1. **控制复杂度**: 根据深度级别限制节点数量
2. **缓存机制**: 对相同输入进行结果缓存
3. **分批处理**: 对大型课程进行分章节处理
4. **质量检查**: 添加输出质量评估机制

## 部署说明

1. 在Dify平台创建新的工作流
2. 按照上述配置添加所有节点和连接
3. 测试工作流的输入输出
4. 发布工作流并获取API端点
5. 在后端代码中配置正确的工作流名称: `knowledge-graph`

通过以上配置，Dify智能体就能够理解数据库中的课程结构化数据，并生成高质量的知识图谱。 