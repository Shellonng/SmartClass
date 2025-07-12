# SmartClass 作业功能修复说明

## 修复内容概述

本次修复解决了 SmartClass 智能教育平台创建作业功能存在的多个问题，包括数据模型不一致、字段映射缺失、数据库表结构问题等。

## 修复的问题

### 1. 数据模型不一致
- **问题**：前端发送作业数据，后端使用 `ExamDTO` 接收并操作 `Exam` 实体
- **解决方案**：创建了专门的 `AssignmentDTO` 和 `AssignmentService`

### 2. 字段映射缺失
- **问题**：`ExamDTO` 中的 `totalScore`、`duration` 等字段在 `Exam` 实体中被标记为非数据库字段
- **解决方案**：在 `Assignment` 实体中添加了所有必要的数据库字段

### 3. 数据库表结构问题
- **问题**：`assignment` 表缺少 `total_score`、`duration` 等重要字段
- **解决方案**：创建了完整的数据库补丁脚本 `fix_assignment_complete.sql`

### 4. 智能组卷功能问题
- **问题**：前端组卷参数未在数据库中存储
- **解决方案**：创建了 `assignment_config` 表和相关实体类

### 5. API 接口不一致
- **问题**：前端调用 `/api/teacher/assignments`，后端使用 `ExamService`
- **解决方案**：修改 `AssignmentController` 使用新的 `AssignmentService`

## 新增文件

### 数据库相关
- `fix_assignment_complete.sql` - 数据库补丁脚本

### 后端代码
- `AssignmentDTO.java` - 作业数据传输对象
- `AssignmentService.java` - 作业服务接口
- `AssignmentServiceImpl.java` - 作业服务实现类
- `AssignmentConfig.java` - 作业配置实体类
- `AssignmentConfigMapper.java` - 作业配置Mapper接口

### 修改文件
- `AssignmentController.java` - 修改为使用新的 `AssignmentService`
- `Assignment.java` - 添加缺失的数据库字段

## 部署步骤

### 1. 执行数据库补丁
```sql
-- 在数据库中执行以下脚本
source fix_assignment_complete.sql;
```

### 2. 重启应用
重启 Spring Boot 应用以加载新的代码更改。

### 3. 验证功能
- 测试创建作业功能
- 测试智能组卷功能
- 测试文件上传型作业
- 测试作业发布和管理

## 数据库表结构

### assignment 表新增字段
- `total_score` INT - 总分
- `duration` INT - 考试时长（分钟）
- `allowed_file_types` TEXT - 允许的文件类型（JSON格式）
- `max_file_size` INT - 最大文件大小（MB）
- `reference_answer` TEXT - 参考答案（用于智能批改）

### assignment_config 表（新建）
- `id` BIGINT - 主键
- `assignment_id` BIGINT - 作业ID
- `knowledge_points` TEXT - 知识点范围（JSON格式）
- `difficulty` ENUM - 难度级别
- `question_count` INT - 题目总数
- `question_types` TEXT - 题型分布（JSON格式）
- `additional_requirements` TEXT - 额外要求
- `create_time` DATETIME - 创建时间
- `update_time` DATETIME - 更新时间

## API 接口变更

### 请求/响应数据类型变更
- 所有作业相关接口的请求和响应数据类型从 `ExamDTO` 改为 `AssignmentDTO`
- 智能组卷接口返回题目列表而不是完整的作业对象

### 新增功能
- 支持文件上传型作业的完整配置
- 支持智能组卷配置的持久化存储
- 支持参考答案的设置和管理

## 待完善功能

以下功能在当前实现中标记为 TODO，需要后续完善：

1. **智能组卷算法**
   - 根据知识点和难度筛选题目
   - 按照题型分布要求选择题目
   - 随机排序或按难度排序

2. **手动选题功能**
   - 验证题目是否存在
   - 保存作业题目关联关系

3. **提交率计算**
   - 查询作业所属课程的学生总数
   - 查询已提交作业的学生数
   - 计算提交率

4. **题目管理功能**
   - 获取知识点列表
   - 按题型获取题目列表

5. **作业提交记录管理**
   - 获取作业提交记录列表
   - 智能批改功能集成

## 注意事项

1. **数据迁移**：执行数据库补丁前请备份现有数据
2. **兼容性**：新的 `AssignmentService` 与原有的 `ExamService` 并存，不影响考试功能
3. **权限控制**：所有作业操作都会验证当前用户权限
4. **错误处理**：增强了输入验证和错误提示

## 技术栈

- **后端框架**：Spring Boot + MyBatis Plus
- **数据库**：MySQL 8.0
- **数据传输**：JSON
- **文件处理**：Jackson（JSON序列化/反序列化）

## 联系方式

如有问题或需要进一步的技术支持，请联系开发团队。