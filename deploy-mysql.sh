#!/bin/bash

# SmartClass MySQL 数据库部署脚本
# 使用方法：./deploy-mysql.sh

echo "🚀 开始部署 SmartClass 数据库..."

# 配置变量
MYSQL_ROOT_PASSWORD="123456"
MYSQL_HOST="127.0.0.1"
MYSQL_PORT="3306"
DATABASE_NAME="education_platform"
SQL_FILE="education_platform.sql"

# 检查MySQL容器是否运行
echo "📋 检查MySQL容器状态..."
if ! docker ps | grep -q mysql_8.0; then
    echo "❌ MySQL容器未运行，请先启动MySQL容器"
    exit 1
fi

# 等待MySQL启动完成
echo "⏳ 等待MySQL启动完成..."
sleep 5

# 测试MySQL连接
echo "🔗 测试MySQL连接..."
docker exec mysql_8.0 mysql -u root -p$MYSQL_ROOT_PASSWORD -e "SELECT 1" > /dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "❌ MySQL连接失败，请检查密码和容器状态"
    exit 1
fi

# 创建数据库
echo "📊 创建数据库..."
docker exec mysql_8.0 mysql -u root -p$MYSQL_ROOT_PASSWORD -e "
CREATE DATABASE IF NOT EXISTS $DATABASE_NAME 
CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
"

# 配置远程访问权限
echo "🔐 配置远程访问权限..."
docker exec mysql_8.0 mysql -u root -p$MYSQL_ROOT_PASSWORD -e "
CREATE USER IF NOT EXISTS 'root'@'%' IDENTIFIED BY '$MYSQL_ROOT_PASSWORD';
GRANT ALL PRIVILEGES ON *.* TO 'root'@'%' WITH GRANT OPTION;
FLUSH PRIVILEGES;
"

# 导入SQL文件（如果存在）
if [ -f "$SQL_FILE" ]; then
    echo "📥 导入SQL文件..."
    docker cp $SQL_FILE mysql_8.0:/tmp/
    docker exec mysql_8.0 mysql -u root -p$MYSQL_ROOT_PASSWORD $DATABASE_NAME -e "source /tmp/$SQL_FILE"
    echo "✅ SQL文件导入完成"
else
    echo "⚠️  SQL文件不存在，跳过导入步骤"
fi

# 验证部署
echo "🔍 验证部署结果..."
TABLE_COUNT=$(docker exec mysql_8.0 mysql -u root -p$MYSQL_ROOT_PASSWORD -e "
USE $DATABASE_NAME; 
SELECT COUNT(*) FROM information_schema.tables WHERE table_schema='$DATABASE_NAME';
" -s -N)

if [ "$TABLE_COUNT" -gt 0 ]; then
    echo "✅ 数据库部署成功！共创建了 $TABLE_COUNT 个表"
    echo "📋 数据库信息："
    echo "   - 主机: $MYSQL_HOST"
    echo "   - 端口: $MYSQL_PORT"
    echo "   - 数据库: $DATABASE_NAME"
    echo "   - 用户名: root"
    echo "   - 密码: $MYSQL_ROOT_PASSWORD"
else
    echo "❌ 数据库部署失败，请检查日志"
    exit 1
fi

# 显示测试命令
echo ""
echo "🧪 测试连接命令："
echo "   mysql -h $MYSQL_HOST -P $MYSQL_PORT -u root -p$MYSQL_ROOT_PASSWORD -e \"USE $DATABASE_NAME; SHOW TABLES;\""
echo ""
echo "🎉 部署完成！" 