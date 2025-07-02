@echo off
echo ====================================
echo  SmartClass 认证系统数据库初始化
echo ====================================
echo.

echo 正在初始化认证系统数据库...
echo.

REM 检查MySQL是否运行
echo 1. 检查MySQL服务状态...
sc query mysql >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ MySQL服务未运行，请先启动MySQL服务
    echo 可以运行以下命令启动MySQL：
    echo   net start mysql
    pause
    exit /b 1
)
echo ✅ MySQL服务正在运行

echo.
echo 2. 执行数据库初始化脚本...
mysql -u root -p < ..\sql\database_init_auth_only.sql

if %errorlevel% equ 0 (
    echo ✅ 数据库初始化成功！
    echo.
    echo 📋 测试账号信息（密码统一为：123456）：
    echo   - 管理员: admin
    echo   - 教师1: teacher1 (张老师)
    echo   - 教师2: teacher2 (李老师)  
    echo   - 学生1: student1 (王同学)
    echo   - 学生2: student2 (刘同学)
    echo   - 学生3: student3 (陈同学)
    echo.
    echo 🚀 可以启动后端服务进行测试了！
) else (
    echo ❌ 数据库初始化失败，请检查MySQL连接和权限
)

echo.
pause 