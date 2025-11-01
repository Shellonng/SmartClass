@echo off
chcp 65001
echo 正在初始化简化版数据库（无JWT）...

:: 数据库连接信息
set DB_HOST=localhost
set DB_PORT=3306
set DB_USER=root
set DB_PASSWORD=123456

:: 执行SQL脚本
echo 连接到MySQL并执行初始化脚本...
mysql -h %DB_HOST% -P %DB_PORT% -u %DB_USER% -p%DB_PASSWORD% < ..\sql\database_init_simple_no_jwt.sql

if %errorlevel% equ 0 (
    echo.
    echo ✅ 简化版数据库初始化成功！
    echo.
    echo 📋 测试账户信息：
    echo    管理员：admin / 123456
    echo    教师1：teacher1 / 123456  
    echo    教师2：teacher2 / 123456
    echo    学生1：student1 / 123456
    echo    学生2：student2 / 123456
    echo    学生3：student3 / 123456
    echo.
    echo 🚀 现在可以启动项目进行测试
) else (
    echo.
    echo ❌ 数据库初始化失败，请检查：
    echo    1. MySQL服务是否启动
    echo    2. 数据库连接信息是否正确
    echo    3. SQL脚本文件是否存在
)

pause 