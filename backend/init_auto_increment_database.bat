@echo off
echo =============================================
echo 初始化教育平台数据库(自增ID版本)
echo =============================================

set MYSQL_PATH=mysql
set DB_HOST=localhost
set DB_PORT=3306
set DB_USER=root
set DB_PASS=root
set DB_NAME=education_platform
set SQL_FILE=..\sql\database_init_auto_increment.sql

echo 正在连接数据库...
%MYSQL_PATH% -h%DB_HOST% -P%DB_PORT% -u%DB_USER% -p%DB_PASS% -e "DROP DATABASE IF EXISTS %DB_NAME%; CREATE DATABASE %DB_NAME% CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;"

if %ERRORLEVEL% NEQ 0 (
    echo 数据库连接失败，请检查MySQL服务是否启动或者用户名密码是否正确。
    goto :EOF
)

echo 数据库连接成功，开始导入数据...
%MYSQL_PATH% -h%DB_HOST% -P%DB_PORT% -u%DB_USER% -p%DB_PASS% %DB_NAME% < %SQL_FILE%

if %ERRORLEVEL% NEQ 0 (
    echo 数据导入失败，请检查SQL文件路径是否正确。
    goto :EOF
)

echo 数据库初始化完成！
echo 用户名: admin, teacher1, teacher2, student1, student2
echo 密码: 123456
echo =============================================

pause 