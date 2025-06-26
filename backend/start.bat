@echo off
echo 正在启动教育平台应用程序...

REM 设置 Java 参数
set JAVA_OPTS=-Xms512m -Xmx1024m -Dfile.encoding=UTF-8

REM 检查 target 目录下是否有 jar 文件
if exist "target\*.jar" (
    echo 找到已构建的 JAR 文件，直接运行...
    for /f "delims=" %%i in ('dir /b "target\*.jar" ^| findstr /v "sources"') do (
        echo 启动应用: %%i
        java %JAVA_OPTS% -Dspring.profiles.active=dev -jar "target\%%i"
        goto :end
    )
) else (
    echo 未找到已构建的 JAR 文件，尝试使用 Maven 启动...
    
    REM 检查是否安装了 Maven
    where mvn >nul 2>nul
    if %errorlevel% equ 0 (
        echo 使用 Maven 启动应用...
        mvn spring-boot:run -Dspring-boot.run.profiles=dev
    ) else (
        echo 错误: 未找到 Maven，请先安装 Maven 或构建项目。
        echo 可以通过运行 'mvn clean package' 来构建项目。
        pause
        exit /b 1
    )
)

:end
echo 应用已关闭
pause 