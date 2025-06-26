@echo off
echo 正在启动教育平台应用程序...

REM 设置 Java 和应用程序参数
set JAVA_OPTS=-Xms512m -Xmx1024m -Dspring.profiles.active=dev -Dfile.encoding=UTF-8
set MAIN_CLASS=com.education.EducationApplication

REM 设置类路径（包含编译的类和依赖项）
set CLASSPATH=target\classes

REM 添加 Maven 本地仓库的依赖项（这里是简化版，实际需要更多依赖）
if exist "%USERPROFILE%\.m2\repository" (
    echo 找到 Maven 本地仓库
    for /r "%USERPROFILE%\.m2\repository" %%i in (*.jar) do (
        set CLASSPATH=!CLASSPATH!;%%i
    )
) else (
    echo 警告: 未找到 Maven 本地仓库，可能需要手动下载依赖
)

echo 使用类路径: %CLASSPATH%
echo 启动主类: %MAIN_CLASS%

REM 启动应用程序
java %JAVA_OPTS% -cp "%CLASSPATH%" %MAIN_CLASS%

pause 