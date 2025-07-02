@echo off
echo 修复TeacherLayout.vue中重复的"首页"问题...

REM 确保Node.js已安装
where node >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
  echo 错误: 未找到Node.js，请先安装Node.js
  exit /b 1
)

REM 运行修复脚本
node fix-breadcrumb.js

if %ERRORLEVEL% EQU 0 (
  echo 修复成功！请重新启动前端服务以查看效果。
) else (
  echo 修复失败，请手动编辑文件。
  echo 文件路径: Vue\frontend\src\components\layout\TeacherLayout.vue
  echo 需要修改的内容:
  echo 1. 找到第432行左右的代码:
  echo    const breadcrumbItems = computed(() => {
  echo      const items = [{ title: '首页', path: '/teacher' }]
  echo      ...
  echo 2. 将其修改为:
  echo    const breadcrumbItems = computed(() => {
  echo      const items = []
  echo      ...
)

pause 