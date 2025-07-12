@echo off
echo ========================================
echo    Dify AI 平台连接测试脚本
echo ========================================
echo.

echo 1. 测试基本网络连接...
ping -n 1 219.216.65.108 > nul
if %errorlevel% == 0 (
    echo    ✅ 网络连接正常
) else (
    echo    ❌ 网络连接失败
    echo    请检查网络设置和防火墙
)
echo.

echo 2. 测试HTTP连接...
curl -s -o nul -w "HTTP状态码: %%{http_code}\n" http://219.216.65.108
if %errorlevel% == 0 (
    echo    ✅ HTTP连接正常
) else (
    echo    ❌ HTTP连接失败
    echo    请检查Dify服务器是否运行
)
echo.

echo 3. 测试API端点...
curl -s -o nul -w "API端点状态: %%{http_code}\n" http://219.216.65.108/v1/
if %errorlevel% == 0 (
    echo    ✅ API端点可访问
) else (
    echo    ❌ API端点不可访问
    echo    请检查API路径配置
)
echo.

echo 4. 配置检查清单:
echo    📋 请确保以下配置正确:
echo    ▸ Dify服务器地址: http://219.216.65.108
echo    ▸ API版本: v1
echo    ▸ 完整API地址: http://219.216.65.108/v1
echo.

echo 5. 下一步操作:
echo    📝 1. 访问 http://219.216.65.108 登录Dify管理后台
echo    📝 2. 创建以下工作流应用:
echo       - paper-generation (组卷工作流)
echo       - auto-grading (自动批改工作流)  
echo       - knowledge-graph (知识图谱生成工作流)
echo    📝 3. 获取每个应用的API密钥
echo    📝 4. 更新backend/src/main/resources/application.yml中的配置:
echo       education:
echo         dify:
echo           api-url: http://219.216.65.108
echo           api-keys:
echo             paper-generation: app-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
echo             auto-grading: app-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
echo             knowledge-graph: app-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
echo    📝 5. 重启应用程序
echo.

echo 📖 详细配置说明请参考: docs/dify-setup-guide.md
echo.

echo ========================================
echo    测试完成
echo ========================================
pause 