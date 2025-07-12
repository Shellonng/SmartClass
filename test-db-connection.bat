@echo off
echo 测试数据库连接...
echo.
echo 正在启动Spring Boot应用以测试数据库连接...
echo 如果看到"APPLICATION STARTED"字样，说明数据库连接成功！
echo.
cd backend
mvn spring-boot:run -Dspring-boot.run.arguments="--server.port=8081"
pause 