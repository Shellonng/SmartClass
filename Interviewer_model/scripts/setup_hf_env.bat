@echo off
echo ============================================================
echo 设置HuggingFace环境变量
echo ============================================================
echo.

REM 设置用户环境变量
setx HF_HOME "E:\HuggingFace_Cache"
setx TRANSFORMERS_CACHE "E:\HuggingFace_Cache\hub"
setx HF_DATASETS_CACHE "E:\HuggingFace_Cache\datasets"

echo [OK] 环境变量已设置:
echo   HF_HOME = E:\HuggingFace_Cache
echo   TRANSFORMERS_CACHE = E:\HuggingFace_Cache\hub
echo   HF_DATASETS_CACHE = E:\HuggingFace_Cache\datasets
echo.
echo ============================================================
echo 注意：需要重新打开终端才能生效
echo ============================================================
pause

