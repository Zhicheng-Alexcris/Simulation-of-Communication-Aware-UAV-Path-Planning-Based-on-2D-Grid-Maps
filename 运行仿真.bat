@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" main.py %*
) else (
    where py >nul 2>nul
    if not errorlevel 1 (
        py -3 main.py %*
    ) else (
        python main.py %*
    )
)
if errorlevel 1 goto failed
echo.
echo 仿真完成。默认结果位于本目录的 outputs 文件夹。
pause
exit /b 0
:failed
echo.
echo 运行失败。如提示缺少依赖，请先双击“安装依赖.bat”。
echo 如提示起终点或地图错误，请根据上方提示修改 config.json。
pause
exit /b 1
