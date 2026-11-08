@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
echo 为毕业设计创建独立 Python 环境并安装依赖...
if exist ".venv\Scripts\python.exe" goto install
where py >nul 2>nul
if not errorlevel 1 (
    py -3 -m venv .venv
) else (
    python -m venv .venv
)
if errorlevel 1 goto failed
:install
".venv\Scripts\python.exe" -c "import sys; assert sys.version_info >= (3,10), 'Python 3.10 or newer is required'"
if errorlevel 1 goto failed
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto failed
echo.
echo 安装完成。现在可以双击“运行仿真.bat”。
pause
exit /b 0
:failed
echo.
echo 安装失败。请确认 Python 3.10 或以上版本已安装，且网络可用。
pause
exit /b 1
