@echo off
setlocal EnableExtensions

powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_dashboard_windows.ps1" %*
set "RC=%ERRORLEVEL%"
exit /b %RC%
