@echo off
setlocal EnableExtensions

call "%~dp0run.bat" %*
exit /b %errorlevel%
