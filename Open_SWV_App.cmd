@echo off
rem Keep the convenient entry point, but use the validated 64-bit launcher.
call "%~dp0launch_app_windows.bat"
exit /b %errorlevel%
