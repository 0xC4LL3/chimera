@echo off
pushd "%~dp0..\.."
if errorlevel 1 exit /b 1
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%CD%\unreal\ChimeraDemo\launch.ps1" %*
set "chimera_exit=%errorlevel%"
popd
exit /b %chimera_exit%
