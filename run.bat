@echo off
rem run <anything pipeline.py understands>, in the environment setup.bat made, from any folder
if exist "%~dp0.venv\Scripts" set "PATH=%~dp0.venv\Scripts;%PATH%"
if /i "%~1"=="fetch-memes" (
  pushd "%~dp0"
  python fetch_memes.py
  popd
) else (
  python "%~dp0pipeline.py" %*
)
