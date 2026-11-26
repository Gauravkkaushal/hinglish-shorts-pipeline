@echo off
rem One-time setup: private Python environment in .venv, all packages, sound effects. Safe to run again.
cd /d "%~dp0"
where python >nul 2>nul || (echo Python is missing. Run: winget install Python.Python.3.13   then open a new terminal. & exit /b 1)
rem an interrupted setup can leave .venv without pip: then it is made again
if exist .venv\Scripts\python.exe .venv\Scripts\python.exe -m pip --version >nul 2>nul || rmdir /s /q .venv 2>nul
if not exist .venv python -m venv .venv || exit /b 1
set "PATH=%~dp0.venv\Scripts;%PATH%"
python -m pip install --upgrade pip -q
python -m pip install -r requirements.txt || exit /b 1
rem rapidocr may bring opencv-python-headless or opencv 5, which clash with opencv-python 4
python -m pip uninstall -y -q opencv-python-headless opencv-contrib-python
python -m pip install -q --force-reinstall --no-deps "opencv-python<5" || exit /b 1
if not exist .env copy .env.example .env >nul
python make_demo_assets.py || exit /b 1
echo.
python pipeline.py doctor
echo.
echo Next: put your Groq key into .env, then:   run fetch-memes   (once)   and   run auto --creator @ZakirKhan
