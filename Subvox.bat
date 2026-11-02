@echo off
cd /d "%~dp0"
if exist .venv\Scripts\pythonw.exe (start "" .venv\Scripts\pythonw.exe -m subvox_win) else (start "" pythonw -m subvox_win)
