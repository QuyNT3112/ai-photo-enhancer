@echo off
title Clarity AI Studio - Super-Resolution Engine
cd /d "%~dp0"
echo ========================================================
echo   Starting Clarity AI Studio (NVIDIA RTX 3060 CUDA)
echo   Local Web URL: http://127.0.0.1:7860
echo ========================================================
python server.py
pause
