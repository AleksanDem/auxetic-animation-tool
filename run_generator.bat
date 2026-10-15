@echo off
chcp 65001 > nul
title Генератор анимаций ауксетиков (PHD_0826)
echo Запуск Streamlit-генератора анимаций...
python -m streamlit run "D:\02_01_Antigrav\Project\PHD_0826_Dissertation\05_src\04_Animation_Generator\app.py"
pause
