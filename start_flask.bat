@echo off

rem 切换路径到虚拟环境并激活
call .venv/Scripts/activate.bat

cd /d C:\Users\Administrator\Desktop\股指策略_训练平台

rem js挂载flask开启
python flask接口.py

pause