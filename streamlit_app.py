# streamlit_app.py - Streamlit Cloud 入口文件
# 此文件仅用于指向实际的应用文件 app.py

import os
import sys

# 将当前目录添加到 Python 路径
sys.path.insert(0, os.path.dirname(__file__))

# 导入主应用文件
# Streamlit 会在导入时自动执行 app.py 中的全局代码
import app
