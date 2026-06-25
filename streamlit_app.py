# streamlit_app.py - Streamlit Cloud 入口文件
# 此文件仅用于指向实际的应用文件 app.py

import sys

# 关键修复：每次 rerun 时强制重新导入 app 模块
# 否则 Streamlit 交互（点击按钮等）时 import app 会因模块缓存而变成空操作，导致白屏
if 'app' in sys.modules:
    del sys.modules['app']

import app
