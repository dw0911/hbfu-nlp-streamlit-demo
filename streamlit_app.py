# -*- coding: utf-8 -*-
# Streamlit Cloud 默认入口文件
# 重要：每次 Streamlit rerun 时，必须强制重新加载 app 模块，
# 否则 Python 的 sys.modules 缓存会导致 import 被跳过 → 白屏
import sys
sys.modules.pop('app', None)  # 清除缓存，确保每次 rerun 都重新执行 app.py
import app
