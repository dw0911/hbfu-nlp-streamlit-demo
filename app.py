# -*- coding: utf-8 -*-
# 这是一个临时修复文件，用于将对话功能独立成标签页
# 请将此文件重命名为 app.py 以使用

import json
from html import escape as _html_escape
import io
import base64
import pandas as pd
import streamlit as st
from PIL import Image
from collections import Counter

from ner import extract_entities, load_ner, LABEL_MAP, TYPE_COLORS
import ocr as _ocr
from extractor import fetch_url_text, parse_html, fetch_url_images

from summarizer import generate_report, classify_topic, chat_with_article
from visualizer import wordcloud_to_base64, build_pyvis_html, build_cooccurrence_matrix, build_mpl_network

st.set_page_config(
    page_title="河北金融学院公众号 NER 实体识别",
    layout="wide",
    page_icon="🎓",
)

# ============================================================
# 自定义样式：让界面更精美
# ============================================================
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.4rem;
        font-weight: 700;
        color: #1a5276;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #5d6d7e;
        margin-bottom: 1.5rem;
    }
    .author-badge {
        display: inline-block;
        background: linear-gradient(90deg, #1a5276, #2e86ab);
        color: white;
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: 500;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 16px;
        border-left: 5px solid #2e86ab;
    }
    .highlight-box {
        max-height: 520px;
        overflow-y: auto;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 16px;
        line-height: 1.8;
        background: #fafafa;
        font-size: 1.05rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="main-header">🎓 河北金融学院公众号 NER 实体识别</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">基于 NLP 的微信公众号文章实体识别、可视化与总结系统</div>',
    unsafe_allow_html=True,
)
st.markdown('<div class="author-badge">👨💻 开发作者：河北金融学院 应统 王平</div>', unsafe_allow_html=True)


@st.cache_resource(show_spinner="加载 OCR 引擎...")
def get_ocr():
    return _ocr.load_ocr()


def highlight_html(text, entities):
    """把实体在原文字符位置上用色块高亮。"""
    entities = sorted(entities, key=lambda x: x["start"])
    parts = []
    pos = 0
    for ent in entities:
        s, e = ent["start"], ent["end"]
        typ = ent["entity"]
        if s < pos:
            continue
        if s > pos:
            parts.append(_html_escape(text[pos:s]))
        color = TYPE_COLORS.get(typ, "#dddddd")
        label = LABEL_MAP.get(typ, typ)
        parts.append(
            f'<span style="background-color:{color}; padding:2px 5px; border-radius:4px; '
            f'margin:0 2px; font-weight:500; box-shadow:0 1px 2px rgba(0,0,0,0.1);" title="{label}">'
            f'{_html_escape(text[s:e])}</span>'
        )
        pos = e
    if pos < len(text):
        parts.append(_html_escape(text[pos:]))
    return "".join(parts)


# ============================================================
# 输入区
# ============================================================
with st.sidebar:
    st.header("⚙️ 输入与设置")
    input_mode = st.radio(
        "输入方式",
        ["粘贴文本", "上传图片（OCR）", "URL 抓取", "上传文件", "批量文件处理"],
        key="input_mode_radio"
    )

    st.markdown("---")
    st.markdown("**🤖 NER 实体识别引擎**")
    ner_backend = st.selectbox(
        "选择 NER 引擎",
        ["glm", "jieba"],
        index=0,  # 默认选择 GLM
        format_func=lambda x: "智谱 GLM-5.2（在线大模型，推荐）" if x == "glm" else "jieba 离线规则（兜底）",
        key="ner_backend_select"
    )
    
    # 检查 GLM API Key 是否配置
    if ner_backend == "glm":
        from ner import _get_glm_api_key
        glm_key = _get_glm_api_key()
        if not glm_key:
            st.warning("⚠️ 未配置 GLM_API_KEY，将自动切换到 jieba 引擎")
            ner_backend = "jieba"

    st.markdown("---")
    st.markdown("**ℹ️ 说明**")
    st.markdown("- **NER 引擎**：GLM 大模型（在线）或 jieba 离线规则。")
    st.markdown("- **大模型已自动配置**：无需手动填写 API Key。")
    st.markdown("- 支持文本、图片 OCR、URL、单文件、批量文件五种输入。")
    st.markdown("- 识别结果分为：实体识别、可视化、文章总结、自由对话四个标签页。")

    if st.button("🗑️ 一键重置", key="reset_all"):
        # 清除所有 session_state
        for key in list(st.session_state.keys()):
            if key != "input_mode_radio" and key != "ner_backend_select":  # 保留输入方式和引擎选择
                del st.session_state[key]
        st.success("已重置所有内容，可以重新开始。")
        st.rerun()


# ============================================================
# 收集文本输入
# ============================================================
text = ""
file_name = ""

if input_mode == "粘贴文本":
    text = st.text_area("请输入需要识别的文本", height=300, placeholder="粘贴河北金融学院公众号文章正文...", key="text_input")

elif input_mode == "上传图片（OCR）":
    img_file = st.file_uploader("上传文章截图（PNG / JPG）", type=["png", "jpg", "jpeg"], key="img_upload")
    if img_file:
        image = Image.open(img_file)
        st.image(image, caption="已上传图片", use_container_width=True)
        engine = get_ocr()
        if engine is None:
            st.error(f"OCR 引擎在当前环境不可用，图片识别功能已禁用。({_ocr._OCR_ERROR})")
        else:
            try:
                with st.spinner("正在进行 OCR 识别..."):
                    text = _ocr.extract_text_from_image(image, engine)
                text = st.text_area("OCR 识别结果（可编辑）", value=text, height=220, key="ocr_result")
            except Exception as e:
                st.error(f"OCR 识别失败：{e}")


elif input_mode == "URL 抓取":
    url = st.text_input("微信公众号文章 URL", placeholder="https://mp.weixin.qq.com/s/...", key="url_input")
    
    # 初始化 session_state 用于缓存 URL 抓取结果
    if "fetched_url" not in st.session_state:
        st.session_state["fetched_url"] = ""
    if "fetched_text" not in st.session_state:
        st.session_state["fetched_text"] = ""
    if "fetched_title" not in st.session_state:
        st.session_state["fetched_title"] = ""
    
    # 抓取按钮（常驻显示）
    fetch_clicked = st.button("🔍 抓取文章", key="fetch_url_btn", disabled=not (url and url.strip()))
    
    if fetch_clicked:
        try:
            with st.spinner("正在抓取文章并识别图片文字..."):
                title, fetched_text, html = fetch_url_text(url)
                engine = get_ocr()
                if engine is not None:
                    image_text = fetch_url_images(html, base_url=url, ocr_engine=engine)
                    if image_text.strip():
                        fetched_text += "\n\n" + image_text
                        st.info(f"已从图片中识别 {len(image_text)} 字符")
                else:
                    st.info("OCR 引擎不可用，跳过图片文字识别。")
            
            # 保存到 session_state
            st.session_state["fetched_url"] = url
            st.session_state["fetched_text"] = fetched_text
            st.session_state["fetched_title"] = title
            
            if title:
                st.success(f"已抓取：{title}")
            else:
                st.info("已提取正文，未获取到标题。")
        except Exception as e:
            st.error(f"抓取失败：{e}")
    
    # 如果已经抓取过，显示内容
    if st.session_state["fetched_text"]:
        text = st.text_area("正文（可编辑）", value=st.session_state["fetched_text"], height=300, key="url_text")


elif input_mode == "上传文件":
    uploaded = st.file_uploader("上传 .txt / .html", type=["txt", "html", "htm"], key="file_upload")
    if uploaded:
        try:
            raw = uploaded.read().decode("utf-8", errors="ignore")
            if uploaded.name.lower().endswith((".html", ".htm")):
                _, text = parse_html(raw)
            else:
                text = raw
            file_name = uploaded.name
            text = st.text_area("文件内容（可编辑）", value=text, height=300, key="file_text")
        except Exception as e:
            st.error(f"读取文件失败：{e}")

elif input_mode == "批量文件处理":
    batch_files = st.file_uploader("批量上传 .txt / .html", type=["txt", "html", "htm"], accept_multiple_files=True, key="batch_upload")
    if batch_files:
        batch_texts = {}
        for uploaded in batch_files:
            try:
                raw = uploaded.read().decode("utf-8", errors="ignore")
                if uploaded.name.lower().endswith((".html", ".htm")):
                    _, content = parse_html(raw)
                else:
                    content = raw
                batch_texts[uploaded.name] = content
            except Exception as e:
                st.error(f"读取 {uploaded.name} 失败：{e}")
        if batch_texts:
            st.success(f"已加载 {len(batch_texts)} 个文件")
            selected_file = st.selectbox("选择预览文件", list(batch_texts.keys()), key="batch_select")
            text = batch_texts[selected_file]
            file_name = selected_file
            text = st.text_area("当前文件内容（可编辑）", value=text, height=300, key="batch_text")
            # 保存到 session_state 供批量导出使用
            st.session_state["batch_texts"] = batch_texts


# 切换到非批量模式时清理残留的批量文本，避免底部导出块错误显示
if input_mode != "批量文件处理" and "batch_texts" in st.session_state:
    del st.session_state["batch_texts"]



# ============================================================
# 初始化 session_state
# ============================================================
if "recognition_done" not in st.session_state:
    st.session_state["recognition_done"] = False
if "recognized_text" not in st.session_state:
    st.session_state["recognized_text"] = ""
if "recognized_entities" not in st.session_state:
    st.session_state["recognized_entities"] = []
if "recognized_report" not in st.session_state:
    st.session_state["recognized_report"] = None



# ============================================================
# 识别按钮与结果处理
# ============================================================
if st.button("🚀 开始识别", type="primary", disabled=not (text and text.strip())):
    try:
        with st.spinner(f"正在使用 {ner_backend.upper()} 引擎识别实体，请稍候..."):
            use_glm = True  # 大模型已自动配置
            entities = extract_entities(text, backend=ner_backend)
            
            # 保存到 session_state
            st.session_state["recognition_done"] = True
            st.session_state["recognized_text"] = text
            st.session_state["recognized_entities"] = entities
            st.session_state["recognized_report"] = generate_report(text, entities, use_glm=use_glm)
            
            st.rerun()  # 重新运行以显示结果
    except Exception as e:
        st.error(f"识别失败：{e}")


# 如果已经识别过，显示结果
if st.session_state["recognition_done"] and st.session_state["recognized_text"]:
    text = st.session_state["recognized_text"]
    entities = st.session_state["recognized_entities"]
    report = st.session_state["recognized_report"]
    
    # 显示识别结果（与原逻辑相同）
    if not entities:
        st.info("未识别到实体，请检查输入内容或更换文章。")
    else:
        engine_name = "智谱 GLM-5.2" if ner_backend == "glm" else "jieba 离线规则"
        st.success(f"识别到 {len(entities)} 个实体 · 当前引擎：{engine_name}")

        df = pd.DataFrame(entities)
        df["类型"] = df["entity"].map(LABEL_MAP).fillna(df["entity"])
        df = df.rename(columns={"word": "实体", "score": "置信度"})
        df = df[["类型", "实体", "置信度"]]

        # 统计指标
        col_metric1, col_metric2, col_metric3, col_metric4 = st.columns(4)
        with col_metric1:
            st.metric("实体总数", len(entities))
        with col_metric2:
            st.metric("实体类型数", df["类型"].nunique())
        with col_metric3:
            top_type = df["类型"].value_counts().idxmax() if not df.empty else "-"
            st.metric("最多类型", top_type)
        with col_metric4:
            top_entity = df["实体"].value_counts().idxmax() if not df.empty else "-"
            st.metric("最高频实体", top_entity)

        tab_rec, tab_viz, tab_summary, tab_chat = st.tabs(["📝 实体识别", "📊 可视化分析", "📋 文章总结", "🤖 自由对话"])

        # ============================================================
        # 识别区
        # ============================================================
        with tab_rec:
            col_left, col_right = st.columns([3, 2])
            with col_left:
                st.subheader("原文高亮")
                highlighted = highlight_html(text, entities)
                st.markdown(
                    f'<div class="highlight-box">{highlighted}</div>',
                    unsafe_allow_html=True,
                )

            with col_right:
                st.subheader("实体列表")
                types = sorted(df["类型"].unique())
                selected = st.multiselect("按类型筛选", types, default=types, key="entity_filter_multiselect")
                filtered = df[df["类型"].isin(selected)]
                st.dataframe(filtered, use_container_width=True, height=420)

                csv = filtered.to_csv(index=False).encode("utf-8-sig")
                col_dl1, col_dl2 = st.columns(2)
                with col_dl1:
                    st.download_button("📥 CSV", csv, "entities.csv", "text/csv")
                with col_dl2:
                    json_str = json.dumps(entities, ensure_ascii=False, indent=2)
                    st.download_button("📥 JSON", json_str, "entities.json", "application/json")

        # ============================================================
        # 可视化区
        # ============================================================
        with tab_viz:
            viz_col1, viz_col2 = st.columns([2, 3])
            with viz_col1:
                st.subheader("实体类型分布")
                type_counts = df["类型"].value_counts()
                st.bar_chart(type_counts, use_container_width=True)

                st.subheader("高频实体 Top 10")
                top_entities = df["实体"].value_counts().head(10).reset_index()
                top_entities.columns = ["实体", "出现次数"]
                st.bar_chart(top_entities.set_index("实体"), use_container_width=True)

            with viz_col2:
                st.subheader("☁️ 词云图")
                wc_b64 = wordcloud_to_base64(text, max_words=100)
                if wc_b64:
                    st.markdown(
                        f'<div style="text-align:center; background:#fff; border-radius:8px; padding:10px; border:1px solid #e0e0e0;">'
                        f'<img src="data:image/png;base64,{wc_b64}" style="max-width:100%;" />'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.info("文本过短，无法生成词云图。")

                st.subheader("🕸️ 实体共现网络")
                cooccur_df = build_cooccurrence_matrix(entities)
                if cooccur_df is not None and not cooccur_df.empty:
                    # 优先尝试交互式 pyvis，失败或不可用时回退到静态网络图
                    try:
                        pyvis_html = build_pyvis_html(entities, height=520)
                        if pyvis_html:
                            st.components.v1.html(pyvis_html, height=540)
                        else:
                            st.info("交互式网络不可用，已切换为静态网络图。")
                            net_img = build_mpl_network(entities, figsize=(10, 6))
                            if net_img:
                                st.image(net_img, use_container_width=True)
                    except Exception as e:
                        st.warning(f"交互式网络加载失败，已切换为静态网络图：{e}")
                        net_img = build_mpl_network(entities, figsize=(10, 6))
                        if net_img:
                            st.image(net_img, use_container_width=True)

                    st.markdown("**共现矩阵 Top 20**")
                    st.dataframe(cooccur_df.head(20), use_container_width=True, height=260)
                else:
                    st.info("实体数量不足或类型单一，无法构建共现网络。")

        # ============================================================
        # 总结区
        # ============================================================
        with tab_summary:
            col_s1, col_s2 = st.columns([3, 2])
            with col_s1:
                st.subheader("📝 文章摘要")
                st.info(report["summary"] if report["summary"] else "文本过短，无法生成摘要。")

                st.subheader("🔑 关键词")
                keywords = report.get("keywords", [])
                if keywords:
                    tags = " ".join([f"<span style='display:inline-block;background:#eaf2f8;color:#1a5276;padding:4px 10px;border-radius:12px;margin:3px;font-size:0.9rem;'>{k}</span>" for k, _ in keywords[:15]])
                    st.markdown(tags, unsafe_allow_html=True)
                else:
                    st.markdown("未提取到关键词。")

            with col_s2:
                st.subheader("📌 文章主题")
                st.metric("主题分类", report["topic"])

                st.subheader("🔍 关键信息")
                info = report["key_info"]
                for label, key in [("🏢 涉及机构", "orgs"), ("👤 人物", "persons"), ("📍 地点", "locations"),
                                   ("🎉 活动/事件", "events"), ("📚 专业/课程", "majors"), ("⏰ 时间", "times")]:
                    vals = info.get(key, [])
                    display_vals = "、".join(vals[:8]) if vals else "暂无"
                    st.markdown(f"**{label}**：{display_vals}")

            report_json = json.dumps(report, ensure_ascii=False, indent=2)
            st.download_button("📥 下载总结报告（JSON）", report_json, "report.json", "application/json", key="report_download")

        # ============================================================
        # 自由对话区（独立标签页）
        # ============================================================
        with tab_chat:
            st.subheader("🤖 自由对话（基于文章内容）")
            st.caption("⚠️ 限制：最多 3 轮对话")

            # 初始化对话状态
            if "chat_history" not in st.session_state:
                st.session_state["chat_history"] = []
            if "chat_round" not in st.session_state:
                st.session_state["chat_round"] = 0
            if "chat_article_hash" not in st.session_state:
                st.session_state["chat_article_hash"] = None

            # 识别完成后，自动初始化对话（将文章摘要作为首轮对话）
            # 使用文章内容的 hash 来追踪是否换了文章
            import hashlib
            current_hash = hashlib.md5(text.encode()).hexdigest()[:16]
            
            if st.session_state["chat_article_hash"] != current_hash:
                # 新文章，重新初始化对话
                if report.get("summary"):
                    summary_intro = f"我已阅读这篇文章，以下是摘要：\n\n{report['summary']}\n\n您可以基于这篇文章提问，我会尽力回答。"
                    st.session_state["chat_history"] = [["请介绍这篇文章的主要内容", summary_intro]]
                    st.session_state["chat_round"] = 1
                    st.session_state["chat_article_hash"] = current_hash

            # 显示对话历史
            if st.session_state["chat_history"]:
                st.markdown("**对话历史**")
                for i, (user_msg, ai_msg) in enumerate(st.session_state["chat_history"]):
                    st.markdown(f"**🧑 用户**：{user_msg}")
                    st.markdown(f"**🤖 AI**：{ai_msg}")
                    if i < len(st.session_state["chat_history"]) - 1:
                        st.markdown("---")

            # 检查是否达到轮次上限
            if st.session_state["chat_round"] >= 3:
                st.warning("已达到对话轮次上限（3 轮）。点击清除对话可重新开始。")
            else:
                # 用户输入（使用 form 避免页面刷新）
                with st.form(key="chat_form", clear_on_submit=True):
                    user_input = st.text_input(
                        f"第 {st.session_state['chat_round'] + 1} 轮对话",
                        placeholder="请输入您的问题（基于文章内容）...",
                        key="chat_input"
                    )
                    submit_button = st.form_submit_button(label="发送")

                    if submit_button and user_input:
                        with st.spinner("AI 正在思考..."):
                            ai_reply, updated_history = chat_with_article(
                                text,
                                st.session_state["chat_history"],
                                user_input
                            )
                            st.session_state["chat_history"] = updated_history
                            st.session_state["chat_round"] = len(updated_history)
                            st.rerun()

            # 清除对话按钮
            if st.session_state["chat_history"]:
                if st.button("🗑️ 清除对话", key="chat_clear"):
                    st.session_state["chat_history"] = []
                    st.session_state["chat_round"] = 0
                    st.session_state["chat_article_hash"] = None
                    st.rerun()


# ============================================================
# 批量导出（仅在批量处理模式显示）
# ============================================================
if input_mode == "批量文件处理" and "batch_texts" in st.session_state:
    st.markdown("---")
    st.subheader("📦 批量处理导出")
    if st.button("一键处理所有文件并导出汇总", type="secondary", key="batch_export_btn"):
        try:
            use_glm = True  # 大模型已自动配置
            all_results = []
            progress = st.progress(0)
            for idx, (fname, content) in enumerate(st.session_state["batch_texts"].items()):
                ents = extract_entities(content, backend=ner_backend)
                report = generate_report(content, ents, use_glm=use_glm)
                all_results.append({
                    "文件名": fname,
                    "实体数": len(ents),
                    "主题": report["topic"],
                    "摘要": report["summary"].replace("\n", " "),
                    "关键词": ", ".join([k for k, _ in report["keywords"][:10]]),
                    "涉及机构": ", ".join(report["key_info"]["orgs"][:10]),
                    "人物": ", ".join(report["key_info"]["persons"][:10]),
                    "地点": ", ".join(report["key_info"]["locations"][:10]),
                    "时间": ", ".join(report["key_info"]["times"][:10]),
                })
                progress.progress((idx + 1) / len(st.session_state["batch_texts"]))
            batch_df = pd.DataFrame(all_results)
            st.dataframe(batch_df, use_container_width=True, height=400)
            csv = batch_df.to_csv(index=False).encode("utf-8-sig")
            st.download_button("📥 下载批量汇总 CSV", csv, "batch_summary.csv", "text/csv", key="batch_csv_download")
        except Exception as e:
            st.error(f"批量导出失败：{e}")
