# -*- coding: utf-8 -*-
import base64
import io
import re
from collections import Counter, defaultdict

import os

# 必须在 import matplotlib.pyplot 之前设置后端，避免无头服务器环境崩溃
import matplotlib
matplotlib.use('Agg')

from wordcloud import WordCloud
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import pandas as pd
import jieba


# 中文停用词表（简化版）
_STOP_WORDS = {
    "的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都", "一", "一个", "上", "也", "很", "到", "说", "要", "去", "你", "会", "着", "没有", "看", "好", "自己", "这", "那", "这些", "那些", "这个", "那个", "之", "与", "及", "等", "或", "但", "而", "因", "于", "则", "以", "被", "把", "给", "让", "向", "从", "对", "关于", "为", "作为", "可以", "进行", "已经", "正在", "开始", "完成", "实现", "提出", "通过", "根据", "随着", "由于", "为了", "以及", "其中", "其", "他", "她", "它", "他们", "她们", "它们", "我们", "你们", "咱们",
    "今天", "明天", "昨天", "今年", "去年", "明年", "目前", "现在", "当时", "期间", "时间", "时候", "日期", "年度", "月", "日", "年",
    "公司", "企业", "单位", "部门", "学院", "大学", "学校", "专业", "学生", "老师", "教授", "同志", "先生", "女士",
    "需要", "表示", "认为", "指出", "介绍", "报道", "记者", "网讯", "本报", "本刊", "网络", "图片", "文章", "报道", "发表", "发布",
}


def _is_meaningful_word(word):
    """判断一个词是否值得放入词云：长度>=2、非纯数字、非停用词。"""
    if not word or len(word.strip()) < 2:
        return False
    if word in _STOP_WORDS:
        return False
    # 过滤纯数字、纯标点、纯英文字母（通常不是中文关键词）
    if re.fullmatch(r"\d+", word):
        return False
    if re.fullmatch(r"[a-zA-Z]+", word):
        return False
    if re.fullmatch(r"[^\u4e00-\u9fa5a-zA-Z0-9]+", word):
        return False
    return True





# 常见跨平台中文字体候选路径，按顺序查找并返回第一个存在的字体
_CANDIDATE_FONTS = [
    # Windows
    r"C:\Windows\Fonts\simhei.ttf",
    r"C:\Windows\Fonts\msyh.ttc",
    r"C:\Windows\Fonts\simsun.ttc",
    r"C:\Windows\Fonts\ARIALUNI.ttf",
    # macOS
    "/System/Library/Fonts/PingFang.ttc",
    "/Library/Fonts/Arial Unicode.ttf",
    # Linux
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]


def _find_font():
    """返回可用的中文字体文件路径，找不到则返回 None。"""
    for path in _CANDIDATE_FONTS:
        if os.path.isfile(path):
            return path
    return None



def _generate_wordcloud_image(text, max_words=100, width=800, height=400):
    """生成词云图：先 jieba 分词，再过滤停用词，最后按词频生成。"""
    if not text or not isinstance(text, str):
        return None

    # 1. jieba 分词
    words = jieba.lcut(text)
    # 2. 过滤无意义词
    words = [w.strip() for w in words if _is_meaningful_word(w.strip())]
    if not words:
        return None

    # 3. 按空格连接成 WordCloud 需要的格式
    word_text = " ".join(words)

    wc = WordCloud(
        font_path=_find_font(),
        width=width,
        height=height,
        max_words=max_words,
        background_color="white",
        colormap="tab20",
        contour_width=1,
        contour_color="steelblue",
    ).generate(word_text)

    return wc.to_image()


def wordcloud_to_base64(text, max_words=100):
    """将词云图转为 base64 字符串，供 HTML img 标签使用。"""
    img = _generate_wordcloud_image(text, max_words)
    if img is None:
        return None
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


def build_cooccurrence(entities, window_size=50):
    """基于实体在文本中的位置窗口，构建共现网络。

    返回：
        nodes: [{id, label, type, count}, ...]
        edges: [{source, target, weight}, ...]
    """
    if not entities:
        return [], []

    # 按类型分组，保留核心实体类型用于共现网络
    focus_types = {"PERSON", "ORG", "GPE", "LOC", "EVENT", "MAJOR", "PRODUCT", "TECH", "LAW"}
    filtered = [e for e in entities if e.get("entity") in focus_types]
    if len(filtered) < 2:
        return [], []

    # 按位置排序
    filtered.sort(key=lambda x: x["start"])

    # 滑动窗口内两两共现
    cooccur = defaultdict(int)
    node_count = Counter()
    for i, ent in enumerate(filtered):
        node_count[(ent["word"], ent["entity"])] += 1
        for j in range(i + 1, len(filtered)):
            other = filtered[j]
            if other["start"] - ent["end"] > window_size:
                break
            if ent["word"] == other["word"]:
                continue
            pair = tuple(sorted([ent["word"], other["word"]]))
            cooccur[pair] += 1

    nodes = []
    node_ids = {}
    for idx, ((word, typ), count) in enumerate(node_count.items()):
        node_id = f"n{idx}"
        node_ids[word] = node_id
        nodes.append({
            "id": node_id,
            "label": word,
            "type": typ,
            "count": count,
        })

    edges = []
    for (a, b), weight in cooccur.items():
        if a in node_ids and b in node_ids:
            edges.append({
                "source": node_ids[a],
                "target": node_ids[b],
                "weight": weight,
            })

    return nodes, edges


# 统一的实体类型颜色映射（用于网络图）
TYPE_COLOR_MAP = {
    "PERSON":  "#e63946",  # 深红
    "ORG":     "#1d3557",  # 深蓝
    "GPE":     "#f97316",  # 深橙
    "LOC":     "#2a9d8f",  # 翠绿
    "EVENT":   "#e9c46a",  # 金黄
    "MAJOR":   "#9b5de5",  # 亮紫
    "PRODUCT": "#6366f1",  # 靛蓝
    "TECH":    "#14b8a6",  # teal
    "LAW":     "#a855f7",  # 紫罗兰
}


def build_pyvis_html(entities, window_size=50, height=600, max_nodes=80):
    """用 pyvis 生成实体共现网络图 HTML。节点过多时自动截断，避免前端卡死。"""
    try:
        from pyvis.network import Network
    except ImportError:
        return None

    try:
        nodes, edges = build_cooccurrence(entities, window_size)
        if not nodes:
            return None

        # 限制节点数量，防止生成的 HTML 过大导致 Streamlit 前端崩溃
        if len(nodes) > max_nodes:
            nodes.sort(key=lambda x: x["count"], reverse=True)
            top_nodes = nodes[:max_nodes]
            node_ids = {n["id"] for n in top_nodes}
            edges = [e for e in edges if e["source"] in node_ids and e["target"] in node_ids]
            nodes = top_nodes

        net = Network(height=f"{height}px", width="100%", bgcolor="#ffffff", font_color="#333333", notebook=False)
        net.barnes_hut(gravity=-2000, central_gravity=0.3, spring_length=120, spring_strength=0.05)

        for node in nodes:
            size = 15 + min(node["count"] * 3, 30)
            net.add_node(
                node["id"],
                label=node["label"],
                title=f"{node['label']} ({node['type']})\n出现次数：{node['count']}",
                color=TYPE_COLOR_MAP.get(node["type"], "#dddddd"),
                size=size,
            )

        for edge in edges:
            width = 1 + min(edge["weight"], 5)
            net.add_edge(edge["source"], edge["target"], value=edge["weight"], title=f"共现次数：{edge['weight']}")

        html = net.generate_html()
        return html
    except Exception:
        return None



def build_cooccurrence_matrix(entities, focus_types=None):
    """构建实体共现矩阵，返回 DataFrame。"""
    if focus_types is None:
        focus_types = {"PERSON", "ORG", "GPE", "LOC", "EVENT", "MAJOR", "PRODUCT", "TECH", "LAW"}
    filtered = [e for e in entities if e.get("entity") in focus_types]
    if len(filtered) < 2:
        return None

    filtered.sort(key=lambda x: x["start"])
    cooccur = defaultdict(int)
    for i, ent in enumerate(filtered):
        for j in range(i + 1, len(filtered)):
            other = filtered[j]
            if other["start"] - ent["end"] > 50:
                break
            if ent["word"] == other["word"]:
                continue
            pair = tuple(sorted([ent["word"], other["word"]]))
            cooccur[pair] += 1

    if not cooccur:
        return None

    rows = []
    for (a, b), count in cooccur.items():
        rows.append({"实体A": a, "实体B": b, "共现次数": count})
    return pd.DataFrame(rows).sort_values("共现次数", ascending=False).reset_index(drop=True)


def build_mpl_network(entities, figsize=(10, 8)):
    """用 matplotlib + networkx 绘制静态共现网络图，返回 PIL Image。"""
    nodes, edges = build_cooccurrence(entities)
    if not nodes:
        return None

    G = nx.Graph()
    node_labels = {}
    node_colors = []
    for node in nodes:
        G.add_node(node["id"], label=node["label"], count=node["count"])
        node_labels[node["id"]] = node["label"]
        node_colors.append(TYPE_COLOR_MAP.get(node["type"], "#dddddd"))

    for edge in edges:
        G.add_edge(edge["source"], edge["target"], weight=edge["weight"])

    font_path = _find_font()
    if font_path:
        font_prop = FontProperties(fname=font_path)
        font_family = font_prop.get_name()
        plt.rcParams["font.sans-serif"] = [font_family]
        plt.rcParams["axes.unicode_minus"] = False
    else:
        font_family = "sans-serif"

    plt.figure(figsize=figsize)
    pos = nx.spring_layout(G, k=0.5, iterations=50)
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=[G.nodes[n]["count"] * 200 + 200 for n in G.nodes])
    nx.draw_networkx_edges(G, pos, width=[d["weight"] * 0.5 for u, v, d in G.edges(data=True)], alpha=0.6)
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=10, font_family=font_family)
    plt.axis("off")
    plt.tight_layout()



    buffer = io.BytesIO()
    plt.savefig(buffer, format="PNG", dpi=100, bbox_inches="tight")
    plt.close()
    buffer.seek(0)
    return buffer


if __name__ == "__main__":
    sample_entities = [
        {"entity": "ORG", "word": "河北金融学院", "start": 0, "end": 6},
        {"entity": "ORG", "word": "金融系", "start": 6, "end": 9},
        {"entity": "PERSON", "word": "李明", "start": 36, "end": 38},
        {"entity": "ORG", "word": "招商银行", "start": 45, "end": 49},
        {"entity": "LOC", "word": "保定", "start": 49, "end": 51},
        {"entity": "EVENT", "word": "讲座", "start": 33, "end": 35},
    ]
    nodes, edges = build_cooccurrence(sample_entities)
    print("Nodes:", nodes)
    print("Edges:", edges)
