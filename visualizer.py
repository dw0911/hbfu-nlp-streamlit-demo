# -*- coding: utf-8 -*-
import base64
import io
import re
from collections import Counter, defaultdict

import os

from wordcloud import WordCloud
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import pandas as pd





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
    """生成词云图，返回 PIL Image 对象。"""
    if not text or not isinstance(text, str):
        return None

    # 简单清洗：去除标点和数字，只保留中文字符
    cleaned = re.sub(r"[^\u4e00-\u9fa5]", " ", text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if not cleaned:
        return None

    wc = WordCloud(
        font_path=_find_font(),
        width=width,
        height=height,
        max_words=max_words,
        background_color="white",
        colormap="Spectral",
        contour_width=1,
        contour_color="steelblue",
    ).generate(cleaned)


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

    # 按类型分组，只保留部分类型用于共现网络
    focus_types = {"PERSON", "ORG", "LOC", "EVENT", "MAJOR"}
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


def build_pyvis_html(entities, window_size=50, height=600):
    """用 pyvis 生成实体共现网络图 HTML。"""
    try:
        from pyvis.network import Network
    except ImportError:
        return None

    nodes, edges = build_cooccurrence(entities, window_size)
    if not nodes:
        return None

    net = Network(height=f"{height}px", width="100%", bgcolor="#ffffff", font_color="#333333", notebook=False)
    net.barnes_hut(gravity=-2000, central_gravity=0.3, spring_length=120, spring_strength=0.05)

    type_color = {
        "PERSON": "#ffadad",
        "ORG": "#caffbf",
        "LOC": "#9bf6ff",
        "EVENT": "#ffc6ff",
        "MAJOR": "#fdffb6",
    }

    for node in nodes:
        size = 15 + min(node["count"] * 3, 30)
        net.add_node(
            node["id"],
            label=node["label"],
            title=f"{node['label']} ({node['type']})\n出现次数：{node['count']}",
            color=type_color.get(node["type"], "#dddddd"),
            size=size,
        )

    for edge in edges:
        width = 1 + min(edge["weight"], 5)
        net.add_edge(edge["source"], edge["target"], value=edge["weight"], title=f"共现次数：{edge['weight']}")

    html = net.generate_html()
    return html


def build_cooccurrence_matrix(entities, focus_types=None):
    """构建实体共现矩阵，返回 DataFrame。"""
    if focus_types is None:
        focus_types = {"PERSON", "ORG", "LOC", "EVENT", "MAJOR"}
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
    type_color = {
        "PERSON": "#ffadad",
        "ORG": "#caffbf",
        "LOC": "#9bf6ff",
        "EVENT": "#ffc6ff",
        "MAJOR": "#fdffb6",
    }
    node_colors = []
    for node in nodes:
        G.add_node(node["id"], label=node["label"], count=node["count"])
        node_labels[node["id"]] = node["label"]
        node_colors.append(type_color.get(node["type"], "#dddddd"))

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
