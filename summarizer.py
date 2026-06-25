import re
import json
import jieba
import jieba.analyse as analyse
from collections import Counter

# 停用词（简单版，可扩展）
_STOPWORDS = set([
    "的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都", "一", "一个", "上", "也", "很", "到", "说", "要", "去", "你", "会", "着", "没有", "看", "好", "自己", "这", "那", "之", "与", "及", "等", "对", "可以", "已经", "但是", "而", "为", "被", "让", "向", "从", "将", "并", "中", "后", "前", "内", "外", "以", "以及", "进行", "通过", "根据", "关于", "关于", "今日", "近日", "今年", "去年", "明年",
])

# 主题分类关键词
_TOPIC_KEYWORDS = {
    "党建思政": ["党委", "党建", "思政", "党员", "党课", "主题教育", "学习", "贯彻", "二十大", "会议精神", "政治"],
    "教学科研": ["教学", "课程", "专业", "学科", "科研", "课题", "论文", "教材", "教改", "教学质量", "课堂", "学术"],
    "学生工作": ["学生", "学生会", "社团", "志愿", "实践", "三下乡", "支教", "辅导员", "班主任", "奖学", "资助", "心理"],
    "招生就业": ["招生", "就业", "招聘", "宣讲", "双选会", "招聘会", "实习", "岗位", "用人单位", "求职"],
    "校园活动": ["讲座", "论坛", "报告会", "比赛", "竞赛", "大赛", "运动会", "晚会", "典礼", "仪式", "展览", "展演"],
    "合作交流": ["合作", "签约", "协议", "交流", "访问", "调研", "校企", "银校", "共建", "战略合作"],
    "通知公告": ["通知", "公告", "公示", "关于开展", "关于举办", "关于组织", "请各", "请相关", "届时"],
}


def _split_sentences(text):
    """按中文句号、感叹号、问号分句，保留句子。"""
    if not text:
        return []
    sentences = re.split(r"(?<=[。！？\n])", text)
    return [s.strip() for s in sentences if s.strip()]


def _clean_sentence(sentence):
    """简单清洗句子。"""
    sentence = sentence.replace("\n", " ").strip()
    sentence = re.sub(r"\s+", " ", sentence)
    return sentence


def summarize(text, top_k=3):
    """抽取式文章摘要：返回摘要文本与关键词列表。"""
    if not text or not isinstance(text, str):
        return "", []

    text = text.strip()
    if not text:
        return "", []

    sentences = _split_sentences(text)
    if not sentences:
        return text, []

    # 提取关键词（带权重）
    keywords = analyse.extract_tags(text, topK=25, withWeight=True)
    keyword_set = {k for k, _ in keywords}
    keyword_weight = {k: w for k, w in keywords}

    scored = []
    for idx, sent in enumerate(sentences):
        sent_clean = _clean_sentence(sent)
        if not sent_clean or len(sent_clean) < 10:
            continue

        score = 0.0
        words = set(jieba.lcut(sent_clean))
        for w in words:
            if w in keyword_weight:
                score += keyword_weight[w]

        # 包含时间、地点、事件、组织的句子加分
        if re.search(r"\d{4}年|\d{1,2}月|\d{1,2}日", sent_clean):
            score += 1.2
        if re.search(r"学院|系|处|部|公司|银行|集团", sent_clean):
            score += 0.8
        if re.search(r"讲座|论坛|报告会|招聘会|比赛|竞赛|大赛|典礼|仪式|会议", sent_clean):
            score += 1.0
        if re.search(r"举办|召开|举行|开展|组织|签约|合作|启动", sent_clean):
            score += 0.8

        # 首句、末句适度加权
        if idx == 0:
            score += 1.0
        if idx == len(sentences) - 1:
            score += 0.5

        scored.append((score, idx, sent_clean))

    if not scored:
        return text, keywords

    # 选择 top_k 句子，并按原文顺序组合
    scored.sort(key=lambda x: x[0], reverse=True)
    top_sentences = scored[:top_k]
    top_sentences.sort(key=lambda x: x[1])

    summary = "\n".join(s for _, _, s in top_sentences)
    return summary, keywords


def classify_topic(text):
    """基于关键词匹配进行简单主题分类。"""
    if not text:
        return "其他"

    text = text.lower()
    scores = {}
    for topic, keywords in _TOPIC_KEYWORDS.items():
        score = sum(text.count(k) for k in keywords)
        if score > 0:
            scores[topic] = score

    if not scores:
        return "其他"
    return max(scores, key=scores.get)


def extract_key_info(text, entities):
    """从实体中汇总关键信息：涉及机构、人物、地点、活动、专业、时间。"""
    if not entities:
        entities = []

    info = {
        "orgs": [],
        "persons": [],
        "locations": [],
        "events": [],
        "majors": [],
        "times": [],
    }

    for ent in entities:
        typ = ent.get("entity")
        word = ent.get("word", "").strip()
        if not word:
            continue
        if typ == "ORG":
            info["orgs"].append(word)
        elif typ == "PERSON":
            info["persons"].append(word)
        elif typ == "LOC":
            info["locations"].append(word)
        elif typ == "EVENT":
            info["events"].append(word)
        elif typ == "MAJOR":
            info["majors"].append(word)
        elif typ == "TIME":
            info["times"].append(word)

    # 去重并保持顺序
    for key in info:
        seen = set()
        unique = []
        for w in info[key]:
            if w not in seen:
                unique.append(w)
                seen.add(w)
        info[key] = unique

    # 如果机构为空，尝试用正则抓取包含学院/系/处的词
    if not info["orgs"]:
        org_candidates = re.findall(r"[\u4e00-\u9fa5]{2,}(?:学院|系|处|部|办公室|中心|所|银行|公司|集团)", text)
        info["orgs"] = list(dict.fromkeys(org_candidates))[:10]

    return info


def generate_report(text, entities):
    """生成完整的文章概括报告。"""
    summary, keywords = summarize(text)
    topic = classify_topic(text)
    key_info = extract_key_info(text, entities)

    report = {
        "topic": topic,
        "summary": summary,
        "keywords": keywords,
        "key_info": key_info,
    }
    return report


if __name__ == "__main__":
    sample = (
        "河北金融学院金融系于2024年5月15日在图书馆报告厅举办金融科技讲座。"
        "李明教授主讲，并与招商银行保定分行签署合作协议。"
        "此次讲座旨在提升学生对金融科技的理解，推动产学研合作。"
        "金融系师生共计200余人参加了本次活动。"
    )
    from ner import extract_entities

    ents = extract_entities(sample)
    report = generate_report(sample, ents)
    print(json.dumps(report, ensure_ascii=False, indent=2))


