# -*- coding: utf-8 -*-
import re
import json
import os
from collections import Counter
import jieba.posseg as pseg
import jieba


# ============================================================
# GLM NER API 配置（与 summarizer.py 共用 API Key）
# ============================================================
_GLM_API_KEY = "02374cf332e343248cebf0bbc430d779.HUQSMSpMC1svAof1"  # 直接配置 API Key

def _get_glm_api_key():
    """惰性读取 GLM API Key（支持运行时设置）。"""
    # 优先使用代码中配置的 Key
    if _GLM_API_KEY:
        return _GLM_API_KEY
    
    # 尝试从环境变量读取
    key = os.getenv("GLM_API_KEY", "")
    if key:
        return key
    
    # 尝试从 streamlit secrets 读取
    try:
        import streamlit as st
        return st.secrets.get("GLM_API_KEY", "")
    except Exception:
        return ""

# ============================================================
# 领域词典：通用中文实体词典
# ============================================================

# 常见组织机构类型
COMMON_ORGS = [
    "公司", "集团", "企业", "银行", "医院", "学校", "大学", "学院", "研究所", "研究院",
    "协会", "基金会", "委员会", "中心", "部", "处", "科", "办公室", "局", "厅", "署",
    "党委", "政府", "厅", "局", "法院", "检察院", "人大", "政协", "工会", "团委",
]

# 常见地点类型
COMMON_LOCATIONS = [
    "省", "市", "区", "县", "镇", "乡", "村", "街道", "路", "街", "大道", "巷", "号",
    "大厦", "楼", "广场", "公园", "机场", "车站", "港口", "图书馆", "博物馆", "体育馆",
    "学校", "医院", "商场", "酒店", "餐厅", "公园", "景区", "校园", "校区",
]

# 常见活动/事件类型
COMMON_EVENTS = [
    "会议", "论坛", "峰会", "研讨会", "座谈会", "讲座", "培训", "展览", "展会",
    "比赛", "竞赛", "大赛", "运动会", "节", "庆典", "典礼", "仪式", "开幕式", "闭幕式",
    "发布会", "发布会", "新闻发布会", "产品发布会", "年会", "晚会", "舞会", "聚会",
    "招聘会", "宣讲会", "双选会", "面试", "笔试", "考试", "比赛", "评选", "表彰",
    "签约仪式", "启动仪式", "竣工仪式", "开业典礼", "婚礼", "葬礼",
]

# 常见专业/学科
COMMON_MAJORS = [
    "经济学", "金融学", "会计学", "管理学", "法学", "文学", "理学", "工学", "医学", "农学",
    "计算机科学", "软件工程", "电子信息", "机械工程", "土木工程", "建筑学", "艺术设计", "音乐", "体育",
    "哲学", "历史学", "教育学", "心理学", "新闻学", "传播学", "社会学", "政治学", "国际关系",
]

# 常见职务/职称
COMMON_WORK_TITLES = [
    "董事长", "总经理", "总裁", "CEO", "总监", "经理", "主管", "主任", "部长", "处长",
    "教授", "副教授", "讲师", "助教", "研究员", "副研究员", "工程师", "高级工程师",
    "书记", "副书记", "委员", "代表", "主席", "副主席", "秘书长", "副秘书长",
    "院长", "副院长", "校长", "副校长", "所长", "副所长", "科长", "副科长",
]

ORG_KEYWORDS = [
    "公司", "集团", "银行", "证券", "保险", "基金", "信托", "投资", "学院", "系", "处", "部", "科",
    "办公室", "中心", "所", "协会", "学会", "联盟", "社团", "联合会", "委员会", "学生会", "团委", "工会",
    "企业", "工厂", "报社", "电视台", "电台", "网站", "平台", "工作室", "事务所", "出版社", "杂志社",
]

LOC_KEYWORDS = [
    "省", "市", "区", "县", "镇", "乡", "村", "街道", "路", "街", "道", "巷", "号", "大厦", "楼", "广场",
    "公园", "机场", "站", "港", "湾", "河", "湖", "山", "岛", "城", "校区", "校园", "图书馆", "教学楼",
    "实验楼", "体育馆", "公寓", "食堂", "报告厅", "会议室", "礼堂", "操场", "门", "中心",
]

EVENT_KEYWORDS = ["讲座", "论坛", "报告会", "研讨会", "座谈会", "招聘会", "宣讲会", "双选会", "供需见面会",
                  "比赛", "竞赛", "大赛", "挑战赛", "运动会", "篮球赛", "足球赛", "排球赛", "乒乓球赛",
                  "羽毛球赛", "晚会", "典礼", "仪式", "开幕式", "闭幕式", "展览", "展演", "志愿服务",
                  "社会实践", "三下乡", "支教", "实习", "见习", "军训", "迎新", "毕业典礼", "开学典礼"]

MAJOR_KEYWORDS = ["专业", "系", "学科", "课程"]

WORK_KEYWORDS = ["书记", "校长", "院长", "系主任", "处长", "科长", "教授", "副教授", "讲师", "辅导员",
                 "班主任", "主任", "主席", "秘书长"]

LABEL_MAP = {
    "PERSON": "人名",
    "ORG": "组织机构",
    "LOC": "地点",
    "TIME": "时间",
    "EVENT": "活动/事件",
    "MAJOR": "专业/课程",
    "WORK": "职务/职称",
    "MONEY": "金额",
    "PERCENT": "百分比",
}

TYPE_COLORS = {
    "PERSON": "#ef4444",  # 红色 - 更醒目的红色
    "ORG": "#10b981",     # 绿色 - 更醒目的绿色
    "LOC": "#3b82f6",     # 蓝色 - 更醒目的蓝色
    "TIME": "#f59e0b",    # 橙色 - 更醒目的橙色
    "EVENT": "#8b5cf6",   # 紫色 - 更醒目的紫色
    "MAJOR": "#06b6d4",   # 青色 - 更醒目的青色
    "WORK": "#ec4899",    # 粉色 - 更醒目的粉色
    "MONEY": "#84cc16",   # 黄绿色 - 更醒目的黄绿色
    "PERCENT": "#6366f1", # 靛蓝色 - 更醒目的靛蓝色
}

# 正则模式：时间、金额、百分比
_TIME_PATTERNS = [
    (re.compile(r"\d{4}年\d{1,2}月\d{1,2}日(?:\s*\d{1,2}时\d{1,2}分)?"), "TIME"),
    (re.compile(r"\d{4}年\d{1,2}月"), "TIME"),
    (re.compile(r"\d{1,2}月\d{1,2}日(?:\s*\d{1,2}时\d{1,2}分)?"), "TIME"),
    (re.compile(r"\d{4}-\d{1,2}-\d{1,2}"), "TIME"),
    (re.compile(r"\d{4}/\d{1,2}/\d{1,2}"), "TIME"),
    (re.compile(r"\d{1,2}:\d{2}(?::\d{2})?"), "TIME"),
    (re.compile(r"\d{1,2}时\d{1,2}分"), "TIME"),
    (re.compile(r"(?:19|20)\d{2}年"), "TIME"),
    (re.compile(r"(?:昨|今|明|后|前)天|(?:上|下|本|前|后)周|(?:上|下|本|前|后)个月|(?:去|今|明|后)年"), "TIME"),
]

_MONEY_PATTERNS = [
    (re.compile(r"(?:\d+(?:,\d{3})*(?:\.\d+)?|\d+\.\d+)(?:[万亿])?(?:元|人民币|美元|欧元|英镑|日元|港币|新台币|韩元)"), "MONEY"),
]

_PERCENT_PATTERNS = [
    (re.compile(r"\d+(?:\.\d+)?%"), "PERCENT"),
    (re.compile(r"(?:百|千|万|亿)分之\d+(?:\.\d+)?"), "PERCENT"),
]


def _add_domain_words():
    """把领域词添加到 jieba 用户词典，提高切分与识别准确率。"""
    for word in set(COMMON_ORGS + COMMON_LOCATIONS + COMMON_EVENTS + COMMON_MAJORS + COMMON_WORK_TITLES):
        if word and len(word) >= 2:
            jieba.add_word(word, freq=1000)


_add_domain_words()


class NEREngine:
    """
    NER 引擎，基于 jieba 词性标注 + 领域词典 + 规则。
    零依赖，离线可用，适合所有部署环境。
    """

    def __init__(self):
        self.backend = "jieba"
        self.info = "jieba 领域词典 + 规则（离线可用）"

    def _is_chinese_name(self, word):
        if not (2 <= len(word) <= 4):
            return False
        if re.search(r"[0-9a-zA-Z\s\u3000]", word):
            return False
        non_person_suffix = ["公司", "集团", "银行", "学院", "系", "处", "部", "讲座", "比赛", "大会", "专业", "校区"]
        if any(word.endswith(s) for s in non_person_suffix):
            return False
        return True

    def _domain_classify(self, word, flag):
        if not word or len(word) < 2:
            return None
        # 检查通用领域词典
        if any(org in word for org in COMMON_ORGS):
            return "ORG"
        if any(loc in word for loc in COMMON_LOCATIONS):
            return "LOC"
        if any(event in word for event in COMMON_EVENTS):
            return "EVENT"
        if any(major in word for major in COMMON_MAJORS):
            return "MAJOR"
        if any(work in word for work in COMMON_WORK_TITLES):
            return "WORK"
        
        # 基于词性标注的分类
        if flag.startswith("nr") and self._is_chinese_name(word):
            return "PERSON"
        if flag == "ns":
            return "LOC"
        if flag == "nt":
            return "ORG"
        
        # 基于关键词后缀的分类
        if any(word.endswith(s) for s in ORG_KEYWORDS):
            return "ORG"
        if any(word.endswith(s) for s in LOC_KEYWORDS):
            return "LOC"
        if any(word.endswith(s) for s in EVENT_KEYWORDS):
            return "EVENT"
        if any(word.endswith(s) for s in MAJOR_KEYWORDS):
            return "MAJOR"
        if any(word.endswith(s) for s in WORK_KEYWORDS):
            return "WORK"
        
        return None

    def _extract_by_jieba(self, text):
        entities = []
        pos = 0
        for word, flag in pseg.cut(text):
            start = text.find(word, pos)
            if start == -1:
                start = pos
            end = start + len(word)
            pos = end
            ent_type = self._domain_classify(word, flag)
            if ent_type:
                entities.append({
                    "entity": ent_type,
                    "score": 0.85,
                    "word": word,
                    "start": start,
                    "end": end,
                })
        return entities

    def _extract_by_regex(self, text):
        entities = []
        for pattern, ent_type in _TIME_PATTERNS + _MONEY_PATTERNS + _PERCENT_PATTERNS:
            for m in pattern.finditer(text):
                entities.append({
                    "entity": ent_type,
                    "score": 0.95,
                    "word": m.group(),
                    "start": m.start(),
                    "end": m.end(),
                })
        return entities

    def _dedup_and_merge(self, entities):
        if not entities:
            return []
        entities.sort(key=lambda x: (x["start"], -x["end"]))
        merged = []
        for ent in entities:
            if not merged:
                merged.append(ent)
                continue
            last = merged[-1]
            if ent["start"] < last["end"] and ent["end"] > last["start"]:
                if (ent["end"] - ent["start"]) > (last["end"] - last["start"]):
                    merged[-1] = ent
                elif ent["score"] > last["score"] and (ent["end"] - ent["start"]) >= (last["end"] - last["start"]):
                    merged[-1] = ent
            else:
                merged.append(ent)
        return merged

    def extract(self, text):
        """对文本进行实体识别，返回实体列表。"""
        if not text or not isinstance(text, str):
            return []
        text = text.strip()
        if not text:
            return []

        entities = []
        entities.extend(self._extract_by_jieba(text))
        entities.extend(self._extract_by_regex(text))
        entities = self._dedup_and_merge(entities)
        entities.sort(key=lambda x: x["start"])
        return entities


def load_ner(backend="jieba"):
    """加载 NER 引擎（统一返回 jieba 引擎）。backend 参数保留以兼容旧调用。"""
    return NEREngine()


def extract_entities(text, nlp=None, max_len=None):
    """兼容旧接口的实体识别函数。"""
    engine = load_ner()
    return engine.extract(text)


class GLMNEREngine:
    """
    基于智谱 GLM 的 NER 引擎。
    通过 API 调用大模型进行实体识别，效果更准，支持更多实体类型。
    需要配置 GLM_API_KEY。
    """

    def __init__(self):
        self.backend = "glm"
        self.info = "智谱 GLM-5.2（在线大模型）"
        self._api_key = _get_glm_api_key()

    def _call_glm_ner(self, text):
        """调用 GLM API 进行实体识别。"""
        if not self._api_key:
            return None
        try:
            from openai import OpenAI
            client = OpenAI(
                api_key=self._api_key,
                base_url="https://open.bigmodel.cn/api/paas/v4/"
            )
            prompt = (
                "请从以下中文文本中识别命名实体，并以 JSON 数组格式返回。"
                "每个实体包含：word（实体文本）、entity（实体类型，可选值：PERSON/ORG/LOC/TIME/EVENT/MAJOR/WORK/MONEY/PERCENT）、start（起始字符位置）、end（结束字符位置）。"
                "只返回 JSON 数组，不要有其他内容。\n\n文本："
            )
            response = client.chat.completions.create(
                model="glm-5.2",
                messages=[
                    {"role": "system", "content": "你是一个命名实体识别助手，只返回 JSON 格式结果。"},
                    {"role": "user", "content": prompt + text[:4000]},
                ],
                max_tokens=1024,
                temperature=0.1,
            )
            result = response.choices[0].message.content.strip()
            # 提取 JSON 数组
            json_match = re.search(r'\[.*\]', result, re.DOTALL)
            if json_match:
                entities = json.loads(json_match.group())
                # 补全 score 字段
                for ent in entities:
                    if "score" not in ent:
                        ent["score"] = 0.95
                return entities
            return None
        except Exception as e:
            print(f"GLM NER 调用失败：{e}")
            return None

    def extract(self, text):
        """对文本进行实体识别，返回实体列表。"""
        if not text or not isinstance(text, str):
            return []
        text = text.strip()
        if not text:
            return []

        # 调用 GLM NER
        entities = self._call_glm_ner(text)
        if entities:
            return entities

        # 回退到 jieba
        print("GLM NER 失败，回退到 jieba 引擎")
        jieba_engine = NEREngine()
        return jieba_engine.extract(text)


def load_ner(backend="glm"):
    """
    加载 NER 引擎。
    backend: "glm"（默认，大模型）或 "jieba"（离线兜底）
    """
    if backend == "glm":
        engine = GLMNEREngine()
        # 检查 API Key 是否可用
        if not engine._api_key:
            print("未配置 GLM_API_KEY，自动切换到 jieba 引擎")
            return NEREngine()
        return engine
    else:
        return NEREngine()


def extract_entities(text, nlp=None, max_len=None, backend="glm"):
    """兼容旧接口的实体识别函数。backend 参数控制使用的引擎。"""
    engine = load_ner(backend=backend)
    return engine.extract(text)


if __name__ == "__main__":
    sample = "河北金融学院金融系于2024年5月15日在图书馆报告厅举办金融科技讲座，李明教授主讲，并与招商银行保定分行签署合作协议。"
    
    # 测试 GLM 引擎
    print("=== GLM 引擎 ===")
    engine = load_ner(backend="glm")
    print(f"引擎：{engine.info}")
    if isinstance(engine, GLMNEREngine) and not engine._api_key:
        print("（GLM API Key 未配置，已自动切换到 jieba）")
    print("实体识别结果：")
    for e in engine.extract(sample):
        print(e)
    
    # 测试 jieba 引擎
    print("\n=== jieba 引擎 ===")
    engine = load_ner(backend="jieba")
    print(f"引擎：{engine.info}")
    for e in engine.extract(sample):
        print(e)
