# -*- coding: utf-8 -*-
import os
import re
import json
from statistics import mean
import jieba.posseg as pseg
import jieba

# ============================================================
# 领域词典：河北金融学院官方公众号常见实体
# ============================================================

HBFU_CORE_ORGS = [
    "河北金融学院",
    "党委",
    "校长办公室",
    "教务处",
    "学生处",
    "科研处",
    "财务处",
    "人事处",
    "招生就业处",
    "国际合作与交流处",
    "团委",
    "工会",
    "图书馆",
    "信息中心",
    "后勤管理处",
    "保卫处",
    "金融系",
    "会计系",
    "保险系",
    "管理系",
    "经济贸易系",
    "计算机系",
    "外语系",
    "法律系",
    "文传系",
    "国际教育学院",
    "继续教育学院",
    "马克思主义学院",
    "体育教学部",
    "学生会",
    "社团联合会",
    "青年志愿者协会",
    "红十字会",
    "心理协会",
    "创业协会",
    "辩论队",
    "广播台",
    "大学生通讯社",
]

HBFU_EXTERNAL_ORGS = [
    "中国人民银行",
    "国家金融监督管理总局",
    "中国证监会",
    "中国银保监会",
    "河北省教育厅",
    "保定市教育局",
    "保定市",
    "河北省",
    "河北金融学院",
    "清华大学",
    "北京大学",
    "中国人民大学",
    "中央财经大学",
    "对外经济贸易大学",
    "河北大学",
    "河北农业大学",
    "保定学院",
]

HBFU_LOCATIONS = [
    "保定",
    "恒祥",
    "金院",
    "图书馆",
    "教学楼",
    "实验楼",
    "体育馆",
    "学生公寓",
    "食堂",
    "报告厅",
    "会议室",
    "礼堂",
    "操场",
    "南门",
    "北门",
    "西门",
    "东门",
    "校园",
    "大学生活动中心",
    "众创空间",
    "孵化基地",
    "实训中心",
    "金融中心",
]

HBFU_EVENTS = [
    "讲座",
    "论坛",
    "学术报告",
    "报告会",
    "研讨会",
    "座谈会",
    "招聘会",
    "宣讲会",
    "双选会",
    "供需见面会",
    "比赛",
    "竞赛",
    "大赛",
    "挑战赛",
    "创新创业大赛",
    "运动会",
    "篮球赛",
    "足球赛",
    "排球赛",
    "乒乓球赛",
    "羽毛球赛",
    "晚会",
    "典礼",
    "仪式",
    "开幕式",
    "闭幕式",
    "展览",
    "展演",
    "志愿服务",
    "社会实践",
    "三下乡",
    "支教",
    "实习",
    "见习",
    "军训",
    "迎新",
    "毕业典礼",
    "开学典礼",
    "升旗仪式",
    "主题党日",
    "主题团日",
    "班会",
]

HBFU_MAJORS = [
    "金融学",
    "会计学",
    "保险学",
    "财务管理",
    "审计学",
    "国际经济与贸易",
    "经济学",
    "经济统计学",
    "计算机科学与技术",
    "软件工程",
    "网络工程",
    "数据科学与大数据技术",
    "英语",
    "商务英语",
    "法学",
    "汉语言文学",
    "新闻学",
    "工商管理",
    "市场营销",
    "人力资源管理",
    "电子商务",
    "统计学",
    "数学与应用数学",
    "应用统计学",
    "投资学",
    "信用管理",
    "金融工程",
    "金融科技",
    "精算学",
    "劳动与社会保障",
    "行政管理",
    "信息与计算科学",
]

HBFU_WORK_TITLES = [
    "党委书记",
    "校长",
    "副校长",
    "党委副书记",
    "纪委书记",
    "院长",
    "副院长",
    "系主任",
    "副主任",
    "教授",
    "副教授",
    "讲师",
    "助教",
    "辅导员",
    "班主任",
    "处长",
    "副处长",
    "科长",
    "书记",
    "副书记",
    "团委书记",
    "学生会主席",
    "社团负责人",
    "秘书长",
    "主任",
    "总工程师",
    "总会计师",
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
    "PERSON": "#ffadad",
    "ORG": "#caffbf",
    "LOC": "#9bf6ff",
    "TIME": "#ffd6a5",
    "EVENT": "#ffc6ff",
    "MAJOR": "#fdffb6",
    "WORK": "#bdb2ff",
    "MONEY": "#ffffcc",
    "PERCENT": "#e2f0cb",
}

SPACY_LABEL_MAP = {
    "PERSON": "PERSON",
    "ORG": "ORG",
    "GPE": "LOC",
    "LOC": "LOC",
    "FAC": "LOC",
    "EVENT": "EVENT",
    "NORP": "ORG",
    "WORK_OF_ART": "EVENT",
    "LAW": "EVENT",
    "PRODUCT": "ORG",
    "DATE": "TIME",
    "TIME": "TIME",
    "MONEY": "MONEY",
    "PERCENT": "PERCENT",
    "CARDINAL": "MONEY",
    "ORDINAL": "TIME",
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
    for word in set(HBFU_CORE_ORGS + HBFU_EXTERNAL_ORGS + HBFU_LOCATIONS + HBFU_EVENTS + HBFU_MAJORS + HBFU_WORK_TITLES):
        if word and len(word) >= 2:
            jieba.add_word(word, freq=1000)


_add_domain_words()


class NEREngine:
    """统一 NER 引擎，支持 spaCy（高级模型）和 jieba（离线备选）。"""

    def __init__(self, backend="auto"):
        self.backend = backend
        self.spacy_nlp = None
        self.info = ""

        if backend in ("auto", "spacy"):
            try:
                import spacy
                # 先检查模型是否已安装，避免直接加载抛出长错误
                if not spacy.util.is_package("zh_core_web_sm"):
                    raise ModuleNotFoundError(
                        "spaCy 中文模型 zh_core_web_sm 未安装。"
                        "如需使用高级模型，请运行：python -m spacy download zh_core_web_sm"
                    )
                self.spacy_nlp = spacy.load("zh_core_web_sm")
                self.backend = "spacy"
                self.info = "spaCy zh_core_web_sm"
            except Exception as e:
                self.backend = "jieba"
                if backend == "auto":
                    self.info = f"jieba 离线规则（spaCy 中文模型未安装，已自动回退）"
                else:
                    raise RuntimeError(
                        f"spaCy 模型加载失败：{e}\n"
                        f"请尝试运行：python -m spacy download zh_core_web_sm"
                    )

        else:
            self.backend = "jieba"
            self.info = "jieba 离线规则"


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
        if word in HBFU_CORE_ORGS or word in HBFU_EXTERNAL_ORGS:
            return "ORG"
        if word in HBFU_LOCATIONS:
            return "LOC"
        if word in HBFU_EVENTS:
            return "EVENT"
        if word in HBFU_MAJORS:
            return "MAJOR"
        if word in HBFU_WORK_TITLES:
            return "WORK"
        if flag.startswith("nr") and self._is_chinese_name(word):
            return "PERSON"
        if flag == "ns":
            return "LOC"
        if flag == "nt":
            return "ORG"
        if flag == "nw":
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
        if any(word.endswith(s) for s in ORG_KEYWORDS) or any(s in word for s in ["公司", "集团", "银行", "学院", "协会"]):
            return "ORG"
        if any(word.endswith(s) for s in LOC_KEYWORDS):
            return "LOC"
        if any(word.endswith(s) for s in EVENT_KEYWORDS):
            return "EVENT"
        if any(word.endswith(s) for s in ["专业", "系", "学科", "课程"]):
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

    def _extract_by_spacy(self, text):
        entities = []
        if self.spacy_nlp is None:
            return entities
        doc = self.spacy_nlp(text)
        for ent in doc.ents:
            typ = SPACY_LABEL_MAP.get(ent.label_, None)
            if typ:
                entities.append({
                    "entity": typ,
                    "score": 0.90,
                    "word": ent.text,
                    "start": ent.start_char,
                    "end": ent.end_char,
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
        if self.backend == "spacy":
            entities.extend(self._extract_by_spacy(text))
        # spaCy 对专业、职务、活动识别有限，用 jieba 补充
        entities.extend(self._extract_by_jieba(text))
        entities.extend(self._extract_by_regex(text))
        entities = self._dedup_and_merge(entities)
        entities.sort(key=lambda x: x["start"])
        return entities


def load_ner(backend="auto"):
    """加载 NER 引擎。backend 可选：auto（自动选择）、spacy、jieba。"""
    return NEREngine(backend=backend)


def extract_entities(text, nlp=None, max_len=None):
    """兼容旧接口的实体识别函数。"""
    if nlp is None:
        nlp = load_ner()
    if not isinstance(nlp, NEREngine):
        nlp = load_ner()
    return nlp.extract(text)


if __name__ == "__main__":
    sample = "河北金融学院金融系于2024年5月15日在图书馆报告厅举办金融科技讲座，李明教授主讲，并与招商银行保定分行签署合作协议。"
    engine = load_ner()
    print(f"引擎：{engine.info}")
    print("实体识别结果：")
    for e in engine.extract(sample):
        print(e)
