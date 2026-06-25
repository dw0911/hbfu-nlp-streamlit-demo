# -*- coding: utf-8 -*-
from PIL import Image
import numpy as np
import os
import traceback

# 设置模型缓存目录，避免每次下载
os.environ.setdefault('RAPIDOCR_HOME', os.path.expanduser('~/.rapidocr'))

try:
    from rapidocr import RapidOCR
    _RAPIDOCR_AVAILABLE = True
    _RAPIDOCR_ERROR = ""
except Exception as e:
    RapidOCR = None
    _RAPIDOCR_AVAILABLE = False
    _RAPIDOCR_ERROR = f"RapidOCR 导入失败: {e}"


def load_ocr():
    """加载 RapidOCR 引擎。若当前环境缺少依赖或初始化失败则返回 None。"""
    global _RAPIDOCR_AVAILABLE, _RAPIDOCR_ERROR
    if not _RAPIDOCR_AVAILABLE:
        return None
    try:
        # RapidOCR 3.x 初始化，设置较小的检测阈值以加快速度
        engine = RapidOCR(
            text_score=0.5,
            box_thresh=0.5,
        )
        return engine
    except TypeError:
        # 如果参数不支持，回退到无参数初始化
        try:
            return RapidOCR()
        except Exception as e:
            _RAPIDOCR_AVAILABLE = False
            _RAPIDOCR_ERROR = f"RapidOCR 初始化失败: {e}"
            return None
    except Exception as e:
        _RAPIDOCR_AVAILABLE = False
        _RAPIDOCR_ERROR = f"RapidOCR 初始化失败: {e}\n{traceback.format_exc()}"
        return None




def extract_text_from_image(image, engine=None):
    """对图片做 OCR，按文本行从上到下排序后返回整段文字。"""
    if engine is None:
        if not _RAPIDOCR_AVAILABLE:
            return ""
        engine = load_ocr()
        if engine is None:
            return ""

    if isinstance(image, Image.Image):
        img = np.array(image.convert("RGB"))
    elif isinstance(image, (str, bytes)):
        img = image
    else:
        img = np.array(image)

    result = engine(img)
    if result is None or not result.txts:
        return ""

    txts = list(result.txts)
    boxes = result.boxes
    if boxes is not None and len(boxes) == len(txts):
        indexed = []
        for txt, box in zip(txts, boxes):
            y = box[:, 1].min()
            x = box[:, 0].min()
            indexed.append((y, x, txt))
        indexed.sort()
        return "\n".join(t[2] for t in indexed)

    return "\n".join(txts)

