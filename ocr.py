# -*- coding: utf-8 -*-
"""OCR 模块 —— 基于 Tesseract，避免 opencv 系统依赖冲突。"""
from PIL import Image
import io

_OCR_AVAILABLE = False
_OCR_ERROR = ""

try:
    import pytesseract
    # 尝试验证 tesseract 可执行文件是否存在
    ver = pytesseract.get_tesseract_version()
    _OCR_AVAILABLE = True
    _OCR_ERROR = ""
except Exception as e:
    pytesseract = None
    _OCR_AVAILABLE = False
    _OCR_ERROR = f"Tesseract OCR 不可用: {e}"


def load_ocr():
    """加载 OCR 引擎。返回 True 表示可用，None 表示不可用。"""
    if not _OCR_AVAILABLE:
        return None
    return True


def extract_text_from_image(image, engine=None):
    """对图片做 OCR，返回识别出的文字。"""
    if engine is None:
        if not _OCR_AVAILABLE:
            return ""
        engine = load_ocr()
        if engine is None:
            return ""

    # 统一转为 PIL Image
    if isinstance(image, Image.Image):
        img = image
    elif isinstance(image, str):
        img = Image.open(image)
    elif isinstance(image, bytes):
        img = Image.open(io.BytesIO(image))
    else:
        # numpy array 等其他格式
        img = Image.fromarray(image)

    # 使用中英文混合识别
    try:
        text = pytesseract.image_to_string(img, lang='chi_sim+eng')
        return text.strip()
    except Exception:
        return ""
