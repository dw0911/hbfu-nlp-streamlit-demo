from PIL import Image
import numpy as np

try:
    from rapidocr import RapidOCR
    _RAPIDOCR_AVAILABLE = True
    _RAPIDOCR_ERROR = ""
except Exception as e:
    RapidOCR = None
    _RAPIDOCR_AVAILABLE = False
    _RAPIDOCR_ERROR = str(e)


def load_ocr():
    """加载 RapidOCR 引擎。若当前环境缺少依赖则返回 None。"""
    if not _RAPIDOCR_AVAILABLE:
        return None
    return RapidOCR()


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

