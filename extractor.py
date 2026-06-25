import re
import io
import requests
from urllib.parse import urljoin
from bs4 import BeautifulSoup
from PIL import Image


HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}


def fetch_url_text(url, timeout=20):
    """抓取 URL 并提取微信公众号文章正文，同时返回原始 HTML。"""
    resp = requests.get(url, headers=HEADERS, timeout=timeout)
    resp.raise_for_status()
    resp.encoding = resp.apparent_encoding or "utf-8"
    html = resp.text
    title, text = parse_html(html)
    return title, text, html



def parse_html(html_text):
    """解析 HTML，提取正文内容。支持多种网站结构。"""
    soup = BeautifulSoup(html_text, "html.parser")
    for tag in soup(["script", "style", "noscript", "iframe", "svg", "nav", "footer", "header"]):
        tag.decompose()

    title = ""
    
    # 尝试多种标题提取方式（按优先级）
    # 1. 常见文章标题 class
    for class_name in ["rich_media_title", "article-title", "post-title", "entry-title", "title", "news-title", "headline"]:
        title_tag = soup.find(class_=class_name)
        if title_tag:
            title = title_tag.get_text(strip=True)
            break
    
    # 2. 尝试 h1, h2 标签
    if not title:
        for tag_name in ["h1", "h2"]:
            title_tag = soup.find(tag_name)
            if title_tag:
                title = title_tag.get_text(strip=True)
                break
    
    # 3. 最后尝试 title 标签
    if not title:
        title_tag = soup.find("title")
        if title_tag:
            title = title_tag.get_text(strip=True)
    
    if title:
        # 清理标题中的多余空格和换行
        title = re.sub(r'\s+', ' ', title).strip()

    # 尝试多种正文内容提取方式（按优先级）
    content = None
    
    # 1. 微信公众号文章
    content = soup.find("div", id="js_content")
    
    # 2. 常见文章容器
    if not content:
        for class_name in ["article-content", "post-content", "entry-content", "content", "article-body", "text"]:
            content = soup.find(class_=class_name)
            if content:
                break
    
    # 3. article 标签
    if not content:
        content = soup.find("article")
    
    # 4. main 标签
    if not content:
        content = soup.find("main")
    
    # 5. 兜底：使用整个页面
    if not content:
        content = soup
    
    if content:
        text = content.get_text(separator="\n", strip=True)
    else:
        text = soup.get_text(separator="\n", strip=True)

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return title, "\n".join(lines)


def extract_image_urls(html_text, base_url=None):
    """从 HTML 中提取图片 URL，优先使用微信文章的 data-src。"""
    soup = BeautifulSoup(html_text, "html.parser")
    urls = []
    for img in soup.find_all("img"):
        url = img.get("data-src") or img.get("src")
        if not url:
            continue
        url = url.strip()
        if url.startswith("data:"):
            continue
        if base_url:
            url = urljoin(base_url, url)

        lower = url.lower()
        # 仅保留常见图片格式或微信公众号图片
        if not (any(ext in lower for ext in [".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"]) or "mmbiz" in lower):
            continue
        # 跳过明显的小图标、二维码、表情
        if any(k in lower for k in ["emoji", "icon", "qrcode", "qr", "weixin_qr"]):
            continue
        if url not in urls:
            urls.append(url)
    return urls


def fetch_image_text(url, ocr_engine=None, timeout=20):
    """下载单张图片并 OCR 识别文字。"""
    try:
        from ocr import extract_text_from_image

        headers = HEADERS.copy()
        headers["Referer"] = url
        resp = requests.get(url, headers=headers, timeout=timeout)
        resp.raise_for_status()
        image = Image.open(io.BytesIO(resp.content)).convert("RGB")
        # 跳过过小图标
        if image.size[0] < 50 or image.size[1] < 50:
            return ""
        text = extract_text_from_image(image, ocr_engine)
        return text.strip() if text else ""
    except Exception:
        return ""


def fetch_url_images(html_text, base_url=None, ocr_engine=None, max_images=20):
    """抓取 HTML 中图片的文字内容，返回合并后的文本。"""
    urls = extract_image_urls(html_text, base_url=base_url)[:max_images]
    if not urls:
        return ""

    texts = []
    for url in urls:
        txt = fetch_image_text(url, ocr_engine=ocr_engine)
        if txt:
            texts.append(txt)
    return "\n\n".join(texts)

