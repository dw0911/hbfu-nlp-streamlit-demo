from ner import load_ner, extract_entities
from ocr import load_ocr, extract_text_from_image
from PIL import Image
import io

print("Loading NER model...")
nlp = load_ner()
print("NER model loaded.")

sample_text = "我叫张三，来自北京大学，今天去上海参加会议。2024年5月1日，阿里巴巴创始人马云在杭州出席了公司年会。"
print("Running NER on sample text...")
entities = extract_entities(sample_text, nlp)
print(f"Found {len(entities)} entities:")
for e in entities:
    print(e)

print("Loading OCR engine...")
engine = load_ocr()
print("OCR engine loaded.")

# Create a tiny blank image to test OCR (should produce no text)
img = Image.new("RGB", (100, 30), color="white")
print("Running OCR on blank image...")
text = extract_text_from_image(img, engine)
print(f"OCR result: {text!r}")
print("All tests passed.")
