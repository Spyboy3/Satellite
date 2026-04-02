import base64
from PIL import Image
import io
import cv2
import numpy as np

def load_image(file_bytes: bytes) -> Image.Image:
    """Convert raw uploaded bytes to PIL Image"""
    image = Image.open(io.BytesIO(file_bytes))
    if image.mode != "RGB":
        image = image.convert("RGB")
    return image

def image_to_base64(image: Image.Image) -> str:
    """Convert PIL image to base64 string for API transport"""
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return encoded

def resize_image(image: Image.Image, max_size: int = 1024) -> Image.Image:
    """Resize image while maintaining aspect ratio"""
    w, h = image.size
    if max(w, h) > max_size:
        scale = max_size / max(w, h)
        image = image.resize((int(w * scale), int(h * scale)))
    return image

def draw_boxes(image: Image.Image, boxes: list, labels: list) -> Image.Image:
    """
    Draw bounding boxes on image
    boxes: list of [x1, y1, x2, y2]
    labels: list of strings
    """
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    for box, label in zip(boxes, labels):
        x1, y1, x2, y2 = [int(c) for c in box]
        cv2.rectangle(img_cv, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img_cv, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    result = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
    return result

def image_to_bytes(image: Image.Image) -> bytes:
    """Convert PIL image back to bytes for API response"""
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return buffer.getvalue()
