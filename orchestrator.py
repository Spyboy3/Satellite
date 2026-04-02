from PIL import Image

# ─────────────────────────────────────────────────────────────
# PERSON 1 — swap these imports when their code is ready
# from captioning import get_caption
# from vqa import get_vqa_answer
# from grounding import get_bounding_boxes
# from scene import get_scene_class
# ─────────────────────────────────────────────────────────────

# Placeholders until Person 1 is done
def get_caption(image: Image.Image) -> str:
    return "Placeholder: Person 1 replaces this"

def get_vqa_answer(image: Image.Image, question: str) -> str:
    return "Placeholder: Person 1 replaces this"

def get_bounding_boxes(image: Image.Image, query: str):
    # Must return (list of [x1,y1,x2,y2], list of label strings)
    return [], []

def get_scene_class(image: Image.Image) -> str:
    return "Placeholder: Person 1 replaces this"

# ─────────────────────────────────────────────────────────────
# Orchestration functions — YOUR code, don't touch after this
# ─────────────────────────────────────────────────────────────

def run_caption_pipeline(image: Image.Image) -> dict:
    try:
        caption = get_caption(image)
        return {
            "status": "success",
            "caption": caption
        }
    except Exception as e:
        return {
            "status": "error",
            "caption": None,
            "error": str(e)
        }

def run_vqa_pipeline(image: Image.Image, question: str) -> dict:
    try:
        answer = get_vqa_answer(image, question)
        return {
            "status": "success",
            "question": question,
            "answer": answer
        }
    except Exception as e:
        return {
            "status": "error",
            "question": question,
            "answer": None,
            "error": str(e)
        }

def run_grounding_pipeline(image: Image.Image, query: str) -> dict:
    try:
        boxes, labels = get_bounding_boxes(image, query)
        return {
            "status": "success",
            "query": query,
            "boxes": boxes,
            "labels": labels
        }
    except Exception as e:
        return {
            "status": "error",
            "query": query,
            "boxes": [],
            "labels": [],
            "error": str(e)
        }

def run_scene_pipeline(image: Image.Image) -> dict:
    try:
        scene = get_scene_class(image)
        return {
            "status": "success",
            "scene": scene
        }
    except Exception as e:
        return {
            "status": "error",
            "scene": None,
            "error": str(e)
        }
