from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
import uvicorn
from utils import load_image, resize_image, draw_boxes, image_to_bytes
from orchestrator import (
    run_caption_pipeline,
    run_vqa_pipeline,
    run_grounding_pipeline,
    run_scene_pipeline
)
from metrics import evaluate_captions, evaluate_boxes, evaluate_vqa

app = FastAPI(title="GeoNLI Satellite API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"status": "GeoNLI backend is running"}

@app.post("/caption")
async def caption(file: UploadFile = File(...)):
    try:
        raw = await file.read()
        image = load_image(raw)
        image = resize_image(image)
        result = run_caption_pipeline(image)
        return JSONResponse(result)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/vqa")
async def vqa(
    file: UploadFile = File(...),
    question: str = Form(...)
):
    try:
        raw = await file.read()
        image = load_image(raw)
        image = resize_image(image)
        result = run_vqa_pipeline(image, question)
        return JSONResponse(result)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/ground")
async def ground(
    file: UploadFile = File(...),
    query: str = Form(...)
):
    try:
        raw = await file.read()
        image = load_image(raw)
        image = resize_image(image)
        result = run_grounding_pipeline(image, query)
        if result["status"] == "success" and result["boxes"]:
            annotated = draw_boxes(image, result["boxes"], result["labels"])
            img_bytes = image_to_bytes(annotated)
            return Response(content=img_bytes, media_type="image/jpeg")
        return JSONResponse(result)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/scene")
async def scene(file: UploadFile = File(...)):
    try:
        raw = await file.read()
        image = load_image(raw)
        image = resize_image(image)
        result = run_scene_pipeline(image)
        return JSONResponse(result)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/metrics")
async def metrics(data: dict):
    """
    Expects JSON body:
    {
        "type": "caption" | "grounding" | "vqa",
        "predictions": [...],
        "ground_truths": [...]
    }
    """
    try:
        t = data.get("type")
        preds = data.get("predictions", [])
        gts = data.get("ground_truths", [])

        if t == "caption":
            result = evaluate_captions(gts, preds)
        elif t == "grounding":
            result = evaluate_boxes(preds, gts)
        elif t == "vqa":
            result = evaluate_vqa(preds, gts)
        else:
            return JSONResponse(
                {"error": "type must be caption, grounding, or vqa"},
                status_code=400
            )

        return JSONResponse(result)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
