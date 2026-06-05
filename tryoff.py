import os
import re
import json
import torch
import requests
import uuid
from io import BytesIO
from PIL import Image
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

from ultralytics import YOLO
from diffusers import DiffusionPipeline
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

FLUX_MODEL_REPO = "black-forest-labs/FLUX.2-klein-4B"
QWEN_MODEL_REPO = "Qwen/Qwen2-VL-2B-Instruct" 
BBOX_MODEL_PATH = "models/bbox2.pt" 
OUTPUT_DIR = "outputs"

TARGET_GARMENT_IDS = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 
    13, 14, 15, 16, 17, 18, 20, 21, 22, 23, 24, 25, 26, 
    27, 35, 46, 47, 48, 49, 50, 51, 52, 53, 54
]

os.makedirs(OUTPUT_DIR, exist_ok=True)

app = FastAPI(title="Fashion Try-Off API")
device = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading Models into VRAM... This will take a moment.")

qwen_processor = AutoProcessor.from_pretrained(QWEN_MODEL_REPO)
qwen_model = Qwen2VLForConditionalGeneration.from_pretrained(
    QWEN_MODEL_REPO,
    torch_dtype=torch.float16,
    device_map="auto"
)
yolo_model = YOLO(BBOX_MODEL_PATH).to(device)
flux_pipe = DiffusionPipeline.from_pretrained(
    FLUX_MODEL_REPO,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
).to(device)
flux_pipe.enable_attention_slicing()

print("✅ Server Ready!")

class ImageRequest(BaseModel):
    image_url: str

@app.post("/process")
async def process_image(request: ImageRequest):
    if not request.image_url:
        raise HTTPException(status_code=400, detail="Image URL is required.")

    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(request.image_url, headers=headers, timeout=15)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content)).convert("RGB")
        img.thumbnail((1024, 1024)) 
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to download image: {str(e)}")

    qwen_prompt = """Analyze the clothing items in this image. For each distinct garment, output a JSON object within a list. Detect:
    1. category
    2. subcategory
    3. color
    4. pattern
    5. fit
    6. material
    Return ONLY the raw JSON list."""

    messages = [{
        "role": "user",
        "content": [{"type": "image", "image": img}, {"type": "text", "text": qwen_prompt}],
    }]

    text = qwen_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    
    inputs = qwen_processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        generated_ids = qwen_model.generate(**inputs, max_new_tokens=512, temperature=0.1)
    
    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    raw_qwen_text = qwen_processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0]
    
    torch.cuda.empty_cache()

    cleaned_text = re.sub(r"^```json\s*", "", raw_qwen_text.strip())
    cleaned_text = re.sub(r"^```\s*", "", cleaned_text)
    cleaned_text = re.sub(r"\s*```$", "", cleaned_text)
    try:
        attributes = json.loads(cleaned_text)
    except:
        attributes = {"raw": raw_qwen_text, "error": "JSON parse failed"}

    yolo_results = yolo_model.predict(img, conf=0.40, classes=TARGET_GARMENT_IDS, verbose=False)[0]
    
    generated_files = []
    
    for i, box in enumerate(yolo_results.boxes):
        coords = box.xyxy[0].tolist()
        cls_id = int(box.cls[0].item())
        garment_name = yolo_results.names[cls_id].split(',')[0].strip()

        crop_img = img.crop((coords[0], coords[1], coords[2], coords[3]))

        flux_prompt = (
            f"TRYOFF extract the {garment_name} on a pure white background, "
                "product photography, NO HUMAN VISIBLE, "
                "garments keep natural 3D shape as if on an invisible mannequin, "
                "clean edges, sharp fabric detail."
        )

        try:
            result_image = flux_pipe(
                image=crop_img,
                prompt=flux_prompt,
                height=1024,
                width=768,
                num_inference_steps=4,
                guidance_scale=3.5,
                generator=torch.Generator(device).manual_seed(42)
            ).images[0]
            
            filename = f"tryoff_{garment_name.replace(' ', '_')}_{uuid.uuid4().hex[:6]}.png"
            filepath = os.path.join(OUTPUT_DIR, filename)
            result_image.save(filepath)
            
            generated_files.append({
                "garment": garment_name,
                "saved_path": filepath
            })
        except Exception as e:
            print(f"FLUX Error: {e}")

    torch.cuda.empty_cache()

    return {
        "status": "success",
        "attributes": attributes,
        "generated_images": generated_files
    }

if __name__ == "__main__":
    uvicorn.run("tryoff:app", host="0.0.0.0", port=8080)
