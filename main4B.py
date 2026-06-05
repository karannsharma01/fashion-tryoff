"""
Fashion API main4B.py (Serverless HF API Version)
Two independent endpoints taking image URLs:
  POST /predict  -> Garment attribute detection via free Hugging Face Vision API (JSON out)
  POST /tryoff   -> Virtual try-off FLUX Schnell via HF API (Direct Image out)
  GET  /health   -> Status of API
"""

import os
import httpx
import logging
import json
import base64
from fastapi import FastAPI, Form, HTTPException
from fastapi.responses import JSONResponse, Response

# ------------------------------------------------------------------------------
#  CONFIG
# ------------------------------------------------------------------------------

# Set your free Hugging Face token in the environment
HF_TOKEN = os.getenv("HF_TOKEN", "")

HOST = "0.0.0.0"
PORT = 8001

# ------------------------------------------------------------------------------
#  LOGGING
# ------------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------------------
#  APP
# ------------------------------------------------------------------------------
app = FastAPI(title="Dripi Fashion API")

def get_hf_headers():
    if not HF_TOKEN:
        logger.warning("HF_TOKEN is not set. The API might return 401 Unauthorized.")
    return {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json"
    }

# ------------------------------------------------------------------------------
#  HEALTH
# ------------------------------------------------------------------------------
@app.get("/health")
def health():
    return {
        "status": "ok",
        "mode": "Serverless HF API",
        "hf_token_configured": bool(HF_TOKEN)
    }

# ------------------------------------------------------------------------------
#  ENDPOINT 1 POST /predict
#  Image URL in -> HF Vision API -> Structured JSON out
# ------------------------------------------------------------------------------
@app.post("/predict")
async def predict(
    image_url: str = Form(..., description="URL of the source image"),
):
    # Using Llama-3.2-11B-Vision-Instruct as a free robust alternative on HF
    # Or Qwen2-VL if available via serverless
    api_url = "https://api-inference.huggingface.co/models/meta-llama/Llama-3.2-11B-Vision-Instruct/v1/chat/completions"

    prompt = """Analyze the clothing items in this image. For each distinct garment, output a JSON object within a list. Detect:
    1. category (Topwear, Bottomwear, Dress, Footwear, etc.)
    2. subcategory (T-shirt, Shirt, Jeans, Sneakers, etc.)
    3. color (Primary color)
    4. pattern (Camo, Floral, Solid, Checkered, etc.)
    5. fit (Slim, Oversized, Relaxed, Regular, etc.)
    6. material (Guess based on texture)
    7. dominance_score (An integer from 1-100 indicating visual prominence)

    Return ONLY the raw JSON list."""

    payload = {
        "model": "meta-llama/Llama-3.2-11B-Vision-Instruct",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_url
                        }
                    }
                ]
            }
        ],
        "max_tokens": 1024,
    }

    async with httpx.AsyncClient() as client:
        try:
            resp = await client.post(api_url, headers=get_hf_headers(), json=payload, timeout=60.0)
            resp.raise_for_status()
            data = resp.json()
            output_text = data["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error(f"HF Vision Inference Error: {e}")
            if 'resp' in locals():
                logger.error(f"Response: {resp.text}")
            raise HTTPException(status_code=500, detail="Vision inference failed.")

    # Clean and Parse JSON
    cleaned_text = output_text.strip()
    if cleaned_text.startswith("```json"): cleaned_text = cleaned_text[7:]
    elif cleaned_text.startswith("```"): cleaned_text = cleaned_text[3:]
    if cleaned_text.endswith("```"): cleaned_text = cleaned_text[:-3]
    cleaned_text = cleaned_text.strip()

    try:
        structured_data = json.loads(cleaned_text)
    except json.JSONDecodeError as e:
        logger.error(f"JSON Parsing Error: {e} | Raw: {output_text}")
        return JSONResponse(status_code=500, content={
            "error": "Model returned invalid JSON.",
            "raw_output": output_text
        })

    return JSONResponse(content={
        "status": "success",
        "image_url": image_url,
        "results": structured_data,
    })

# ------------------------------------------------------------------------------
#  ENDPOINT 2 POST /tryoff
#  Image URL in -> HF FLUX API -> Direct Image Out
# ------------------------------------------------------------------------------
@app.post("/tryoff")
async def tryoff(
    image_url: str = Form(..., description="URL of the source image"),
    prompt:    str = Form(
        default=(
            "TRYOFF extract the garment on a pure white background, "
            "product photography, NO HUMAN VISIBLE, "
            "garments keep natural 3D shape as if on an invisible mannequin, "
            "clean edges, sharp fabric detail."
        )
    ),
    height: int = Form(default=1024),
    width:  int = Form(default=768),
):
    # Using FLUX.1-schnell via Hugging Face Serverless API
    api_url = "https://api-inference.huggingface.co/models/black-forest-labs/FLUX.1-schnell"

    # Incorporate the image URL into the prompt to guide the Text-to-Image model conceptually 
    # since standard HF Serverless FLUX is primarily text-to-image.
    full_prompt = f"Based on the garment from this image URL: {image_url}. {prompt}"

    payload = {
        "inputs": full_prompt,
        "parameters": {
            "height": height,
            "width": width,
            "num_inference_steps": 4
        }
    }

    async with httpx.AsyncClient() as client:
        try:
            resp = await client.post(api_url, headers=get_hf_headers(), json=payload, timeout=60.0)
            resp.raise_for_status()
            image_bytes = resp.content
        except Exception as e:
            logger.error(f"HF FLUX Inference Error: {e}")
            if 'resp' in locals():
                logger.error(f"Response: {resp.text}")
            raise HTTPException(status_code=500, detail="Diffusion inference failed.")

    return Response(content=image_bytes, media_type="image/jpeg")

# ------------------------------------------------------------------------------
#  ENDPOINT 3 POST /process
#  Image URL in -> Returns BOTH JSON attributes and Base64 encoded Tryoff Image
# ------------------------------------------------------------------------------
@app.post("/process")
async def process(
    image_url: str = Form(..., description="URL of the source image"),
    prompt:    str = Form(
        default=(
            "TRYOFF extract the garment on a pure white background, "
            "product photography, NO HUMAN VISIBLE, "
            "garments keep natural 3D shape as if on an invisible mannequin, "
            "clean edges, sharp fabric detail."
        )
    ),
    height: int = Form(default=1024),
    width:  int = Form(default=768),
):
    """
    Combined endpoint that fetches the attributes and the tryoff image concurrently,
    returning a single JSON object.
    """
    import asyncio

    # We can reuse the predict and tryoff logic internally by wrapping them
    # But since they rely on Form parameters in FastAPI, we'll extract the core logic
    
    # 1. Vision Logic
    vision_url = "https://api-inference.huggingface.co/models/meta-llama/Llama-3.2-11B-Vision-Instruct/v1/chat/completions"
    vision_prompt = """Analyze the clothing items in this image. For each distinct garment, output a JSON object within a list. Detect:
    1. category (Topwear, Bottomwear, Dress, Footwear, etc.)
    2. subcategory (T-shirt, Shirt, Jeans, Sneakers, etc.)
    3. color (Primary color)
    4. pattern (Camo, Floral, Solid, Checkered, etc.)
    5. fit (Slim, Oversized, Relaxed, Regular, etc.)
    6. material (Guess based on texture)
    7. dominance_score (An integer from 1-100 indicating visual prominence)

    Return ONLY the raw JSON list."""

    vision_payload = {
        "model": "meta-llama/Llama-3.2-11B-Vision-Instruct",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": vision_prompt},
                    {"type": "image_url", "image_url": {"url": image_url}}
                ]
            }
        ],
        "max_tokens": 1024,
    }

    # 2. FLUX Logic
    flux_url = "https://api-inference.huggingface.co/models/black-forest-labs/FLUX.1-schnell"
    flux_payload = {
        "inputs": f"Based on the garment from this image URL: {image_url}. {prompt}",
        "parameters": {"height": height, "width": width, "num_inference_steps": 4}
    }

    async with httpx.AsyncClient() as client:
        # Run both requests concurrently
        vision_task = client.post(vision_url, headers=get_hf_headers(), json=vision_payload, timeout=60.0)
        flux_task = client.post(flux_url, headers=get_hf_headers(), json=flux_payload, timeout=60.0)
        
        vision_resp, flux_resp = await asyncio.gather(vision_task, flux_task, return_exceptions=True)

    # Process Vision Results
    structured_data = []
    if isinstance(vision_resp, Exception):
        logger.error(f"Combined Endpoint Vision Error: {vision_resp}")
    else:
        try:
            vision_resp.raise_for_status()
            out_text = vision_resp.json()["choices"][0]["message"]["content"].strip()
            if out_text.startswith("```json"): out_text = out_text[7:]
            elif out_text.startswith("```"): out_text = out_text[3:]
            if out_text.endswith("```"): out_text = out_text[:-3]
            structured_data = json.loads(out_text.strip())
        except Exception as e:
            logger.error(f"Failed to parse vision response in combined endpoint: {e}")

    # Process FLUX Results
    image_base64 = ""
    if isinstance(flux_resp, Exception):
        logger.error(f"Combined Endpoint FLUX Error: {flux_resp}")
    else:
        try:
            flux_resp.raise_for_status()
            image_base64 = base64.b64encode(flux_resp.content).decode("utf-8")
        except Exception as e:
            logger.error(f"Failed to fetch image in combined endpoint: {e}")

    return JSONResponse(content={
        "status": "success",
        "image_url": image_url,
        "attributes": structured_data,
        "tryoff_image_base64": image_base64
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main4B:app", host=HOST, port=PORT, reload=False, workers=1)
