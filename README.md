# Fashion Try-Off

This repository contains a robust local FastAPI server (`tryoff.py`) that utilizes `Qwen2-VL-2B-Instruct` for garment attribute detection, `YOLO` for bounding box extraction, and `FLUX.2-klein-4B` for virtual try-off image generation.

## Features
- **POST /process**: Takes an image URL, downloads the image, and then:
  1. Detects garment attributes (category, color, fit, etc.) using Qwen2-VL.
  2. Uses YOLO to detect bounding boxes of specific garments in the image.
  3. Crops the image based on the YOLO detection and uses FLUX to generate a clean product photography try-off image for each detected garment.

## Setup & Installation

1. **Install dependencies:**
   Make sure you have python installed. Then, install the required packages. Note that this setup requires a system with a CUDA-enabled GPU and sufficient VRAM.
   ```bash
   pip install -r requirements.txt
   ```
   *Make sure you also place your trained YOLO model weights at `models/bbox2.pt` as specified in the script!*

2. **Run the API Server:**
   ```bash
   python tryoff.py
   ```
   The server will start at `http://0.0.0.0:8080`. Note: Initial model loading into VRAM will take a moment.

## Running Inference

You can test the API by running the provided `inference.py` client script. This script automatically pings the `/process` endpoint.
```bash
python inference.py
```

## Results

When you run `inference.py` with a test image URL, the script will output the detected attributes from Qwen2-VL, process the try-off through FLUX, and save the resulting images locally to the `outputs/` directory.

### Example Output

```text
Starting Dripi Fashion API Test
Endpoint: http://127.0.0.1:8080/process
Image:   https://images.pexels.com/photos/15869955/pexels-photo-15869955.jpeg
Waiting for AI generation (this usually takes 10-20 seconds)...

SUCCESS! (Took 15.4 seconds)

--- DETECTED ATTRIBUTES (Qwen2-VL) ---
[
  {
    "category": "Topwear",
    "subcategory": "Jacket",
    "color": "Black",
    "pattern": "Solid",
    "fit": "Regular",
    "material": "Leather"
  }
]

--- GENERATED TRY-OFFS (YOLO + FLUX) ---
Garment: JACKET
 Saved To: outputs\tryoff_jacket_a1b2c3.png
```

Check the `outputs/` folder in your project directory to view the generated try-off images!
