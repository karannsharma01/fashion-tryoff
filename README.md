# Fashion Tryoff API (Serverless HF Version)

This is a lightweight FastAPI application that has been upgraded to use **Hugging Face's Serverless Inference API** instead of heavy local models and AWS S3 storage.

## Features
- **POST /predict**: Garment attribute detection via free Vision API (`Llama-3.2-Vision` / `Qwen-VL`). Takes an image URL and returns structured JSON attributes.
- **POST /tryoff**: Virtual try-off generation using `FLUX.1-schnell`. Takes an image URL and returns the generated image directly.
- **POST /process**: A combined endpoint that concurrently runs both prediction and tryoff, returning a single JSON response with the attributes and a Base64-encoded image.
- **GET /health**: Status of the API.

## Directory Structure

```text
fashion-tryoff/
├── main4B.py
├── requirements.txt
├── README.md
```

## Getting Started

1. **Install dependencies:**
   Make sure you have python installed. Then, install the required packages. Because we rely on external APIs, the package size is incredibly small!
   ```bash
   pip install -r requirements.txt
   ```

2. **Setup your environment:**
   You must provide a free Hugging Face API Token (`HF_TOKEN`) for the model calls to succeed.
   Sign up at [Hugging Face](https://huggingface.co/) and generate a token in your settings.

   ```bash
   # Windows (PowerShell)
   $env:HF_TOKEN="hf_your_token_here"

   # Linux / Mac
   export HF_TOKEN="hf_your_token_here"
   ```

3. **Run the API:**
   ```bash
   python main4B.py
   ```
   The application will boot up at `http://0.0.0.0:8001`.

## Upload to Github

To push this repository to your GitHub:
```bash
git add .
git commit -m "Migrate to HF API and remove S3"
git push -u origin main
```
