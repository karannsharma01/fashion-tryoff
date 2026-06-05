import requests
import json
import time
 
API_URL = "http://127.0.0.1:8080/process"
TEST_IMAGE_URL = "https://images.pexels.com/photos/15869955/pexels-photo-15869955.jpeg"

def run_inference():
    payload = {
        "image_url": TEST_IMAGE_URL
    }
    print("Starting Fashion API Test")
    print(f"Endpoint: {API_URL}")
    print(f"Image:   {TEST_IMAGE_URL}")
    print("Waiting for AI generation (this usually takes 10-20 seconds)...\n")

    start_time = time.time()

    try:
        response = requests.post(
            API_URL,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=300 
        )

        response.raise_for_status()
        
        result = response.json()
        end_time = time.time()

        print(f"SUCCESS! (Took {round(end_time - start_time, 2)} seconds)\n")
        print("--- DETECTED ATTRIBUTES (Qwen2-VL) ---")
        attributes = result.get("attributes", [])
        print(json.dumps(attributes, indent=2))
        print("\n")

        print("--- GENERATED TRY-OFFS (YOLO + FLUX) ---")
        generated_images = result.get("generated_images", [])
        
        if generated_images:
            for img in generated_images:
                print(f"Garment: {img.get('garment', 'Unknown').upper()}")
                print(f" Saved To: {img.get('saved_path', 'Unknown')}\n")
        else:
            print(" No images were generated.")
            print(f" YOLO Detections: {result.get('yolo_items_detected', 0)}")

        errors = result.get("flux_errors", [])
        if errors:
            print("--- PIPELINE ERRORS ---")
            for err in errors:
                print(f" - Failed on {err.get('garment')}: {err.get('error')}")

    except requests.exceptions.ConnectionError:
        print("ERROR: Could not connect to the API. Is your FastAPI server running?")
    except requests.exceptions.Timeout:
        print("ERROR: The request timed out. The models might be taking too long to load.")
    except requests.exceptions.RequestException as e:
        print(f"ERROR: HTTP Request failed: {e}")
        try:
            print(f"Server replied: {response.json()}")
        except:
            print(f"Server text: {response.text}")

if __name__ == "__main__":
    run_inference()
