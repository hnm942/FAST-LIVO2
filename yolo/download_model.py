import os
import requests
from pathlib import Path

def download_file(url, save_path):
    """
    Download a file from URL to the specified path
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Download the file
    print(f"Downloading {url} to {save_path}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    # Get total file size
    total_size = int(response.headers.get('content-length', 0))
    block_size = 8192
    downloaded = 0
    
    # Write to file
    with open(save_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=block_size):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                # Print progress
                done = int(50 * downloaded / total_size) if total_size > 0 else 0
                print(f"\r[{'=' * done}{' ' * (50 - done)}] {downloaded}/{total_size} bytes", end='')
    
    print("\nDownload complete!")
    return save_path

if __name__ == "__main__":
    # YOLOv8 model URL - using YOLOv8n (nano) as it's smaller but still effective
    model_url = "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt"
    
    # Save path
    save_path = Path("weights/yolov8n.pt")
    
    # Download the model
    download_file(model_url, save_path)
    print(f"Model saved to {save_path.absolute()}")
