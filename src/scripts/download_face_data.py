"""Helper script to download YOLOv7-face weights and setup dataset folders."""

import argparse
import os
import urllib.request
from pathlib import Path

WEIGHTS_URL = "https://github.com/derronqi/yolov7-face/releases/download/v0.1/yolov7-lite-t.pt"
# Using the lite-t weights as a reasonable default for fast evaluation.

def download_weights(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "yolov7-lite-t.pt"
    
    if out_path.exists():
        print(f"Weights already exist at {out_path}")
        return
        
    print(f"Downloading YOLOv7-face weights to {out_path}...")
    try:
        urllib.request.urlretrieve(WEIGHTS_URL, out_path)
        print("Download complete.")
    except Exception as e:
        print(f"Failed to download weights: {e}")

def download_yolov7_repo(third_party_dir: Path):
    third_party_dir.mkdir(parents=True, exist_ok=True)
    repo_dir = third_party_dir / "yolov7_face"
    
    if repo_dir.exists():
        print(f"YOLOv7-face repo already exists at {repo_dir}")
        return
        
    import zipfile
    import io
    
    url = "https://github.com/derronqi/yolov7-face/archive/refs/heads/main.zip"
    print(f"Downloading YOLOv7-face repo from {url}...")
    try:
        response = urllib.request.urlopen(url)
        with zipfile.ZipFile(io.BytesIO(response.read())) as z:
            z.extractall(third_party_dir)
        
        # Rename extracted folder
        extracted_dir = third_party_dir / "yolov7-face-main"
        if extracted_dir.exists():
            extracted_dir.rename(repo_dir)
        print("Repo downloaded and extracted.")
    except Exception as e:
        print(f"Failed to download repo: {e}")

def setup_dataset_folders(data_root: Path):
    wider_dir = data_root / "widerface"
    wider_dir.mkdir(parents=True, exist_ok=True)
    
    ego4d_dir = data_root / "ego4d"
    ego4d_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Created dataset directories at:")
    print(f"  - {wider_dir}")
    print(f"  - {ego4d_dir}")
    
    print("\nPlease manually download and extract datasets into these folders.")
    print("WIDERFACE: http://shuoyang1213.me/WIDERFACE/")
    print("Ego4D Social: https://ego4d-data.org/docs/benchmarks/social/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default="data", help="Root directory for data")
    args = parser.parse_args()
    
    root = Path(args.data_root)
    download_weights(root / "weights")
    download_yolov7_repo(Path("src/third_party"))
    setup_dataset_folders(root)
