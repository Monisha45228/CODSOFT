"""
download_models.py
-------------------
Fetches the pre-trained DNN face-detector weights (ResNet-10 SSD, Caffe
format) from OpenCV's official GitHub repos and stores them under
../models/. Haar cascades need no download -- they ship inside opencv-python.

Run once before using method="dnn":
    python src/download_models.py
"""

import urllib.request
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

FILES = {
    "deploy.prototxt": (
        "https://raw.githubusercontent.com/opencv/opencv/master/"
        "samples/dnn/face_detector/deploy.prototxt"
    ),
    "res10_300x300_ssd_iter_140000.caffemodel": (
        "https://raw.githubusercontent.com/opencv/opencv_3rdparty/"
        "dnn_samples_face_detector_20170830/"
        "res10_300x300_ssd_iter_140000.caffemodel"
    ),
}


def download():
    for filename, url in FILES.items():
        dest = MODELS_DIR / filename
        if dest.exists() and dest.stat().st_size > 0:
            print(f"[skip] {filename} already present")
            continue
        print(f"[downloading] {filename} ...")
        urllib.request.urlretrieve(url, dest)
        print(f"[done] saved to {dest} ({dest.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    download()
