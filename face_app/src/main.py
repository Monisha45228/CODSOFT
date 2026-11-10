"""
main.py
-------
Command-line entry point for the face detection & recognition app.

Examples
--------
# Detect faces only, in an image, drawing boxes, save result:
python src/main.py detect --input photo.jpg --output outputs/result.jpg --method dnn

# Detect faces in a video file, save annotated video:
python src/main.py detect --input clip.mp4 --output outputs/result.mp4 --method dnn

# Detect+track faces live from the webcam (press 'q' to quit):
python src/main.py detect --input webcam --method dnn

# Register a new known person from a photo (crops + stores their face):
python src/main.py register --name "Alice" --input alice1.jpg
python src/main.py register --name "Alice" --input alice2.jpg

# Recognize faces in a new image using the registered database:
python src/main.py recognize --input group_photo.jpg --output outputs/recognized.jpg --backend lbph

# Recognize faces live from the webcam:
python src/main.py recognize --input webcam --backend lbph
"""

import argparse
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))
from detector import FaceDetector, draw_boxes
from recognizer import FaceRecognizer


def is_image_file(path: str) -> bool:
    return Path(path).suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def open_source(input_arg: str):
    """Returns (cv2.VideoCapture-or-None, is_webcam, is_image)"""
    if input_arg == "webcam":
        return cv2.VideoCapture(0), True, False
    if is_image_file(input_arg):
        return None, False, True
    return cv2.VideoCapture(input_arg), False, False


def cmd_detect(args):
    detector = FaceDetector(method=args.method, confidence_threshold=args.confidence)

    if is_image_file(args.input):
        frame = cv2.imread(args.input)
        if frame is None:
            raise FileNotFoundError(f"Could not read image: {args.input}")
        boxes = detector.detect(frame)
        print(f"Detected {len(boxes)} face(s).")
        annotated = draw_boxes(frame, boxes)
        out_path = args.output or "outputs/detected.jpg"
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(out_path, annotated)
        print(f"Saved annotated image to {out_path}")
        return

    cap, is_webcam, _ = open_source(args.input)
    if not cap or not cap.isOpened():
        raise IOError(f"Could not open video source: {args.input}")

    writer = None
    if args.output and not is_webcam:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(args.output, fourcc, fps, (w, h))

    frame_count = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        boxes = detector.detect(frame)
        annotated = draw_boxes(frame, boxes)
        frame_count += 1

        if writer:
            writer.write(annotated)
        if is_webcam or args.show:
            cv2.imshow("Face Detection", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    if writer:
        writer.release()
        print(f"Saved annotated video to {args.output}")
    cv2.destroyAllWindows()
    print(f"Processed {frame_count} frame(s).")


def cmd_register(args):
    detector = FaceDetector(method=args.method, confidence_threshold=args.confidence)
    recognizer = FaceRecognizer(backend=args.backend)

    frame = cv2.imread(args.input)
    if frame is None:
        raise FileNotFoundError(f"Could not read image: {args.input}")
    boxes = detector.detect(frame)
    if not boxes:
        print("No face found in image; nothing registered.")
        return

    # take the largest detected face
    x, y, w, h = max(boxes, key=lambda b: b[2] * b[3])
    face_crop = frame[y:y + h, x:x + w]
    recognizer.register_face(args.name, face_crop)
    print(f"Registered 1 face for '{args.name}' (backend={args.backend}).")


def cmd_recognize(args):
    detector = FaceDetector(method=args.method, confidence_threshold=args.confidence)
    recognizer = FaceRecognizer(backend=args.backend)

    if is_image_file(args.input):
        frame = cv2.imread(args.input)
        if frame is None:
            raise FileNotFoundError(f"Could not read image: {args.input}")
        boxes = detector.detect(frame)
        labels = []
        for (x, y, w, h) in boxes:
            crop = frame[y:y + h, x:x + w]
            name, score = recognizer.recognize(crop)
            labels.append(name if score is None else f"{name} ({score:.1f})")
            print(f"  face at ({x},{y},{w},{h}) -> {name}"
                  + (f", score={score:.2f}" if score is not None else ""))
        annotated = draw_boxes(frame, boxes, labels)
        out_path = args.output or "outputs/recognized.jpg"
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(out_path, annotated)
        print(f"Saved annotated image to {out_path}")
        return

    cap, is_webcam, _ = open_source(args.input)
    if not cap or not cap.isOpened():
        raise IOError(f"Could not open video source: {args.input}")

    writer = None
    if args.output and not is_webcam:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(args.output, fourcc, fps, (w, h))

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        boxes = detector.detect(frame)
        labels = []
        for (x, y, w, h) in boxes:
            crop = frame[y:y + h, x:x + w]
            name, score = recognizer.recognize(crop)
            labels.append(name)
        annotated = draw_boxes(frame, boxes, labels)

        if writer:
            writer.write(annotated)
        if is_webcam or args.show:
            cv2.imshow("Face Recognition", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    if writer:
        writer.release()
        print(f"Saved annotated video to {args.output}")
    cv2.destroyAllWindows()


def build_parser():
    p = argparse.ArgumentParser(description="Face detection & recognition CLI")
    sub = p.add_subparsers(dest="command", required=True)

    p_detect = sub.add_parser("detect", help="Detect faces in an image/video/webcam")
    p_detect.add_argument("--input", required=True, help="Path to image/video, or 'webcam'")
    p_detect.add_argument("--output", help="Path to save annotated output (image or .mp4)")
    p_detect.add_argument("--method", choices=["haar", "dnn"], default="dnn")
    p_detect.add_argument("--confidence", type=float, default=0.5)
    p_detect.add_argument("--show", action="store_true", help="Show live preview window for video files")
    p_detect.set_defaults(func=cmd_detect)

    p_register = sub.add_parser("register", help="Register a known face from a photo")
    p_register.add_argument("--name", required=True)
    p_register.add_argument("--input", required=True, help="Path to a photo containing that person's face")
    p_register.add_argument("--method", choices=["haar", "dnn"], default="dnn")
    p_register.add_argument("--confidence", type=float, default=0.5)
    p_register.add_argument("--backend", choices=["lbph", "embedding"], default="lbph")
    p_register.set_defaults(func=cmd_register)

    p_recognize = sub.add_parser("recognize", help="Detect + recognize faces against known database")
    p_recognize.add_argument("--input", required=True, help="Path to image/video, or 'webcam'")
    p_recognize.add_argument("--output", help="Path to save annotated output (image or .mp4)")
    p_recognize.add_argument("--method", choices=["haar", "dnn"], default="dnn")
    p_recognize.add_argument("--confidence", type=float, default=0.5)
    p_recognize.add_argument("--backend", choices=["lbph", "embedding"], default="lbph")
    p_recognize.add_argument("--show", action="store_true")
    p_recognize.set_defaults(func=cmd_recognize)

    return p


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)
