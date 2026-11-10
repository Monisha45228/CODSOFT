"""
recognizer.py
-------------
Face recognition module. Two backends are provided behind one interface:

1. "lbph"      - Local Binary Patterns Histograms (cv2.face.LBPHFaceRecognizer).
                 Classic ML approach, trains in milliseconds on a handful of
                 photos per person, no GPU or extra heavy deps needed
                 (ships in opencv-contrib-python). Good baseline accuracy.

2. "embedding" - Deep-embedding based recognition (128-d face embeddings via
                 the `face_recognition` library, which wraps a dlib
                 ResNet model conceptually similar to FaceNet/ArcFace:
                 a CNN maps each face to a point in embedding space such
                 that same-person faces are close together and different
                 people are far apart). Recognition = nearest neighbour
                 in embedding space. Higher accuracy, esp. across pose/
                 lighting variation, but requires the `dlib` +
                 `face_recognition` packages (heavier install).

Both backends share the same on-disk "known faces" database format so you
can register faces once and try either backend.
"""

import json
import pickle
from pathlib import Path

import cv2
import numpy as np

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
KNOWN_FACES_DIR = DATA_DIR / "known_faces"
LBPH_MODEL_PATH = DATA_DIR / "lbph_model.yml"
LBPH_LABELS_PATH = DATA_DIR / "lbph_labels.json"
EMBEDDINGS_PATH = DATA_DIR / "embeddings.pkl"

FACE_SIZE = (200, 200)  # normalized size for LBPH training/inference


class FaceRecognizer:
    def __init__(self, backend: str = "lbph"):
        self.backend = backend.lower()
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        KNOWN_FACES_DIR.mkdir(parents=True, exist_ok=True)

        if self.backend == "lbph":
            self.model = cv2.face.LBPHFaceRecognizer_create()
            self.label_to_name = {}
            self._load_lbph()
        elif self.backend == "embedding":
            try:
                import face_recognition  # noqa: F401
            except ImportError as e:
                raise ImportError(
                    "backend='embedding' requires the `face_recognition` "
                    "(and `dlib`) packages. Install with:\n"
                    "    pip install dlib face_recognition\n"
                    "or use backend='lbph' instead."
                ) from e
            self.known_embeddings = []  # list of (name, 128-d np.array)
            self._load_embeddings()
        else:
            raise ValueError("backend must be 'lbph' or 'embedding'")

    # ------------------------------------------------------------------ #
    # Registration
    # ------------------------------------------------------------------ #
    def register_face(self, name: str, face_img: np.ndarray):
        """
        Add one face crop (BGR image, already cropped to just the face) to
        the known-faces database for `name`, then persist + retrain.
        """
        person_dir = KNOWN_FACES_DIR / name
        person_dir.mkdir(parents=True, exist_ok=True)
        idx = len(list(person_dir.glob("*.png")))
        save_path = person_dir / f"{idx:04d}.png"
        cv2.imwrite(str(save_path), face_img)

        if self.backend == "lbph":
            self._train_lbph()
        else:
            self._add_embedding(name, face_img)
            self._save_embeddings()

    def build_database_from_known_faces_dir(self):
        """Rebuild the whole database from images already present under data/known_faces/<name>/*.png"""
        if self.backend == "lbph":
            self._train_lbph()
        else:
            self.known_embeddings = []
            import face_recognition
            for person_dir in sorted(KNOWN_FACES_DIR.iterdir()):
                if not person_dir.is_dir():
                    continue
                name = person_dir.name
                for img_path in person_dir.glob("*.png"):
                    img = cv2.imread(str(img_path))
                    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    encs = face_recognition.face_encodings(rgb)
                    if encs:
                        self.known_embeddings.append((name, encs[0]))
            self._save_embeddings()

    # ------------------------------------------------------------------ #
    # Recognition
    # ------------------------------------------------------------------ #
    def recognize(self, face_img: np.ndarray, threshold: float = None):
        """
        Identify a single cropped face.

        Returns:
            (name: str, confidence_or_distance: float)
            name == "Unknown" if no match passes the threshold.
        """
        if self.backend == "lbph":
            return self._recognize_lbph(face_img, threshold if threshold is not None else 80.0)
        return self._recognize_embedding(face_img, threshold if threshold is not None else 0.6)

    # ------------------------------------------------------------------ #
    # LBPH internals
    # ------------------------------------------------------------------ #
    def _preprocess_for_lbph(self, face_img: np.ndarray):
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY) if face_img.ndim == 3 else face_img
        gray = cv2.equalizeHist(gray)
        return cv2.resize(gray, FACE_SIZE)

    def _train_lbph(self):
        images, labels = [], []
        label_to_name = {}
        next_label = 0

        for person_dir in sorted(KNOWN_FACES_DIR.iterdir()):
            if not person_dir.is_dir():
                continue
            name = person_dir.name
            img_paths = list(person_dir.glob("*.png"))
            if not img_paths:
                continue
            label_to_name[next_label] = name
            for img_path in img_paths:
                img = cv2.imread(str(img_path))
                images.append(self._preprocess_for_lbph(img))
                labels.append(next_label)
            next_label += 1

        if not images:
            return  # nothing to train on yet

        self.model.train(images, np.array(labels))
        self.label_to_name = label_to_name
        self.model.write(str(LBPH_MODEL_PATH))
        with open(LBPH_LABELS_PATH, "w") as f:
            json.dump(label_to_name, f)

    def _load_lbph(self):
        if LBPH_MODEL_PATH.exists() and LBPH_LABELS_PATH.exists():
            self.model.read(str(LBPH_MODEL_PATH))
            with open(LBPH_LABELS_PATH) as f:
                self.label_to_name = {int(k): v for k, v in json.load(f).items()}

    def _recognize_lbph(self, face_img: np.ndarray, threshold: float):
        if not self.label_to_name:
            return "Unknown", None
        processed = self._preprocess_for_lbph(face_img)
        label, distance = self.model.predict(processed)
        # LBPH: LOWER distance = better match
        if distance <= threshold and label in self.label_to_name:
            return self.label_to_name[label], distance
        return "Unknown", distance

    # ------------------------------------------------------------------ #
    # Embedding (deep-learning) internals
    # ------------------------------------------------------------------ #
    def _add_embedding(self, name: str, face_img: np.ndarray):
        import face_recognition
        rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        encs = face_recognition.face_encodings(rgb)
        if encs:
            self.known_embeddings.append((name, encs[0]))

    def _recognize_embedding(self, face_img: np.ndarray, threshold: float):
        import face_recognition
        if not self.known_embeddings:
            return "Unknown", None
        rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        encs = face_recognition.face_encodings(rgb)
        if not encs:
            return "Unknown", None
        query = encs[0]
        names = [n for n, _ in self.known_embeddings]
        vectors = np.array([v for _, v in self.known_embeddings])
        dists = np.linalg.norm(vectors - query, axis=1)
        best_idx = int(np.argmin(dists))
        best_dist = float(dists[best_idx])
        # embedding distance: LOWER = better match (typical face_recognition threshold ~0.6)
        if best_dist <= threshold:
            return names[best_idx], best_dist
        return "Unknown", best_dist

    def _save_embeddings(self):
        with open(EMBEDDINGS_PATH, "wb") as f:
            pickle.dump(self.known_embeddings, f)

    def _load_embeddings(self):
        if EMBEDDINGS_PATH.exists():
            with open(EMBEDDINGS_PATH, "rb") as f:
                self.known_embeddings = pickle.load(f)
        else:
            self.known_embeddings = []
