"""Image-quality metrics for diagnosing *why* an observation is unreliable.

Each metric is computed per image. Face-dependent metrics (face size, head
pose, occlusion, detector confidence) use MediaPipe when available and fall
back to an OpenCV Haar cascade. Pose/occlusion are approximate and flagged as
limited in the quality report (Milestone 2 roadmap, sections 14-15).

Output schema (one row per image):
    image_id, brightness, contrast, blur_score, face_area_ratio,
    yaw, pitch, roll, occlusion_score, motion_score, detector_confidence
"""

import math

import cv2
import numpy as np

QUALITY_COLUMNS = [
    "brightness",
    "contrast",
    "blur_score",
    "face_area_ratio",
    "yaw",
    "pitch",
    "roll",
    "occlusion_score",
    "motion_score",
    "detector_confidence",
]

# Canonical 3D face model points (nose, chin, eyes, mouth) for solvePnP.
_MODEL_POINTS = np.array(
    [
        (0.0, 0.0, 0.0),       # nose tip
        (0.0, -63.6, -12.5),   # chin
        (-43.3, 32.7, -26.0),  # left eye outer corner
        (43.3, 32.7, -26.0),   # right eye outer corner
        (-28.9, -28.9, -24.1), # left mouth corner
        (28.9, -28.9, -24.1),  # right mouth corner
    ],
    dtype=np.float64,
)
# Corresponding MediaPipe FaceMesh landmark indices.
_MESH_IDS = [1, 152, 263, 33, 291, 61]


class QualityAnalyzer:
    """Lazily initializes MediaPipe / Haar detectors and analyzes images."""

    def __init__(self, use_mediapipe: bool = True):
        self.use_mediapipe = use_mediapipe
        self._face_detection = None
        self._face_mesh = None
        self._haar = None
        self.backend = "none"
        if use_mediapipe:
            try:
                import mediapipe as mp

                self._mp = mp
                self._face_detection = mp.solutions.face_detection.FaceDetection(
                    model_selection=1, min_detection_confidence=0.3
                )
                self._face_mesh = mp.solutions.face_mesh.FaceMesh(
                    static_image_mode=True, max_num_faces=1,
                    refine_landmarks=False, min_detection_confidence=0.3,
                )
                self.backend = "mediapipe"
            except Exception:  # noqa: BLE001 - fall back to Haar
                self.use_mediapipe = False
        if not self.use_mediapipe:
            cascade = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self._haar = cv2.CascadeClassifier(cascade)
            self.backend = "haar"

    # --- low-level, face-independent metrics ---------------------------------
    @staticmethod
    def brightness(gray):
        return float(gray.mean())

    @staticmethod
    def contrast(gray):
        return float(gray.std())

    @staticmethod
    def blur_score(gray):
        """Variance of the Laplacian (higher = sharper)."""
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    # --- face detection ------------------------------------------------------
    def _detect_face_box(self, image_rgb):
        """Return (x, y, w, h, detector_confidence) or None."""
        h, w = image_rgb.shape[:2]
        if self.backend == "mediapipe":
            res = self._face_detection.process(image_rgb)
            if res.detections:
                det = max(res.detections, key=lambda d: d.score[0])
                box = det.location_data.relative_bounding_box
                x, y = int(box.xmin * w), int(box.ymin * h)
                bw, bh = int(box.width * w), int(box.height * h)
                return x, y, max(bw, 1), max(bh, 1), float(det.score[0])
            return None
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        faces = self._haar.detectMultiScale(gray, 1.1, 5, minSize=(20, 20))
        if len(faces) == 0:
            return None
        x, y, bw, bh = max(faces, key=lambda f: f[2] * f[3])
        return int(x), int(y), int(bw), int(bh), float("nan")  # Haar: no score

    # --- head pose (MediaPipe FaceMesh + solvePnP) ---------------------------
    def _head_pose(self, image_rgb):
        if self.backend != "mediapipe":
            return float("nan"), float("nan"), float("nan")
        h, w = image_rgb.shape[:2]
        res = self._face_mesh.process(image_rgb)
        if not res.multi_face_landmarks:
            return float("nan"), float("nan"), float("nan")
        lms = res.multi_face_landmarks[0].landmark
        try:
            image_points = np.array(
                [(lms[i].x * w, lms[i].y * h) for i in _MESH_IDS], dtype=np.float64
            )
        except IndexError:
            return float("nan"), float("nan"), float("nan")
        focal = w
        cam = np.array([[focal, 0, w / 2], [0, focal, h / 2], [0, 0, 1]], dtype=np.float64)
        ok, rvec, _ = cv2.solvePnP(
            _MODEL_POINTS, image_points, cam, np.zeros((4, 1)),
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok:
            return float("nan"), float("nan"), float("nan")
        rmat, _ = cv2.Rodrigues(rvec)
        sy = math.sqrt(rmat[0, 0] ** 2 + rmat[1, 0] ** 2)
        if sy > 1e-6:
            pitch = math.degrees(math.atan2(-rmat[2, 0], sy))
            yaw = math.degrees(math.atan2(rmat[1, 0], rmat[0, 0]))
            roll = math.degrees(math.atan2(rmat[2, 1], rmat[2, 2]))
        else:
            pitch = math.degrees(math.atan2(-rmat[2, 0], sy))
            yaw = 0.0
            roll = math.degrees(math.atan2(-rmat[1, 2], rmat[1, 1]))
        return round(yaw, 2), round(pitch, 2), round(roll, 2)

    @staticmethod
    def _occlusion_proxy(image_rgb, box):
        """Crude left/right symmetry-based occlusion proxy in [0, 1].

        Higher = more asymmetric (a possible sign of partial occlusion). This is
        a coarse heuristic and is documented as a limitation in the report.
        """
        if box is None:
            return float("nan")
        x, y, w, h, _ = box
        H, W = image_rgb.shape[:2]
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(W, x + w), min(H, y + h)
        crop = image_rgb[y0:y1, x0:x1]
        if crop.size == 0 or crop.shape[1] < 4:
            return float("nan")
        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY).astype(np.float64)
        mid = gray.shape[1] // 2
        left = gray[:, :mid]
        right = np.fliplr(gray[:, gray.shape[1] - mid:])
        m = min(left.shape[1], right.shape[1])
        if m == 0:
            return float("nan")
        diff = np.abs(left[:, :m] - right[:, :m]).mean() / 255.0
        return round(float(np.clip(diff, 0.0, 1.0)), 4)

    # --- public API ----------------------------------------------------------
    def analyze(self, image_rgb):
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        H, W = image_rgb.shape[:2]
        box = self._detect_face_box(image_rgb)
        if box is not None:
            x, y, bw, bh, det_conf = box
            face_area_ratio = round((bw * bh) / float(W * H), 4)
        else:
            face_area_ratio, det_conf = 0.0, float("nan")
        yaw, pitch, roll = self._head_pose(image_rgb)
        return {
            "brightness": round(self.brightness(gray), 2),
            "contrast": round(self.contrast(gray), 2),
            "blur_score": round(self.blur_score(gray), 2),
            "face_area_ratio": face_area_ratio,
            "yaw": yaw,
            "pitch": pitch,
            "roll": roll,
            "occlusion_score": self._occlusion_proxy(image_rgb, box),
            "motion_score": float("nan"),  # undefined for a single still image
            "detector_confidence": (round(det_conf, 4)
                                    if not math.isnan(det_conf) else float("nan")),
        }

    def analyze_path(self, path):
        bgr = cv2.imread(str(path))
        if bgr is None:
            return {c: float("nan") for c in QUALITY_COLUMNS}
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return self.analyze(rgb)

    # --- helpers reused by the evidence-signal extractor ---------------------
    def face_box(self, image_rgb):
        """Public access to the detected face box (x, y, w, h, conf) or None."""
        return self._detect_face_box(image_rgb)

    def landmark_reliability(self, image_rgb):
        """Fraction of FaceMesh landmarks detected inside the frame ([0, 1]).

        Returns NaN when MediaPipe is unavailable (Haar fallback has no mesh).
        """
        if self.backend != "mediapipe":
            return float("nan")
        res = self._face_mesh.process(image_rgb)
        if not res.multi_face_landmarks:
            return 0.0
        lms = res.multi_face_landmarks[0].landmark
        in_bounds = sum(1 for lm in lms if 0.0 <= lm.x <= 1.0 and 0.0 <= lm.y <= 1.0)
        return round(in_bounds / max(len(lms), 1), 4)

    def close(self):
        for obj in (self._face_detection, self._face_mesh):
            try:
                obj.close()
            except Exception:  # noqa: BLE001
                pass
