"""Webcam capture + MediaPipe FaceMesh landmark extraction."""
import cv2
import numpy as np
import mediapipe as mp
from .features import frame_vector


class FaceCamera:
    def __init__(self, index: int = 0):
        self.cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera {index}")
        self.mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)

    def read(self):
        """Returns (bgr_image, feature_vector or None, landmarks or None)."""
        ok, img = self.cap.read()
        if not ok:
            return None, None, None
        res = self.mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if not res.multi_face_landmarks:
            return img, None, None
        lm = np.array([[p.x, p.y, p.z] for p in res.multi_face_landmarks[0].landmark])
        return img, frame_vector(lm), lm

    def close(self):
        self.cap.release()
        self.mesh.close()
