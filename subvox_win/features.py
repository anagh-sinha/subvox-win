"""Turn raw landmark frames into a motion feature sequence."""
import numpy as np

# MediaPipe FaceMesh indices: lips (outer+inner), chin, nose tip, eye outer corners.
LIPS = [61, 291, 0, 17, 37, 267, 84, 314, 78, 308, 13, 14, 82, 312, 87, 317, 40, 270, 91, 321]
CHIN, NOSE, EYE_L, EYE_R = 152, 1, 33, 263
KEEP = LIPS + [CHIN]


def frame_vector(lm: np.ndarray) -> np.ndarray:
    """lm: (468+,3) normalised landmarks -> pose-invariant vector (2*len(KEEP),).

    Origin at nose tip, rotated so the eye line is horizontal, scaled by eye distance,
    so head translation, roll and distance to camera do not leak into the signal.
    """
    pts = lm[:, :2].astype(float)
    eye = pts[EYE_R] - pts[EYE_L]
    scale = np.linalg.norm(eye) or 1.0
    ang = np.arctan2(eye[1], eye[0])
    c, s = np.cos(-ang), np.sin(-ang)
    rot = np.array([[c, -s], [s, c]])
    rel = (pts[KEEP] - pts[NOSE]) @ rot.T / scale
    return rel.ravel()


def segment_features(frames: np.ndarray, trim: float = 0.1) -> np.ndarray:
    """frames: (T,D) -> mean-centred (T',D) with quiet lead-in/out trimmed."""
    if len(frames) < 4:
        return np.zeros((0, frames.shape[1] if frames.ndim == 2 else 0))
    x = frames - frames.mean(0)
    energy = np.linalg.norm(np.diff(frames, axis=0), axis=1)
    if energy.max() > 0:
        active = np.where(energy > trim * energy.max())[0]
        if len(active) >= 3:
            x = x[active[0] : active[-1] + 2]
    return x
