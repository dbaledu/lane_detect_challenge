"""Video quality and environment analyzer."""
import cv2
import numpy as np
from typing import Optional, Dict, Any
from ..common.data_models import QualityMetric


class QualityAnalyzer:
    """Analyzes lighting, blur, and environmental conditions of road frames."""

    def __init__(self, cfg: Optional[Dict[str, Any]] = None):
        self.cfg = cfg or {}
        q_cfg = self.cfg.get("quality", {})
        self.brightness_night_th = float(q_cfg.get("brightness_night_threshold", 70.0))
        self.blur_laplacian_th = float(q_cfg.get("blur_laplacian_threshold", 45.0))

    def analyze(self, frame: np.ndarray, gray: Optional[np.ndarray] = None) -> QualityMetric:
        """Analyzes frame quality and returns a QualityMetric instance."""
        if gray is None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        brightness = float(np.mean(gray))
        is_night = brightness < self.brightness_night_th

        # Laplacian variance measures edge sharpness / blur
        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        is_blurred = lap_var < self.blur_laplacian_th

        return QualityMetric(
            brightness=brightness,
            is_night=is_night,
            blur_score=lap_var,
            is_blurred=is_blurred
        )
