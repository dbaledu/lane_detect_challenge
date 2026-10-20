"""Image preprocessing module optimized for night and urban dashcam conditions."""
import cv2
import numpy as np
from typing import Tuple, Optional, Dict, Any


class FramePreprocessor:
    """Preprocesses road images using CLAHE, Auto-Canny, and morphological filtering."""

    def __init__(self, cfg: Optional[Dict[str, Any]] = None):
        self.cfg = cfg or {}
        prep_cfg = self.cfg.get("preprocessing", {})

        self.use_clahe = prep_cfg.get("use_clahe", True)
        clip_limit = float(prep_cfg.get("clahe_clip_limit", 2.5))
        grid_size = tuple(prep_cfg.get("clahe_grid_size", [8, 8]))
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)

        self.blur_kernel = tuple(prep_cfg.get("gaussian_kernel", [5, 5]))
        self.auto_canny = prep_cfg.get("auto_canny", True)
        self.sigma = float(prep_cfg.get("auto_canny_sigma", 0.33))
        self.fixed_low = int(prep_cfg.get("canny_low", 60))
        self.fixed_high = int(prep_cfg.get("canny_high", 150))
        self.use_morph_close = prep_cfg.get("morph_close", True)
        self.morph_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    def compute_auto_canny_thresholds(self, img_single_channel: np.ndarray, mask: Optional[np.ndarray] = None) -> Tuple[int, int]:
        """Calculates dynamic Canny low and high thresholds based on median intensity."""
        if mask is not None and np.count_nonzero(mask) > 100:
            pixels = img_single_channel[mask > 0]
            v = np.median(pixels)
        else:
            v = np.median(img_single_channel)

        lower = int(max(25, (1.0 - self.sigma) * v))
        upper = int(min(220, (1.0 + self.sigma) * v))
        if lower >= upper:
            lower = max(20, upper - 40)
        return lower, upper

    def extract_color_features(self, frame: np.ndarray) -> np.ndarray:
        """Extracts white and yellow lane features in HLS and LAB color spaces."""
        hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)

        # L channel from HLS: prominent under headlights
        l_channel = hls[:, :, 1]
        _, white_mask = cv2.threshold(l_channel, 200, 255, cv2.THRESH_BINARY)

        # b channel from LAB: separates yellow center lines even at night
        b_channel = lab[:, :, 2]
        _, yellow_mask = cv2.threshold(b_channel, 145, 255, cv2.THRESH_BINARY)

        combined_color_mask = cv2.bitwise_or(white_mask, yellow_mask)
        return combined_color_mask

    def process(self, frame: np.ndarray, roi_mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Runs complete preprocessing pipeline.

        Returns:
            gray: grayscale image
            enhanced: CLAHE enhanced contrast image
            edges: binary edge image
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.use_clahe:
            enhanced = self.clahe.apply(gray)
        else:
            enhanced = gray

        blurred = cv2.GaussianBlur(enhanced, self.blur_kernel, 0)

        if self.auto_canny:
            low_th, high_th = self.compute_auto_canny_thresholds(blurred, roi_mask)
        else:
            low_th, high_th = self.fixed_low, self.fixed_high

        edges = cv2.Canny(blurred, low_th, high_th)

        if self.use_morph_close:
            edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, self.morph_kernel)

        return gray, enhanced, edges
