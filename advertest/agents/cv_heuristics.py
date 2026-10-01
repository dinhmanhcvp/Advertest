import cv2
import numpy as np

class FastHeuristicFilter:
    """
    Lightweight Computer Vision heuristics to run on the hot path (Streaming).
    Replaces synchronous VLM calls. Categorizes obvious physical errors 
    in milliseconds before delegating complex ambiguous cases to VLM asynchronously.
    """

    @staticmethod
    def compute_blur_score(image_bgr: np.ndarray) -> float:
        """Laplacian variance to detect motion/defocus blur."""
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var()

    @staticmethod
    def compute_exposure_scores(image_bgr: np.ndarray) -> tuple[float, float]:
        """Detect overexposure (blown highlights) or underexposure."""
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        pixels = gray.size
        
        # Percentage of pixels near 255 (overexposed)
        over_pct = (np.sum(hist[245:]) / pixels) * 100
        # Percentage of pixels near 0 (underexposed)
        under_pct = (np.sum(hist[:10]) / pixels) * 100
        
        return float(over_pct), float(under_pct)

    @classmethod
    def triage(cls, image_bgr: np.ndarray) -> str:
        """
        Fast triage gate.
        Returns a specific error class, or 'ambiguous' if it requires VLM analysis.
        """
        blur_score = cls.compute_blur_score(image_bgr)
        over_pct, under_pct = cls.compute_exposure_scores(image_bgr)

        # Thresholds derived empirically from Egocentric datasets
        if blur_score < 50.0:  
            return "error_blur"
        elif over_pct > 15.0:
            return "error_overexposure"
        elif under_pct > 25.0:
            return "error_lowlight"
        else:
            # Structurally fine but YOLO failed -> Probably Occlusion or complex scenario -> Send to VLM
            return "ambiguous_needs_vlm"
