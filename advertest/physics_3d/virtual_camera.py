"""3D Egocentric Virtual Camera (Phase 2 Upgrade).

Uses mathematical projection and camera intrinsics/extrinsics to map a 3D
reconstructed scene (e.g., from NeRF or 3D Gaussian Splatting) onto an
egocentric virtual plane simulating a chest-mounted GoPro Hero 10.

Requirements:
    pip install numpy scipy opencv-python
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

try:
    from scipy.spatial.transform import Rotation
except ImportError:
    raise ImportError("scipy is required: pip install scipy")

logger = logging.getLogger(__name__)


class EgoCentricVirtualCamera:
    """Virtual Camera representing a body-worn action camera."""

    def __init__(self, image_width: int = 1920, image_height: int = 1080) -> None:
        self.width = image_width
        self.height = image_height

        # ────────────────────────────────────────────────
        # INTRINSICS: GoPro Hero 10 (SuperView Approximation)
        # ────────────────────────────────────────────────
        # Extremely wide FOV (approx 122 degrees HFOV).
        # We model this with a very short focal length and extreme barrel distortion.
        fx = 800.0  # Approx focal length in pixels for 1920 width
        fy = 800.0
        cx = self.width / 2.0
        cy = self.height / 2.0

        # Camera Intrinsic Matrix (K)
        self.K = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0,  0,  1]
        ], dtype=np.float32)

        # Distortion Coefficients (k1, k2, p1, p2, k3)
        # Deep negative k1 simulates intense barrel distortion (fisheye warp)
        self.dist_coeffs = np.array([-0.35, 0.12, 0.0, 0.0, -0.01], dtype=np.float32)

        # ────────────────────────────────────────────────
        # EXTRINSICS: "Chest-Mount" Perspective
        # ────────────────────────────────────────────────
        # Origin (0,0,0) is assumed to be eye-level of the wearer, looking straight ahead (+Z).
        # Y is down, X is right.
        
        # Translation: Move down 35cm (0.35m) along Y-axis, slightly forward (0.1m) on Z.
        self.tvec = np.array([0.0, 0.35, 0.10], dtype=np.float32)
        
        # Rotation: Chest naturally points slightly downwards or straight, but action 
        # cams are often tilted up (~15 degrees) to capture faces rather than the ground.
        # Pitch up: negative rotation around X-axis.
        rot = Rotation.from_euler('x', -15, degrees=True)
        self.R = rot.as_matrix().astype(np.float32)
        
        logger.info("Initialized EgoCentricVirtualCamera (GoPro Hero 10 SuperView Profile).")

    def get_projection_matrix(self) -> np.ndarray:
        """Get the full 3x4 projection matrix P = K [R | t]"""
        Rt = np.hstack((self.R, self.tvec.reshape(3, 1)))
        P = self.K @ Rt
        return P

    def project_3d_to_2d(self, points_3d: np.ndarray) -> np.ndarray:
        """Project an N x 3 array of 3D points onto the 2D egocentric image plane.
        
        Includes the non-linear lens distortion modeling (Fisheye).
        
        Parameters
        ----------
        points_3d : np.ndarray
            Shape (N, 3). The 3D coordinates in the world/eye-level frame.
            
        Returns
        -------
        np.ndarray
            Shape (N, 2). The projected 2D pixel coordinates (u, v).
        """
        if points_3d.ndim == 1:
            points_3d = points_3d.reshape(1, 3)

        # Convert to camera coordinate system (R*P + t)
        # Using cv2.projectPoints handles the perspective divide and lens distortion.
        
        # Note: cv2.projectPoints expects rvec instead of R matrix
        rvec, _ = cv2.Rodrigues(self.R)
        
        points_2d, _ = cv2.projectPoints(
            points_3d.astype(np.float32),
            rvec,
            self.tvec,
            self.K,
            self.dist_coeffs
        )
        
        return points_2d.reshape(-1, 2)

    def render_from_3dgs(self, point_cloud_path: str, trajectory_poses: List[Dict[str, Any]]) -> None:
        """Mock integration: Render bounding boxes from a 3D Gaussian Splatting scene.
        
        This demonstrates how we extract spatial ground truth (3D bounding boxes 
        of faces) from a reconstructed WIDER FACE scene and project them into 
        our newly generated egocentric viewpoint.
        
        Parameters
        ----------
        point_cloud_path : str
            Path to the .ply file containing the 3DGS scene.
        trajectory_poses : List[Dict[str, Any]]
            A list of time-series poses (simulating the walking movement of the chest).
        """
        logger.info("Loading 3D Gaussian Splatting scene from: %s", point_cloud_path)
        
        # 1. Mock: Load 3D bounding boxes from the scene (e.g., 3D faces)
        # Shape: (N, 8, 3) - 8 corners for each 3D box
        logger.info("Extracting 3D face bounding boxes from the scene...")
        mock_3d_box = np.array([
            [-0.1, -0.1, 2.0], [ 0.1, -0.1, 2.0], [ 0.1,  0.1, 2.0], [-0.1,  0.1, 2.0], # Front face
            [-0.1, -0.1, 2.2], [ 0.1, -0.1, 2.2], [ 0.1,  0.1, 2.2], [-0.1,  0.1, 2.2]  # Back face
        ])
        
        for frame_idx, pose in enumerate(trajectory_poses):
            # In a real pipeline, we update self.R and self.tvec based on the walking trajectory
            # Here we apply the base Chest-Mount transform.
            
            logger.info("Rendering Frame %d...", frame_idx)
            
            # 2. Project the 8 corners of the 3D bounding box to 2D
            corners_2d = self.project_3d_to_2d(mock_3d_box)
            
            # 3. Compute the 2D axis-aligned bounding box from the projected corners
            u_min, v_min = np.min(corners_2d, axis=0)
            u_max, v_max = np.max(corners_2d, axis=0)
            
            # Clip to image boundaries
            u_min = max(0, u_min)
            v_min = max(0, v_min)
            u_max = min(self.width, u_max)
            v_max = min(self.height, v_max)
            
            # Convert to YOLO format
            w = (u_max - u_min) / self.width
            h = (v_max - v_min) / self.height
            xc = (u_min / self.width) + (w / 2)
            yc = (v_min / self.height) + (h / 2)
            
            logger.info(
                "  Projected 3D Face to 2D YOLO BBox (Class 0): [%.4f, %.4f, %.4f, %.4f]",
                xc, yc, w, h
            )
            
            # Note: The actual pixels of the image would be rendered by the 3DGS 
            # rasterizer (e.g. diff-gaussian-rasterization) using self.K, self.R, self.tvec.
            logger.info("  Rasterizing Gaussian Splats for Egocentric view... (Mocked)")


if __name__ == "__main__":
    # Test the Virtual Camera projection
    logging.basicConfig(level=logging.INFO)
    cam = EgoCentricVirtualCamera()
    
    # Simulate a walking trajectory
    mock_trajectory = [{"time": 0.0}, {"time": 0.1}, {"time": 0.2}]
    
    cam.render_from_3dgs("data/scenes/wider_face_reconstruction.ply", mock_trajectory)
