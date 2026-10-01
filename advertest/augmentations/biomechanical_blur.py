import cv2
import numpy as np
import math

class BiomechanicalBlur:
    """
    Simulates non-linear egocentric camera motion (e.g., walking head-bob or quick panning)
    by generating a mathematically modeled 2D blur kernel and convolving it with the image.
    """
    def __init__(self, severity=1):
        self.severity = max(1, min(5, severity))
        # Kernel size scales with severity (e.g., severity 1 -> 15, severity 5 -> 55)
        self.kernel_size = 5 + (self.severity * 10)
        self.kernel = self._generate_imu_kernel()

    def _generate_imu_kernel(self):
        """
        Generates a non-linear motion blur kernel using sine waves to simulate a 2Hz walking gait.
        """
        k = self.kernel_size
        kernel = np.zeros((k, k), dtype=np.float32)
        center = k // 2

        # Parameters for biomechanical trajectory (Head-bobbing or panning)
        amplitude = k * 0.4  # Max spread
        frequency = 2.0      # Simulating ~2Hz human walking gait
        num_points = 200     # Granularity of the curve

        points = []
        for i in range(num_points):
            t = i / float(num_points)
            # Parametric equation for human gait (figure-8 / sine wave pattern)
            # x(t) = linear pan + slight sway
            x = center + (t - 0.5) * amplitude
            # y(t) = bounce (sine wave squared simulates stepping impact)
            y = center + amplitude * 0.3 * math.sin(2 * math.pi * frequency * t)

            x, y = int(round(x)), int(round(y))
            
            # Ensure points stay within kernel bounds
            if 0 <= x < k and 0 <= y < k:
                points.append((x, y))

        # Draw the trajectory onto the kernel
        for pt in points:
            # Accumulate intensity to simulate exposure time at that pixel
            kernel[pt[1], pt[0]] += 1.0

        # Smooth the trajectory slightly so it's not aliased
        kernel = cv2.GaussianBlur(kernel, (3, 3), 0)

        # Normalize the kernel so the total sum is 1.0 (preserves image brightness)
        if np.sum(kernel) > 0:
            kernel /= np.sum(kernel)
        else:
            kernel[center, center] = 1.0

        return kernel

    def __call__(self, image, bounding_boxes=None):
        """
        Applies the biomechanical blur to the image.
        bounding_boxes are passed through unchanged.
        """
        if not isinstance(image, np.ndarray):
            raise ValueError("Input image must be a numpy array (cv2 format).")

        # Apply 2D convolution with the biomechanical kernel
        blurred_image = cv2.filter2D(image, -1, self.kernel)
        
        if bounding_boxes is None:
            return blurred_image
        return blurred_image, bounding_boxes
