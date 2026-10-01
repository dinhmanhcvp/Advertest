import numpy as np

class Track:
    def __init__(self, bbox, track_id):
        self.id = track_id
        self.bbox = np.array(bbox[:4], dtype=np.float32)  # [x1, y1, x2, y2]
        self.conf = bbox[4] if len(bbox) > 4 else 1.0
        self.velocity = np.zeros(4, dtype=np.float32)
        self.time_since_update = 0
        self.hits = 1

    def predict(self):
        """Predicts the next bounding box position based on velocity."""
        self.bbox += self.velocity
        self.time_since_update += 1
        return self.bbox

    def update(self, bbox):
        """Updates the track with a new matched observation."""
        new_bbox = np.array(bbox[:4], dtype=np.float32)
        # Simple velocity estimate: difference between new and old box
        # Using a slight EMA to smooth the velocity
        new_velocity = new_bbox - self.bbox
        self.velocity = 0.5 * self.velocity + 0.5 * new_velocity
        
        self.bbox = new_bbox
        self.conf = bbox[4] if len(bbox) > 4 else 1.0
        self.time_since_update = 0
        self.hits += 1


def calculate_iou(box1, box2):
    """Calculates Intersection over Union between two bounding boxes [x1, y1, x2, y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    if inter_area == 0:
        return 0.0

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    return inter_area / float(box1_area + box2_area - inter_area)


class TemporalTracker:
    """
    A lightweight IoU-based tracker to persist bounding boxes 
    across highly degraded frames (e.g. extreme egocentric motion blur).
    """
    def __init__(self, max_age=3, min_iou=0.3):
        self.max_age = max_age
        self.min_iou = min_iou
        self.tracks = []
        self.next_id = 1

    def update(self, detections):
        """
        Takes current frame detections: list of [x1, y1, x2, y2, conf]
        Returns temporarily smoothed and predicted bounding boxes.
        """
        # Step 1: Predict new locations for all existing tracks
        for track in self.tracks:
            track.predict()

        matched_tracks = set()
        matched_detections = set()

        # Step 2: Match detections to existing tracks via IoU greedy assignment
        for d_idx, det in enumerate(detections):
            best_iou = self.min_iou
            best_t_idx = -1
            
            for t_idx, track in enumerate(self.tracks):
                if t_idx in matched_tracks:
                    continue
                    
                iou = calculate_iou(det, track.bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_t_idx = t_idx
                    
            if best_t_idx != -1:
                self.tracks[best_t_idx].update(det)
                matched_tracks.add(best_t_idx)
                matched_detections.add(d_idx)

        # Step 3: Create new tracks for unmatched detections
        for d_idx, det in enumerate(detections):
            if d_idx not in matched_detections:
                new_track = Track(det, self.next_id)
                self.tracks.append(new_track)
                self.next_id += 1

        # Step 4: Purge dead tracks (dropped for more than max_age frames)
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]

        # Step 5: Format output
        # Return bounding boxes of all tracks that are currently valid
        # We include tracks that were predicted (time_since_update > 0) to maintain persistence
        smoothed_detections = []
        for t in self.tracks:
            # Optionally, you can filter out tracks that haven't had enough 'hits' 
            # to avoid false positives, but for red-teaming we want high recall.
            det = [t.bbox[0], t.bbox[1], t.bbox[2], t.bbox[3], t.conf]
            smoothed_detections.append(det)

        return smoothed_detections
