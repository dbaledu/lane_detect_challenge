"""Object tracking module with ID persistence and lead vehicle dynamics analysis."""
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from collections import deque
from ..common.data_models import ObjectBox, LaneResult
from ..common.geometry import calculate_iou


class TrackRecord:
    """Historical state of an individual tracked object."""

    def __init__(self, track_id: int, obj: ObjectBox):
        self.track_id = track_id
        self.class_id = obj.class_id
        self.label = obj.label
        self.confidence = obj.confidence
        self.bbox = obj.bbox
        self.center = obj.center
        self.missed_frames = 0
        self.history = deque(maxlen=20)
        self.history.append((obj.bbox, obj.center))
        self.velocity = (0.0, 0.0)

    def update(self, obj: ObjectBox):
        """Updates track with a new matched detection."""
        prev_cx, prev_cy = self.center
        self.confidence = obj.confidence
        self.bbox = obj.bbox
        self.center = obj.center
        self.missed_frames = 0

        vx = self.center[0] - prev_cx
        vy = self.center[1] - prev_cy
        self.velocity = (0.3 * vx + 0.7 * self.velocity[0], 0.3 * vy + 0.7 * self.velocity[1])
        self.history.append((obj.bbox, obj.center))

    def predict(self):
        """Extrapolates position during missed/skipped frames."""
        self.missed_frames += 1
        x1, y1, x2, y2 = self.bbox
        vx, vy = self.velocity
        new_x1 = int(round(x1 + vx))
        new_y1 = int(round(y1 + vy))
        new_x2 = int(round(x2 + vx))
        new_y2 = int(round(y2 + vy))
        self.bbox = (new_x1, new_y1, new_x2, new_y2)
        self.center = ((new_x1 + new_x2) / 2.0, (new_y1 + new_y2) / 2.0)
        self.history.append((self.bbox, self.center))


class ObjectTracker:
    """Maintains consistent IDs across frames and analyzes lead vehicle movement."""

    def __init__(self, cfg: Optional[Dict[str, Any]] = None):
        self.cfg = cfg or {}
        tr_cfg = self.cfg.get("tracker", {})

        self.max_distance = float(tr_cfg.get("max_distance_px", 85.0))
        self.max_missed_frames = int(tr_cfg.get("max_missed_frames", 10))
        self.lead_margin = float(tr_cfg.get("lead_vehicle_corridor_margin_px", 130.0))

        self.tracks: Dict[int, TrackRecord] = {}
        self.next_id: int = 1

    def update(
        self,
        detections: List[ObjectBox],
        detection_executed: bool,
        lane: Optional[LaneResult],
        frame_shape: Tuple[int, int, int],
        road_markings: Optional[Any] = None
    ) -> List[ObjectBox]:
        """Matches detections to existing tracks and computes dynamics."""
        h, w = frame_shape[:2]
        lane_center = lane.lane_center_x if (lane and lane.lane_center_x is not None) else (w / 2.0)

        # 1. If detection was executed, match detections to tracks
        if detection_executed:
            matched_track_ids = set()
            unmatched_dets = []

            for det in detections:
                best_id = None
                best_score = -1.0  # Combined IoU and distance metric

                for tid, track in self.tracks.items():
                    if tid in matched_track_ids:
                        continue
                    if track.label != det.label and det.label != "car" and track.label != "truck":
                        continue

                    iou = calculate_iou(track.bbox, det.bbox)
                    dist = np.hypot(track.center[0] - det.center[0], track.center[1] - det.center[1])

                    if iou > 0.25:
                        score = iou + 1.0
                    elif dist < self.max_distance:
                        score = 1.0 - (dist / self.max_distance)
                    else:
                        score = -1.0

                    if score > best_score and score > 0.15:
                        best_score = score
                        best_id = tid

                if best_id is not None:
                    self.tracks[best_id].update(det)
                    matched_track_ids.add(best_id)
                else:
                    unmatched_dets.append(det)

            # Age unmatched tracks
            for tid in list(self.tracks.keys()):
                if tid not in matched_track_ids:
                    self.tracks[tid].predict()
                    if self.tracks[tid].missed_frames > self.max_missed_frames:
                        del self.tracks[tid]

            # Register new tracks
            for det in unmatched_dets:
                new_track = TrackRecord(self.next_id, det)
                self.tracks[self.next_id] = new_track
                self.next_id += 1

        else:
            # Predict for all tracks during skipped frames
            for tid in list(self.tracks.keys()):
                self.tracks[tid].predict()
                if self.tracks[tid].missed_frames > self.max_missed_frames:
                    del self.tracks[tid]

        # 2. Build tracked objects list and identify Lead Vehicle
        active_objects: List[ObjectBox] = []
        lead_candidate_id = None
        max_bottom_y = -1

        for tid, track in self.tracks.items():
            cx, cy = track.center
            x1, y1, x2, y2 = track.bbox

            # Evaluate distance trend over history
            trend = "STABLE"
            if len(track.history) >= 8:
                past_bbox, _ = track.history[0]
                past_area = max(1, (past_bbox[2] - past_bbox[0]) * (past_bbox[3] - past_bbox[1]))
                curr_area = max(1, (x2 - x1) * (y2 - y1))
                area_ratio = curr_area / float(past_area)

                if area_ratio > 1.15:
                    trend = "APPROACHING"
                elif area_ratio < 0.85:
                    trend = "RECEDING"

            # Determine motion status
            speed = np.hypot(track.velocity[0], track.velocity[1])
            status = "STOPPED" if speed < 1.2 else "MOVING"

            obj_box = ObjectBox(
                track_id=tid,
                bbox=track.bbox,
                class_id=track.class_id,
                label=track.label,
                confidence=track.confidence,
                center=track.center,
                velocity=track.velocity,
                is_lead=False,
                relative_distance_trend=trend,
                status=status
            )
            active_objects.append(obj_box)

            # Check if candidate for Lead Vehicle (in front of our lane)
            if track.label in ("car", "truck", "bus"):
                if abs(cx - lane_center) < self.lead_margin and y2 > (h * 0.50):
                    if y2 > max_bottom_y:
                        max_bottom_y = y2
                        lead_candidate_id = tid

        # Mark lead vehicle
        if lead_candidate_id is not None:
            for obj in active_objects:
                if obj.track_id == lead_candidate_id:
                    obj.is_lead = True

        # Mark pedestrians situated on/near detected crosswalks
        if road_markings and road_markings.crosswalk_detected:
            for obj in active_objects:
                if obj.label == "person":
                    ox1, oy1, ox2, oy2 = obj.bbox
                    for cx1, cy1, cx2, cy2 in road_markings.crosswalk_boxes:
                        # Check horizontal and vertical bounding overlap
                        if (ox2 >= cx1 - 15 and ox1 <= cx2 + 15) and (oy2 >= cy1 - 30 and oy1 <= cy2 + 30):
                            obj.is_on_crosswalk = True
                            break

        return active_objects
