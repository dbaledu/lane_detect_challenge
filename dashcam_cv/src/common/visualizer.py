"""HUD visualizer for autonomous vision video analysis."""
import cv2
import numpy as np
from typing import List, Optional, Tuple, Dict, Any
from .data_models import (
    LaneResult,
    ObjectBox,
    RoadMarkingResult,
    MotionResult,
    DrivingStateResult,
    FrameTimings,
    QualityMetric
)
from .geometry import get_roi_polygon


class HUDVisualizer:
    """Renders professional autonomous driving HUD overlays on video frames."""

    def __init__(self, cfg: Optional[Dict[str, Any]] = None):
        self.cfg = cfg or {}
        self.vis_cfg = self.cfg.get("visualization", {})
        self.show_corridor = self.vis_cfg.get("show_lane_polygon", True)
        self.show_center_path = self.vis_cfg.get("show_center_path", True)
        self.show_hud_card = self.vis_cfg.get("show_hud_card", True)
        self.show_debug_minimap = self.vis_cfg.get("show_debug_minimap", True)
        self.show_objects = self.vis_cfg.get("show_objects", True)
        self.show_bbox = self.vis_cfg.get("show_bbox", True)
        self.show_object_labels = self.vis_cfg.get(
            "show_object_labels",
            self.vis_cfg.get("show_box_labels", self.cfg.get("detector", {}).get("show_labels", True))
        )

    def draw(
        self,
        frame: np.ndarray,
        frame_idx: int,
        timestamp_sec: float,
        lane: LaneResult,
        objects: List[ObjectBox],
        markings: RoadMarkingResult,
        motion: MotionResult,
        state: DrivingStateResult,
        timings: FrameTimings,
        quality: QualityMetric,
        debug_edge_map: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Composes all visual components onto the frame."""
        h, w = frame.shape[:2]
        vis = frame.copy()
        overlay = frame.copy()

        roi_corners = get_roi_polygon(w, h, self.cfg.get("roi"))

        # 1. Draw Dual ROI Guidelines & Bounding Boxes (cv1021.py) or default ROI
        if lane.left_corners is not None and lane.right_corners is not None:
            if not lane.is_crosswalk:
                if self.show_bbox:
                    if lane.left_bbox is not None:
                        lx1, ly1, lx2, ly2 = lane.left_bbox
                        cv2.rectangle(overlay, (lx1, ly1), (lx2, ly2), (0, 255, 255), 1)
                        cv2.putText(overlay, f"L-BBox ({lx2-lx1}px)", (lx1, max(12, ly1 - 6)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

                    if lane.right_bbox is not None:
                        rx1, ry1, rx2, ry2 = lane.right_bbox
                        cv2.rectangle(overlay, (rx1, ry1), (rx2, ry2), (255, 255, 0), 1)
                        cv2.putText(overlay, f"R-BBox ({rx2-rx1}px)", (rx1, max(12, ry1 - 6)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

                cv2.polylines(overlay, [lane.left_corners], isClosed=True, color=(255, 200, 0), thickness=1)
                cv2.polylines(overlay, [lane.right_corners], isClosed=True, color=(0, 200, 255), thickness=1)
        else:
            cv2.polylines(overlay, [roi_corners], isClosed=True, color=(70, 70, 70), thickness=1)


        # 2. Draw Driving Corridor & Lane Boundaries
        if self.show_corridor and lane.left_line and lane.right_line and lane.status != "LOST":
            lx1, ly1, lx2, ly2 = lane.left_line
            rx1, ry1, rx2, ry2 = lane.right_line
            corridor_pts = np.array([[
                (lx1, ly1), (lx2, ly2), (rx2, ry2), (rx1, ry1)
            ]], dtype=np.int32)

            # Corridor color changes depending on warning decision state
            corridor_color = (0, 200, 0)  # Green
            if state.fusion_state == "SLOW_DOWN":
                corridor_color = (0, 180, 240)  # Amber
            elif state.fusion_state == "STOP_REQUIRED":
                corridor_color = (40, 40, 220)  # Red

            cv2.fillPoly(overlay, corridor_pts, corridor_color)

        # Alpha blend corridor overlay
        vis = cv2.addWeighted(overlay, 0.30, vis, 0.70, 0)

        # Draw Left/Right Lane lines on top of blend
        if lane.left_line:
            lx1, ly1, lx2, ly2 = lane.left_line
            cv2.line(vis, (lx1, ly1), (lx2, ly2), (255, 120, 0), 4, cv2.LINE_AA)
        if lane.right_line:
            rx1, ry1, rx2, ry2 = lane.right_line
            cv2.line(vis, (rx1, ry1), (rx2, ry2), (0, 100, 255), 4, cv2.LINE_AA)

        # Draw Center Driving Path
        if self.show_center_path and lane.lane_center_x is not None and lane.status != "LOST":
            cx = int(lane.lane_center_x)
            top_y = int(roi_corners[0][0][1])
            bot_y = int(roi_corners[0][3][1])
            # Dashed center line
            for y in range(top_y, bot_y, 16):
                cv2.line(vis, (cx, y), (cx, min(bot_y, y + 8)), (0, 255, 180), 2, cv2.LINE_AA)
            # Vehicle bottom frame center marker
            frame_center_x = w // 2
            cv2.circle(vis, (frame_center_x, bot_y), 6, (255, 255, 0), -1)
            cv2.line(vis, (frame_center_x, bot_y), (cx, bot_y), (0, 255, 255), 2)

        # 3. Draw Road Markings (Crosswalks and Stoplines)
        if markings.crosswalk_detected:
            for cbox in markings.crosswalk_boxes:
                bx1, by1, bx2, by2 = cbox
                cv2.rectangle(vis, (bx1, by1), (bx2, by2), (0, 240, 240), 2)
                cv2.putText(vis, "CROSSWALK", (bx1, max(18, by1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 240, 240), 1, cv2.LINE_AA)

        if markings.stopline_detected:
            for sbox in markings.stopline_boxes:
                bx1, by1, bx2, by2 = sbox
                cv2.rectangle(vis, (bx1, by1), (bx2, by2), (0, 0, 255), 3)
                cv2.putText(vis, "STOPLINE", (bx1, max(18, by1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2, cv2.LINE_AA)

        # 4. Draw Detected Objects
        if self.show_objects:
            for obj in objects:
                x1, y1, x2, y2 = obj.bbox
                if obj.is_on_crosswalk:
                    box_color = (0, 69, 255)  # High-visibility Red-Orange for pedestrians on crosswalk
                    box_thickness = 3
                    tag = f"CROSSWALK PED #{obj.track_id} {obj.confidence:.2f}"
                elif obj.is_lead:
                    box_color = (0, 0, 255)  # Red for lead vehicle
                    box_thickness = 3
                    tag = f"LEAD CAR #{obj.track_id} [{obj.status}|{obj.relative_distance_trend}]"
                elif obj.label == "person":
                    box_color = (0, 255, 255)  # Yellow for pedestrians
                    box_thickness = 2
                    tag = f"PED #{obj.track_id} {obj.confidence:.2f}"
                else:
                    box_color = (255, 180, 0)  # Cyan/Blue for other vehicles
                    box_thickness = 2
                    tag = f"{obj.label} #{obj.track_id}"

                cv2.rectangle(vis, (x1, y1), (x2, y2), box_color, box_thickness)
                # 박태훈 10/23 수정. 객체 라벨 표시여부 제어(on/off) 
                if self.show_object_labels:
                    # Label tag background
                    (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                    cv2.rectangle(vis, (x1, max(0, y1 - th - 6)), (x1 + tw + 6, max(th + 6, y1)), box_color, -1)
                    cv2.putText(vis, tag, (x1 + 3, max(th, y1 - 3)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

        # 4.1 Draw Detected Traffic Light
        if state.traffic_light and state.traffic_light.detected:
            tl = state.traffic_light
            color_map = {
                "RED": (0, 0, 255),
                "YELLOW": (0, 220, 255),
                "GREEN": (0, 255, 0)
            }
            tl_box_color = color_map.get(tl.color, (0, 220, 255))
            if tl.bbox:
                tx1, ty1, tx2, ty2 = tl.bbox
                cv2.rectangle(vis, (tx1, ty1), (tx2, ty2), tl_box_color, 2)
                tl_tag = f"TL: {tl.color} ({tl.confidence:.2f})"
                (tw, th), _ = cv2.getTextSize(tl_tag, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                cv2.rectangle(vis, (tx1, max(0, ty1 - th - 6)), (tx1 + tw + 6, max(th + 6, ty1)), tl_box_color, -1)
                cv2.putText(vis, tl_tag, (tx1 + 3, max(th, ty1 - 3)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
            if tl.circle:
                tcx, tcy, tr = tl.circle
                cv2.circle(vis, (tcx, tcy), tr, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.circle(vis, (tcx, tcy), max(2, tr - 2), tl_box_color, -1)

        # 5. Draw HUD Dashboard Card (Top Left)
        if self.show_hud_card:
            card_w, card_h = 380, 172
            card_x, card_y = 15, 15

            # Translucent dark background
            hud_sub = vis[card_y:card_y + card_h, card_x:card_x + card_w]
            hud_bg = np.zeros_like(hud_sub)
            cv2.rectangle(hud_bg, (0, 0), (card_w, card_h), (25, 25, 25), -1)
            vis[card_y:card_y + card_h, card_x:card_x + card_w] = cv2.addWeighted(hud_sub, 0.25, hud_bg, 0.75, 0)
            cv2.rectangle(vis, (card_x, card_y), (card_x + card_w, card_y + card_h), (120, 120, 120), 1)

            # Header
            cv2.putText(vis, "DASHCAM VISION SYSTEM", (card_x + 12, card_y + 22),
                        cv2.FONT_HERSHEY_DUPLEX, 0.55, (240, 240, 240), 1, cv2.LINE_AA)
            cv2.line(vis, (card_x + 10, card_y + 28), (card_x + card_w - 10, card_y + 28), (80, 80, 80), 1)

            # Frame & Timestamp & FPS
            eff_fps = (1000.0 / timings.total_ms) if timings.total_ms > 0 else 0.0
            info_str = f"F#{frame_idx:04d} | {timestamp_sec:.2f}s | {eff_fps:4.1f} FPS ({timings.total_ms:4.1f}ms)"
            cv2.putText(vis, info_str, (card_x + 12, card_y + 46),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

            # Lead Vehicle State & Traffic Light
            lead_color = (255, 100, 100) if state.lead_state in ("STOPPED", "APPROACHING") else (180, 220, 180)
            tl_tag_hud = f" | TL:{state.traffic_light.color}" if (state.traffic_light and state.traffic_light.detected and state.traffic_light.color) else ""
            cv2.putText(vis, f"LEAD: {state.lead_state} | LC: {state.lane_change_status}{tl_tag_hud}", (card_x + 12, card_y + 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, lead_color, 1, cv2.LINE_AA)

            # Fusion Decision State & Reason (주행 판단 및 사유 유지)
            cv2.putText(vis, f"DECISION: {state.fusion_state}", (card_x + 12, card_y + 93),
                        cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 230, 255), 1, cv2.LINE_AA)
            reason_trunc = state.reason if len(state.reason) < 40 else state.reason[:38] + ".."
            cv2.putText(vis, f"  -> {reason_trunc}", (card_x + 12, card_y + 113),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (220, 220, 220), 1, cv2.LINE_AA)

            # Lane Status & Offset
            if lane.is_crosswalk:
                lane_str = "LANE: CROSSWALK (ROI PAUSED)"
                lane_color = (0, 140, 255)  # Orange warning for crosswalk
            elif lane.status == "BOTH_DETECTED":
                lane_str = f"LANE: KALMAN (L+R)" if lane.lost_frames == 0 else f"LANE: PREDICTING ({lane.lost_frames}f)"
                lane_color = (0, 255, 0)
            elif lane.status == "LEFT_ONLY":
                lane_str = f"LANE: LEFT ONLY" if lane.lost_frames == 0 else f"LANE: PREDICTING (L)"
                lane_color = (255, 200, 0)
            elif lane.status == "RIGHT_ONLY":
                lane_str = f"LANE: RIGHT ONLY" if lane.lost_frames == 0 else f"LANE: PREDICTING (R)"
                lane_color = (0, 200, 255)
            elif lane.status == "TEMPORARY_LOST":
                lane_str = f"LANE: PREDICTING ({lane.lost_frames}f)"
                lane_color = (0, 180, 255)
            else:
                lane_str = "LANE: LOST"
                lane_color = (0, 0, 255)

            err_str = f"Δx:{int(lane.lane_error_px):+d}px" if lane.lane_error_px is not None else "Δx:N/A"
            cv2.putText(vis, f"{lane_str} | {err_str}", (card_x + 12, card_y + 136),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, lane_color, 1, cv2.LINE_AA)

            # Motion & Lighting
            night_tag = "NIGHT" if quality.is_night else "DAY"
            cv2.putText(vis, f"FLOW: {motion.magnitude:.2f} px/f | LIGHT: {night_tag} (br:{quality.brightness:.0f})",
                        (card_x + 12, card_y + 156),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (190, 190, 190), 1, cv2.LINE_AA)

        # 6. Draw Debug Minimap (Top Right - cv1021 Dual Canny Minimap)
        if self.show_debug_minimap and debug_edge_map is not None:
            mw, mh = int(w * 0.28), int(h * 0.20)
            mx = w - mw - 15
            my = 15

            if len(debug_edge_map.shape) == 2:
                mini_bgr = cv2.cvtColor(debug_edge_map, cv2.COLOR_GRAY2BGR)
            else:
                mini_bgr = debug_edge_map.copy()

            # cv1021.py 좌/우 분리 미니맵 라벨링
            if mini_bgr.shape[1] == 640 and mini_bgr.shape[0] == 360:
                cv2.line(mini_bgr, (320, 0), (320, 360), (0, 255, 255), 2)
                cv2.putText(mini_bgr, "LEFT (Canny)", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2)
                cv2.putText(mini_bgr, "RIGHT (Canny)", (340, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

            mini_resized = cv2.resize(mini_bgr, (mw, mh))
            vis[my:my + mh, mx:mx + mw] = mini_resized
            cv2.rectangle(vis, (mx, my), (mx + mw, my + mh), (0, 220, 255), 2)
            cv2.putText(vis, "Dual Canny Minimap", (mx + 8, my + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 255), 1, cv2.LINE_AA)

        return vis
