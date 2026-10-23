"""Main pipeline orchestrator for dashcam_cv system.

Executes end-to-end video decoding, perception, state fusion, and logging.
"""
import argparse
import os
import sys
import time
from pathlib import Path
import cv2
import yaml

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.common.data_models import FrameTimings
from src.common.geometry import get_roi_polygon
from src.common.visualizer import HUDVisualizer
from src.video.video_reader import VideoReader
from src.video.preprocessing import FramePreprocessor
from src.video.quality_analyzer import QualityAnalyzer
from src.lane.lane_detector import LaneDetector
from src.lane.lane_tracker import LaneTracker
from src.lane.lane_change import LaneChangeDetector

from src.objects.detector import ObjectDetector
from src.objects.object_tracker import ObjectTracker
from src.objects.road_markings import RoadMarkingDetector
from src.state.motion_estimator import MotionEstimator
from src.state.decision_fusion import DecisionFusion
from src.output.video_writer import AnnotatedVideoWriter
from src.output.csv_logger import CSVLogger
from src.output.metrics import MetricsCollector


def load_config(config_path: str) -> dict:
    """Loads YAML configuration file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_args():
    """Parses command-line arguments and flags."""
    parser = argparse.ArgumentParser(
        description="dashcam_cv: Lightweight Autonomous Vision Video Analysis System"
    )
    parser.add_argument("--config", default="config/settings.yaml", help="Path to settings.yaml")
    parser.add_argument("--input", help="Override input video path")
    parser.add_argument("--output-dir", help="Override output directory")
    parser.add_argument("--width", type=int, help="Override target frame width (e.g. 640 or 960)")
    parser.add_argument("--stride", type=int, help="Override frame stride (e.g. 1, 2, 3)")
    parser.add_argument("--detect-interval", type=int, help="Override object detection frame interval (e.g. 1, 5, 10)")
    parser.add_argument("--max-frames", type=int, help="Process only first N frames")
    parser.add_argument("--display", action="store_true", help="Display GUI playback window")
    parser.add_argument("--no-display", action="store_true", help="Disable GUI playback window")
    return parser.parse_args()


def run_pipeline(cfg: dict, args: argparse.Namespace):
    """Executes the full computer vision analysis pipeline."""
    # Apply CLI overrides
    if args.input:
        cfg["video"]["input_path"] = args.input
    if args.width:
        cfg["video"]["target_width"] = args.width
    if args.stride:
        cfg["video"]["frame_stride"] = args.stride
    if args.max_frames:
        cfg["video"]["max_frames"] = args.max_frames
    if args.detect_interval:
        cfg["detector"]["detect_interval"] = args.detect_interval
    if args.display:
        cfg["visualization"]["display_window"] = True
    elif args.no_display:
        cfg["visualization"]["display_window"] = False

    if args.output_dir:
        out_dir = Path(args.output_dir)
        cfg["video"]["output_video"] = str(out_dir / "annotated_video.mp4")
        cfg["video"]["output_frame_log"] = str(out_dir / "frame_log.csv")
        cfg["video"]["output_perf_log"] = str(out_dir / "performance.csv")
        cfg["video"]["output_summary"] = str(out_dir / "summary.json")

    v_cfg = cfg["video"]
    input_path = v_cfg["input_path"]
    out_video_path = v_cfg["output_video"]
    out_frame_log = v_cfg["output_frame_log"]
    out_perf_log = v_cfg["output_perf_log"]
    out_summary = v_cfg["output_summary"]

    target_w = int(v_cfg.get("target_width", 960))
    stride = int(v_cfg.get("frame_stride", 1))
    max_frames = int(v_cfg.get("max_frames", 0))

    print("=" * 65)
    print("  DASHCAM COMPUTER VISION ANALYSIS PIPELINE")
    print(f"  Input: {input_path}")
    print(f"  Output Video: {out_video_path}")
    print(f"  Target Width: {target_w} | Stride: {stride} | Detect Interval: {cfg['detector']['detect_interval']}")
    print("=" * 65)

    # 1. Initialize Video Reader
    reader = VideoReader(input_path, target_width=target_w, frame_stride=stride)
    meta = reader.meta
    print(f"[Video Info] Source: {meta.width}x{meta.height} @ {meta.fps:.2f} FPS | "
          f"Frames: {meta.frame_count} ({meta.duration_sec:.1f}s)")
    print(f"[Processing Size] {meta.target_width}x{meta.target_height}")

    # 2. Initialize Core Perception Modules
    preprocessor = FramePreprocessor(cfg)
    quality_analyzer = QualityAnalyzer(cfg)
    lane_detector = LaneDetector(cfg)
    lane_tracker = LaneTracker(cfg)
    lane_change_detector = LaneChangeDetector(cfg)
    object_detector = ObjectDetector(cfg)
    object_tracker = ObjectTracker(cfg)
    road_markings = RoadMarkingDetector(cfg)
    motion_estimator = MotionEstimator(cfg)
    decision_fusion = DecisionFusion(cfg)
    visualizer = HUDVisualizer(cfg)

    # 3. Initialize Output Writers
    video_writer = AnnotatedVideoWriter(
        out_video_path,
        meta.fps / float(stride),
        (meta.target_width, meta.target_height),
        codec=v_cfg.get("codec", "mp4v")
    )
    csv_logger = CSVLogger(out_frame_log, out_perf_log)
    metrics = MetricsCollector(out_summary, {
        "input_path": input_path,
        "source_resolution": [meta.width, meta.height],
        "target_resolution": [meta.target_width, meta.target_height],
        "fps": meta.fps,
        "total_source_frames": meta.frame_count
    })

    display_active = cfg.get("visualization", {}).get("display_window", False)
    paused = False

    try:
        for ok, frame, frame_idx, timestamp in reader.frames(max_frames=max_frames):
            if not ok or frame is None:
                continue

            t_total_start = time.perf_counter()

            # Preprocessing & Quality
            t0 = time.perf_counter()
            roi_corners = get_roi_polygon(meta.target_width, meta.target_height, cfg.get("roi"))
            gray, enhanced, edges = preprocessor.process(frame)
            quality = quality_analyzer.analyze(frame, gray)
            t_preprocess = (time.perf_counter() - t0) * 1000.0

            # Lane Detection & Tracking (cv1021.py dual-ROI Canny + Kalman Filter)
            t0 = time.perf_counter()
            curr_left_fit, curr_right_fit, is_crosswalk, lane_debug = lane_detector.detect(frame)
            lane_res = lane_tracker.update(
                curr_left_fit=curr_left_fit,
                curr_right_fit=curr_right_fit,
                raw_confidence=lane_debug.get("confidence", 0.0),
                w=meta.target_width,
                h=meta.target_height,
                is_crosswalk=is_crosswalk,
                debug_info=lane_debug,
            )
            lane_change_st = lane_change_detector.update(lane_res)
            t_lane = (time.perf_counter() - t0) * 1000.0

            # Road Markings (Crosswalks and Stop lines)
            t0 = time.perf_counter()
            markings_res = road_markings.detect(frame, enhanced, edges)
            if is_crosswalk:
                markings_res.crosswalk_detected = True
                if lane_debug.get("cw_boxes") and not markings_res.crosswalk_boxes:
                    markings_res.crosswalk_boxes = lane_debug.get("cw_boxes", [])
            t_markings = (time.perf_counter() - t0) * 1000.0

            # Object Detection
            t0 = time.perf_counter()
            raw_objects, det_updated = object_detector.detect(frame, frame_idx)
            t_detect = (time.perf_counter() - t0) * 1000.0

            # Object Tracking
            t0 = time.perf_counter()
            tracked_objects = object_tracker.update(
                raw_objects, det_updated, lane_res, frame.shape, markings_res
            )
            t_tracking = (time.perf_counter() - t0) * 1000.0

            # Background Motion Estimation
            t0 = time.perf_counter()
            motion_res = motion_estimator.estimate(gray, tracked_objects)
            t_motion = (time.perf_counter() - t0) * 1000.0

            # Decision Fusion
            t0 = time.perf_counter()
            state_res = decision_fusion.evaluate(
                motion_res, quality, lane_res, tracked_objects, markings_res, lane_change_st, frame.shape
            )
            t_decision = (time.perf_counter() - t0) * 1000.0

            # HUD Visualization Overlay
            t0 = time.perf_counter()
            dummy_timings = FrameTimings(
                preprocess_ms=t_preprocess,
                lane_ms=t_lane,
                detect_ms=t_detect,
                tracking_ms=t_tracking,
                marking_ms=t_markings,
                motion_ms=t_motion,
                decision_ms=t_decision,
                render_ms=0.0,
                total_ms=0.0,
                detection_updated=det_updated
            )
            # Intermediate total for HUD display
            dummy_timings.total_ms = (t_preprocess + t_lane + t_detect + t_tracking +
                                     t_markings + t_motion + t_decision)

            debug_map = lane_debug.get("combined_edges") if visualizer.show_debug_minimap else None
            vis_frame = visualizer.draw(
                frame, frame_idx, timestamp, lane_res, tracked_objects,
                markings_res, motion_res, state_res, dummy_timings, quality,
                debug_edge_map=debug_map
            )
            t_render = (time.perf_counter() - t0) * 1000.0

            t_total = (time.perf_counter() - t_total_start) * 1000.0

            timings = FrameTimings(
                decode_ms=0.0,
                preprocess_ms=t_preprocess,
                lane_ms=t_lane,
                detect_ms=t_detect,
                tracking_ms=t_tracking,
                marking_ms=t_markings,
                motion_ms=t_motion,
                decision_ms=t_decision,
                render_ms=t_render,
                total_ms=t_total,
                detection_updated=det_updated
            )

            # Write outputs
            video_writer.write(vis_frame)
            csv_logger.log(
                frame_idx, timestamp, lane_res, tracked_objects,
                markings_res, motion_res, state_res, timings
            )
            metrics.record(timings, lane_res, state_res, markings_res, len(tracked_objects))

            # Progress log every 30 frames
            if frame_idx % 30 == 0:
                fps_val = (1000.0 / t_total) if t_total > 0 else 0.0
                print(f"[Frame {frame_idx:04d} | {timestamp:5.2f}s] "
                      f"FPS: {fps_val:4.1f} | Ego: {state_res.ego_state:<20} | "
                      f"Lane: {lane_res.status:<14} | Objs: {len(tracked_objects):2d}")

            if display_active:
                cv2.imshow("dashcam_cv", vis_frame)
                key = cv2.waitKey(1 if not paused else 0) & 0xFF
                if key == ord('q'):
                    print("[Pipeline] User requested quit.")
                    break
                elif key == ord('p'):
                    paused = not paused
                elif key == ord('d'):
                    visualizer.show_debug_minimap = not visualizer.show_debug_minimap
                elif key == ord('b'):
                    visualizer.show_bbox = not visualizer.show_bbox


    finally:
        reader.release()
        video_writer.release()
        csv_logger.close()
        if display_active:
            cv2.destroyAllWindows()

    summary = metrics.generate_summary()
    print("=" * 65)
    print("  ANALYSIS COMPLETE")
    print(f"  Processed Frames: {summary.get('performance', {}).get('processed_frames', 0)}")
    print(f"  Effective FPS:    {summary.get('performance', {}).get('effective_processing_fps', 0):.2f}")
    print(f"  Mean Latency:     {summary.get('performance', {}).get('mean_total_ms', 0):.2f} ms")
    print(f"  P95 Latency:      {summary.get('performance', {}).get('p95_total_ms', 0):.2f} ms")
    print(f"  Output Video:     {out_video_path}")
    print(f"  Frame Log:        {out_frame_log}")
    print(f"  Perf Log:         {out_perf_log}")
    print(f"  Summary Report:   {out_summary}")
    print("=" * 65)


def main():
    args = parse_args()
    config = load_config(args.config)
    run_pipeline(config, args)


if __name__ == "__main__":
    main()
