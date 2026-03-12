import cv2
import numpy as np
import torch
from ultralytics import YOLO
from collections import defaultdict, deque
import argparse
import os


class DeepSORTTracker:
    def __init__(self, max_disappeared=30, max_distance=100):
        """
        Simple DeepSORT-like tracker implementation

        Args:
            max_disappeared: Maximum frames an object can be missing before deletion
            max_distance: Maximum distance for object association
        """
        self.next_object_id = 0
        self.objects = {}
        self.disappeared = {}
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def register(self, centroid, bbox, confidence, class_id):
        """Register a new object"""
        self.objects[self.next_object_id] = {
            "centroid": centroid,
            "bbox": bbox,
            "confidence": confidence,
            "class_id": class_id,
            "track": deque(maxlen=50),
        }
        self.objects[self.next_object_id]["track"].append(centroid)
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1

    def deregister(self, object_id):
        """Remove an object from tracking"""
        del self.objects[object_id]
        del self.disappeared[object_id]

    def update(self, detections):
        """
        Update tracker with new detections

        Args:
            detections: List of (bbox, confidence, class_id) tuples
        """
        if len(detections) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects

        input_centroids = []
        input_bboxes = []
        input_confidences = []
        input_class_ids = []

        for detection in detections:
            bbox, confidence, class_id = detection
            x1, y1, x2, y2 = bbox
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            input_centroids.append((cx, cy))
            input_bboxes.append(bbox)
            input_confidences.append(confidence)
            input_class_ids.append(class_id)

        if len(self.objects) == 0:
            for i in range(len(input_centroids)):
                self.register(
                    input_centroids[i],
                    input_bboxes[i],
                    input_confidences[i],
                    input_class_ids[i],
                )
        else:
            object_centroids = [obj["centroid"] for obj in self.objects.values()]
            object_ids = list(self.objects.keys())

            distances = np.linalg.norm(
                np.array(object_centroids)[:, np.newaxis] - np.array(input_centroids),
                axis=2,
            )

            rows = distances.min(axis=1).argsort()
            cols = distances.argmin(axis=1)[rows]

            used_row_indices = set()
            used_col_indices = set()

            for row, col in zip(rows, cols):
                if row in used_row_indices or col in used_col_indices:
                    continue

                if distances[row, col] > self.max_distance:
                    continue

                object_id = object_ids[row]
                self.objects[object_id]["centroid"] = input_centroids[col]
                self.objects[object_id]["bbox"] = input_bboxes[col]
                self.objects[object_id]["confidence"] = input_confidences[col]
                self.objects[object_id]["class_id"] = input_class_ids[col]
                self.objects[object_id]["track"].append(input_centroids[col])
                self.disappeared[object_id] = 0

                used_row_indices.add(row)
                used_col_indices.add(col)

            unused_row_indices = set(range(0, distances.shape[0])).difference(
                used_row_indices
            )
            unused_col_indices = set(range(0, distances.shape[1])).difference(
                used_col_indices
            )

            if distances.shape[0] >= distances.shape[1]:
                for row in unused_row_indices:
                    object_id = object_ids[row]
                    self.disappeared[object_id] += 1

                    if self.disappeared[object_id] > self.max_disappeared:
                        self.deregister(object_id)

            else:
                for col in unused_col_indices:
                    self.register(
                        input_centroids[col],
                        input_bboxes[col],
                        input_confidences[col],
                        input_class_ids[col],
                    )

        return self.objects


class YOLODeepSORTTracker:
    def __init__(self, model_path, confidence_threshold=0.5):
        """
        Initialize YOLO + DeepSORT tracker

        Args:
            model_path: Path to YOLO model (.pt file)
            confidence_threshold: Minimum confidence for detections
        """
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        self.tracker = DeepSORTTracker()
        self.colors = {}

    def generate_color(self, track_id):
        """Generate a unique color for each track ID"""
        if track_id not in self.colors:
            # Generate a random color
            np.random.seed(track_id)
            self.colors[track_id] = tuple(map(int, np.random.randint(0, 255, 3)))
        return self.colors[track_id]

    def process_frame(self, frame):
        """
        Process a single frame

        Args:
            frame: Input frame (BGR format)

        Returns:
            Annotated frame with tracking information
        """
        results = self.model(frame, verbose=False)

        detections = []
        if len(results[0].boxes) > 0:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confidences = results[0].boxes.conf.cpu().numpy()
            class_ids = results[0].boxes.cls.cpu().numpy().astype(int)

            for i, (box, conf, cls_id) in enumerate(zip(boxes, confidences, class_ids)):
                if conf >= self.confidence_threshold:
                    detections.append((box, conf, cls_id))

        tracked_objects = self.tracker.update(detections)

        annotated_frame = self.draw_tracks(frame, tracked_objects)

        return annotated_frame, tracked_objects

    def draw_tracks(self, frame, tracked_objects):
        """
        Draw bounding boxes and tracking lines

        Args:
            frame: Input frame
            tracked_objects: Dictionary of tracked objects

        Returns:
            Annotated frame
        """
        annotated_frame = frame.copy()

        for track_id, obj in tracked_objects.items():
            bbox = obj["bbox"]
            centroid = obj["centroid"]
            confidence = obj["confidence"]
            class_id = obj["class_id"]
            track = obj["track"]

            color = self.generate_color(track_id)

            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)

            class_name = (
                self.model.names[class_id]
                if hasattr(self.model, "names")
                else f"Class_{class_id}"
            )
            label = f"ID:{track_id} {class_name} {confidence:.2f}"

            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )

            cv2.rectangle(
                annotated_frame,
                (x1, y1 - text_height - 10),
                (x1 + text_width, y1),
                color,
                -1,
            )

            cv2.putText(
                annotated_frame,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

            if len(track) > 1:
                points = np.array(list(track), dtype=np.int32)
                cv2.polylines(annotated_frame, [points], False, color, 2)

            cv2.circle(annotated_frame, centroid, 4, color, -1)

        return annotated_frame

    def process_video(self, input_path, output_path=None, display=True):
        """
        Process a video file

        Args:
            input_path: Path to input video
            output_path: Path to save output video (optional)
            display: Whether to display video while processing
        """
        cap = cv2.VideoCapture(input_path)

        if not cap.isOpened():
            print(f"Error: Could not open video {input_path}")
            return

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"Processing video: {width}x{height} @ {fps}fps, {total_frames} frames")

        out = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_count = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                print(f"Processing frame {frame_count}/{total_frames}", end="\r")

                annotated_frame, tracked_objects = self.process_frame(frame)

                if out:
                    out.write(annotated_frame)

                if display:
                    cv2.imshow("YOLO + DeepSORT Tracking", annotated_frame)

                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        break
                    elif key == ord("p"):
                        cv2.waitKey(0)

        except KeyboardInterrupt:
            print("\nProcessing interrupted by user")

        finally:
            cap.release()
            if out:
                out.release()
            if display:
                cv2.destroyAllWindows()

        print(f"\nProcessing complete. Processed {frame_count} frames.")
        if output_path:
            print(f"Output saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="YOLO v8 + DeepSORT Object Tracking")
    parser.add_argument("--model", required=True, help="Path to YOLO model (.pt file)")
    parser.add_argument("--video", required=True, help="Path to input video")
    parser.add_argument("--output", help="Path to output video (optional)")
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.5,
        help="Confidence threshold (default: 0.5)",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        default=True,
        help="Don't display video during processing",
    )

    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"Error: Model file {args.model} not found")
        return

    if not os.path.exists(args.video):
        print(f"Error: Video file {args.video} not found")
        return

    print("Initializing YOLO + DeepSORT tracker...")
    tracker = YOLODeepSORTTracker(args.model, args.confidence)

    tracker.process_video(
        input_path=args.video, output_path=args.output, display=not args.no_display
    )


if __name__ == "__main__":
    main()
