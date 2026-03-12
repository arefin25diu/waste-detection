import cv2
import numpy as np
from ultralytics import YOLO
from .deep_sort_tracker import DeepSORTTracker


class YOLODeepSORTTracker:
    def __init__(self, model_path, confidence_threshold=0.5):
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        self.tracker = DeepSORTTracker()
        self.colors = {}

    def generate_color(self, track_id):
        if track_id not in self.colors:
            np.random.seed(track_id)
            self.colors[track_id] = tuple(map(int, np.random.randint(0, 255, 3)))
        return self.colors[track_id]

    def process_frame(self, frame):
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
