import streamlit as st
import os
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from collections import defaultdict
import tempfile
import os
import time
from src.deep_sort_tracker import DeepSORTTracker
from src.yolo_tracker import YOLODeepSORTTracker


# -------------------------------
# PAGE CONFIG
# -------------------------------

st.set_page_config(
    page_title="AI Object Tracking Dashboard",
    page_icon="🎯",
    layout="wide",
)

# -------------------------------
# CUSTOM CSS
# -------------------------------

st.markdown(
    """
<style>

.main-title{
    font-size:42px;
    font-weight:700;
    text-align:center;
    background: linear-gradient(90deg,#06b6d4,#3b82f6,#9333ea);
    -webkit-background-clip:text;
    -webkit-text-fill-color:transparent;
}

.subtitle{
    text-align:center;
    color:gray;
    margin-bottom:30px;
}

.metric-card{
    background:#111827;
    padding:20px;
    border-radius:12px;
    text-align:center;
    color:white;
}

.metric-value{
    font-size:28px;
    font-weight:700;
    color:#22c55e;
}

.metric-label{
    font-size:14px;
    color:#9ca3af;
}

.upload-box{
    border:2px dashed #3b82f6;
    padding:30px;
    border-radius:10px;
    text-align:center;
}

.stButton>button{
    background:linear-gradient(90deg,#3b82f6,#9333ea);
    color:white;
    border:none;
    padding:12px 30px;
    border-radius:10px;
    font-size:18px;
}

</style>
""",
    unsafe_allow_html=True,
)

# -------------------------------
# HEADER
# -------------------------------

st.markdown(
    '<p class="main-title">🎯 YOLO + DeepSORT Object Tracking</p>',
    unsafe_allow_html=True,
)

st.markdown(
    '<p class="subtitle">AI Powered Real-Time Object Detection & Tracking System</p>',
    unsafe_allow_html=True,
)

# -------------------------------
# SIDEBAR
# -------------------------------

with st.sidebar:
    st.title("⚙️ Settings")

    model_files = os.listdir("model")
    model_files = [f for f in model_files if f.endswith(".pt")]

    model_option = st.selectbox(
        "Select YOLO Model",
        model_files,
    )

    model_path = "model/" + model_option

    st.subheader("Detection Parameters")

    confidence_threshold = st.slider(
        "Confidence Threshold",
        0.1,
        0.9,
        0.5,
    )

    max_disappeared = st.slider(
        "Max Disappeared Frames",
        10,
        100,
        30,
    )

    max_distance = st.slider(
        "Max Association Distance",
        50,
        200,
        100,
    )

    st.divider()

    st.info("Upload a video and click **Start Tracking**")


# -------------------------------
# VIDEO UPLOAD
# -------------------------------

st.markdown("### 📹 Upload Video")

st.markdown('<div class="upload-box">', unsafe_allow_html=True)

uploaded_video = st.file_uploader(
    "Drag & Drop Video File",
    type=["mp4", "avi", "mov", "mkv"],
)

st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------
# PROCESS VIDEO
# -------------------------------

if uploaded_video is not None:

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
        tmp_file.write(uploaded_video.read())
        video_path = tmp_file.name

    cap = cv2.VideoCapture(video_path)

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps

    cap.release()

    st.markdown("## 📊 Video Information")

    col1, col2, col3, col4 = st.columns(4)

    col1.markdown(
        f"""
        <div class="metric-card">
        <div class="metric-value">{width}x{height}</div>
        <div class="metric-label">Resolution</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col2.markdown(
        f"""
        <div class="metric-card">
        <div class="metric-value">{fps}</div>
        <div class="metric-label">FPS</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col3.markdown(
        f"""
        <div class="metric-card">
        <div class="metric-value">{total_frames}</div>
        <div class="metric-label">Frames</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col4.markdown(
        f"""
        <div class="metric-card">
        <div class="metric-value">{duration:.1f}s</div>
        <div class="metric-label">Duration</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("")

    if st.button("🚀 Start Tracking"):

        try:

            with st.spinner("Initializing YOLO + DeepSORT..."):
                tracker = YOLODeepSORTTracker(model_path, confidence_threshold)
                tracker.tracker.max_disappeared = max_disappeared
                tracker.tracker.max_distance = max_distance

            st.success("Tracker initialized!")

            output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name

            cap = cv2.VideoCapture(video_path)

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")

            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            progress_bar = st.progress(0)

            frame_placeholder = st.empty()

            status_text = st.empty()

            stats_placeholder = st.empty()

            frame_count = 0

            object_counts = defaultdict(int)

            start_time = time.time()

            while True:

                ret, frame = cap.read()

                if not ret:
                    break

                frame_count += 1

                annotated_frame, tracked_objects = tracker.process_frame(frame)

                out.write(annotated_frame)

                current_objects = {}

                for obj_id, obj in tracked_objects.items():

                    class_name = (
                        tracker.model.names[obj["class_id"]]
                        if hasattr(tracker.model, "names")
                        else f'Class_{obj["class_id"]}'
                    )

                    if class_name not in current_objects:
                        current_objects[class_name] = 0

                    current_objects[class_name] += 1

                    object_counts[class_name] = max(
                        object_counts[class_name], current_objects[class_name]
                    )

                if frame_count % 10 == 0 or frame_count == total_frames:

                    progress = frame_count / total_frames

                    progress_bar.progress(progress)

                    elapsed = time.time() - start_time

                    fps_current = frame_count / elapsed if elapsed > 0 else 0

                    status_text.text(
                        f"Frame {frame_count}/{total_frames} | FPS: {fps_current:.2f}"
                    )

                    display = cv2.resize(annotated_frame, (640, 480))

                    display = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)

                    frame_placeholder.image(display)

                    if current_objects:

                        stats = " | ".join(
                            [f"{k}: {v}" for k, v in current_objects.items()]
                        )

                        stats_placeholder.markdown(
                            f"**Current Objects:** {stats}"
                        )

            cap.release()

            out.release()

            st.success("Processing Complete!")

            # -------------------------
            # STATS
            # -------------------------

            st.markdown("## 📊 Tracking Statistics")

            cols = st.columns(len(object_counts))

            for i, (cls, count) in enumerate(object_counts.items()):
                cols[i].metric(cls, count)

            # -------------------------
            # DOWNLOAD
            # -------------------------

            st.markdown("## 📥 Download Result")

            with open(output_path, "rb") as f:
                video_bytes = f.read()

            st.download_button(
                "⬇ Download Processed Video",
                data=video_bytes,
                file_name=f"tracked_{uploaded_video.name}",
                mime="video/mp4",
            )

        except Exception as e:

            st.error(f"Error: {str(e)}")
            st.exception(e)