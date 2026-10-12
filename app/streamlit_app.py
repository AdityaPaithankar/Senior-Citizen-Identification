from pathlib import Path

code = r'''from pathlib import Path
from datetime import datetime
import csv
import threading
import time

import av
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from streamlit_webrtc import VideoProcessorBase, webrtc_streamer

from pipeline import SeniorCitizenPipeline


# -------------------- PATHS & SETTINGS --------------------
BASE_DIR = Path(__file__).resolve().parent.parent
LOG_DIR = BASE_DIR / "logs"
LOG_FILE = LOG_DIR / "visitor_log.csv"
LOG_DIR.mkdir(parents=True, exist_ok=True)

LOG_COLUMNS = [
    "estimated_age",
    "predicted_gender",
    "visit_time",
    "senior_status",
]

# One shared lock protects CSV writes from image and webcam processing.
CSV_LOCK = threading.Lock()

# Avoid logging the same approximate detection signature too frequently.
LIVE_LOG_COOLDOWN_SECONDS = 30


# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="Senior Citizen Identification",
    page_icon="👴",
    layout="wide",
)

st.title("👴 Senior Citizen Identification")
st.caption(
    "Multi-face detection with estimated age, predicted gender, "
    "and configurable senior-status classification."
)


# -------------------- MODEL --------------------
@st.cache_resource
def load_pipeline():
    return SeniorCitizenPipeline()


try:
    pipeline = load_pipeline()
except Exception as exc:
    st.error(f"Could not load the model: {exc}")
    st.info(
        "Check that model/best_age_gender_model.keras exists and "
        "that app/pipeline.py defines SeniorCitizenPipeline correctly."
    )
    st.stop()


# -------------------- STUN / TURN CONFIGURATION --------------------
# STUN can help discover network routes but does not relay media.
# For restrictive networks and reliable cloud connectivity, add TURN
# credentials in Streamlit Cloud -> App -> Settings -> Secrets.
def get_rtc_configuration():
    ice_servers = [
        {"urls": ["stun:stun.l.google.com:19302"]},
    ]

    try:
        turn_url = st.secrets.get("TURN_SERVER_URL", "")
        turn_username = st.secrets.get("TURN_USERNAME", "")
        turn_credential = st.secrets.get("TURN_CREDENTIAL", "")

        if turn_url and turn_username and turn_credential:
            ice_servers.append(
                {
                    "urls": [turn_url],
                    "username": turn_username,
                    "credential": turn_credential,
                }
            )
    except Exception:
        # Local runs without a secrets.toml file use STUN only.
        pass

    return {"iceServers": ice_servers}


RTC_CONFIGURATION = get_rtc_configuration()


# -------------------- CSV LOGGING --------------------
def save_logs(results):
    """Append one CSV row per supplied detection."""
    if not results:
        return

    with CSV_LOCK:
        file_needs_header = (
            not LOG_FILE.exists() or LOG_FILE.stat().st_size == 0
        )

        with LOG_FILE.open("a", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=LOG_COLUMNS)
            if file_needs_header:
                writer.writeheader()

            timestamp = datetime.now().astimezone().isoformat(
                timespec="seconds"
            )

            for person in results:
                writer.writerow(
                    {
                        "estimated_age": int(person["age"]),
                        "predicted_gender": str(person["gender"]),
                        "visit_time": timestamp,
                        "senior_status": (
                            "Senior Citizen"
                            if person["senior"]
                            else "Non-Senior"
                        ),
                    }
                )


def annotate_frame(frame, results):
    """Draw face boxes and labels on a BGR frame."""
    output = frame.copy()

    for person in results:
        x, y, w, h = map(int, person["box"])
        status = (
            "Senior Citizen"
            if person["senior"]
            else "Non-Senior"
        )
        label = f'{person["gender"]} | Age: {person["age"]} | {status}'

        cv2.rectangle(
            output,
            (x, y),
            (x + w, y + h),
            (0, 200, 0),
            2,
        )
        cv2.putText(
            output,
            label,
            (x, max(y - 10, 25)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 200, 0),
            2,
            cv2.LINE_AA,
        )

    return output


# -------------------- LIVE VIDEO PROCESSOR --------------------
class SeniorCitizenVideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.pipeline = load_pipeline()
        self.last_inference_time = 0.0
        self.last_results = []
        self.error_message = None
        self.last_logged_at = {}

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        image = frame.to_ndarray(format="bgr24")
        now = time.monotonic()

        # Limit inference rate to reduce CPU usage.
        if now - self.last_inference_time >= 0.5:
            try:
                self.last_results = self.pipeline.predict_faces(image)
                self.error_message = None

                # Debounced event logging. This is not identity tracking:
                # a person continuously visible may be logged again later.
                current_signatures = set()
                for person in self.last_results:
                    signature = (
                        int(person["age"]) // 5,
                        str(person["gender"]),
                        bool(person["senior"]),
                    )
                    current_signatures.add(signature)

                    last_logged = self.last_logged_at.get(signature, 0.0)
                    if now - last_logged >= LIVE_LOG_COOLDOWN_SECONDS:
                        save_logs([person])
                        self.last_logged_at[signature] = now

                # Drop stale signatures so this dictionary stays small.
                self.last_logged_at = {
                    signature: timestamp
                    for signature, timestamp in self.last_logged_at.items()
                    if now - timestamp < LIVE_LOG_COOLDOWN_SECONDS * 3
                }

            except Exception as exc:
                self.error_message = str(exc)
                self.last_results = []

            self.last_inference_time = now

        annotated = annotate_frame(image, self.last_results)
        return av.VideoFrame.from_ndarray(annotated, format="bgr24")


# -------------------- IMAGE / SNAPSHOT TABS --------------------
tab_upload, tab_snapshot, tab_live = st.tabs(
    ["📤 Upload Image", "📷 Take Photo", "🎥 Live Webcam"]
)

with tab_upload:
    uploaded_file = st.file_uploader(
        "Upload an image containing one or more people",
        type=["jpg", "jpeg", "png"],
        key="senior_image_uploader",
    )

with tab_snapshot:
    camera_file = st.camera_input(
        "Capture a photo using your camera",
        key="senior_camera_snapshot",
    )

with tab_live:
    st.subheader("Live Multi-Face Detection")
    st.write(
        "Allow camera access in your browser. The live feed labels each "
        "detected face and writes debounced detection events to the CSV log."
    )
    st.info(
        "Live logging is event-based, not person identification. "
        "It cannot tell whether a person seen later is the same individual."
    )

    webrtc_context = webrtc_streamer(
        key="senior-citizen-live-webcam",
        video_processor_factory=SeniorCitizenVideoProcessor,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    if webrtc_context.state.playing:
        st.success("Live webcam is running.")
        if webrtc_context.video_processor is not None:
            processor_error = webrtc_context.video_processor.error_message
            if processor_error:
                st.error(f"Video processing error: {processor_error}")


# -------------------- UPLOADED IMAGE / SNAPSHOT PROCESSING --------------------
image_file = uploaded_file if uploaded_file is not None else camera_file

if image_file is not None:
    try:
        image = Image.open(image_file).convert("RGB")
        rgb_image = np.array(image)
        frame_bgr = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

        with st.spinner("Detecting faces and estimating attributes..."):
            results = pipeline.predict_faces(frame_bgr)

        annotated_bgr = annotate_frame(frame_bgr, results)
        annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original Image")
            st.image(rgb_image, use_container_width=True)
        with col2:
            st.subheader("Detection Results")
            st.image(annotated_rgb, use_container_width=True)

        if results:
            st.success(f"Detected {len(results)} face(s).")
            result_df = pd.DataFrame(
                [
                    {
                        "Estimated Age": person["age"],
                        "Predicted Gender": person["gender"],
                        "Status": (
                            "Senior Citizen"
                            if person["senior"]
                            else "Non-Senior"
                        ),
                    }
                    for person in results
                ]
            )
            st.dataframe(result_df, use_container_width=True)

            image_bytes = image_file.getvalue()
            # Stable hash within this running Python process is sufficient
            # to prevent duplicate logging caused by ordinary Streamlit reruns.
            image_key = str(hash(image_bytes))
            if st.session_state.get("last_logged_image") != image_key:
                save_logs(results)
                st.session_state["last_logged_image"] = image_key
                st.success("Image detections saved to the CSV log.")
        else:
            st.warning("No faces detected. Try a clear, front-facing image.")

    except Exception as exc:
        st.error(f"Could not process this image: {exc}")


# -------------------- VISITOR LOG --------------------
st.divider()
st.subheader("📋 Detection Log")

if LOG_FILE.exists() and LOG_FILE.stat().st_size > 0:
    try:
        with CSV_LOCK:
            logs_df = pd.read_csv(LOG_FILE)

        st.dataframe(logs_df.tail(100), use_container_width=True)
        st.download_button(
            "Download CSV Log",
            data=LOG_FILE.read_bytes(),
            file_name="visitor_log.csv",
            mime="text/csv",
            key="download_visitor_log",
        )
    except Exception as exc:
        st.warning(f"Could not read the log file: {exc}")
else:
    st.write("No detections have been logged yet.")

st.caption(
    "Age and gender are model estimates, not verified facts. "
    "The senior-status threshold is configured in pipeline.py. "
    "Use an appropriate privacy notice and obtain consent before using "
    "a camera to monitor people."
)
'''

