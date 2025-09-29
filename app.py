import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
from ultralytics import YOLO
import cv2

import os
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"

model = YOLO("models/best.pt")

st.title("Deteksi Objek Realtime dengan Kamera")

class VideoProcessor(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        results = model(img, conf=0.5)

        annotated_frame = results[0].plot()
        
        return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

webrtc_streamer(
    key="example",
    video_processor_factory=VideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
)
