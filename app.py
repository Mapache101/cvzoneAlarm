import streamlit as st
from streamlit_webrtc import webrtc_streamer, RTCConfiguration

st.set_page_config(page_title="Webcam Test", page_icon="🎥")

st.title("🎥 Basic Webcam Test")
st.write("If you can see yourself below, the camera connection is working perfectly!")

# Setup STUN servers so the video stream works across different internet networks
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# Start the simple webcam streamer (No AI processing yet)
webrtc_streamer(
    key="basic-webcam",
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False} # Video only
)
