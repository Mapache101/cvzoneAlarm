# ==============================================================================
# SQUAT TRACKER - STREAMLIT + CVZONE
# ==============================================================================
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import cv2
import numpy as np
import av
import cvzone
from cvzone.PoseModule import PoseDetector

st.set_page_config(page_title="AI Squat Tracker", page_icon="🏋️")
st.title("🏋️ AI Squat Tracker (CVZone)")
st.write("Make sure your full body is visible to the camera. The tracker monitors your right leg angles!")

# Setup STUN servers so the video stream works across different internet networks
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

def calculate_angle(a, b, c):
    """Calculates the angle between three points."""
    a = np.array(a) 
    b = np.array(b) 
    c = np.array(c) 
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
        
    return angle

class SquatVideoProcessor(VideoProcessorBase):
    def __init__(self):
        # Initialize cvzone PoseDetector inside the class so it persists
        self.detector = PoseDetector()
        self.counter = 0
        self.stage = "UP"

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        # Convert the incoming webcam frame to an OpenCV format
        img = frame.to_ndarray(format="bgr24")
        
        # --- CVZONE PROCESSING ---
        # Automatically draws the skeleton on the image
        img = self.detector.findPose(img)
        
        # Extract position data without drawing redundant circles
        position_data = self.detector.findPosition(img, draw=False)
        
        # Handle the data depending on cvzone's return structure
        if position_data:
            lmList = position_data[0] if isinstance(position_data, tuple) else position_data
            
            # Check if we have detected the full body (at least 28 landmarks)
            if len(lmList) > 28:
                # Use Right leg points: Hip (24), Knee (26), Ankle (28)
                hip = lmList[24][:2]
                knee = lmList[26][:2]
                ankle = lmList[28][:2]
                
                # Calculate Knee Angle
                angle = calculate_angle(hip, knee, ankle)
                
                # Draw angle near the knee
                cv2.putText(img, str(int(angle)), tuple(knee), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
                
                # Squat Counting Logic
                if angle > 160:
                    self.stage = "UP"
                if angle < 90 and self.stage == 'UP':
                    self.stage = "DOWN"
                    self.counter += 1

        # Draw nice UI banners using cvzone's built-in putTextRect
        cvzone.putTextRect(img, f'SQUATS: {self.counter}', (20, 50), scale=2, thickness=2, colorR=(0, 200, 0), offset=10)
        cvzone.putTextRect(img, f'STAGE: {self.stage}', (20, 110), scale=2, thickness=2, colorR=(200, 0, 0), offset=10)
        
        # Convert processed image back to WebRTC frame
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# Start the WebRTC streamer
webrtc_streamer(
    key="squat-tracker",
    mode=1, # SENDRECV mode
    rtc_configuration=RTC_CONFIGURATION,
    video_processor_factory=SquatVideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True
)
