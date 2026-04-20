# ==============================================================================
# SQUAT TRACKER - STREAMLIT CLOUD EDITION
# ==============================================================================
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import cv2
import numpy as np
import mediapipe as mp
import av

# Initialize MediaPipe Pose components
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Setup STUN/TURN servers to ensure video streams work on public networks
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

def calculate_angle(a, b, c):
    """Calculates the angle between three points."""
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
        
    return angle

class SquatVideoProcessor(VideoProcessorBase):
    """
    This class handles the video processing frame-by-frame.
    We use a class (instead of a simple function) so we can remember 
    the squat count and stage between frames!
    """
    def __init__(self):
        self.pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.counter = 0
        self.stage = "UP"

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        # Convert the frame from the webcam into an OpenCV image array
        img = frame.to_ndarray(format="bgr24")
        
        # Recolor image to RGB for MediaPipe
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_rgb.flags.writeable = False
      
        # Make detection
        results = self.pose.process(img_rgb)
    
        # Recolor back to BGR for drawing
        img_rgb.flags.writeable = True
        img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        
        # Extract landmarks and calculate squat logic
        if results.pose_landmarks:
            try:
                landmarks = results.pose_landmarks.landmark
                
                # We'll use the left leg for this example. 
                # Points: 23 (Hip), 25 (Knee), 27 (Ankle)
                hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                
                # Calculate angle
                angle = calculate_angle(hip, knee, ankle)
                
                # Get the pixel coordinates of the knee to draw the text
                h, w, c = img.shape
                knee_pos = tuple(np.multiply(knee, [w, h]).astype(int))
                
                # Visualize angle
                cv2.putText(img, str(int(angle)), knee_pos, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                
                # Squat counter logic
                if angle > 160:
                    self.stage = "UP"
                if angle < 90 and self.stage == 'UP':
                    self.stage = "DOWN"
                    self.counter += 1
            except:
                pass
            
            # Draw MediaPipe skeleton
            mp_drawing.draw_landmarks(
                img, 
                results.pose_landmarks, 
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2), 
                mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
            )
            
        # Setup nice UI banners on the video feed
        cv2.rectangle(img, (0,0), (280, 110), (245, 117, 16), -1)
        
        # Stage data
        cv2.putText(img, 'STAGE', (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(img, self.stage, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3, cv2.LINE_AA)
        
        # Rep data
        cv2.putText(img, 'REPS', (140, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(img, str(self.counter), (135, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3, cv2.LINE_AA)
        
        # Convert the processed OpenCV image back to a WebRTC frame
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- Streamlit UI Layout ---
st.set_page_config(page_title="AI Squat Tracker", page_icon="🏋️")
st.title("🏋️ AI Squat Tracker")
st.write("Make sure your full body is visible to the camera. The tracker monitors your left leg angles!")

# Start the WebRTC streamer
webrtc_streamer(
    key="squat-tracker",
    mode=1, # SENDRECV mode (sends webcam, receives processed video)
    rtc_configuration=RTC_CONFIGURATION,
    video_processor_factory=SquatVideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True
)

st.markdown("---")
st.markdown("Built with Streamlit, WebRTC, and MediaPipe.")
