import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
import cv2
import mediapipe as mp
from fer import FER
import numpy as np
import av
import threading
import time
import queue
import pandas as pd
import gc

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="NeuroScan: Behavioral Analysis", layout="wide", page_icon="🧠")

st.markdown("""
    <style>
    .stApp { background-color: #0e1117; }
    .prompt-box {
        background-color: #262730;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #4CAF50;
        font-size: 20px;
        margin-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

if "emotion_log" not in st.session_state: st.session_state["emotion_log"] = []
if "blink_count" not in st.session_state: st.session_state["blink_count"] = 0
if "chart_data" not in st.session_state: st.session_state["chart_data"] = []
if "data_queue" not in st.session_state: st.session_state["data_queue"] = queue.Queue()
data_queue = st.session_state["data_queue"]

# --- 2. LIGHTWEIGHT SYSTEM STARTUP ---
@st.cache_resource
def load_models():
    # 1. Load MediaPipe
    mp_face = mp.solutions.face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)
    
    # 2. Load FER (Lightweight Emotion Engine)
    # mtcnn=False means we use the fast OpenCV detector, not the slow deep learning one
    detector = FER(mtcnn=False) 
    
    return mp_face, detector

with st.spinner('Initializing Cloud-Safe AI Engine...'):
    face_mesh, emotion_detector = load_models()

# --- 3. VIDEO PROCESSOR ---
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

class NeuroProcessor(VideoTransformerBase):
    def __init__(self):
        self.frame_counter = 0
        self.blink_consec_frames = 0
        self.blink_threshold = 0.20
        self.processing = False
        self.last_emotion = "Neutral"
        self.emotion_color = (255, 255, 0)
        
    def calculate_ear(self, landmarks, w, h):
        def dist(p1, p2): return ((p2.x - p1.x)**2 + (p2.y - p1.y)**2)**0.5
        l_p = [landmarks[i] for i in LEFT_EYE]
        l_ear = (dist(l_p[1], l_p[5]) + dist(l_p[2], l_p[4])) / (2.0 * dist(l_p[0], l_p[3]))
        r_p = [landmarks[i] for i in RIGHT_EYE]
        r_ear = (dist(r_p[1], r_p[5]) + dist(r_p[2], r_p[4])) / (2.0 * dist(r_p[0], r_p[3]))
        return (l_ear + r_ear) / 2.0

    def analyze_emotion_thread(self, img_bgr):
        try:
            # FER Analysis
            # This returns a list like [{'box': ..., 'emotions': {'angry': 0.1, 'happy': 0.9...}}]
            analysis = emotion_detector.detect_emotions(img_bgr)
            
            dominant = "Neutral"
            if analysis:
                # Get the first face found
                emotions = analysis[0]['emotions']
                
                # Manual Dominance Logic
                # We check the highest value
                max_score = 0
                for emo, score in emotions.items():
                    if score > max_score:
                        max_score = score
                        dominant = emo.upper()
                
                # Sensitivity Tweaks
                if emotions['sad'] > 0.20: dominant = "SAD"
                if emotions['fear'] > 0.20: dominant = "FEAR"
                if emotions['angry'] > 0.20: dominant = "ANGRY"
            
            self.last_emotion = dominant
            
            # Colors
            if dominant == 'SAD': self.emotion_color = (0, 0, 255) 
            elif dominant == 'FEAR': self.emotion_color = (128, 0, 128) 
            elif dominant == 'ANGRY': self.emotion_color = (0, 0, 139) 
            elif dominant == 'HAPPY': self.emotion_color = (0, 255, 0) 
            else: self.emotion_color = (255, 255, 0) 
            
            data_queue.put({"type": "emotion", "value": dominant})
        except: pass
        finally: self.processing = False

    def recv(self, frame):
        try:
            img = frame.to_ndarray(format="bgr24")
            h, w, c = img.shape
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(img_rgb)

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    ear = self.calculate_ear(face_landmarks.landmark, w, h)
                    if ear < self.blink_threshold: self.blink_consec_frames += 1
                    else:
                        if self.blink_consec_frames >= 1: data_queue.put({"type": "blink"})
                        self.blink_consec_frames = 0
                    
                    # Run every 10 frames (Fast & Light)
                    if self.frame_counter % 10 == 0 and not self.processing:
                        self.processing = True
                        threading.Thread(target=self.analyze_emotion_thread, args=(img.copy(),)).start()

                    cv2.rectangle(img, (0, 0), (350, 50), (0, 0, 0), -1)
                    cv2.putText(img, f"Behavior: {self.last_emotion}", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.emotion_color, 2)
            
            self.frame_counter += 1
            return av.VideoFrame.from_ndarray(img, format="bgr24")
        except Exception:
            return frame

# --- 4. UI LAYOUT ---
st.title("🧠 NeuroScan: Behavioral Risk Assessment")
st.markdown("Automated screening for non-verbal psychopathology markers.")

st.markdown("""
<div class="prompt-box">
    <b>🗣️ Clinical Prompt:</b> <br>
    "Please look at the camera and describe a stressful event from your week, 
    or simply sit quietly and reflect on your current mood for 30 seconds."
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns([2, 1])

with col1:
    ctx = webrtc_streamer(
        key="neuro-scanner",
        video_processor_factory=NeuroProcessor,
        rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True
    )

with col2:
    st.markdown("#### Live Behavioral Metrics")
    emo_metric = st.empty()
    emo_metric.metric("Dominant Expression", "Waiting...")
    blink_metric = st.empty()
    blink_metric.metric("Blink Rate (Arousal)", "0")
    st.markdown("#### Emotional Volatility")
    chart_placeholder = st.empty()

# --- 5. DATA LOOP ---
if ctx.state.playing:
    while True:
        try:
            data = data_queue.get(timeout=0.1)
            
            if data["type"] == "emotion":
                curr_emo = data["value"]
                st.session_state["emotion_log"].append(curr_emo)
                
                display_label = curr_emo
                if curr_emo == "SAD": display_label = "Dysphoria (Sadness)"
                if curr_emo == "NEUTRAL": display_label = "Flat Affect (Neutral)"
                if curr_emo == "FEAR": display_label = "Anxiety (Fear)"
                
                emo_metric.metric("Dominant Expression", display_label)
                
                stress_val = 1 if curr_emo in ["SAD", "FEAR", "ANGRY"] else 0
                st.session_state["chart_data"].append(stress_val)
                if len(st.session_state["chart_data"]) > 50: st.session_state["chart_data"].pop(0)
                chart_placeholder.line_chart(st.session_state["chart_data"])
                
            if data["type"] == "blink":
                st.session_state["blink_count"] += 1
                blink_metric.metric("Blink Rate (Arousal)", st.session_state["blink_count"])
                
        except queue.Empty: continue
        except Exception: break

# --- 6. REPORT ---
st.markdown("---")
if st.button("Reset / Clear Data"):
    st.session_state["emotion_log"] = []
    st.session_state["blink_count"] = 0
    st.session_state["chart_data"] = []
    st.rerun()

if st.button("Generate Behavioral Analysis Report"):
    if len(st.session_state["emotion_log"]) > 0:
        st.subheader("📋 Psychological Risk Assessment")
        
        total = len(st.session_state["emotion_log"])
        # FER returns lowercase usually, ensuring safety
        logs = [e.upper() for e in st.session_state["emotion_log"]]
        
        sad_pct = (logs.count("SAD") / total) * 100
        neutral_pct = (logs.count("NEUTRAL") / total) * 100
        fear_pct = (logs.count("FEAR") / total) * 100
        angry_pct = (logs.count("ANGRY") / total) * 100
        
        observations = []
        risk_level = "Low"
        
        if sad_pct > 20:
            observations.append(f"🔴 **Dysphoria Detected ({sad_pct:.1f}%):** Significant duration of negative affect.")
            risk_level = "Moderate"
        if sad_pct + neutral_pct > 85:
            observations.append("🔴 **Flat Affect / Anhedonia:** Lack of emotional reactivity.")
            risk_level = "High"
        if fear_pct > 15:
            observations.append(f"🟠 **Hyper-Arousal ({fear_pct:.1f}%):** Micro-expressions of fear/worry.")
            if risk_level == "Low": risk_level = "Moderate"
        if st.session_state["blink_count"] > 20:
             observations.append(f"🟠 **High Psychomotor Activity:** Rapid blinking ({st.session_state['blink_count']} blinks).")
        if angry_pct > 15:
             observations.append(f"🟡 **Irritability:** Signs of agitation detected.")
        if risk_level == "Low" and not observations:
            observations.append("🟢 **Euthymic Mood:** Emotional range appears normal.")

        c1, c2 = st.columns([2, 1])
        with c1:
            st.info(f"**Overall Behavioral Risk:** {risk_level}")
            for obs in observations: st.markdown(obs)
        with c2:
            chart_df = pd.DataFrame({
                "Affect": ["Negative", "Anxiety", "Agitation", "Neutral", "Positive"],
                "Duration": [sad_pct, fear_pct, angry_pct, neutral_pct, logs.count("HAPPY")/total*100]
            })
            st.bar_chart(chart_df.set_index("Affect"))
            
        csv = chart_df.to_csv().encode('utf-8')
        st.download_button("📥 Download Clinical Data", csv, "behavior_report.csv", "text/csv")
    else:
        st.warning("No session data. Please Start the camera.")