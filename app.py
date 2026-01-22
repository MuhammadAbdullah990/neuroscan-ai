import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
import cv2
import mediapipe as mp
from deepface import DeepFace
import numpy as np
import av
import threading
import time
import queue
import pandas as pd

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="NeuroScan: Behavioral Analysis", layout="wide", page_icon="🧠")

# Custom CSS for "Medical" Look
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

@st.cache_resource
def load_mediapipe():
    return mp.solutions.face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)
face_mesh = load_mediapipe()

if "data_queue" not in st.session_state: st.session_state["data_queue"] = queue.Queue()
data_queue = st.session_state["data_queue"]

LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

# --- 2. AI ENGINE (SAME LOGIC) ---
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
            # Pre-processing for better detection
            face_norm = cv2.normalize(img_bgr, None, 0, 255, cv2.NORM_MINMAX)
            obj = DeepFace.analyze(face_norm, actions=['emotion'], enforce_detection=False)
            
            # RAW EMOTIONS
            emotions = obj[0]['emotion']
            
            # BEHAVIOR MAPPING (Logic to catch hidden signals)
            dominant = "Neutral"
            
            # Logic: Even small sadness triggers "Dysphoria"
            if emotions['sad'] > 15: dominant = "SAD" 
            elif emotions['fear'] > 10: dominant = "FEAR"
            elif emotions['angry'] > 15: dominant = "ANGRY"
            elif emotions['happy'] > 50: dominant = "HAPPY"
            else: dominant = "NEUTRAL"
            
            self.last_emotion = dominant
            
            # Map Colors
            if dominant == 'SAD': self.emotion_color = (0, 0, 255) # Red
            elif dominant == 'FEAR': self.emotion_color = (128, 0, 128) # Purple
            elif dominant == 'ANGRY': self.emotion_color = (0, 0, 139) # Dark Red
            elif dominant == 'HAPPY': self.emotion_color = (0, 255, 0) # Green
            else: self.emotion_color = (255, 255, 0) # Yellow
            
            data_queue.put({"type": "emotion", "value": dominant})
        except: pass
        finally: self.processing = False

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        h, w, c = img.shape
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(img_rgb)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                ear = self.calculate_ear(face_landmarks.landmark, w, h)
                
                # Blink Detection
                if ear < self.blink_threshold: self.blink_consec_frames += 1
                else:
                    if self.blink_consec_frames >= 1: data_queue.put({"type": "blink"})
                    self.blink_consec_frames = 0
                
                # Emotion Thread
                if self.frame_counter % 10 == 0 and not self.processing:
                    self.processing = True
                    threading.Thread(target=self.analyze_emotion_thread, args=(img.copy(),)).start()

                # Clean Overlay
                cv2.rectangle(img, (0, 0), (350, 50), (0, 0, 0), -1)
                cv2.putText(img, f"Behavior: {self.last_emotion}", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.emotion_color, 2)

        self.frame_counter += 1
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- 3. UI LAYOUT ---

st.title("🧠 NeuroScan: Behavioral Risk Assessment")
st.markdown("Automated screening for non-verbal psychopathology markers.")

# *** NEW: THE PROMPT BOX ***
# This answers the "Why are they sitting there?" question
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
        media_stream_constraints={"video": True, "audio": False}
    )

with col2:
    st.markdown("#### Live Behavioral Metrics")
    emo_metric = st.empty()
    emo_metric.metric("Dominant Expression", "Waiting...")
    
    blink_metric = st.empty()
    blink_metric.metric("Blink Rate (Arousal)", "0")
    
    st.markdown("#### Emotional Volatility")
    chart_placeholder = st.empty()

# --- 4. DATA LOOP ---
if ctx.state.playing:
    while True:
        try:
            data = data_queue.get(timeout=0.1)
            
            if data["type"] == "emotion":
                curr_emo = data["value"]
                st.session_state["emotion_log"].append(curr_emo)
                
                # REFRAMING: Emotion -> Behavior Label on Screen
                display_label = curr_emo
                if curr_emo == "SAD": display_label = "Dysphoria (Sadness)"
                if curr_emo == "NEUTRAL": display_label = "Flat Affect (Neutral)"
                if curr_emo == "FEAR": display_label = "Anxiety (Fear)"
                
                emo_metric.metric("Dominant Expression", display_label)
                
                # Chart Data (1=Negative, 0=Positive/Neutral)
                stress_val = 1 if curr_emo in ["SAD", "FEAR", "ANGRY"] else 0
                st.session_state["chart_data"].append(stress_val)
                if len(st.session_state["chart_data"]) > 50: st.session_state["chart_data"].pop(0)
                chart_placeholder.line_chart(st.session_state["chart_data"])
                
            if data["type"] == "blink":
                st.session_state["blink_count"] += 1
                blink_metric.metric("Blink Rate (Arousal)", st.session_state["blink_count"])
                
        except queue.Empty: continue
        except Exception: break

# --- 5. BEHAVIORAL REPORT (The "Prediction") ---
st.markdown("---")
if st.button("Generate Behavioral Analysis Report"):
    if len(st.session_state["emotion_log"]) > 0:
        st.subheader("📋 Psychological Risk Assessment")
        
        # Calculate Data
        total = len(st.session_state["emotion_log"])
        sad_pct = (st.session_state["emotion_log"].count("SAD") / total) * 100
        neutral_pct = (st.session_state["emotion_log"].count("NEUTRAL") / total) * 100
        fear_pct = (st.session_state["emotion_log"].count("FEAR") / total) * 100
        angry_pct = (st.session_state["emotion_log"].count("ANGRY") / total) * 100
        happy_pct = (st.session_state["emotion_log"].count("HAPPY") / total) * 100
        
        # --- THE BEHAVIOR TRANSLATION LAYER ---
        # This converts "Emotions" into "Medical Predictions"
        
        observations = []
        risk_level = "Low"
        
        # 1. Anhedonia / Depression Check
        if sad_pct > 20:
            observations.append(f"🔴 **Dysphoria Detected ({sad_pct:.1f}%):** Significant duration of negative affect.")
            risk_level = "Moderate"
        if sad_pct + neutral_pct > 85 and happy_pct < 5:
            observations.append("🔴 **Flat Affect / Anhedonia:** Lack of emotional reactivity (Potential Depressive Symptom).")
            risk_level = "High"

        # 2. Anxiety / Arousal Check
        if fear_pct > 15:
            observations.append(f"🟠 **Hyper-Arousal ({fear_pct:.1f}%):** Micro-expressions of fear/worry detected.")
            if risk_level == "Low": risk_level = "Moderate"
        
        if st.session_state["blink_count"] > 20:
             observations.append(f"🟠 **High Psychomotor Activity:** Rapid blinking ({st.session_state['blink_count']} blinks) suggests situational anxiety or stress.")

        # 3. Stress / Irritability Check
        if angry_pct > 15:
             observations.append(f"🟡 **Irritability:** Signs of agitation detected.")

        # 4. Normal Check
        if risk_level == "Low" and not observations:
            observations.append("🟢 **Euthymic Mood:** Emotional range appears within normal limits.")

        # Display Report
        c1, c2 = st.columns([2, 1])
        with c1:
            st.info(f"**Overall Behavioral Risk:** {risk_level}")
            st.write("### Observed Behavioral Markers:")
            for obs in observations:
                st.markdown(obs)
                
        with c2:
            st.write("**Affect Distribution:**")
            # Create a clean DataFrame for the chart
            chart_df = pd.DataFrame({
                "Affect": ["Negative (Sad)", "Anxiety (Fear)", "Agitation (Angry)", "Neutral", "Positive"],
                "Duration": [sad_pct, fear_pct, angry_pct, neutral_pct, happy_pct]
            })
            st.bar_chart(chart_df.set_index("Affect"))
            
        # Download
        csv = chart_df.to_csv().encode('utf-8')
        st.download_button("📥 Download Clinical Data", csv, "behavior_report.csv", "text/csv")
        
    else:
        st.warning("No session data. Please Start the camera.")