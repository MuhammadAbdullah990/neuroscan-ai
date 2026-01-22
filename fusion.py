import cv2
import mediapipe as mp
from deepface import DeepFace
import threading
import time
import numpy as np
from collections import Counter

# --- 1. CONFIGURATION ---
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

# --- TUNING (HYPERSENSITIVE MODE) ---
BLINK_THRESHOLD = 0.20
BLINK_CONSEC_FRAMES = 1

# SENSITIVITY SETTINGS (Lower = Easier to trigger)
SAD_THRESHOLD = 5.0     # Trigger if Sadness is > 5% (Very sensitive)
FEAR_THRESHOLD = 10.0   # Trigger if Fear is > 10%
ANGRY_THRESHOLD = 15.0  # Trigger if Anger is > 15%

# --- 2. VARIABLES ---
current_emotion = "Neutral"
emotion_details = "" 
emotion_color = (255, 0, 0)
blink_count = 0
frame_blink_counter = 0
emotion_history = []  
start_time = time.time()
processing_active = False

# --- 3. HELPER FUNCTIONS ---
def distance(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    return ((x2 - x1)**2 + (y2 - y1)**2)**0.5

def calculate_ear(landmarks, indices, w, h):
    coords = []
    for idx in indices:
        lm = landmarks[idx]
        coords.append((int(lm.x * w), int(lm.y * h)))
    v1 = distance(coords[1], coords[5])
    v2 = distance(coords[2], coords[4])
    horizontal = distance(coords[0], coords[3])
    return (v1 + v2) / (2.0 * horizontal), coords

# --- 4. THE HYPERSENSITIVE ENGINE ---
def analyze_emotion_thread(cropped_face):
    global current_emotion, emotion_color, emotion_history, processing_active, emotion_details
    try:
        # STEP A: ENHANCE IMAGE (New!)
        # This increases contrast so the AI sees facial lines better
        face_norm = cv2.normalize(cropped_face, None, 0, 255, cv2.NORM_MINMAX)

        # STEP B: ANALYZE
        result = DeepFace.analyze(face_norm, actions=['emotion'], enforce_detection=False)
        emotions = result[0]['emotion']
        
        # STEP C: AGGRESSIVE LOGIC
        dominant = "Neutral"
        score = 0
        
        # Check Negative Emotions with LOW thresholds
        if emotions['sad'] > SAD_THRESHOLD:
            dominant = "SAD"
            score = emotions['sad']
        elif emotions['fear'] > FEAR_THRESHOLD:
            dominant = "FEAR"
            score = emotions['fear']
        elif emotions['angry'] > ANGRY_THRESHOLD:
            dominant = "ANGRY"
            score = emotions['angry']
        elif emotions['happy'] > 40: # Happy still needs to be genuine
            dominant = "HAPPY"
            score = emotions['happy']
        else:
            dominant = "NEUTRAL"
            score = emotions['neutral']

        current_emotion = dominant
        emotion_details = f"{int(score)}%" 
        emotion_history.append(dominant)
        
        # Color Map
        if dominant == 'SAD': emotion_color = (0, 0, 255)       
        elif dominant == 'FEAR': emotion_color = (128, 0, 128)  
        elif dominant == 'ANGRY': emotion_color = (0, 0, 139)   
        elif dominant == 'HAPPY': emotion_color = (0, 255, 0)   
        elif dominant == 'NEUTRAL': emotion_color = (255, 255, 0)  
        else: emotion_color = (255, 255, 255) 
            
    except Exception:
        pass
    finally:
        processing_active = False

# --- 5. MAIN LOOP ---
mp_face_mesh = mp.solutions.face_mesh
cap = cv2.VideoCapture(0)

print("--- NEUROSCAN HYPERSENSITIVE MODE ---")
print(f"Sadness Threshold: > {SAD_THRESHOLD}% (Very Low)")

with mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True) as face_mesh:
    
    while True:
        success, image = cap.read()
        if not success: break
        
        h, w, c = image.shape
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image_rgb)

        avg_ear = 0.0
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                landmarks = face_landmarks.landmark
                
                # --- SMART CROP ---
                x_min, y_min = w, h
                x_max, y_max = 0, 0
                for lm in landmarks:
                    x, y = int(lm.x * w), int(lm.y * h)
                    if x < x_min: x_min = x
                    if y < y_min: y_min = y
                    if x > x_max: x_max = x
                    if y > y_max: y_max = y
                
                x_min = max(0, x_min - 20)
                y_min = max(0, y_min - 40) 
                x_max = min(w, x_max + 20)
                y_max = min(h, y_max + 20)
                
                try:
                    face_roi = image[y_min:y_max, x_min:x_max]
                    if not processing_active and face_roi.size > 0:
                        processing_active = True
                        threading.Thread(target=analyze_emotion_thread, args=(face_roi,)).start()
                except:
                    pass

                # --- GEOMETRIC SENSOR ---
                left_ear, l_coords = calculate_ear(landmarks, LEFT_EYE, w, h)
                right_ear, r_coords = calculate_ear(landmarks, RIGHT_EYE, w, h)
                avg_ear = (left_ear + right_ear) / 2.0
                
                cv2.polylines(image, [np.array(l_coords)], True, (0, 255, 0), 1)
                cv2.polylines(image, [np.array(r_coords)], True, (0, 255, 0), 1)

                if avg_ear < BLINK_THRESHOLD:
                    frame_blink_counter += 1
                else:
                    if frame_blink_counter >= BLINK_CONSEC_FRAMES:
                        blink_count += 1
                    frame_blink_counter = 0

        # --- LIVE DISPLAY ---
        elapsed_time = int(time.time() - start_time)
        bpm = int(blink_count / (elapsed_time / 60)) if elapsed_time > 5 else 0

        cv2.rectangle(image, (0, 0), (350, 160), (0, 0, 0), -1)
        
        cv2.putText(image, f"Status: {current_emotion} {emotion_details}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, emotion_color, 2)
        cv2.putText(image, f"Blinks: {blink_count} | BPM: {bpm}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(image, f"EAR: {avg_ear:.3f}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.putText(image, f"Samples: {len(emotion_history)}", (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)

        cv2.imshow('NeuroScan - Hypersensitive', image)

        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()

# --- DETAILED REPORT ---
print("\n" + "="*60)
print("NEUROSCAN FINAL DIAGNOSTIC REPORT")
print("="*60)

total_samples = len(emotion_history)
if total_samples == 0: total_samples = 1
counts = Counter(emotion_history)
def get_pct(emotion): return (counts[emotion] / total_samples) * 100

print(f"Session Duration: {elapsed_time} seconds")
print(f"Total Data Points: {total_samples}")
print("-" * 30)
print("EMOTIONAL SPECTRUM ANALYSIS:")
print(f" > NEUTRAL:  {get_pct('NEUTRAL'):.1f}%")
print(f" > HAPPY:    {get_pct('HAPPY'):.1f}%")
print(f" > SAD:      {get_pct('SAD'):.1f}%")
print(f" > FEAR:     {get_pct('FEAR'):.1f}%")
print(f" > ANGRY:    {get_pct('ANGRY'):.1f}%")
print("-" * 30)
print(f" > Blink Rate: {bpm} BPM")
print("="*60)

diagnosis = []

# REPORT LOGIC (Matched to Hypersensitivity)
if get_pct('SAD') > 15: # Lowered from 25 to 15
    diagnosis.append("HIGH RISK: Major Depressive Disorder Indicators")
elif get_pct('SAD') + get_pct('NEUTRAL') > 85:
    diagnosis.append("MODERATE RISK: Possible Depressive State (Flat Affect)")

if get_pct('FEAR') > 10: # Lowered from 15 to 10
    diagnosis.append("HIGH RISK: Generalized Anxiety Symptoms")
if bpm > 30:
    diagnosis.append("MODERATE RISK: Panic/High Arousal State")
if get_pct('ANGRY') > 15:
    diagnosis.append("MODERATE RISK: High Stress / Irritability")

if not diagnosis:
    diagnosis.append("LOW RISK: Normal Emotional Baseline")

for d in diagnosis:
    print(f" -> {d}")
print("="*60)