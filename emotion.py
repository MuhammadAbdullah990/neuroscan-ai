import cv2
import mediapipe as mp
from deepface import DeepFace
import threading

# --- SETUP ---
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

cap = cv2.VideoCapture(0)

# Variables to hold the current emotion
current_emotion = "Analyzing..."
emotion_color = (0, 255, 255) # Yellow

# This function runs in the background
def analyze_emotion(face_img):
    global current_emotion, emotion_color
    try:
        # DeepFace analysis
        result = DeepFace.analyze(face_img, actions=['emotion'], enforce_detection=False)
        emotion = result[0]['dominant_emotion']
        current_emotion = emotion.upper()
        
        # Color coding logic
        if emotion in ['sad', 'fear', 'angry']:
            emotion_color = (0, 0, 255) # Red (Risk)
        elif emotion in ['happy']:
            emotion_color = (0, 255, 0) # Green (Good)
        else:
            emotion_color = (255, 0, 0) # Blue (Neutral)
            
    except Exception as e:
        pass # Ignore errors during thread

# --- MAIN LOOP ---
frame_counter = 0

print("Starting AI... Press ESC to quit.")

with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as face_mesh:

    while True:
        success, image = cap.read()
        if not success: break

        # 1. Processing
        image.flags.writeable = False
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image_rgb)
        image.flags.writeable = True

        # 2. Draw the "Face Net" (The part you missed!)
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                
                # Draw the mesh (The net)
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())

                # Draw the contours (Eyes/Lips outlines)
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())

                # 3. Emotion Logic (Runs every 30 frames)
                if frame_counter % 30 == 0:
                    threading.Thread(target=analyze_emotion, args=(image,)).start()

        # 4. Display Status
        # Draw a semi-transparent header
        cv2.rectangle(image, (0, 0), (400, 50), (0, 0, 0), -1)
        cv2.putText(image, f"Detected: {current_emotion}", (10, 35), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, emotion_color, 2)

        cv2.imshow('NeuroScan - Day 2 Complete', image)
        frame_counter += 1

        if cv2.waitKey(5) & 0xFF == 27: # ESC to exit
            break

cap.release()
cv2.destroyAllWindows()