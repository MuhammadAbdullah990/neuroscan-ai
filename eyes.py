import cv2
import mediapipe as mp
import numpy as np

# --- 1. SETUP ---
mp_face_mesh = mp.solutions.face_mesh
cap = cv2.VideoCapture(0)

# These are the specific Landmark points for the eyes in MediaPipe
# (Don't worry about memorizing these numbers, they are standard)
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

# --- 2. MATH FUNCTIONS ---
def distance(p1, p2):
    '''Calculate distance between two points'''
    x1, y1 = p1
    x2, y2 = p2
    return ((x2 - x1)**2 + (y2 - y1)**2)**0.5

def calculate_ear(landmarks, indices, w, h):
    '''Calculate Eye Aspect Ratio (EAR)'''
    # Extract the 6 coordinates for the eye
    coords = []
    for idx in indices:
        lm = landmarks[idx]
        coords.append((int(lm.x * w), int(lm.y * h)))

    # Vertical lines (Height of eye)
    v1 = distance(coords[1], coords[5])
    v2 = distance(coords[2], coords[4])
    
    # Horizontal line (Width of eye)
    horizontal = distance(coords[0], coords[3])

    # The EAR Formula
    ear = (v1 + v2) / (2.0 * horizontal)
    return ear, coords

# --- 3. VARIABLES ---
blink_count = 0
is_blinking = False

print("Starting Eye Tracker... Press ESC to exit.")

with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as face_mesh:

    while True:
        success, image = cap.read()
        if not success: break

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image_rgb)
        h, w, c = image.shape

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Get the list of all 468 landmarks
                landmarks = face_landmarks.landmark

                # Calculate EAR for both eyes
                left_ear, left_coords = calculate_ear(landmarks, LEFT_EYE, w, h)
                right_ear, right_coords = calculate_ear(landmarks, RIGHT_EYE, w, h)

                # Average the two eyes
                avg_ear = (left_ear + right_ear) / 2.0

                # --- 4. BLINK LOGIC ---
                # Threshold: If EAR drops below 0.25, it's a blink
                if avg_ear < 0.25:
                    if not is_blinking:
                        blink_count += 1
                        is_blinking = True
                        print(f"Blink detected! Total: {blink_count}")
                else:
                    is_blinking = False

                # --- 5. DRAWING ---
                # Draw green lines around the eyes
                cv2.polylines(image, [np.array(left_coords)], True, (0, 255, 0), 1)
                cv2.polylines(image, [np.array(right_coords)], True, (0, 255, 0), 1)

                # Display the data
                cv2.putText(image, f"Blinks: {blink_count}", (30, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                cv2.putText(image, f"EAR: {avg_ear:.2f}", (30, 90), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                # Visual Alert if closing eyes
                if avg_ear < 0.25:
                    cv2.putText(image, "BLINKING", (200, 200), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

        cv2.imshow('NeuroScan - Day 3 (Eye Tracking)', image)
        
        if cv2.waitKey(5) & 0xFF == 27: # ESC
            break

cap.release()
cv2.destroyAllWindows()