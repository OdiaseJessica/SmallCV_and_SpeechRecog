import cv2
import mediapipe as mp
import numpy as np
import sounddevice as sd
import librosa
import queue
import threading

# -------------------------
# AUDIO SETUP
# -------------------------
audio_queue = queue.Queue()

def audio_callback(indata, frames, time, status):
    audio_queue.put(indata.copy())

def process_audio():
        if not audio_queue.empty():
            data = audio_queue.get()
            y = data.flatten()

            # Extract basic features
            energy = np.mean(y**2)

            try:
                pitch = np.mean(librosa.yin(y, fmin=50, fmax=300))
            except:
                pitch = 0

            # Simple mood heuristic
            if energy > 0.01 and pitch > 150:
                mood = "Energetic"
            elif energy < 0.005:
                mood = "Calm"
            else:
                mood = "Neutral"

            print(f"[AUDIO] Energy: {energy:.4f}, Pitch: {pitch:.2f} → {mood}")

# Start audio stream
stream = sd.InputStream(callback=audio_callback, channels=1, samplerate=22050)
stream.start()

audio_thread = threading.Thread(target=process_audio, daemon=True)
audio_thread.start()

# -------------------------
# VIDEO + FACE EMOTION
# -------------------------
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh()

cap = cv2.VideoCapture(0)

def fake_emotion_classifier(landmarks):
    # Placeholder logic (replace with ML model later)
    mouth_open = landmarks[13].y - landmarks[14].y
    eyebrow_raise = landmarks[65].y - landmarks[159].y

    if mouth_open > 0.02:
        return "Surprised / Engaged"
    elif eyebrow_raise > 0.01:
        return "Focused"
    else:
        return "Neutral"

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    emotion = "No face"

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            landmarks = face_landmarks.landmark
            emotion = fake_emotion_classifier(landmarks)

            # Draw a few points for visualization
            for lm in landmarks[:50]:
                h, w, _ = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 1, (0, 255, 0), -1)

    cv2.putText(frame, f"Emotion: {emotion}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Mood Tracker", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()