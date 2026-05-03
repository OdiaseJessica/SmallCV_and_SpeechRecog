import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions
from mediapipe.tasks.python.core.base_options import BaseOptions

def run_hand_tracking():
    cap = cv2.VideoCapture(0)

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path="hand_landmarker.task"),
        num_hands=2
    )

    with HandLandmarker.create_from_options(options) as landmarker:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            result = landmarker.detect(mp_image)
            
            if result.hand_landmarks:
                for hand_landmarks in result.hand_landmarks:
                    for landmark in hand_landmarks: 
                        h, w, _ = frame.shape
                        cx, cy = int(landmark.x * w), int(landmark.y * h)

                    # draw points
                    cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
                    pass

            # (optional: draw later if you want)

            cv2.imshow("Hand Tracking", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()



if __name__ == "__main__":
    run_hand_tracking()