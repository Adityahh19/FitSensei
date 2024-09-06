import cv2
import mediapipe as mp
import threading
import numpy as np
from gtts import gTTS
import playsound
import os

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Global variables for feedback and rep counting
feedback_playing = False
feedback_lock = threading.Lock()
rep_count = 0
stage_left = None
stage_right = None

def preload_audio(text, filename):
    tts = gTTS(text=text, lang='en')
    tts.save(filename)

def play_audio(filename):
    global feedback_playing
    with feedback_lock:
        if feedback_playing:
            return
        feedback_playing = True

    def audio_thread():
        playsound.playsound(filename)
        os.remove(filename)
        with feedback_lock:
            feedback_playing = False

    threading.Thread(target=audio_thread, daemon=True).start()

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    v1 = a - b
    v2 = c - b
    
    angle = np.arccos(np.clip(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)), -1.0, 1.0))
    return np.degrees(angle)

def main():
    global rep_count, stage_left, stage_right

    # Pre-load audio files
    preload_audio("Keep your elbows in line with your shoulders.", "feedback.mp3")
    
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Reduce the resolution for faster processing
        frame = cv2.resize(frame, (640, 480))
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            # Get coordinates for left side
            shoulder_left = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            elbow_left = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            wrist_left = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

            # Get coordinates for right side
            shoulder_right = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            elbow_right = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            wrist_right = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

            # Calculate angles at elbows
            angle_left = calculate_angle(shoulder_left, elbow_left, wrist_left)
            angle_right = calculate_angle(shoulder_right, elbow_right, wrist_right)

            print(f"Left Angle: {angle_left}, Right Angle: {angle_right}")

            # Implement rep counting logic for left arm
            if angle_left > 160:
                stage_left = "down"
            if angle_left < 90 and stage_left == 'down':
                stage_left = "up"
            if angle_left > 160 and stage_left == 'up':
                stage_left = "down"
                rep_count += 1

            # Implement rep counting logic for right arm
            if angle_right > 160:
                stage_right = "down"
            if angle_right < 90 and stage_right == 'down':
                stage_right = "up"
            if angle_right > 160 and stage_right == 'up':
                stage_right = "down"
                rep_count += 1

            # Check for posture correction only during the 'up' phase
            if (stage_left == 'up' and (angle_left < 70 or angle_left > 110)) or (stage_right == 'up' and (angle_right < 70 or angle_right > 110)):
                play_audio("feedback.mp3")
                cv2.putText(frame, "Posture correction needed!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                cv2.putText(frame, "Good posture!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Draw lines for shoulders, elbows, and wrists
            height, width, _ = frame.shape
            shoulder_x_left, shoulder_y_left = int(shoulder_left[0] * width), int(shoulder_left[1] * height)
            elbow_x_left, elbow_y_left = int(elbow_left[0] * width), int(elbow_left[1] * height)
            wrist_x_left, wrist_y_left = int(wrist_left[0] * width), int(wrist_left[1] * height)

            shoulder_x_right, shoulder_y_right = int(shoulder_right[0] * width), int(shoulder_right[1] * height)
            elbow_x_right, elbow_y_right = int(elbow_right[0] * width), int(elbow_right[1] * height)
            wrist_x_right, wrist_y_right = int(wrist_right[0] * width), int(wrist_right[1] * height)

            cv2.line(frame, (shoulder_x_left, shoulder_y_left), (elbow_x_left, elbow_y_left), (0, 255, 0), 2)
            cv2.line(frame, (elbow_x_left, elbow_y_left), (wrist_x_left, wrist_y_left), (0, 255, 0), 2)

            cv2.line(frame, (shoulder_x_right, shoulder_y_right), (elbow_x_right, elbow_y_right), (0, 255, 0), 2)
            cv2.line(frame, (elbow_x_right, elbow_y_right), (wrist_x_right, wrist_y_right), (0, 255, 0), 2)

            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        cv2.putText(frame, f"Reps: {rep_count}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow("Shoulder Press Posture Correction", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
              
