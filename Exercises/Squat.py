import cv2
import mediapipe as mp
import numpy as np
import threading
from playsound import playsound  # Importing playsound

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Function to calculate the angle between three points
def calculate_angle(a, b, c):
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle

# Function to play audio feedback
def play_audio_feedback():
    playsound("/Users/Aditya/Minor Project/Program Files/police-short-whistle-615.wav")  # Replace with the path to your audio file

# Squat posture correction
def squat_correction(frame):
    with mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)

        if results.pose_landmarks:
            # Get landmarks
            landmarks = results.pose_landmarks.landmark

            # Define key points for squat
            hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

            # Calculate angle
            angle = calculate_angle(hip, knee, ankle)

            # Feedback
            if angle < 90:
                cv2.putText(frame, "Keep your back straight and knees aligned.", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                threading.Thread(target=play_audio_feedback).start()  # Play audio feedback in a new thread
            elif angle >= 90 and angle <= 180:
                cv2.putText(frame, "Good squat posture!", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            # Draw landmarks
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    return frame

# Example usage
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = squat_correction(frame)
    cv2.imshow('Squat Posture Correction', frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


