import cv2
import mediapipe as mp
import numpy as np
from gtts import gTTS
from playsound import playsound
import threading
import os
import time

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Function to calculate the slope between two points
def calculate_slope(a, b):
    return (b[1] - a[1]) / (b[0] - a[0] + 1e-5)

# Global variable to track if feedback is playing
feedback_playing = False
feedback_lock = threading.Lock()

def give_feedback(text):
    global feedback_playing
    with feedback_lock:
        if feedback_playing:
            return
        feedback_playing = True

    # Generate and save the audio feedback
    tts = gTTS(text=text, lang='en')
    audio_file = "feedback.mp3"
    tts.save(audio_file)
    
    def play_audio():
        playsound(audio_file)
        os.remove(audio_file)
        global feedback_playing
        with feedback_lock:
            feedback_playing = False

    # Play the audio in a separate thread
    threading.Thread(target=play_audio).start()

# Open video feed from the camera
cap = cv2.VideoCapture(0)

# Timer parameters
GRACE_PERIOD = 10  # Time in seconds
start_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)

    # Initialize variables for timer text
    timer_text = ""

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        # Extract the coordinates of relevant joints
        shoulder_left = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        shoulder_right = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
        hip_left = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
        hip_right = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
        ankle_left = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
        ankle_right = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

        # Calculate the average positions for symmetry
        shoulder_avg = [(shoulder_left[0] + shoulder_right[0]) / 2, (shoulder_left[1] + shoulder_right[1]) / 2]
        hip_avg = [(hip_left[0] + hip_right[0]) / 2, (hip_left[1] + hip_right[1]) / 2]
        ankle_avg = [(ankle_left[0] + ankle_right[0]) / 2, (ankle_left[1] + ankle_right[1]) / 2]

        # Calculate the slope of the line from shoulders to ankles
        body_slope = calculate_slope(shoulder_avg, ankle_avg)

        # Check for common errors
        error_detected = False

        # Hip sagging: If hips are significantly lower than shoulders and ankles
        if hip_avg[1] > (shoulder_avg[1] + ankle_avg[1]) / 2 + 0.05:
            give_feedback("Raise your hips slightly to align with your shoulders and ankles.")
            error_detected = True
        # Hip piking: If hips are significantly higher than shoulders and ankles
        elif hip_avg[1] < (shoulder_avg[1] + ankle_avg[1]) / 2 - 0.05:
            give_feedback("Lower your hips slightly to align with your shoulders and ankles.")
            error_detected = True

        # Draw the landmarks and connections with colored lines for errors
        color = (0, 0, 255) if error_detected else (0, 255, 0)  # Red for error, green for correct posture

        # Draw lines on the body parts of interest
        height, width, _ = frame.shape
        cv2.line(frame, (int(shoulder_avg[0] * width), int(shoulder_avg[1] * height)),
                 (int(hip_avg[0] * width), int(hip_avg[1] * height)), color, 4)
        cv2.line(frame, (int(hip_avg[0] * width), int(hip_avg[1] * height)),
                 (int(ankle_avg[0] * width), int(ankle_avg[1] * height)), color, 4)

        # Draw the landmarks
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # Update and display the timer
    elapsed_time = time.time() - start_time
    remaining_time = max(0, GRACE_PERIOD - elapsed_time)

    if remaining_time > 0:
        timer_text = f"Get ready: {int(remaining_time)}s"

    # Display the timer text on the frame if the timer is active
    if timer_text:
        cv2.putText(frame, timer_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow('Plank Posture Detection', frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
