import cv2
import mediapipe as mp
import numpy as np
import math

# Initialize MediaPipe Pose and Drawing
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Function to calculate the angle between two vectors
def get_angle(v1, v2):
    dot = np.dot(v1, v2)
    mod_v1 = np.linalg.norm(v1)
    mod_v2 = np.linalg.norm(v2)
    cos_theta = dot / (mod_v1 * mod_v2)
    theta = math.acos(cos_theta)
    return theta

# Function to calculate the length of a vector
def get_length(v):
    return np.dot(v, v) ** 0.5

# Function to extract parameters for squat posture analysis
def get_squat_params(results):
    if results.pose_landmarks is None:
        return np.zeros((1, 4))

    points = {}
    left_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
    points["LEFT_SHOULDER"] = np.array([left_shoulder.x, left_shoulder.y, left_shoulder.z])
    right_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    points["RIGHT_SHOULDER"] = np.array([right_shoulder.x, right_shoulder.y, right_shoulder.z])
    left_hip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
    points["LEFT_HIP"] = np.array([left_hip.x, left_hip.y, left_hip.z])
    right_hip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]
    points["RIGHT_HIP"] = np.array([right_hip.x, right_hip.y, right_hip.z])
    left_knee = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE]
    points["LEFT_KNEE"] = np.array([left_knee.x, left_knee.y, left_knee.z])
    right_knee = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE]
    points["RIGHT_KNEE"] = np.array([right_knee.x, right_knee.y, right_knee.z])
    left_ankle = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE]
    points["LEFT_ANKLE"] = np.array([left_ankle.x, left_ankle.y, left_ankle.z])
    right_ankle = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE]
    points["RIGHT_ANKLE"] = np.array([right_ankle.x, right_ankle.y, right_ankle.z])

    points["MID_SHOULDER"] = (points["LEFT_SHOULDER"] + points["RIGHT_SHOULDER"]) / 2
    points["MID_HIP"] = (points["LEFT_HIP"] + points["RIGHT_HIP"]) / 2

    # Calculate angles
    theta_neck = get_angle(np.array([0, 0, -1]), points["MID_SHOULDER"] - points["MID_HIP"])
    theta_k1 = get_angle(points["RIGHT_HIP"] - points["RIGHT_KNEE"], points["RIGHT_ANKLE"] - points["RIGHT_KNEE"])
    theta_k2 = get_angle(points["LEFT_HIP"] - points["LEFT_KNEE"], points["LEFT_ANKLE"] - points["LEFT_KNEE"])
    theta_k = (theta_k1 + theta_k2) / 2
    theta_h1 = get_angle(points["RIGHT_KNEE"] - points["RIGHT_HIP"], points["RIGHT_SHOULDER"] - points["RIGHT_HIP"])
    theta_h2 = get_angle(points["LEFT_KNEE"] - points["LEFT_HIP"], points["LEFT_SHOULDER"] - points["LEFT_HIP"])
    theta_h = (theta_h1 + theta_h2) / 2

    # Calculate vertical distance for knee and foot correction
    left_foot_y = (points["LEFT_ANKLE"][1] + points["LEFT_ANKLE"][1] + points["LEFT_ANKLE"][1]) / 3
    right_foot_y = (points["RIGHT_ANKLE"][1] + points["RIGHT_ANKLE"][1] + points["RIGHT_ANKLE"][1]) / 3
    left_ky = points["LEFT_KNEE"][1] - left_foot_y
    right_ky = points["RIGHT_KNEE"][1] - right_foot_y
    ky = (left_ky + right_ky) / 2

    # Combine into parameter array
    params = np.array([theta_neck, theta_k, theta_h, ky])
    return np.round(params, 2)

# Main function for live webcam feed and squat posture correction
def squat_posture_correction():
    cap = cv2.VideoCapture(0)
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            # Convert the BGR image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            # Process the image and get pose landmarks
            results = pose.process(image)

            # Convert the image color back to BGR for OpenCV
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Draw pose landmarks on the image
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Get squat posture parameters
            squat_params = get_squat_params(results)

            # Check if the parameters were correctly calculated
            if len(squat_params) == 4:
                # Display parameters on the image
                cv2.putText(image, f'Neck Angle: {squat_params[0]}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                cv2.putText(image, f'Knee Angle: {squat_params[1]}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                cv2.putText(image, f'Hip Angle: {squat_params[2]}', (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                cv2.putText(image, f'Knee-Y Distance: {squat_params[3]}', (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            # Show the image in a window
            cv2.imshow('Squat Posture Correction', image)

            if cv2.waitKey(5) & 0xFF == 27:
                break

    cap.release()
    cv2.destroyAllWindows()

# Run the squat posture correction function
if __name__ == "__main__":
    squat_posture_correction()
