import cv2
import mediapipe as mp
import numpy as np
import math

# Initialize MediaPipe Pose.
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Function to draw landmarks on the image.
def draw_landmarks(image, landmarks, connections):
    for connection in connections:
        start_idx, end_idx = connection
        if landmarks[start_idx].visibility > 0.5 and landmarks[end_idx].visibility > 0.5:
            start_point = (int(landmarks[start_idx].x * image.shape[1]), int(landmarks[start_idx].y * image.shape[0]))
            end_point = (int(landmarks[end_idx].x * image.shape[1]), int(landmarks[end_idx].y * image.shape[0]))
            cv2.line(image, start_point, end_point, (0, 255, 0), 2)
            cv2.circle(image, start_point, 5, (0, 0, 255), -1)
            cv2.circle(image, end_point, 5, (0, 0, 255), -1)

# Function to calculate the Euclidean distance between two points.
def euclidean_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

# Function to calculate angle between three points.
def calculate_angle(point1, point2, point3):
    a = np.array(point1)
    b = np.array(point2)
    c = np.array(point3)
    
    ba = a - b
    bc = c - b
    
    angle = np.arccos(np.clip(np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc)), -1.0, 1.0))
    return np.degrees(angle)

# Function to calculate REBA score.
def calculate_reba_score(neck_angle, trunk_angle, upper_arm_angle, lower_arm_angle):
    score = 0
    if neck_angle < 20:
        score += 1
    elif neck_angle >20:
        score += 2
    if trunk_angle < 20:
        score += 2
    elif trunk_angle > 20:
        score += 3
    elif trunk_angle > 60:
        score += 4
    if upper_arm_angle < 20:
        score += 2
    elif 90>upper_arm_angle > 45:
        score += 3
    elif upper_arm_angle > 120:
        score += 4
    if lower_arm_angle < 20:
        score += 1
    elif lower_arm_angle > 20:
        score += 2
        
    return score

# Open video capture.
video_path = '/home/wndrsn/Downloads/gettyimages-86201221-640_adpp.mp4'  # Replace with your video path
cap = cv2.VideoCapture(video_path)

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Recolor the frame to RGB.
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Make detection.
        results = pose.process(image_rgb)
        
        # Recolor back to BGR.
        frame = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            # Extract landmarks for neck, trunk, upper arm, and lower arm.
            neck = [
                mp_pose.PoseLandmark.NOSE.value,
                mp_pose.PoseLandmark.LEFT_SHOULDER.value,
                mp_pose.PoseLandmark.RIGHT_SHOULDER.value
            ]
            trunk = [
                mp_pose.PoseLandmark.LEFT_SHOULDER.value,
                mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
                mp_pose.PoseLandmark.LEFT_HIP.value
            ]
            upper_arm = [
                mp_pose.PoseLandmark.LEFT_SHOULDER.value,
                mp_pose.PoseLandmark.LEFT_ELBOW.value,
                mp_pose.PoseLandmark.LEFT_WRIST.value
            ]
            lower_arm = [
                mp_pose.PoseLandmark.LEFT_ELBOW.value,
                mp_pose.PoseLandmark.LEFT_WRIST.value,
                mp_pose.PoseLandmark.LEFT_KNEE.value
            ]
            
            # Calculate angles.
            neck_angle = calculate_angle(
                (landmarks[neck[0]].x, landmarks[neck[0]].y),
                (landmarks[neck[1]].x, landmarks[neck[1]].y),
                (landmarks[neck[2]].x, landmarks[neck[2]].y)
            )
            
            trunk_angle = calculate_angle(
                (landmarks[trunk[0]].x, landmarks[trunk[0]].y),
                (landmarks[trunk[1]].x, landmarks[trunk[1]].y),
                (landmarks[trunk[2]].x, landmarks[trunk[2]].y)
            )
            
            upper_arm_angle = calculate_angle(
                (landmarks[upper_arm[0]].x, landmarks[upper_arm[0]].y),
                (landmarks[upper_arm[1]].x, landmarks[upper_arm[1]].y),
                (landmarks[upper_arm[2]].x, landmarks[upper_arm[2]].y)
            )
            
            lower_arm_angle = calculate_angle(
                (landmarks[lower_arm[0]].x, landmarks[lower_arm[0]].y),
                (landmarks[lower_arm[1]].x, landmarks[lower_arm[1]].y),
                (landmarks[lower_arm[2]].x, landmarks[lower_arm[2]].y)
            )
            
            # Calculate REBA score
            reba_score = calculate_reba_score(neck_angle, trunk_angle, upper_arm_angle, lower_arm_angle)
            
            # Display REBA score
            cv2.putText(frame, f'REBA Score: {reba_score:.2f}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Display the frame.
        cv2.imshow('MediaPipe Pose', frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
