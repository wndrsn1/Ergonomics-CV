import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Pose.
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Function to draw landmarks on the image.
def draw_landmarks(image, landmarks, connections):
    for connection in connections:
        start_idx = connection[0]
        end_idx = connection[1]

        if landmarks[start_idx].visibility > 0.5 and landmarks[end_idx].visibility > 0.5:
            start_point = (int(landmarks[start_idx].x * image.shape[1]),
                           int(landmarks[start_idx].y * image.shape[0]))
            end_point = (int(landmarks[end_idx].x * image.shape[1]),
                         int(landmarks[end_idx].y * image.shape[0]))

            cv2.line(image, start_point, end_point, (0, 255, 0), 2)
            cv2.circle(image, start_point, 5, (0, 0, 255), -1)
            cv2.circle(image, end_point, 5, (0, 0, 255), -1)

# Function to calculate the Euclidean distance between two points.
def euclidean_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

# Open video capture.
video_path = '/home/wndrsn/Downloads/gettyimages-86201221-640_adpp.mp4'  # Replace with your video path
cap = cv2.VideoCapture(video_path)

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    prev_distance = None
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
            
            # Extract landmarks for the back and legs.
            back_landmarks = [
                mp_pose.PoseLandmark.LEFT_SHOULDER.value,
                mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
                mp_pose.PoseLandmark.LEFT_HIP.value,
                mp_pose.PoseLandmark.RIGHT_HIP.value
            ]
            
            leg_landmarks = [
                mp_pose.PoseLandmark.LEFT_HIP.value,
                mp_pose.PoseLandmark.RIGHT_HIP.value,
                mp_pose.PoseLandmark.LEFT_KNEE.value,
                mp_pose.PoseLandmark.RIGHT_KNEE.value,
                mp_pose.PoseLandmark.LEFT_ANKLE.value,
                mp_pose.PoseLandmark.RIGHT_ANKLE.value
            ]
            
            # Calculate the distance between shoulders and hips.
            left_shoulder = (landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y)
            right_shoulder = (landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y)
            left_hip = (landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y)
            right_hip = (landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y)
            
            shoulder_distance = euclidean_distance(left_shoulder, right_shoulder)
            hip_distance = euclidean_distance(left_hip, right_hip)
            total_distance = shoulder_distance + hip_distance
            
            # Calculate the percentage change in distance.
            if prev_distance is not None:
                distance_change = prev_distance - total_distance
                percentage_change = (distance_change / prev_distance) * 100 if prev_distance != 0 else 0
                
                # Display change in distance.
                cv2.putText(frame, f'Distance Change: {distance_change:.2f}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                
                # Check if the back is rounding.
                if distance_change >= 0.01:  # Adjust the threshold as needed.
                    cv2.putText(frame, 'Rounding Detected', (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            
            # Update previous distance.
            prev_distance = total_distance
            
            # Draw back landmarks and connections.
            back_connections = [(back_landmarks[i], back_landmarks[j]) for i in range(len(back_landmarks)) for j in range(i+1, len(back_landmarks))]
            draw_landmarks(frame, landmarks, back_connections)
            
            # Draw leg landmarks and connections.
            leg_connections = [(leg_landmarks[i], leg_landmarks[j]) for i in range(len(leg_landmarks)) for j in range(i+1, len(leg_landmarks))]
            draw_landmarks(frame, landmarks, leg_connections)
        
        # Display the frame.
        cv2.imshow('MediaPipe Pose', frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
