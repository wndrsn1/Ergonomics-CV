import cv2
import mediapipe as mp
import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QLineEdit, QPushButton, QFileDialog

class InputDialog(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.layout = QVBoxLayout()

        self.height_label = QLabel('Enter your height (in cm):', self)
        self.layout.addWidget(self.height_label)

        self.height_input = QLineEdit(self)
        self.layout.addWidget(self.height_input)

        self.weight_label = QLabel('Enter your weight (in kg):', self)
        self.layout.addWidget(self.weight_label)

        self.weight_input = QLineEdit(self)
        self.layout.addWidget(self.weight_input)

        self.video_path_label = QLabel('Select video file:', self)
        self.layout.addWidget(self.video_path_label)

        self.video_path_button = QPushButton('Browse', self)
        self.video_path_button.clicked.connect(self.browse_file)
        self.layout.addWidget(self.video_path_button)

        self.submit_button = QPushButton('Submit', self)
        self.submit_button.clicked.connect(self.submit)
        self.layout.addWidget(self.submit_button)

        self.setLayout(self.layout)

    def browse_file(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        file_path, _ = QFileDialog.getOpenFileName(self, "Select video file", "", "MP4 files (*.mp4);;All files (*)", options=options)
        if file_path:
            self.video_path = file_path
            self.video_path_label.setText(f'Selected: {file_path}')

    def submit(self):
        self.height = float(self.height_input.text())
        self.weight = float(self.weight_input.text())
        self.close()

    def closeEvent(self, event):
        
        QApplication.instance().quit()

def get_height_weight_video_path():
    app = QApplication([])
    dialog = InputDialog()
    dialog.show()
    app.exec_()
    return dialog.height, dialog.weight, dialog.video_path

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
def calculate_reba_score(angles):
    neck_angle, trunk_angle, upper_arm_angle, lower_arm_angle, leg_angle = angles
    score = 0
    
    # Score calculation based on angles
    if neck_angle <= 20:
        score += 1
    elif neck_angle <= 30:
        score += 2
    elif neck_angle <= 45:
        score += 3
    else:
        score += 4
    
    if trunk_angle <= 20:
        score += 2
    elif trunk_angle <= 45:
        score += 3
    elif trunk_angle <= 60:
        score += 4
    else:
        score += 5
    
    if upper_arm_angle <= 20:
        score += 2
    elif upper_arm_angle <= 45:
        score += 3
    elif upper_arm_angle <= 90:
        score += 4
    else:
        score += 5
    
    if lower_arm_angle < 20:
        score += 1
    elif lower_arm_angle <= 45:
        score += 2
    elif lower_arm_angle <= 90:
        score += 3
    else:
        score += 4

    if 60 > leg_angle > 30:
        score += 1
    elif 30 < leg_angle <= 60:
        score += 2

    return score

def main():
    height, weight, video_path = get_height_weight_video_path()

    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    output_path = 'reba_output.mp4'  # Replace with desired output path

    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for mp4 format
    fps = cap.get(cv2.CAP_PROP_FPS)  # Frames per second of the video
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Create VideoWriter object
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
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
                
                # Extract landmarks for neck, trunk, upper arm, lower arm, and legs.
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
                leg = [
                    mp_pose.PoseLandmark.LEFT_HIP.value,
                    mp_pose.PoseLandmark.LEFT_KNEE.value,
                    mp_pose.PoseLandmark.LEFT_ANKLE.value
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
                
                leg_angle = calculate_angle(
                    (landmarks[leg[0]].x, landmarks[leg[0]].y),
                    (landmarks[leg[1]].x, landmarks[leg[1]].y),
                    (landmarks[leg[2]].x, landmarks[leg[2]].y)
                )
                
                angles = (neck_angle, trunk_angle, upper_arm_angle, lower_arm_angle, leg_angle)
                reba_score = calculate_reba_score(angles)
                
                # Draw landmarks on the frame.
                draw_landmarks(frame, landmarks, mp_pose.POSE_CONNECTIONS)
                
                # Display REBA score on the frame.
                cv2.putText(frame, f'REBA Score: {reba_score}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            
            # Write the frame to the output video.
            out.write(frame)
            
            # Display the frame.
            cv2.imshow('REBA Assessment', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    # Release the video capture and writer objects.
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
    