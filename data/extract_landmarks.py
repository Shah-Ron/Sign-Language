import os
import cv2
import numpy as np
import mediapipe as mp
from tqdm import tqdm

VIDEOS_DIR = r"C:\Users\shahr\OneDrive\Desktop\Self Study\Sign-Language\data\raw\raw_videos"
OUTPUT_DIR = r"C:\Users\shahr\OneDrive\Desktop\Self Study\Sign-Language\data\processed"
os.makedirs(OUTPUT_DIR, exist_ok=True)

mp_holistic = mp.solutions.holistic

# Function to extract and flatten keypoints from a MediaPipe result
def extract_keypoints(results):
    def flatten_landmarks(landmarks, expected_len):
        if landmarks:
            return np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in landmarks]).flatten()
        else:
            return np.zeros(expected_len * 4)

    pose = flatten_landmarks(results.pose_landmarks.landmark if results.pose_landmarks else None, 33)
    left_hand = flatten_landmarks(results.left_hand_landmarks.landmark if results.left_hand_landmarks else None, 21)
    right_hand = flatten_landmarks(results.right_hand_landmarks.landmark if results.right_hand_landmarks else None, 21)

    return np.concatenate([pose, left_hand, right_hand])  # shape = (300,)

# Process each video file
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    for video_file in tqdm(os.listdir(VIDEOS_DIR)):
        if not video_file.endswith(".mp4"):
            continue

        video_path = os.path.join(VIDEOS_DIR, video_file)
        cap = cv2.VideoCapture(video_path)

        all_frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(frame)
            keypoints = extract_keypoints(results)
            all_frames.append(keypoints)

        cap.release()

        # Save landmarks as .npy
        video_id = os.path.splitext(video_file)[0]
        np.save(os.path.join(OUTPUT_DIR, f"{video_id}.npy"), np.array(all_frames))
