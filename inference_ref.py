import cv2
import json
import os
import numpy as np

import sys
# Change this to the actual folder where your 'mmpose' directory lives
# Based on your error, it looks like it's in /home/hanan/Documents/RTMPose/
sys.path.append('/home/hanan/Documents/RTMPose/mmpose') 

from mmpose.apis import init_model, inference_topdown
from core_utils import normalize_keypoints

# --- CONFIGURATION ---
VIDEO_FOLDER = '/home/hanan//Documents/mediapipe-pose-python/mediapipe-pose-python-master/data' 
OUTPUT_FOLDER = '/home/hanan/Documents/RTMPose/reference_videos'
CONFIG = '/home/hanan/Documents/RTMPose/mmpose/configs/body_2d_keypoint/rtmpose/coco/rtmpose-m_8xb256-50e-256x192.py'
CHECKPOINT = '/home/hanan/Documents/RTMPose/work_dirs/rtmpose_m_workout/epoch_50.pth'

if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

def bulk_process_references():
    model = init_model(CONFIG, CHECKPOINT, device='cuda:0')
    
    video_files = [f for f in os.listdir(VIDEO_FOLDER) if f.endswith(('.mp4', '.mov', '.avi'))]
    
    for video_name in video_files:
        video_path = os.path.join(VIDEO_FOLDER, video_name)
        workout_label = os.path.splitext(video_name)[0] # Filename becomes Workout Name
        
        cap = cv2.VideoCapture(video_path)
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        hop = max(1, int(video_fps / 30)) # Force 30 FPS sampling
        
        ref_frames = []
        f_idx = 0
        
        print(f"Processing: {workout_label}...")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            if f_idx % hop == 0:
                results = inference_topdown(model, frame)
                if results and len(results[0].pred_instances.keypoints) > 0:
                    inst = results[0].pred_instances
                    # Normalizing and storing
                    norm = normalize_keypoints(inst.keypoints[0], inst.bboxes[0])
                    ref_frames.append(norm.tolist())
            f_idx += 1
        
        # Save JSON
        with open(os.path.join(OUTPUT_FOLDER, f"{workout_label}.json"), "w") as f:
            json.dump({"workout": workout_label, "data": ref_frames}, f)
        
        cap.release()
        print(f"✅ Saved {workout_label}.json")

if __name__ == "__main__":
    bulk_process_references()