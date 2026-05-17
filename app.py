import streamlit as st
import cv2
import time
import tempfile
import numpy as np
import os
import shutil
import sys

# Making the script portable using relative paths
base_path = os.path.dirname(os.path.abspath(__file__))
mmpose_folder = os.path.join(base_path, 'mmpose')
sys.path.append(mmpose_folder)

from mmpose.apis import init_model, inference_topdown
from core_utils import (normalize_keypoints, analyze_workout_type, 
                        SKELETON_LINKS, calculate_rolling_metrics, MAJOR_JOINTS)

# Relative Configurations & Paths
CONFIG_PATH = os.path.join(base_path, 'mmpose/configs/body_2d_keypoint/rtmpose/coco/rtmpose-m_8xb256-50e-256x192.py')
CHECKPOINT_PATH = os.path.join(base_path, 'work_dirs/rtmpose_m_workout/epoch_50.pth')
REFS_DIR = os.path.join(base_path, 'reference_videos')

st.set_page_config(page_title="Kinematic AI Partner", layout="wide")
st.title("🏋️ Workout Analysis Dashboard")

@st.cache_resource
def load_rtm_model():
    try:
        return init_model(CONFIG_PATH, CHECKPOINT_PATH, device='cuda:0')
    except:
        return init_model(CONFIG_PATH, CHECKPOINT_PATH, device='cpu')

model = load_rtm_model()

TEMP_FRAME_DIR = "session_frames"
if os.path.exists(TEMP_FRAME_DIR):
    shutil.rmtree(TEMP_FRAME_DIR)
os.makedirs(TEMP_FRAME_DIR)

upload = st.file_uploader("Upload Your Video", type=['mp4', 'mov'])

if upload:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(upload.read())
    
    cap = cv2.VideoCapture(tfile.name)
    fps_in = cap.get(cv2.CAP_PROP_FPS)
    hop = max(1, int(fps_in / 30)) 
    
    user_data = []
    
    with st.status("Analyzing Movement at 30 FPS...", expanded=True) as status:
        frame_idx = 0
        saved_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            if frame_idx % hop == 0:
                results = inference_topdown(model, frame)
                if results and len(results[0].pred_instances.keypoints) > 0:
                    inst = results[0].pred_instances
                    raw_kpts = inst.keypoints[0] 
                    conf_scores = inst.keypoint_scores[0]
                    
                    kpts_with_conf = np.hstack([raw_kpts, conf_scores.reshape(-1, 1)])
                    norm_kpts = normalize_keypoints(kpts_with_conf, inst.bboxes[0])
                    
                    frame_path = os.path.join(TEMP_FRAME_DIR, f"frame_{saved_count}.jpg")
                    cv2.imwrite(frame_path, frame)
                    
                    user_data.append({
                        "epoch": time.time(),
                        "norm": norm_kpts,
                        "raw": kpts_with_conf, 
                        "frame_path": frame_path
                    })
                    saved_count += 1
            frame_idx += 1
        status.update(label="Analysis Complete!", state="complete", expanded=False)

    if user_data:
        u_norm_list = [f["norm"] for f in user_data]
        workout_name, ref_seq, path = analyze_workout_type(u_norm_list, refs_dir=REFS_DIR)
        velocities, rolling_std, total_movement = calculate_rolling_metrics(u_norm_list)

        if np.mean(velocities) > 0.08:
            st.warning("Workout is too fast! Slow down your reps for better muscle control.")
        
        if total_movement > 130:
            st.info("High intensity detected! Consider increasing protein intake today.")

        error_scores = []
        target_joints = [7, 8, 9, 10] if "bench" in workout_name.lower() else MAJOR_JOINTS
        
        for u_idx, r_idx in path:
            valid_dists = [np.linalg.norm(user_data[u_idx]["norm"][j, :2] - ref_seq[r_idx][j, :2]) 
                           for j in target_joints if user_data[u_idx]["norm"][j, 2] > 0.3]
            
            score = np.mean(valid_dists) if valid_dists else 0.0001
            error_scores.append((u_idx, r_idx, score))
        
        sorted_errors = sorted(error_scores, key=lambda x: x[2], reverse=True)
        top_5 = []
        used_indices = []
        for err in sorted_errors:
            if all(abs(err[0] - used) > 50 for used in used_indices):
                top_5.append(err)
                used_indices.append(err[0])
            if len(top_5) == 5: break

        st.subheader(f"Detected: :green[{workout_name}] | Activeness: {total_movement:.2f}")
        st.write("### 🚩 Key Form Analysis")
        
        cols = st.columns(5)
        for i, (u_idx, r_idx, score) in enumerate(top_5):
            u_f = user_data[u_idx]
            ref_f = ref_seq[r_idx]
            img = cv2.imread(u_f["frame_path"])
            
            for start_j, end_j in SKELETON_LINKS:
                if u_f["raw"][start_j, 2] > 0.3 and u_f["raw"][end_j, 2] > 0.3:
                    p1 = (int(u_f["raw"][start_j][0]), int(u_f["raw"][start_j][1]))
                    p2 = (int(u_f["raw"][end_j][0]), int(u_f["raw"][end_j][1]))
                    cv2.line(img, p1, p2, (255, 255, 255), 2)
            
            for j in target_joints:
                if u_f["raw"][j, 2] > 0.25: 
                    diff = np.linalg.norm(u_f["norm"][j, :2] - ref_f[j, :2])
                    color = (0, 255, 0) if diff < 0.15 else (0, 0, 255)
                    pos = (int(u_f["raw"][j][0]), int(u_f["raw"][j][1]))
                    cv2.circle(img, pos, 10, color, -1)
            
            status_text = "✅ Correct" if score < 0.15 else f"Error: {score:.2f}"
            cols[i].image(img, channels="BGR", caption=f"Rank {i+1} | {status_text}")

        st.divider()
        g1, g2 = st.columns(2)
        with g1:
            st.write("### Stability (Fatigue Tracking)")
            st.line_chart(rolling_std)
        with g2:
            st.write("### Repetition Pulse")
            y_vals = [f[10, 1] for f in u_norm_list]
            y_norm = (y_vals - np.min(y_vals)) / (np.max(y_vals) - np.min(y_vals) + 1e-6)
            st.line_chart(y_norm)