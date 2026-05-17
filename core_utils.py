import numpy as np
import json
import os
from scipy.spatial.distance import cosine
from fastdtw import fastdtw

# COCO indices for "Major Joints"
MAJOR_JOINTS = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14]

# COCO Skeleton connections for drawing lines
SKELETON_LINKS = [
    (15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11), (6, 12), (5, 6),
    (5, 7), (6, 8), (7, 9), (8, 10), (1, 2), (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6)
]

def normalize_keypoints(kpts, bbox):
    """Normalize kpts based on BBox center and scale."""
    x1, y1, x2, y2 = bbox[:4]
    center = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
    scale = max(x2 - x1, y2 - y1) + 1e-6
    
    norm_kpts = (kpts[:, :2] - center) / scale
    return np.hstack([norm_kpts, kpts[:, 2:3]])

def analyze_workout_type(user_norm_seq, refs_dir='/home/hanan/Documents/RTMPose/reference_videos'):
    """Finds best match by comparing user to all JSONs in the folder."""
    best_match = "Unknown"
    min_dist = float('inf')
    best_ref_data = None
    best_path = None

    if not os.path.exists(refs_dir):
        return best_match, None, None

    for file in os.listdir(refs_dir):
        if file.endswith('.json'):
            with open(os.path.join(refs_dir, file), 'r') as f:
                ref_json = json.load(f)
            
            ref_norm_seq = np.array(ref_json['data'])
            u_feat = [f[MAJOR_JOINTS, :2].flatten() for f in user_norm_seq]
            r_feat = [f[MAJOR_JOINTS, :2].flatten() for f in ref_norm_seq]
            
            dist, path = fastdtw(u_feat, r_feat, dist=cosine)
            
            if dist < min_dist:
                min_dist = dist
                best_match = ref_json['workout']
                best_ref_data = ref_norm_seq
                best_path = path
                
    return best_match, best_ref_data, best_path

def calculate_rolling_metrics(u_norms, window=60):
    """Calculates velocity and rolling standard deviation for fatigue."""
    velocities = [np.linalg.norm(u_norms[i][:, :2] - u_norms[i-1][:, :2]) for i in range(1, len(u_norms))]
    total_movement = np.sum(velocities)
    
    rolling_std = []
    for i in range(len(velocities) - window + 1):
        window_slice = velocities[i : i + window]
        rolling_std.append(np.std(window_slice))
        
    return velocities, rolling_std, total_movement