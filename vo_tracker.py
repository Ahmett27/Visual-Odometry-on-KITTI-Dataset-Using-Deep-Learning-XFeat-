import os
import cv2
import numpy as np
import torch
import csv
import matplotlib.pyplot as plt

# --- KONFİGÜRASYON ---
DATASET_PATH = "/home/ahmet/Desktop/VS_CODE/KITTI_dataset"
POSES_PATH = "/home/ahmet/Desktop/VS_CODE/poses"
SEQUENCE_ID = "00"

# KITTI Sequence 00 için Kamera Intrinsic Matrisi
K_MATRIX = np.array([[718.856, 0.0, 607.192],
                     [0.0, 718.856, 185.215],
                     [0.0, 0.0, 1.0]], dtype=np.float32)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Çalışan cihaz: {device}")

# --- SINIFLAR VE FONKSİYONLAR ---

class KittiDataLoader:
    def __init__(self, dataset_path, poses_path, sequence_id):
        self.image_path = os.path.join(dataset_path, f"sequences/{sequence_id}/image_0")
        self.pose_path = os.path.join(poses_path, f"{sequence_id}.txt")
        
        self.images = sorted([os.path.join(self.image_path, f) for f in os.listdir(self.image_path) if f.endswith('.png')])
        
        self.gt_poses = []
        if os.path.exists(self.pose_path):
            with open(self.pose_path, 'r') as f:
                for line in f:
                    p = np.fromstring(line, sep=' ', dtype=np.float32)
                    self.gt_poses.append(p.reshape(3, 4))
        else:
            print(f"Uyarı: Ground Truth dosyası bulunamadı: {self.pose_path}")

    def __len__(self):
        return len(self.images)

    def get_frame(self, index):
        img = cv2.imread(self.images[index])
        gt_pose = self.gt_poses[index] if index < len(self.gt_poses) else None
        return img, gt_pose

class XFeatFeatureTracker:
    def __init__(self):
        print("XFeat modeli Torch Hub üzerinden (local önbellekten) yükleniyor...")
        self.xfeat = torch.hub.load('verlab/accelerated_features', 'XFeat', pretrained=True, top_k=2048).to(device).eval()
        self.matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

    def detect_and_compute(self, img):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
        
        with torch.no_grad():
            outputs = self.xfeat.detectAndCompute(img_tensor, top_k=2048)[0]
            
        keypoints = outputs['keypoints'].cpu().numpy() # Nx2
        descriptors = outputs['descriptors'].cpu().numpy() # Nx64
        return keypoints, descriptors

    def match_features(self, desc1, desc2, kp1, kp2):
        matches = self.matcher.knnMatch(desc1, desc2, k=2)
        
        good_matches = []
        pts1 = []
        pts2 = []
        
        ratio_threshold = 0.70
        for m, n in matches:
            if m.distance < ratio_threshold * n.distance:
                good_matches.append(m)
                pts1.append(kp1[m.queryIdx])
                pts2.append(kp2[m.trainIdx])
                
        return np.array(pts1), np.array(pts2), good_matches

def get_absolute_scale(gt_poses, frame_id):
    if frame_id == 0 or not gt_poses:
        return 1.0, (0.0, 0.0, 0.0)
    
    T_prev = gt_poses[frame_id - 1]
    T_curr = gt_poses[frame_id]
    
    prev_coords = T_prev[:, 3]
    curr_coords = T_curr[:, 3]
    
    scale = np.linalg.norm(curr_coords - prev_coords)
    return scale, tuple(curr_coords)

def draw_trajectory(canvas, est_pos, gt_pos, frame_id):
    scale_factor = 1.0 
    offset_x = 400
    offset_z = 100

    draw_x_est = int(est_pos[0] * scale_factor) + offset_x
    draw_z_est = int(est_pos[2] * scale_factor) + offset_z
    
    draw_x_gt = int(gt_pos[0] * scale_factor) + offset_x
    draw_z_gt = int(gt_pos[2] * scale_factor) + offset_z

    cv2.circle(canvas, (draw_x_est, draw_z_est), 1, (0, 0, 255), 2) 
    cv2.circle(canvas, (draw_x_gt, draw_z_gt), 1, (0, 255, 0), 2)   

    cv2.rectangle(canvas, (10, 10), (300, 70), (0,0,0), -1)
    text_est = f"Est: x:{est_pos[0]:.2f} y:{est_pos[1]:.2f} z:{est_pos[2]:.2f}"
    text_gt  = f"GT : x:{gt_pos[0]:.2f} y:{gt_pos[1]:.2f} z:{gt_pos[2]:.2f}"
    cv2.putText(canvas, f"Frame: {frame_id}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    cv2.putText(canvas, text_est, (20, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255), 1)
    cv2.putText(canvas, text_gt, (20, 63), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0), 1)

def main():
    if not os.path.exists(DATASET_PATH):
        print(f"Hata: Veri seti yolu bulunamadı: {DATASET_PATH}")
        return

    loader = KittiDataLoader(DATASET_PATH, POSES_PATH, SEQUENCE_ID)
    tracker = XFeatFeatureTracker()

    log_file_name = f"vo_log_seq_{SEQUENCE_ID}.csv"
    csv_file = open(log_file_name, mode='w', newline='')
    log_writer = csv.writer(csv_file)
    log_writer.writerow(["Frame", "Est_X", "Est_Y", "Est_Z", "GT_X", "GT_Y", "GT_Z"])

    prev_img, _ = loader.get_frame(0)
    h, w = prev_img.shape[:2]
    match_width = w * 2 
    match_height = h
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    match_video = cv2.VideoWriter('matches_output.mp4', fourcc, 10.0, (match_width, match_height))
    traj_video = cv2.VideoWriter('trajectory_output.mp4', fourcc, 10.0, (800, 800))

    traj_canvas = np.zeros((800, 800, 3), dtype=np.uint8) 
    cur_R = np.eye(3) 
    cur_t = np.zeros((3, 1)) 

    prev_kp, prev_desc = tracker.detect_and_compute(prev_img)

    print("Görsel Odometri başlatılıyor... Çıktılar .mp4 dosyası olarak kaydedilecek.")

    all_est_coords = []
    all_gt_coords = []

    for i in range(1, len(loader)):
        curr_img, gt_pose = loader.get_frame(i)
        
        curr_kp, curr_desc = tracker.detect_and_compute(curr_img)
        pts1, pts2, good_matches = tracker.match_features(prev_desc, curr_desc, prev_kp, curr_kp)

        if len(pts1) < 10:
            prev_kp, prev_desc, prev_img = curr_kp, curr_desc, curr_img
            continue

        E, mask = cv2.findEssentialMat(pts2, pts1, cameraMatrix=K_MATRIX, method=cv2.RANSAC, prob=0.999, threshold=0.5)
        
        if mask is None:
            prev_kp, prev_desc, prev_img = curr_kp, curr_desc, curr_img
            continue

        _, R, t, pose_mask = cv2.recoverPose(E, pts2, pts1, cameraMatrix=K_MATRIX, mask=mask)

        scale, gt_coords = get_absolute_scale(loader.gt_poses, i)

        if scale > 0.05: 
            cur_t = cur_t + scale * cur_R.dot(t)
            cur_R = cur_R.dot(R)

        est_coords = (cur_t[0][0], cur_t[1][0], cur_t[2][0])
        log_writer.writerow([i, est_coords[0], est_coords[1], est_coords[2], gt_coords[0], gt_coords[1], gt_coords[2]])

        all_est_coords.append(est_coords)
        all_gt_coords.append(gt_coords)

        match_canvas = cv2.drawMatches(prev_img, 
                                       [cv2.KeyPoint(p[0], p[1], 1) for p in prev_kp], 
                                       curr_img, 
                                       [cv2.KeyPoint(p[0], p[1], 1) for p in curr_kp], 
                                       good_matches, 
                                       None, 
                                       matchColor=(0, 255, 0), 
                                       singlePointColor=(0, 0, 255), 
                                       matchesMask=mask.ravel().tolist(), 
                                       flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        
        match_video.write(match_canvas)

        draw_trajectory(traj_canvas, est_coords, gt_coords, i)
        traj_video.write(traj_canvas)

        prev_img = curr_img
        prev_kp = curr_kp
        prev_desc = curr_desc

        if i % 100 == 0:
            print(f"İşlenen Frame: {i} / {len(loader)}")

    print(f"İşlem bitti! Log dosyası: {log_file_name}")
    print("Videolar oluşturuldu: 'matches_output.mp4' ve 'trajectory_output.mp4'")
    
    csv_file.close()
    match_video.release()
    traj_video.release()

    est_arr = np.array(all_est_coords)
    gt_arr = np.array(all_gt_coords)

    if not np.all(gt_arr == 0):
        errors = np.linalg.norm(gt_arr - est_arr, axis=1)
        rmse = np.sqrt(np.mean(errors**2))
        print(f"\n--- FİNAL SONUÇLARI ---")
        print(f"İncelenen Frame Sayısı: {len(est_arr)}")
        print(f"Kalibre Edilmiş RMSE Hata: {rmse:.4f} Metre")

        plt.figure(figsize=(8, 8))
        plt.plot(gt_arr[:, 0], gt_arr[:, 2], label='Ground Truth', color='black', linestyle='--')
        plt.plot(est_arr[:, 0], est_arr[:, 2], label='XFeat Estimated', color='blue')
        plt.title(f'KITTI {SEQUENCE_ID} VO (RMSE: {rmse:.2f}m)')
        plt.xlabel('X (m)')
        plt.ylabel('Z (m)')
        plt.legend()
        plt.axis('equal')
        plt.grid(True)
        plt.savefig('final_trajectory_plot.png')
        print("Final grafik 'final_trajectory_plot.png' olarak kaydedildi.")
    else:
        print("\nUYARI: Ground Truth verileri okunamadığı için RMSE hesaplanamadı.")

if __name__ == "__main__":
    main()