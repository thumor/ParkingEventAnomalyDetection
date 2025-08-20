import cv2
import os
from ultralytics import YOLO
import numpy as np
import pandas as pd

# --- 1. 設定 ---
model = YOLO('yolov8s.pt')

# ## 改為設定影片來源資料夾 ##
input_video_dir = 'data/videos/exit' # 先處理 'entry' 資料夾
# input_video_dir = 'data/videos/exit' # 之後再換成 'exit'

# 設定輸出路徑
output_video_dir = f"output/trajectory_videos/{os.path.basename(input_video_dir)}"
output_data_dir = f"output/trajectories_data/{os.path.basename(input_video_dir)}"
os.makedirs(output_video_dir, exist_ok=True)
os.makedirs(output_data_dir, exist_ok=True)

# --- 2. 主程式邏輯 ---
def process_video(video_path, output_video_path, output_data_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"*** 錯誤: 無法開啟影片 {video_path} ***")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps == 0: # 修正 fps 為 0 的情況
        fps = 30 

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    trajectory_points = []
    frame_number = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, verbose=False)
        
        target_car_box = None
        max_area = 0

        for result in results:
            for box in result.boxes:
                if int(box.cls) == 2: # class 2 is 'car'
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    area = (x2 - x1) * (y2 - y1)
                    if area > max_area:
                        max_area = area
                        target_car_box = (x1, y1, x2, y2)
        
        if target_car_box is not None:
            x1, y1, x2, y2 = target_car_box
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            
            # 儲存 (影格編號, x, y)
            trajectory_points.append([frame_number, center_x, center_y])
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)

        if len(trajectory_points) > 1:
            # 只用座標點來繪圖
            points_for_drawing = np.array([p[1:] for p in trajectory_points], dtype=np.int32)
            cv2.polylines(frame, [points_for_drawing], isClosed=False, color=(0, 255, 255), thickness=2)

        out.write(frame)
        frame_number += 1
        # 為了批次處理，我們移除即時顯示
        # cv2.imshow('Trajectory Generation', frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

    # 釋放資源
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # 將軌跡資料儲存為 CSV
    if trajectory_points:
        df_trajectory = pd.DataFrame(trajectory_points, columns=['frame', 'x', 'y'])
        df_trajectory.to_csv(output_data_path, index=False)
        print(f"已儲存軌跡資料至: {output_data_path}")

def main():
    video_files = [f for f in os.listdir(input_video_dir) if f.endswith('.mp4')]
    for video_file in video_files:
        video_path = os.path.join(input_video_dir, video_file)
        output_video_path = os.path.join(output_video_dir, f"traj_{video_file}")
        output_data_path = os.path.join(output_data_dir, f"data_{os.path.splitext(video_file)[0]}.csv")
        
        print(f"\n--- 正在處理影片: {video_file} ---")
        process_video(video_path, output_video_path, output_data_path)

if __name__ == '__main__':
    main()