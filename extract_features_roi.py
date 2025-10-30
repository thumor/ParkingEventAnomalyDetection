import pandas as pd
import numpy as np
import os
import glob

# 讀取 ROI 設定檔
roi_config_file = "roi_config.csv" 
# 軌跡資料來源
base_data_dir = "output/trajectories_data"
output_feature_file = "output/trajectory_features_roi.csv"

def calculate_features(df_trajectory):
    if len(df_trajectory) < 5: # 提高對最短軌跡長度的要求
        return None 
    points = df_trajectory[['x', 'y']].to_numpy()
    distances = np.sqrt(np.sum(np.diff(points, axis=0)**2, axis=1))
    frame_diffs = np.diff(df_trajectory['frame'])
    speeds = distances / np.where(frame_diffs == 0, 1, frame_diffs)
    features = {}
    features['duration_frames'] = int(df_trajectory['frame'].max() - df_trajectory['frame'].min())
    features['total_distance'] = np.sum(distances)
    start_point = points[0]
    end_point = points[-1]
    features['displacement'] = np.sqrt(np.sum((end_point - start_point)**2))
    if features['displacement'] > 1e-6: # 避免除以非常小的數
        features['tortuosity'] = features['total_distance'] / features['displacement']
    else:
        features['tortuosity'] = 1.0
    features['avg_speed'] = np.mean(speeds)
    features['speed_std'] = np.std(speeds)
    return features

def main():
    print("--- ROI 特徵提取腳本開始執行 ---")

    # 讀取 ROI 設定，並轉換為字典以便快速查找
    if not os.path.exists(roi_config_file):
        print(f"*** 錯誤: 找不到 ROI 設定檔 '{roi_config_file}' ***")
        return
    df_roi = pd.read_csv(roi_config_file)
    roi_dict = {row.filename: (row.roi_x1, row.roi_y1, row.roi_x2, row.roi_y2) for index, row in df_roi.iterrows()}
    print(f"步驟 1: 成功讀取 {len(roi_dict)} 筆 ROI 設定。")

    all_features = []
    
    # 只處理在 ROI 設定檔中列出的影片
    for video_filename, roi_coords in roi_dict.items():
        trajectory_csv_name = f"data_{os.path.splitext(video_filename)[0]}.csv"
        possible_paths = [
            os.path.join(base_data_dir, 'entry', trajectory_csv_name),
            os.path.join(base_data_dir, 'exit', trajectory_csv_name)
        ]
        
        file_path = None
        for path in possible_paths:
            if os.path.exists(path):
                file_path = path
                break
        
        if file_path is None:
            print(f"警告: 找不到 '{video_filename}' 的軌跡資料檔案，跳過。")
            continue

        df_traj = pd.read_csv(file_path)
        
        roi_x1, roi_y1, roi_x2, roi_y2 = roi_coords
        df_traj_filtered = df_traj[
            (df_traj['x'] >= roi_x1) & (df_traj['x'] <= roi_x2) &
            (df_traj['y'] >= roi_y1) & (df_traj['y'] <= roi_y2)
        ]
        
        features = calculate_features(df_traj_filtered)
        
        if features:
            features['filename'] = trajectory_csv_name
            features['event_type'] = os.path.basename(os.path.dirname(file_path))
            all_features.append(features)

    print(f"步驟 2: 完成 {len(all_features)} 筆資料的特徵計算。")

    df_features = pd.DataFrame(all_features)
    cols_order = ['filename', 'event_type', 'duration_frames', 'total_distance', 
                  'displacement', 'tortuosity', 'avg_speed', 'speed_std']
    df_features = df_features[cols_order]

    df_features.to_csv(output_feature_file, index=False)
    
    print(f"\n--- 成功！已將 ROI 過濾後的特徵儲存至 '{output_feature_file}' ---")
    print("\n新特徵資料預覽:")
    print(df_features.head())

if __name__ == '__main__':
    main()
