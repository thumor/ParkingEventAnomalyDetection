import pandas as pd
import numpy as np
import os
import glob

base_data_dir = "output/trajectories_data"
output_feature_file = "output/trajectory_features.csv"

def calculate_features(df_trajectory):
    """從軌跡 DataFrame 計算一組特徵"""
    if len(df_trajectory) < 2:
        return None # 軌跡太短，無法計算特徵

    points = df_trajectory[['x', 'y']].to_numpy()
    
    # 計算每一步的距離
    distances = np.sqrt(np.sum(np.diff(points, axis=0)**2, axis=1))
    
    # 計算每一步的速度 (距離/影格差)
    frame_diffs = np.diff(df_trajectory['frame'])
    # 避免除以零
    speeds = distances / np.where(frame_diffs == 0, 1, frame_diffs)

    # 計算特徵
    features = {}
    features['duration_frames'] = int(df_trajectory['frame'].max() - df_trajectory['frame'].min())
    features['total_distance'] = np.sum(distances)
    
    start_point = points[0]
    end_point = points[-1]
    features['displacement'] = np.sqrt(np.sum((end_point - start_point)**2))
    
    # 彎曲度：避免除以零
    if features['displacement'] > 0:
        features['tortuosity'] = features['total_distance'] / features['displacement']
    else:
        features['tortuosity'] = 1.0 # 如果位移為零，定義為直線
        
    features['avg_speed'] = np.mean(speeds)
    features['speed_std'] = np.std(speeds)
    
    return features

def main():
    """遍歷所有軌跡資料檔，計算特徵並存成一個 CSV"""
    print("--- 特徵提取腳本開始執行 ---")
    
    entry_files = glob.glob(os.path.join(base_data_dir, 'entry', '*.csv'))
    exit_files = glob.glob(os.path.join(base_data_dir, 'exit', '*.csv'))
    all_files = entry_files + exit_files
    
    if not all_files:
        print(f"*** 錯誤: 在 '{base_data_dir}' 中找不到任何軌跡資料 (.csv) 檔案 ***")
        print("請先執行升級版的 generate_trajectory.py 來產生資料。")
        return

    print(f"找到 {len(all_files)} 個軌跡資料檔案，開始計算特徵...")
    
    all_features = []
    
    for file_path in all_files:
        df_traj = pd.read_csv(file_path)
        
        # 計算特徵
        features = calculate_features(df_traj)
        
        if features:
            # 加入檔案基本資訊
            features['filename'] = os.path.basename(file_path)
            # 從路徑判斷事件類型
            features['event_type'] = os.path.basename(os.path.dirname(file_path))
            all_features.append(features)

    # 將所有特徵轉換為 DataFrame
    df_features = pd.DataFrame(all_features)
    
    # 調整欄位順序
    cols_order = ['filename', 'event_type', 'duration_frames', 'total_distance', 
                  'displacement', 'tortuosity', 'avg_speed', 'speed_std']
    df_features = df_features[cols_order]

    # 儲存為 CSV
    df_features.to_csv(output_feature_file, index=False)
    
    print(f"\n--- 成功！已將 {len(df_features)} 筆資料的特徵儲存至 '{output_feature_file}' ---")
    print("\n特徵資料預覽:")
    print(df_features.head())

if __name__ == '__main__':
    main()
