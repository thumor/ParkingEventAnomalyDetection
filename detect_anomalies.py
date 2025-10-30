import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import os

input_feature_file = "output/trajectory_features_roi.csv"
output_scored_file = "output/trajectory_features_with_scores.csv"

FEATURES = [
    'duration_frames', 
    'total_distance', 
    'displacement', 
    'tortuosity', 
    'avg_speed', 
    'speed_std'
]

def main():
    """
    讀取特徵檔案，訓練隔離森林模型，找出異常事件。
    """
    print("--- 異常檢測腳本開始執行 ---")

    # 檢查特徵檔案是否存在
    if not os.path.exists(input_feature_file):
        print(f"*** 錯誤: 找不到特徵檔案 '{input_feature_file}' ***")
        print("請先執行 extract_features.py 來產生這個檔案。")
        return

    df = pd.read_csv(input_feature_file)
    print(f"步驟 1: 成功讀取 {len(df)} 筆特徵資料。")

    X = df[FEATURES]

    print("步驟 2: 正在訓練隔離森林 (Isolation Forest) 模型...")
    model = IsolationForest(contamination='auto', random_state=42)
    model.fit(X)
    print("模型訓練完成。")

    print("步驟 3: 正在計算每個事件的異常分數...")
    anomaly_scores = model.decision_function(X)
    
    df['anomaly_score'] = anomaly_scores
    print("分數計算完成。")

    df.to_csv(output_scored_file, index=False)
    print(f"步驟 4: 已將帶有異常分數的結果儲存至 '{output_scored_file}'")

    print("\n--- 檢測結果分析 ---")
    
    for event_type in ['entry', 'exit']:
        print(f"\n【最可疑的 10 個 '{event_type}' 事件 (分數由低到高)】:")
        
        suspicious_events = df[df['event_type'] == event_type].sort_values(by='anomaly_score').head(10)
        
        print(suspicious_events[['filename', 'anomaly_score', 'duration_frames', 'tortuosity', 'speed_std']])

# --- 執行主程式 ---
if __name__ == '__main__':
    main()
