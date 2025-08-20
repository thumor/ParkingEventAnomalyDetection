import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import os

# --- 1. 設定 ---
input_feature_file = "output/trajectory_features_roi.csv"
output_scored_file = "output/trajectory_features_with_scores.csv"

# 選擇用來訓練模型的特徵欄位
# 我們排除了 'filename' 和 'event_type' 這些非數字特徵
FEATURES = [
    'duration_frames', 
    'total_distance', 
    'displacement', 
    'tortuosity', 
    'avg_speed', 
    'speed_std'
]

# --- 2. 主程式 ---
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

    # 讀取特徵資料
    df = pd.read_csv(input_feature_file)
    print(f"步驟 1: 成功讀取 {len(df)} 筆特徵資料。")

    # 準備訓練資料 (只使用我們選擇的特徵欄位)
    # 我們假設大部分數據是正常的，所以用全部數據來訓練模型找出異常點
    X = df[FEATURES]

    # 初始化並訓練隔離森林模型
    # contamination 參數可以理解為「預期的異常比例」，設為 'auto' 是 scikit-learn 的推薦做法
    # random_state 設為一個固定數字，可以保證每次執行的結果都一樣
    print("步驟 2: 正在訓練隔離森林 (Isolation Forest) 模型...")
    model = IsolationForest(contamination='auto', random_state=42)
    model.fit(X)
    print("模型訓練完成。")

    # 進行預測並獲取異常分數
    # decision_function 會給出異常分數，分數越低代表越異常
    print("步驟 3: 正在計算每個事件的異常分數...")
    anomaly_scores = model.decision_function(X)
    
    # 將分數加回原始的 DataFrame
    df['anomaly_score'] = anomaly_scores
    print("分數計算完成。")

    # 將帶有分數的結果儲存到新的 CSV 檔案
    df.to_csv(output_scored_file, index=False)
    print(f"步驟 4: 已將帶有異常分數的結果儲存至 '{output_scored_file}'")

    # 找出最可疑的事件並顯示出來
    print("\n--- 檢測結果分析 ---")
    
    # 分別處理 'entry' 和 'exit'
    for event_type in ['entry', 'exit']:
        print(f"\n【最可疑的 10 個 '{event_type}' 事件 (分數由低到高)】:")
        
        # 篩選出特定類型的事件，並根據異常分數排序
        suspicious_events = df[df['event_type'] == event_type].sort_values(by='anomaly_score').head(10)
        
        # 為了方便查看，只顯示關鍵欄位
        print(suspicious_events[['filename', 'anomaly_score', 'duration_frames', 'tortuosity', 'speed_std']])

# --- 執行主程式 ---
if __name__ == '__main__':
    main()