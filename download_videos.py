import pandas as pd
import os
import requests
from tqdm import tqdm

# --- 設定 ---
# 這個腳本會讀取我們上一步產生的 Excel 檔案
input_excel_file = 'video_url_list.xlsx'

# 設定影片下載後要存放的基礎路徑
base_output_dir = "data/videos"

# --- 主程式 ---
def download_video(url, filepath):
    """下載單一影片並顯示進度條"""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status() # 檢查請求是否成功

        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024 # 1KB

        with open(filepath, 'wb') as file, tqdm(
            desc=os.path.basename(filepath),
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
            leave=False
        ) as bar:
            for data in response.iter_content(block_size):
                bar.update(len(data))
                file.write(data)
        return True
    except requests.exceptions.RequestException as e:
        print(f"\n下載失敗: {url}, 錯誤: {e}")
        return False

def main():
    """
    讀取 Excel 清單，遍歷並下載所有影片到分類資料夾。
    """
    print("--- 影片下載腳本開始執行 ---")

    # 檢查 Excel 清單是否存在
    if not os.path.exists(input_excel_file):
        print(f"*** 錯誤: 找不到影片清單檔案 '{input_excel_file}' ***")
        print("請先執行 organize_urls.py 來產生這個檔案。")
        return

    # 讀取 Excel 檔案
    df = pd.read_excel(input_excel_file)
    total_videos = len(df)
    print(f"在 '{input_excel_file}' 中找到 {total_videos} 筆影片資料。")

    # 遍歷 DataFrame 中的每一行來下載影片
    download_count = 0
    for index, row in tqdm(df.iterrows(), total=total_videos, desc="總進度"):
        event_type = row['event_type']
        url = row['url']
        filename = row['filename']

        # 建立分類資料夾 (例如: data/videos/entry)
        output_dir = os.path.join(base_output_dir, event_type)
        os.makedirs(output_dir, exist_ok=True)

        # 組合完整的本地檔案路徑
        filepath = os.path.join(output_dir, filename)

        # 如果檔案已存在，則跳過
        if not os.path.exists(filepath):
            if download_video(url, filepath):
                download_count += 1
        else:
            # 如果檔案存在，我們也算它一個進度
            pass

    print(f"\n--- 全部處理完成 ---")
    print(f"總共需要下載 {total_videos} 個影片，本次新下載了 {download_count} 個影片。")
    print(f"影片已分類儲存於 '{os.path.join(base_output_dir, 'entry')}' 和 '{os.path.join(base_output_dir, 'exit')}' 資料夾中。")

# --- 執行主程式 ---
if __name__ == '__main__':
    main()