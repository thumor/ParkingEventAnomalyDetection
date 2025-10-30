import pandas as pd
import os

csv_input_file = 'video_list_source.csv'
excel_output_file = 'video_url_list.xlsx'

# --- 主程式 ---
def main():
    """
    讀取原始 CSV 檔案，整理成乾淨的格式，並儲存為 Excel 檔案。
    """
    print("--- 腳本開始執行 ---")
    
    print(f"步驟 1: 正在檢查檔案 '{csv_input_file}' 是否存在...")
    if not os.path.exists(csv_input_file):
        print(f"*** 錯誤：找不到輸入的 CSV 檔案 '{csv_input_file}' ***")
        print("請確認您已將 CSV 檔名更改為 'video_list_source.csv'，並且它與本腳本在同一個資料夾中。")
        return # 找不到檔案就直接退出

    print("檔案存在，繼續執行。")
    print("步驟 2: 正在使用 pandas 讀取 CSV 檔案...")
    
    try:
        df = pd.read_csv(csv_input_file)
        print("CSV 檔案讀取成功。")
    except Exception as e:
        print(f"*** 錯誤：讀取 CSV 檔案時發生問題：{e} ***")
        return

    print("步驟 3: 正在整理資料 (分類、標籤)...")
    df_exit = pd.DataFrame({'url': df['Type 0']}).dropna()
    df_exit['event_type'] = 'exit'

    df_entry = pd.DataFrame({'url': df['Type 1']}).dropna()
    df_entry['event_type'] = 'entry'

    df_final = pd.concat([df_entry, df_exit], ignore_index=True)
    df_final['filename'] = df_final['url'].str.split('/').str[-1]
    print("資料整理完成。")

    print(f"步驟 4: 正在將結果寫入 Excel 檔案 '{excel_output_file}'...")
    try:
        df_final.to_excel(excel_output_file, index=False, engine='openpyxl')
        print(f"--- 成功！已將 {len(df_final)} 筆資料儲存至 '{excel_output_file}' ---")
        print("\n資料預覽:")
        print(df_final.head())
    except Exception as e:
        print(f"*** 錯誤：寫入 Excel 檔案時發生問題：{e} ***")

# --- 執行主程式 ---
if __name__ == '__main__':
    main()
