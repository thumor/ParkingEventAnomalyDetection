import cv2
import os

# --- 全域變數 ---
ref_point = []
drawing = False
roi_defined = False
frame_copy = None

def on_mouse(event, x, y, flags, param):
    """滑鼠回呼函式，處理滑鼠事件"""
    global ref_point, drawing, roi_defined, frame_copy

    if event == cv2.EVENT_LBUTTONDOWN:
        ref_point = [(x, y)]
        drawing = True
        roi_defined = False

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            # 畫出即時的矩形預覽
            temp_frame = frame_copy.copy()
            cv2.rectangle(temp_frame, ref_point[0], (x, y), (0, 255, 0), 2)
            cv2.imshow("Select ROI", temp_frame)

    elif event == cv2.EVENT_LBUTTONUP:
        ref_point.append((x, y))
        drawing = False
        roi_defined = True
        # 畫出最終的矩形
        cv2.rectangle(frame_copy, ref_point[0], ref_point[1], (0, 255, 0), 2)
        cv2.imshow("Select ROI", frame_copy)

def main():
    """主程式：讀取影片，讓使用者選擇 ROI"""
    global frame_copy
    
    # ##################################################################
    # ##  請修改這裡，填入您想定義 ROI 的影片路徑                   ##
    # ##################################################################
    video_path = 'data/videos/entry/ad806d5678df11f0b5ed48b02d5582d6.mp4' 
    
    if not os.path.exists(video_path):
        print(f"*** 錯誤: 找不到影片檔案 '{video_path}' ***")
        return

    cap = cv2.VideoCapture(video_path)
    # 只讀取影片的第一幀
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("*** 錯誤: 無法讀取影片的第一幀 ***")
        return

    frame_copy = frame.copy()
    window_name = "Select ROI"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, on_mouse)

    print("--- ROI 選擇工具 ---")
    print("操作說明:")
    print("1. 在彈出的視窗中，用滑鼠左鍵拖曳來畫出停車格的區域 (ROI)。")
    print("2. 畫好後，按下鍵盤 's' 鍵來儲存並印出座標。")
    print("3. 如果不滿意，可以按下 'r' 鍵來重置並重新畫。")
    print("4. 完成後，按下 'q' 鍵退出。")

    while True:
        cv2.imshow(window_name, frame_copy)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('r'): # 按 'r' 重置
            frame_copy = frame.copy()
            print("已重置，請重新畫 ROI。")
        
        elif key == ord('s'): # 按 's' 儲存
            if roi_defined:
                # 確保 x1 < x2 且 y1 < y2
                x1, y1 = ref_point[0]
                x2, y2 = ref_point[1]
                roi_x = min(x1, x2)
                roi_y = min(y1, y2)
                roi_w = abs(x2 - x1)
                roi_h = abs(y2 - y1)
                
                # 在終端機印出結果，格式為 CSV
                video_filename = os.path.basename(video_path)
                # 我們儲存左上角和右下角座標
                roi_x1, roi_y1 = roi_x, roi_y
                roi_x2, roi_y2 = roi_x + roi_w, roi_y + roi_h
                
                print("\n--- ROI 座標已儲存 ---")
                print("請將以下這行文字，複製到你的 roi_config.csv 檔案中:")
                print(f"{video_filename},{roi_x1},{roi_y1},{roi_x2},{roi_y2}")
                
            else:
                print("尚未定義 ROI，請先用滑鼠畫出區域。")

        elif key == ord('q'): # 按 'q' 退出
            break
            
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()