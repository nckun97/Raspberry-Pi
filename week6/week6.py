import cv2
import requests
import numpy as np

RP_IP = "172.20.10.3" 
URL = f"http://{RP_IP}:5000/upload"

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.resize(frame, (640, 480))
    _, img_encoded = cv2.imencode('.jpg', frame)
    
    try:
        # 將影像推送到樹莓派，並等待回傳 (設定 Timeout 為 5 秒以涵蓋推論時間)
        response = requests.post(URL, data=img_encoded.tobytes(), timeout=5.0)
        
        # 驗證 HTTP 狀態碼
        if response.status_code == 200:
            # 讀取樹莓派回傳的二進位影像並解碼
            nparr = np.frombuffer(response.content, np.uint8)
            result_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if result_img is not None:
                # 在筆電端安全地調用 GUI 顯示結果
                cv2.imshow('RPi YOLO Detection', result_img)
        else:
            print(f"Server returned status: {response.status_code}")

    except Exception as e:
        print(f"Network or Timeout Error: {e}")

    # 維持 GUI 事件迴圈
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()