import cv2
import numpy as np
from flask import Flask, request, Response
from ultralytics import YOLO

app = Flask(__name__)
model = YOLO('yolov8n.pt')

@app.route('/upload', methods=['POST'])
def upload():
    file = request.data
    nparr = np.frombuffer(file, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is not None:
        # 1. 執行推論
        results = model(img, stream=False, conf=0.4)
        
        # 2. 在記憶體中繪製 Bounding Box
        annotated_frame = results[0].plot()
        
        # 3. 將繪製完成的矩陣編碼為 JPG 格式
        success, encoded_img = cv2.imencode('.jpg', annotated_frame)
        if success:
            # 4. 作為 HTTP Response 回傳給筆電
            return Response(encoded_img.tobytes(), mimetype='image/jpeg')

    return "Processing Failed", 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=False)