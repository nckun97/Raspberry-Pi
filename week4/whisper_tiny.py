import os
import time
from transformers import pipeline, logging

# 1. 隱藏不必要的 Transformers 警告訊息，讓終端機保持乾淨
logging.set_verbosity_error()

print("正在載入模型，這可能需要一點時間...")

# 2. 載入語音辨識 Pipeline
# 優化：device=cpu 語意比 -1 更清晰
# 優化：加入 chunk_length_s=30 避免長音檔造成樹莓派記憶體不足 (OOM)
asr = pipeline(
    "automatic-speech-recognition", 
    model="openai/whisper-tiny",
    device="cpu",
    chunk_length_s=30
)

audio_path = "test_audio.wav" 

if os.path.exists(audio_path):
    print("模型載入完成！正在處理語音，請稍候...")
    
    # 紀錄開始時間，方便評估樹莓派的處理速度
    start_time = time.time()
    
    # 3. 執行推理
    # 優化：加入 generate_kwargs 直接指定語言為繁體中文，並設定任務為轉錄
    # 這樣可以跳過模型前期的語言偵測步驟，加快處理速度
    try:
        result = asr(
            audio_path,
            generate_kwargs={"language": "chinese", "task": "transcribe"}
        )
        
        end_time = time.time()
        
        print("\n--- 語音辨識結果 ---")
        print(f"辨識內容 {result['text'].strip()}")
        print(f"處理耗時 {end_time - start_time:.2f} 秒")
        
    except Exception as e:
        print(f"處理語音時發生錯誤 {e}")

else:
    print(f"找不到音檔，請確認 {audio_path} 是否在正確的目錄下。")