import sherpa_onnx
import time
import os

def create_tts(thread_count):
    # 模型路徑
    model_dir = "./models/vits-piper-en_US-glados"
    model_path = f"{model_dir}/en_US-glados.onnx"
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"找不到模型檔: {model_path}")

    vits_config = sherpa_onnx.OfflineTtsVitsModelConfig(
        model=model_path,
        lexicon=f"{model_dir}/en_US-glados.onnx",
        tokens=f"{model_dir}/tokens.txt",
        data_dir=f"{model_dir}/espeak-ng-data"
    )

    model_config = sherpa_onnx.OfflineTtsModelConfig(
        vits=vits_config,
        num_threads=4,
        debug=False
    )

    tts_config = sherpa_onnx.OfflineTtsConfig(model=model_config)
    return sherpa_onnx.OfflineTts(tts_config)

def main():
    text = "測試多執行緒效能。邊緣運算實驗開始。"
    # TODO: 請自己新增 thread!
    threads_to_test = [1,2,4] 
    
    print(f"{'Threads':<10} | {'Inference (s)':<15} | {'RTF':<10}")
    print("-" * 45)

    for t in threads_to_test:
        try:
            # 初始化引擎
            tts = create_tts(t)
            
            # 預熱一次 (Warm-up)，讓 CPU 進入狀態並分配好記憶體
            _ = tts.generate("預熱")
            
            # 正式計時推理
            start = time.time()
            audio = tts.generate(text)
            end = time.time()
            
            # 計算數據
            exec_time = end - start
            duration = len(audio.samples) / audio.sample_rate
            rtf = exec_time / duration
            
            print(f"{t:<10} | {exec_time:<15.4f} | {rtf:<10.4f}")
            
        except Exception as e:
            print(f"執行緒 {t} 測試出錯: {e}")

if __name__ == "__main__":
    main()