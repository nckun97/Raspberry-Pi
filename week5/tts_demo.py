##import sherpa_onnx
import time
import os

def main():
    # 檔案路徑與輸出設定
    model_dir = "./models/vits-piper-en_US-glados"
    output_filename = "output.wav"
    
    # 模型設定
    vits_config = sherpa_onnx.OfflineTtsVitsModelConfig(
        model=f"{model_dir}/en_US-glados.onnx",
        lexicon=f"{model_dir}/lexicon.txt",
        tokens=f"{model_dir}/tokens.txt",
        data_dir=f"{model_dir}/espeak-ng-data",
        noise_scale=0.667,
        noise_scale_w=0.8,
        length_scale=1.0
    )

    model_config = sherpa_onnx.OfflineTtsModelConfig(
        vits=vits_config,
        num_threads=4,
        debug=False # 儲存檔案時可以關閉 debug 讓畫面乾淨點
    )

    tts_final_config = sherpa_onnx.OfflineTtsConfig(
        model=model_config,
        max_num_sentences=1
    )

    # 初始化引擎
    tts = sherpa_onnx.OfflineTts(tts_final_config)

    text = "Increasing the number of threads did not lead to performance improvement and even resulted in performance degradation."
    
    print(f"正在執行合成: {text}")
    start_time = time.time()
    
    # 執行生成
    # 如果模型支援切換語者聲音，可以更改 sid 的值
    audio = tts.generate(text, sid=1)
    
    end_time = time.time()
    
    if len(audio.samples) > 0:
        # 儲存為音檔
        # 使用 sherpa_onnx 內建的 write_wave 函式
        # 參數依序為：(檔名, 採樣數據, 採樣率)
        sherpa_onnx.write_wave(output_filename, audio.samples, audio.sample_rate)
        
        duration = len(audio.samples) / audio.sample_rate
        print("-" * 30)
        print(f"合成成功！")
        print(f"輸出檔案: {os.path.abspath(output_filename)}")
        print(f"合成耗時: {end_time - start_time:.2f} 秒")
        print(f"語音長度: {duration:.2f} 秒")
        print(f"實時係數 (RTF): {(end_time - start_time) / duration:.4f}")
        print("-" * 30)
    else:
        print("合成失敗，未產生音訊樣本。")

if __name__ == "__main__":
    main()