```markdown
# 🎤 Whisper Speech Recognition on Raspberry Pi (C/C++)

This project demonstrates speech recognition on Raspberry Pi using whisper.cpp.

---

## 🚀 Features

- C/C++ inference (no PyTorch)
- Runs on Raspberry Pi CPU
- Chinese speech recognition
- Lightweight and deployable

---

## ⚙️ Setup

```bash
git clone https://github.com/ggerganov/whisper.cpp
cd whisper.cpp
make
./models/download-ggml-model.sh base




▶️ Run

./run.sh

📊 Experiment

Models Tested
ggml-tiny.bin
ggml-base.bin
Results
Model	Time	Accuracy
Tiny	Fast (~51s)	Lower
Base	Slower (~113s)	Higher

🧠 Key Insight

Tiny model is fast but less accurate
Base model improves transcription quality
Trade-off between performance and accuracy is critical in embedded systems

🎯 Conclusion

On Raspberry Pi:

Tiny and Base models provide the best balance between performance and accuracy.

📌 Note

Model files are not included due to size limitations.
