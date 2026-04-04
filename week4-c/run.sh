#!/bin/bash

./build/bin/whisper-cli \
  -m models/ggml-base.bin \
  -f ../audio/test_audio.wav \
  -l zh \
  --no-timestamps
