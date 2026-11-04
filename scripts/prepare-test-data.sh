#!/usr/bin/env bash
# Downloads the official Silero VAD ONNX models and whisper.cpp's ggml-silero
# files, converts them to GGUF, and computes ONNX Runtime reference
# probabilities for the test audio. Output: models/ (untracked).
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PY:-.venv/bin/python}
if [ ! -x "$PY" ]; then
    uv venv -q .venv
    uv pip install -q --python .venv/bin/python onnxruntime onnx numpy soundfile scipy gguf kaldi-native-fbank transformers
fi

mkdir -p models/official models/ref
for v in v5.1.2 v6.2; do
    f=models/official/silero_vad_$v.onnx
    [ -f "$f" ] || curl -sfL -o "$f" "https://github.com/snakers4/silero-vad/raw/$v/src/silero_vad/data/silero_vad.onnx"
    (cd tools && ../$PY convert_silero_onnx.py ../$f ../models/silero-$v.gguf $v)
done
mkdir -p models/official/firered
for f in fireredvad_stream_vad.onnx fireredvad_vad.onnx cmvn.ark; do
    [ -f models/official/firered/$f ] || curl -sfL -o models/official/firered/$f "https://github.com/FireRedTeam/FireRedVAD/raw/main/pretrained_models/onnx_models/$f"
done
(cd tools && ../$PY convert_firered_onnx.py ../models/official/firered/fireredvad_stream_vad.onnx ../models/official/firered/cmvn.ark ../models/firered-stream-vad.gguf)
(cd tools && ../$PY convert_firered_onnx.py ../models/official/firered/fireredvad_vad.onnx ../models/official/firered/cmvn.ark ../models/firered-vad.gguf)
mkdir -p models/official/smart-turn
f=models/official/smart-turn/smart-turn-v3.2-gpu.onnx
[ -f "$f" ] || curl -sfL -o "$f" "https://huggingface.co/pipecat-ai/smart-turn-v3/resolve/main/smart-turn-v3.2-gpu.onnx"
(cd tools && ../$PY convert_smart_turn_onnx.py ../$f ../models/smart-turn-v3.2.gguf v3.2)
for v in v5.1.2 v6.2.0; do
    f=models/ggml-silero-$v.bin
    [ -f "$f" ] || curl -sfL -o "$f" "https://huggingface.co/ggml-org/whisper-vad/resolve/main/ggml-silero-$v.bin"
done

# quantized variants (quality not guaranteed; tests only report their deviation)
for m in silero-v6.2 firered-stream-vad firered-vad smart-turn-v3.2; do
    for t in F16 Q8_0; do
        $PY scripts/quantize-gguf.py models/$m.gguf models/$m-$(echo $t | tr A-Z a-z).gguf --type $t > /dev/null
    done
done

# a deliberately invalid file: "pos" is used by an add, not a mul_mat -> vad_model_load must refuse it
$PY scripts/quantize-gguf.py models/smart-turn-v3.2.gguf models/invalid-smart-turn-pos-q8_0.gguf --type F32 --tensor-type 'pos=Q8_0' > /dev/null

(cd tools && ../$PY make_reference.py ../assets/jfk.wav ../models/ref ../models/official/silero_vad_v5.1.2.onnx ../models/official/silero_vad_v6.2.onnx ../models/official/firered/fireredvad_stream_vad.onnx ../models/official/firered/fireredvad_vad.onnx ../models/official/smart-turn/smart-turn-v3.2-gpu.onnx)

# Nemotron-3-Diarization (~500 MB): the checkpoint, NVIDIA's own GGUF (loaded as is),
# a four-speaker test recording from MIT-licensed voices, and references from the
# official transformers implementation (needs PyTorch: set VAD_SKIP_DIARIZATION=1 to skip)
if [ "${VAD_SKIP_DIARIZATION:-0}" != 1 ]; then
    if ! $PY -c "import torch, librosa" 2> /dev/null; then
        uv pip install -q --python $PY torch --index-url https://download.pytorch.org/whl/cpu
        uv pip install -q --python $PY librosa safetensors
    fi
    d=models/official/nemotron
    mkdir -p $d models/official/audio
    for f in config.json processor_config.json model.safetensors Nemotron-3-Diarization.q8_0.gguf; do
        [ -f $d/$f ] || curl -sfL -o $d/$f "https://huggingface.co/nvidia/Nemotron-3-Diarization/resolve/main/$f"
    done
    for f in en-Alice_woman.wav en-Carter_man.wav en-Frank_man.wav; do
        [ -f models/official/audio/$f ] || curl -sfL -o models/official/audio/$f "https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/$f"
    done
    (cd tools && ../$PY convert_sortformer.py ../$d ../models/nemotron-3-diarization.gguf)
    a=models/official/audio
    $PY tools/make_conversation.py assets/jfk.wav $a/en-Alice_woman.wav $a/en-Carter_man.wav $a/en-Frank_man.wav models/ref/conversation.f32
    $PY tools/make_reference_sortformer.py $d models/ref/conversation.f32 models/ref/conversation.sortformer
fi
echo "test data ready in models/"
