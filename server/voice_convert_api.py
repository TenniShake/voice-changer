import argparse
import os
import shutil
import sys
import tempfile
import threading
import wave
from math import gcd
from typing import Literal

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import uvicorn
from starlette.background import BackgroundTask
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from scipy.signal import resample_poly

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from mods.log_control import VoiceChangaerLogger
import voice_changer.VoiceChangerManager as voice_changer_manager_module
from voice_changer.VoiceChangerManager import VoiceChangerManager
from voice_changer.VoiceChangerParamsManager import VoiceChangerParamsManager
from voice_changer.utils.VoiceChangerParams import VoiceChangerParams


Mode = Literal["neutral", "male", "deep_male"]

TARGET_SAMPLE_RATE = 48000
CONVERT_CHUNK_SIZE = 48000
TARGET_RMS_DBFS = -24.0
MAX_NORMALIZE_GAIN = 128.0
MODE_TO_TRANSPOSE = {
    "neutral": 0,
    "male": -5,
    "deep_male": -10,
}


app = FastAPI(title="Voice Convert API")
convert_lock = threading.Lock()


def create_voice_changer_params(model_dir: str = "model_dir", sample_mode: str = "production"):
    return VoiceChangerParams(
        model_dir=model_dir,
        content_vec_500="pretrain/checkpoint_best_legacy_500.pt",
        content_vec_500_onnx="pretrain/content_vec_500.onnx",
        content_vec_500_onnx_on=True,
        hubert_base="pretrain/hubert_base.pt",
        hubert_base_jp="pretrain/rinna_hubert_base_jp.pt",
        hubert_soft="pretrain/hubert/hubert-soft-0d54a1f4.pt",
        nsf_hifigan="pretrain/nsf_hifigan/model",
        sample_mode=sample_mode,
        crepe_onnx_full="pretrain/crepe_onnx_full.onnx",
        crepe_onnx_tiny="pretrain/crepe_onnx_tiny.onnx",
        rmvpe="pretrain/rmvpe.pt",
        rmvpe_onnx="pretrain/rmvpe.onnx",
        whisper_tiny="pretrain/whisper_tiny.pt",
    )


def get_voice_changer_manager() -> VoiceChangerManager:
    manager = getattr(app.state, "voice_changer_manager", None)
    if manager is None:
        params = create_voice_changer_params()
        VoiceChangerParamsManager.get_instance().setParams(params)
        voice_changer_manager_module.getSampleInfos = lambda _mode: []
        manager = VoiceChangerManager.get_instance(params)
        app.state.voice_changer_manager = manager
    return manager


@app.on_event("startup")
def startup():
    VoiceChangaerLogger.get_instance().initialize(initialize=False)
    get_voice_changer_manager()


@app.post("/api/voice/convert")
async def convert_voice(
    audio: UploadFile = File(...),
    mode: Mode = Form(...),
):
    if not audio.filename.lower().endswith(".wav"):
        raise HTTPException(status_code=400, detail="audio must be a .wav file")

    manager = get_voice_changer_manager()
    if manager.voiceChanger is None:
        raise HTTPException(
            status_code=503,
            detail="voice changer model is not loaded; load/select a model slot before converting",
        )

    work_dir = tempfile.mkdtemp(prefix="voice_convert_")
    input_path = os.path.join(work_dir, "input.wav")
    output_path = os.path.join(work_dir, "converted.wav")

    try:
        with open(input_path, "wb") as output:
            output.write(await audio.read())

        sample_rate, samples = read_pcm16_wav(input_path)
        samples = ensure_sample_rate(samples, sample_rate, TARGET_SAMPLE_RATE)

        with convert_lock:
            manager.update_settings("inputSampleRate", TARGET_SAMPLE_RATE)
            manager.update_settings("outputSampleRate", TARGET_SAMPLE_RATE)
            manager.update_settings("tran", MODE_TO_TRANSPOSE[mode])
            converted = convert_samples(manager, samples)

        converted = normalize_output_level(converted, samples)
        write_pcm16_wav(output_path, TARGET_SAMPLE_RATE, converted)
        return FileResponse(
            output_path,
            media_type="audio/wav",
            filename="converted.wav",
            background=BackgroundTask(shutil.rmtree, work_dir, ignore_errors=True),
        )
    except Exception:
        shutil.rmtree(work_dir, ignore_errors=True)
        raise


def read_pcm16_wav(path: str) -> tuple[int, np.ndarray]:
    with wave.open(path, "rb") as wav:
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        sample_rate = wav.getframerate()
        frame_count = wav.getnframes()
        raw = wav.readframes(frame_count)

    if sample_width != 2:
        raise HTTPException(status_code=400, detail="only 16-bit PCM WAV is supported")
    if channels < 1:
        raise HTTPException(status_code=400, detail="invalid WAV channel count")

    samples = np.frombuffer(raw, dtype="<i2")
    if channels > 1:
        samples = samples.reshape(-1, channels).astype(np.int32).mean(axis=1)

    return sample_rate, samples.astype(np.int16)


def write_pcm16_wav(path: str, sample_rate: int, samples: np.ndarray):
    clipped = np.clip(samples, np.iinfo(np.int16).min, np.iinfo(np.int16).max).astype("<i2")
    with wave.open(path, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(clipped.tobytes())


def ensure_sample_rate(samples: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate:
        return samples.astype(np.int16)

    divisor = gcd(source_rate, target_rate)
    up = target_rate // divisor
    down = source_rate // divisor
    resampled = resample_poly(samples.astype(np.float32), up, down)
    return np.clip(resampled, np.iinfo(np.int16).min, np.iinfo(np.int16).max).astype(np.int16)


def convert_samples(manager: VoiceChangerManager, samples: np.ndarray) -> np.ndarray:
    outputs = []
    for offset in range(0, len(samples), CONVERT_CHUNK_SIZE):
        chunk = samples[offset : offset + CONVERT_CHUNK_SIZE]
        if len(chunk) == 0:
            continue
        if len(chunk) < CONVERT_CHUNK_SIZE:
            chunk = np.pad(chunk, (0, CONVERT_CHUNK_SIZE - len(chunk)), mode="constant")

        converted, _performance = manager.changeVoice(chunk.astype(np.int16))
        if len(converted) > 1:
            outputs.append(converted.astype(np.int16))

    if not outputs:
        raise HTTPException(status_code=500, detail="voice conversion returned no audio")

    converted = np.concatenate(outputs)
    return converted[: len(samples)]


def normalize_output_level(converted: np.ndarray, source: np.ndarray) -> np.ndarray:
    source_rms = rms(source)
    converted_rms = rms(converted)
    if source_rms <= 0 or converted_rms <= 0:
        return converted

    target_rms = max(source_rms, dbfs_to_amplitude(TARGET_RMS_DBFS))
    gain = min(target_rms / converted_rms, MAX_NORMALIZE_GAIN)

    peak = float(np.max(np.abs(converted.astype(np.float64))))
    if peak > 0:
        gain = min(gain, 32767.0 / peak)

    return np.clip(
        converted.astype(np.float64) * gain,
        np.iinfo(np.int16).min,
        np.iinfo(np.int16).max,
    ).astype(np.int16)


def rms(samples: np.ndarray) -> float:
    values = samples.astype(np.float64)
    return float(np.sqrt(np.mean(values * values))) if values.size else 0.0


def dbfs_to_amplitude(dbfs: float) -> float:
    return float(32768.0 * (10.0 ** (dbfs / 20.0)))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    uvicorn.run("voice_convert_api:app", host=args.host, port=args.port, reload=False)
