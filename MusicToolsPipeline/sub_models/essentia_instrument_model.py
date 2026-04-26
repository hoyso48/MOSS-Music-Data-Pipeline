# -*- coding: utf-8 -*-
"""
Essentia 乐器变化检测模型
基于 Discogs EffNet 嵌入 + MtgJamendo Instrument 分类器
"""

from __future__ import annotations

import io
import logging
import os
from typing import List, Optional

import numpy as np

from .base_model import BaseModel
from audio_info import AudioInfo

logger = logging.getLogger(__name__)


DEFAULT_DISCOGS_PATH = "/inspire/hdd/project/embodied-multimodality/public/wxw/MusicData_proc_pipeline/essentia/models/discogs-effnet-bs64-1.pb"
DEFAULT_INSTRUMENT_PATH = "/inspire/hdd/project/embodied-multimodality/public/wxw/MusicData_proc_pipeline/essentia/models/mtg_jamendo_instrument-discogs-effnet-1.pb"
INSTRUMENT_LABELS = [
    "accordion", "acousticbassguitar", "acousticguitar", "bass", "beat",
    "bell", "bongo", "brass", "cello", "clarinet", "classicalguitar", "computer",
    "doublebass", "drummachine", "drums", "electricguitar", "electricpiano",
    "flute", "guitar", "harmonica", "harp", "horn", "keyboard", "oboe",
    "orchestra", "organ", "pad", "percussion", "piano", "pipeorgan", "rhodes",
    "sampler", "saxophone", "strings", "synthesizer", "trombone", "trumpet",
    "viola", "violin", "voice"
]


class EssentiaInstrumentModel(BaseModel):
    """
    使用 Essentia TensorFlow 推理乐器激活并检测变化
    """

    def __init__(
        self,
        model_name: str,
        model_path: Optional[str] = None,
        discogs_model_path: Optional[str] = None,
        instrument_model_path: Optional[str] = None,
        discogs_output: str = "PartitionedCall:1",
        sample_rate: int = 16000,
        threshold: float = 0.5,
        return_probabilities: bool = False,
        **kwargs,
    ) -> None:
        self.sample_rate = sample_rate
        self.threshold = threshold
        self.return_probabilities = return_probabilities

        self.discogs_model_path = (
            discogs_model_path
            or os.environ.get("ESSENTIA_DISCOGS_MODEL_PATH")
            or os.environ.get("DISCOGS_MODEL_PATH")
            or DEFAULT_DISCOGS_PATH
        )
        self.instrument_model_path = (
            instrument_model_path
            or os.environ.get("ESSENTIA_INSTRUMENT_MODEL_PATH")
            or os.environ.get("INSTRUMENT_MODEL_PATH")
            or DEFAULT_INSTRUMENT_PATH
        )
        self.discogs_output = discogs_output

        if not os.path.exists(self.discogs_model_path):
            raise FileNotFoundError(f"Discogs embedding model not found: {self.discogs_model_path}")
        if not os.path.exists(self.instrument_model_path):
            raise FileNotFoundError(f"Instrument classifier model not found: {self.instrument_model_path}")

        self.es = None
        self.embedding_model = None
        self.instrument_model = None

        super().__init__(model_name=model_name, model_path=model_path, **kwargs)

    def _load_model(self) -> None:
        try:
            import essentia
            essentia.log.warningActive= False
            import essentia.standard as es

            self.es = es
            self.loader=es.MonoLoader()
            self.embedding_model = es.TensorflowPredictEffnetDiscogs(
                graphFilename=self.discogs_model_path,
                output=self.discogs_output,
            )
            self.instrument_model = es.TensorflowPredict2D(
                graphFilename=self.instrument_model_path
            )
            logger.info(
                "EssentiaInstrumentModel loaded (discogs=%s, instrument=%s)",
                self.discogs_model_path,
                self.instrument_model_path,
            )
        except Exception as exc:
            logger.error("Failed to initialize Essentia instrument models: %s", exc)
            raise

    def generate(self, inputs: List[AudioInfo], **kwargs) -> List[AudioInfo]:
        results: List[AudioInfo] = []
        for audio_info in inputs:
            if audio_info._extra is None:
                audio_info._extra = {}

            audio_array = self._load_audio_array(audio_info)
            if audio_array is None:
                results.append(audio_info)
                continue

            try:
                embeddings = self.embedding_model(audio_array)  # type: ignore[misc]
                predictions = np.array(self.instrument_model(embeddings))  # type: ignore[misc]
                change_summary = self._detect_changes(
                    predictions,
                    len(audio_array) / float(self.sample_rate),
                )

                audio_info._extra["music_gpu"] = change_summary
                if self.return_probabilities:
                    audio_info._extra["instrument_probabilities"] = predictions.tolist()
            except Exception as exc:
                logger.warning(
                    "Failed to run instrument inference for %s: %s",
                    audio_info.get_audio_identifier(),
                    exc,
                )
                audio_info._extra["music_gpu_error"] = str(exc)

            results.append(audio_info)
        return results

    def generate_batch(self, batch_data: List[AudioInfo]) -> List[AudioInfo]:  # type: ignore[override]
        normalized_inputs: List[AudioInfo] = []
        for item in batch_data:
            if isinstance(item, AudioInfo):
                audio_info = item
            elif isinstance(item, dict):
                audio_info = AudioInfo.from_dict(item)
            else:
                audio_info = AudioInfo(audio_path=str(item))
            if audio_info._extra is None:
                audio_info._extra = {}
            normalized_inputs.append(audio_info)
        return self.generate(normalized_inputs)

    def _load_audio_array(self, audio_info: AudioInfo) -> Optional[np.ndarray]:
        audio: Optional[np.ndarray] = None
        # 优先 audio_bytes
        if audio_info.audio_bytes is not None:
            try:
                import soundfile as sf

                buf = io.BytesIO(audio_info.audio_bytes if isinstance(audio_info.audio_bytes, bytes) else bytes(audio_info.audio_bytes))
                audio_array, sr = sf.read(buf, dtype="float32")
                if audio_array.ndim > 1:
                    audio_array = np.mean(audio_array, axis=1)
                if sr != self.sample_rate:
                    resampler = self.es.Resample(  # type: ignore[union-attr]
                        inputSampleRate=sr,
                        outputSampleRate=self.sample_rate,
                    )
                    audio_array = resampler(audio_array)
                audio = audio_array.astype(np.float32)
            except Exception as exc:
                audio_info._extra["music_gpu_error"] = f"Failed to decode audio bytes: {exc}"
                return None
        else:
            audio_path = audio_info.audio_path or audio_info.url or audio_info.path
            if not audio_path:
                audio_info._extra["music_gpu_error"] = audio_info._extra.get(
                    "music_gpu_error", "Missing audio_path/url/path"
                )
                return None
            try:
                self.loader.configure(filename=audio_path, sampleRate=self.sample_rate)  # type: ignore[union-attr]
                audio = self.loader()
            except Exception as exc:
                audio_info._extra["music_gpu_error"] = f"Failed to load audio: {exc}"
                return None

        if audio is None:
            audio_info._extra["music_gpu_error"] = "Unable to decode audio data"
            return None

        if audio.ndim > 1:
            audio = np.mean(audio, axis=0)
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        return audio

    def _detect_changes(self, predictions: np.ndarray, duration: float) -> dict:
        num_frames = predictions.shape[0]
        if num_frames == 0:
            return {"changes": [], "frame_count": 0, "duration_sec": round(duration, 3)}

        timestamps = np.linspace(0, duration, num_frames, endpoint=False)
        changes = []
        prev_active = set()

        for idx in range(num_frames):
            frame_pred = predictions[idx]
            active = {
                INSTRUMENT_LABELS[i]
                for i, prob in enumerate(frame_pred)
                if float(prob) >= self.threshold
            }
            if active != prev_active:
                changes.append(
                    {
                        "time": round(float(timestamps[idx]), 3),
                        "active": sorted(active),
                    }
                )
                prev_active = active

        return {
            "changes": changes,
            "frame_count": int(num_frames),
            "duration_sec": round(duration, 3),
        }

    def cleanup(self) -> None:
        """模型无额外资源需要清理"""
        self.embedding_model = None
        self.instrument_model = None
        self.es = None


