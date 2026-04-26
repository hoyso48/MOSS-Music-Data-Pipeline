# -*- coding: utf-8 -*-
"""
音乐分析 CPU 流水线模型

提供两个 Pipeline 类：
- MusicCpuPipelineModel:   全曲处理（Chordino + BeatNet + Essentia），输出 chords/beatnet/key/melody
- MusicCpuLitePipelineModel: 切片处理（Chordino + Essentia key-only），输出 chords/key
"""
from typing import List
from .base_model import BaseModel
from audio_info import AudioInfo
import logging
import ray

logger = logging.getLogger(__name__)


class MusicCpuPipelineModel(BaseModel):
    """
    全曲 CPU 流水线：Chordino → BeatNet → Essentia
    输出 music_cpu: {chords, beatnet, melody, key}

    用于 Step 1b 全曲 MIR 特征提取。
    """

    def _load_model(self):
        try:
            from workers.model_worker import (
                create_chordino_worker,
                create_beatnet_worker,
                create_essentia_worker,
            )
            self.chordino_worker = create_chordino_worker(model_name="Chordino")
            self.beatnet_worker = create_beatnet_worker(model_name="BeatNet")
            self.essentia_worker = create_essentia_worker(model_name="Essentia")
        except Exception as e:
            logger.error(f"Failed to init MusicCpuPipelineModel: {e}")
            raise

    def generate(self, inputs: List[AudioInfo], **kwargs) -> List[AudioInfo]:
        for audio_info in inputs:
            if audio_info._extra is None:
                audio_info._extra = {}

        current_results = inputs

        # 1) Chordino
        try:
            chordino_future = self.chordino_worker.generate_batch.remote(current_results)
            current_results = ray.get(chordino_future, timeout=300.0)
            logger.debug(f"Chordino processed {len(current_results)} items")
        except Exception as e:
            logger.warning(f"Chordino failed: {e}, continuing")
            for ai in current_results:
                ai._extra.setdefault("chords_error", f"Chordino failed: {e}")

        # 2) BeatNet
        try:
            beatnet_future = self.beatnet_worker.generate_batch.remote(current_results)
            current_results = ray.get(beatnet_future, timeout=300.0)
            logger.debug(f"BeatNet processed {len(current_results)} items")
        except Exception as e:
            logger.warning(f"BeatNet failed: {e}, continuing")
            for ai in current_results:
                ai._extra.setdefault("beatnet_error", f"BeatNet failed: {e}")

        # 3) Essentia
        try:
            essentia_future = self.essentia_worker.generate_batch.remote(current_results)
            current_results = ray.get(essentia_future, timeout=300.0)
            logger.debug(f"Essentia processed {len(current_results)} items")
        except Exception as e:
            logger.warning(f"Essentia failed: {e}")
            for ai in current_results:
                ai._extra.setdefault("essentia_error", f"Essentia failed: {e}")

        for audio_info in current_results:
            if audio_info._extra is None:
                audio_info._extra = {}

            music_cpu = {}

            if "chords" in audio_info._extra:
                music_cpu["chords"] = audio_info._extra["chords"]
            if "chords_error" in audio_info._extra:
                music_cpu["chords_error"] = audio_info._extra["chords_error"]

            if "beatnet" in audio_info._extra:
                music_cpu["beatnet"] = audio_info._extra["beatnet"]
            if "beatnet_error" in audio_info._extra:
                music_cpu["beatnet_error"] = audio_info._extra["beatnet_error"]

            if "melody" in audio_info._extra:
                music_cpu["melody"] = audio_info._extra["melody"]
            if "key" in audio_info._extra:
                music_cpu["key"] = audio_info._extra["key"]
            if "essentia_error" in audio_info._extra:
                music_cpu["essentia_error"] = audio_info._extra["essentia_error"]

            audio_info._extra["music_cpu"] = music_cpu

            audio_info._extra.pop("chords", None)
            audio_info._extra.pop("chords_error", None)
            audio_info._extra.pop("beatnet", None)
            audio_info._extra.pop("beatnet_error", None)
            audio_info._extra.pop("melody", None)
            audio_info._extra.pop("key", None)
            audio_info._extra.pop("essentia_error", None)

        return current_results

    def generate_batch(self, batch_data: List[AudioInfo]) -> List[AudioInfo]:
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

    def cleanup(self):
        pass


class MusicCpuLitePipelineModel(BaseModel):
    """
    切片 CPU 流水线：Chordino → Essentia (key only)
    输出 music_cpu: {chords, key}  —— 不含 BeatNet / melody

    用于 Step 3b 对 SongFormer 切片做 key 分析。
    """

    def _load_model(self):
        try:
            from workers.model_worker import (
                create_chordino_worker,
                create_essentia_worker,
            )
            self.chordino_worker = create_chordino_worker(model_name="Chordino")
            self.essentia_worker = create_essentia_worker(model_name="Essentia")
        except Exception as e:
            logger.error(f"Failed to init MusicCpuLitePipelineModel: {e}")
            raise

    def generate(self, inputs: List[AudioInfo], **kwargs) -> List[AudioInfo]:
        for audio_info in inputs:
            if audio_info._extra is None:
                audio_info._extra = {}

        current_results = inputs

        # 1) Chordino
        try:
            chordino_future = self.chordino_worker.generate_batch.remote(current_results)
            current_results = ray.get(chordino_future, timeout=300.0)
            logger.debug(f"Chordino processed {len(current_results)} items")
        except Exception as e:
            logger.warning(f"Chordino failed: {e}, continuing")
            for ai in current_results:
                ai._extra.setdefault("chords_error", f"Chordino failed: {e}")

        # 2) Essentia (key only)
        try:
            essentia_future = self.essentia_worker.generate_batch.remote(current_results)
            current_results = ray.get(essentia_future, timeout=300.0)
            logger.debug(f"Essentia processed {len(current_results)} items")
        except Exception as e:
            logger.warning(f"Essentia failed: {e}")
            for ai in current_results:
                ai._extra.setdefault("essentia_error", f"Essentia failed: {e}")

        for audio_info in current_results:
            if audio_info._extra is None:
                audio_info._extra = {}

            music_cpu = {}

            if "chords" in audio_info._extra:
                music_cpu["chords"] = audio_info._extra["chords"]
            if "chords_error" in audio_info._extra:
                music_cpu["chords_error"] = audio_info._extra["chords_error"]

            if "key" in audio_info._extra:
                music_cpu["key"] = audio_info._extra["key"]
            if "essentia_error" in audio_info._extra:
                music_cpu["essentia_error"] = audio_info._extra["essentia_error"]

            audio_info._extra["music_cpu"] = music_cpu

            audio_info._extra.pop("chords", None)
            audio_info._extra.pop("chords_error", None)
            audio_info._extra.pop("key", None)
            audio_info._extra.pop("essentia_error", None)
            audio_info._extra.pop("melody", None)
            audio_info._extra.pop("beatnet", None)
            audio_info._extra.pop("beatnet_error", None)

        return current_results

    def generate_batch(self, batch_data: List[AudioInfo]) -> List[AudioInfo]:
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

    def cleanup(self):
        pass
