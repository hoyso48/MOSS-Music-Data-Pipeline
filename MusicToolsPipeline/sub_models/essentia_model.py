# -*- coding: utf-8 -*-
"""
Essentia 音高与调性提取模型
"""
from typing import List
from .base_model import BaseModel
from audio_info import AudioInfo
import logging
import numpy as np
import io
import os

logger = logging.getLogger(__name__)


class EssentiaModel(BaseModel):
    """使用 Essentia 提取旋律与调性"""

    def _load_model(self):
        try:
            import essentia
            import essentia.standard as es
            self.es = es
        except Exception as e:
            logger.error(f"Failed to init Essentia: {e}")
            raise

    def generate(self, inputs, **kwargs):
        # 支持 AudioInfo 列表或字符串列表（向后兼容）
        from typing import Union
        results: List[AudioInfo] = []
        for item in inputs:
            # 如果输入是字符串，转换为 AudioInfo
            if isinstance(item, str):
                audio_info = AudioInfo(audio_path=item)
            elif isinstance(item, AudioInfo):
                audio_info = item
            else:
                # 尝试从字典创建 AudioInfo
                audio_info = AudioInfo.from_dict(item) if isinstance(item, dict) else AudioInfo(audio_path=str(item))
            
            if audio_info._extra is None:
                audio_info._extra = {}
            
            # 优先使用 audio_bytes（Lance 数据集），直接解码为 numpy array
            audio = None
            
            if audio_info.audio_bytes is not None and isinstance(audio_info.audio_bytes, bytes):
                # 如果有 bytes 数据，直接解码为 numpy array（避免临时文件）
                try:
                    import soundfile as sf
                    import librosa
                    
                    # 使用 soundfile 解码 bytes → numpy
                    audio_bytes_io = io.BytesIO(audio_info.audio_bytes)
                    audio_array, sample_rate = sf.read(audio_bytes_io, dtype='float32')
                    
                    # 转换为单声道
                    if len(audio_array.shape) > 1:
                        audio_array = np.mean(audio_array, axis=1)
                    
                    # 确保是 float32
                    audio_array = audio_array.astype(np.float32)
                    
                    # 重采样到 44100 Hz（如果需要）
                    if sample_rate != 44100:
                        audio_array = librosa.resample(
                            audio_array,
                            orig_sr=sample_rate,
                            target_sr=44100
                        )
                    
                    audio = audio_array
                except Exception as e:
                    audio_info._extra["essentia_error"] = f"Failed to decode audio_bytes: {e}"
                    results.append(audio_info)
                    continue
            
            # 使用文件路径
            elif audio_info.audio_path or audio_info.url or audio_info.path:
                audio_path = audio_info.audio_path or audio_info.url or audio_info.path
                try:
                    audio = self.es.MonoLoader(filename=audio_path, sampleRate=44100)()
                except Exception as e:
                    audio_info._extra["essentia_error"] = f"Failed to load audio from file: {e}"
                    results.append(audio_info)
                    continue
            else:
                audio_info.error = audio_info.error or "No audio_path or audio_bytes in input"
                results.append(audio_info)
                continue
            
            # 如果还没有音频数据，跳过
            if audio is None:
                audio_info.error = audio_info.error or "No valid audio data found"
                results.append(audio_info)
                continue
            
            try:
                # 确保音频格式正确（单声道，float32）
                if len(audio.shape) > 1:
                    audio = np.mean(audio, axis=0)
                if audio.dtype != np.float32:
                    audio = audio.astype(np.float32)
                
                # 直接使用 numpy array 传给算法，不需要 MonoLoader！
                pitch, pitch_confidence = self.es.PitchMelodia()(audio)
                voiced = pitch[pitch > 0]
                melody_summary = {
                    "pitch_mean_hz": float(voiced.mean()) if voiced.size else 0.0,
                    "pitch_median_hz": float(np.median(voiced)) if voiced.size else 0.0,
                    "num_voiced_frames": int((pitch > 0).sum()),
                }
                key, scale, strength = self.es.KeyExtractor()(audio)
                key_summary = {"key": key, "scale": scale, "strength": float(strength)}
                audio_info._extra["melody"] = melody_summary
                audio_info._extra["key"] = key_summary
            except Exception as e:
                audio_info._extra["essentia_error"] = str(e)
            results.append(audio_info)
        return results

    def generate_batch(self, batch_data: List[AudioInfo]) -> List[AudioInfo]:  # type: ignore[override]
        """Essentia 直接接受 AudioInfo 列表，不使用基类的字符串提取逻辑。"""
        normalized_inputs: List[AudioInfo] = []
        for item in batch_data:
            if isinstance(item, AudioInfo):
                audio_info = item
            elif isinstance(item, dict):
                audio_info = AudioInfo.from_dict(item)
            else:
                # 假设是音频路径字符串
                audio_info = AudioInfo(audio_path=str(item))
            if audio_info._extra is None:
                audio_info._extra = {}
            normalized_inputs.append(audio_info)
        return self.generate(normalized_inputs)

    def cleanup(self):
        # 无需特殊清理
        pass


